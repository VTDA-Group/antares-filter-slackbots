# Functions for querying YSE and converting to ANTARES loci for processing.

import os
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path

import json
import datetime
from io import BytesIO

import numpy as np
from django.db.models import Q
import astropy.units as u
import astropy.coordinates as cd
from astropy.time import Time
from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module
import pandas as pd
import antares_client
from alerce.core import Alerce
from iinuclear.utils import get_galaxy_center, get_data, check_nuclear

from antares_filter_slackbots.antares_ranker import RankingFilter
from antares_filter_slackbots.slack_formatters import YSESlackPoster
from antares_filter_slackbots.quality_filters import standard_quality_check, yse_quality_check, relaxed_quality_check
from antares_filter_slackbots.auth import login_ysepz, password_ysepz


class Retriever:
    def __init__(self, lookback_days):
        self._lookback = lookback_days
        self.host_properties = [
            'host_redshift',
            'host_prob',
            'host_sep_arcsec',
            'best_host_catalog',
            'host_absmag',
            'best_redshift',
            'host_name',
            'best_cat',
            'high_host_confidence',
            'peak_abs_mag',
        ]
        
    def check_watch(self):
        """Update current time, in MJD.
        """
        self._mjd = Time.now().mjd
        
        
    def is_it_nuclear(self, ra, dec):
        """Applies isitnuclear package to determine
        whether loci are likely nuclear.
        """
        coord = (ra, dec)
        ras, decs, ztf_name, iau_name, catalog_result, _, _ = get_data(*coord, save_all=False)
        if (catalog_result is not None) and (len(catalog_result) > 0):
            ra_galaxy, dec_galaxy, error_arcsec = get_galaxy_center(catalog_result)
            _, _, nuclear_bool = check_nuclear(
                ras, decs, ra_galaxy, dec_galaxy, error_arcsec,
                p_threshold=0.05
            )
            if (nuclear_bool is None) or np.isnan(nuclear_bool):
                return False
            return nuclear_bool
        return False
        
        
        
class ANTARESRetriever(Retriever):
    """Class to retrieve ANTARES loci and convert to timeseries
    dataframes.
    """
    def __init__(self, lookback_days):
        super().__init__(lookback_days)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST.csv",
        )
        self._query = []
        self._alerce_client = Alerce()
        
        
    def add_alerce_phot(self, lc, ztf_name):
        """If there's a ZTF name, query ALeRCE because
        they probably have forced photometry!!
        TODO: once ANTARES adds forced phot, this will be removed.
        """
        forced_detections = self._alerce_client.query_forced_photometry(ztf_name, format="pandas")
        forced_detections.dropna(axis=1, inplace=True, ignore_index=True)
        
        if len(forced_detections) == 0:
            return lc

        alerce_lc = forced_detections[['mjd', 'fid', 'mag', 'e_mag']]
        alerce_lc['ant_ra'] = lc['ant_ra'].iloc[0]
        alerce_lc['ant_dec'] = lc['ant_dec'].iloc[0]
        
        alerce_lc.loc[alerce_lc.fid == 1, 'fid'] = 'g'
        alerce_lc.loc[alerce_lc.fid == 2, 'fid'] = 'R'
        alerce_lc.loc[alerce_lc.fid == 3, 'fid'] = 'i'
        
        alerce_lc.rename(
            columns={
                'mjd': 'ant_mjd',
                'fid': 'ant_passband',
                'mag': 'ant_mag',
                'e_mag': 'ant_magerr',
            },
            inplace=True
        )
        combined_lc = pd.concat([lc, alerce_lc], ignore_index=True)
        combined_lc.sort_values(by='ant_mjd', inplace=True)
        return combined_lc
    
    
    def quality_check(self, ts):
        return standard_quality_check(ts)
    
    def reset_query(self):
        """Reset query."""
        self._query = []
        
    def constrain_query_mjd(self):
        """Constrain MJD range of ANTARES query.
        """
        self._query.append(
            {
                "range": {
                    "properties.newest_alert_observation_time": {
                        "gte": self._mjd - self._lookback
                    }
                }
            }
        )

    def apply_filter(self, filt: RankingFilter):
        """Apply filter constraints to query."""
        self._query = filt.modify_query(self._query)
        
    def star_catalog_check(self, locus):
        """Check whether locus is in star or AGN catalog.
        """
        var_catalogs = [
            "gaia_dr3_variability",
            "sdss_stars",
            "bright_guide_star_cat",
            "asassn_variable_catalog_v2_20190802",
            "vsx",
            "linear_ll",
            "veron_agn_qso", # agn/qso
            "milliquas", # qso
        ]
        for cat in locus.catalog_objects:
            if cat in var_catalogs:
                return False
            
            if cat == 'gaia_dr3_gaia_source':
                info = locus.catalog_objects[cat][0]
                if (info['parallax'] is not None) and ~np.isnan(info['parallax']):
                    return False
            
        return True
    
    
    def run_query(self):
        """Run the query as-is. Returns
        list of loci."""
        full_query = {
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": self._query
                        }
                    }
                }
            }
        }
        return antares_client.search.search(full_query)
    
    
    def process_query_results(self, loci, filt: RankingFilter):
        """Filter query results by preprocessing checks.
        Returns dataframe.
        """
        processed_dicts = []
        if os.path.exists(self._prost_path):
            full_table = pd.read_csv(self._prost_path)
        else:
            full_table = None
            
        ts_dict = {}
            
        for i, locus in enumerate(loci):
            if i % 100 == 0:
                print(f"Processed {i} loci...")
                                
            if (full_table is not None) and (locus.locus_id in full_table['name'].to_numpy()):
                locus_dict = full_table.loc[
                    full_table['name'] == locus.locus_id,
                    ['name', 'ra', 'dec', 'nuclear']
                ].iloc[0].to_dict()
                
            else:
                if not self.star_catalog_check(locus):
                    continue
                    
                try:
                    ts = locus.lightcurve[[
                        'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                    ]].to_pandas() # may have to turn this back on for actual submission
                except:
                    ts = locus.lightcurve[[
                        'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                    ]]

                # removes rows with nan values (aka upper limits)
                ts.dropna(inplace=True, axis=0, ignore_index=True)
                
                if not self.quality_check(ts):
                    continue

                # Is it nuclear?
                nuclear_flag = self.is_it_nuclear(locus.ra, locus.dec)

                locus_dict = {
                    'name': locus.locus_id,
                    'ra': locus.ra,
                    'dec': locus.dec,
                    'nuclear': nuclear_flag,
                }
                
            try:
                lc = locus.lightcurve[[
                    'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                ]].to_pandas() # may have to turn this back on for actual submission
            except:
                lc = locus.lightcurve[[
                    'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                ]]
            lc.dropna(inplace=True, axis=0, ignore_index=True)
            new_lc = self.add_alerce_phot(lc, locus.properties['ztf_object_id'])
            ts_dict[locus.locus_id] = new_lc
            
            if 'tns_public_objects' in locus.catalog_objects:
                tns = locus.catalog_objects['tns_public_objects'][0]
                tns_name, tns_cls, tns_redshift = tns['name'], tns['type'], tns['redshift']
                #if tns_cls == 'SN Ia':
                #    continue # ignoring the Ia's because I can
                if tns_cls == '':
                    tns_cls = '---'
                if tns_redshift is None:
                    tns_redshift = np.nan
            else:
                tns_name, tns_cls, tns_redshift = '---', '---', np.nan
                
            locus_dict['tns_name'] = tns_name
            
            if not isinstance(tns_cls, str):
                print(tns_cls)
                tns_cls = '---'
                
            locus_dict['tns_class'] = tns_cls
            locus_dict['tns_redshift'] = tns_redshift
            locus_dict['brightest_alert_magnitude'] = locus.properties['brightest_alert_magnitude']
            locus_dict['peak_phase'] = self._mjd - locus.properties['brightest_alert_observation_time']
            
            for p in filt._filt.input_properties:
                if (p not in locus_dict) and (p not in self.host_properties):
                    locus_dict[p] = locus.properties[p]
            
            print(locus_dict)
            processed_dicts.append(locus_dict)
                    
        if len(processed_dicts) == 0:
            return None, None
            
        locus_df = pd.DataFrame.from_records(processed_dicts)
        return locus_df, ts_dict
    
    
    def retrieve_candidates(self, filt, max_num):
        """Generate candidate df for the ranker. MUST return a dataframe
        with set columns.
        """
        print("Running ANTARESRanker for filter "+filt.name)
        filt.setup()
        self.reset_query()
        self.check_watch()
        self.constrain_query_mjd()
        self.apply_filter(filt)
        loci = self.run_query()
        out1, out2 = self.process_query_results(loci, filt)
        return out1, out2
        

    
class YSERetriever(Retriever):
    """Class to retrieve YSE data and convert to timeseries dataframes.
    """
    def __init__(self, lookback_days=1.):
        """All URLs go here."""
        super().__init__(lookback_days)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST_YSE.csv"
        )
        
        self._base_url  = "https://ziggy.ucolick.org/yse/api"
        self._current_time = datetime.datetime.utcnow()
        self._current_mjd = Time.now().mjd
        self._instrument_names = [
            "https://ziggy.ucolick.org/yse/api/instruments/21/",
            "https://ziggy.ucolick.org/yse/api/instruments/83/",
        ]
        # bands with GPC1 and GPC2 links as instruments ^
        self._gpc_bands = [
            "https://ziggy.ucolick.org/yse/api/photometricbands/14/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/15/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/36/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/37/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/39/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/40/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/41/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/54/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/66/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/71/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/121/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/216/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/217/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/218/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/219/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/220/",
            "https://ziggy.ucolick.org/yse/api/photometricbands/221/"
        ]
        
        self._yse_tag = "https://ziggy.ucolick.org/yse/api/transienttags/29/"
        
        self._ignore_status = [
            "https://ziggy.ucolick.org/yse/api/transientstatuses/8/", # needs template
            "https://ziggy.ucolick.org/yse/api/transientstatuses/5/", # follow-up done
            "https://ziggy.ucolick.org/yse/api/transientstatuses/2/", # follow-up requested already
            "https://ziggy.ucolick.org/yse/api/transientstatuses/3/", # ignore
        ]
                
        self.save_prefix = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            ) # src
        ), "data", "yse")
        os.makedirs(self.save_prefix, exist_ok=True)
        
        self._auth = HTTPBasicAuth(login_ysepz, password_ysepz)
        
        
    def quality_check(self, ts):
        return yse_quality_check(ts)
        
        
    def retrieve_yse_photometry(self, transient_name):
        """Retrieve YSE photometry for a given transient name.
        """
        url = f'https://ziggy.ucolick.org/yse/download_photometry/{transient_name}'
        phot_data = requests.get(url, auth=self._auth).content
        try:
            df = pd.read_table(BytesIO(phot_data), sep='\s+', comment='#')
        except:
            return None
        ts = df[['MJD', 'FLT', 'MAG', 'MAGERR']]
        ts.rename(
            columns={
                'MJD': 'ant_mjd',
                'MAG': 'ant_mag',
                'MAGERR': 'ant_magerr',
                'FLT': 'ant_passband',
            },
            inplace=True
        )
        # if magerr == 5.0 or is negative, not a valid datapoint
        mask = (ts['ant_magerr'] == 5.0) | (ts['ant_magerr'] <= 0.0)
        ts = ts.loc[~mask]
        # removes rows with nan values (aka upper limits)
        ts.dropna(inplace=True, axis=0, ignore_index=True)
        return ts
    
    
    def query_all_yse_recent(self):
        """Query all YSE transients with last detection in a certain time range."""
        updated_transients = set()
        filt_date_oldest = (
            datetime.datetime.utcnow()-datetime.timedelta(200.)
        ).replace(microsecond=0).isoformat() + "Z"
        filt_date_newest = (
            datetime.datetime.utcnow()-datetime.timedelta(self._lookback)
        ).replace(microsecond=0).isoformat() + "Z"
            
        # get transients within that range
        transient_url = os.path.join(
            self._base_url,
            f"transients/?created_date_gte={filt_date_oldest}"
        )
        transient_url += "&tag_in=YSE&limit=10000"
        transient_url += f"&modified_date_gte={filt_date_newest}"

        response = requests.get(transient_url, auth=self._auth).json()
        try:
            transients = response['results']
        except:
            return []

        transient_names = []

        for t in transients:
            if self._yse_tag not in t['tags']:
                continue
            if self.parse_modified_mjd(t['modified_date']) < self._mjd - self._lookback:      
                print(t['modified_date'])
                continue
            if t['status'] in self._ignore_status:
                continue
            if t["TNS_spec_class"] == 'SN Ia':
                continue

            transient_names.append(t['name'])
            
        print(len(transient_names))

        #updated_transients.update(transient_names)
            
        return transient_names
    
    
    def parse_modified_mjd(self, date_str):
        """Extract MJD from modified_date."""
        t = Time(date_str, format='isot')
        return t.mjd
    
    
    def brightest_time(self, ts):
        """Find brightest time."""
        bright_idx = (ts['ant_mag'] + ts['ant_magerr']).idxmin()
        return ts.loc[bright_idx, 'ant_mjd']
    
    
    def get_tns_info(self, transient_name):
        """Get TNS info from transient (plus RA and DEC)."""
        transient_url = os.path.join(
            self._base_url, f"transients/?name={transient_name}"
        )
        transient = requests.get(transient_url, auth=self._auth).json()['results'][0]
        spec_url = transient['best_spec_class']
        
        if spec_url is not None:
            spec_class = requests.get(spec_url, auth=self._auth).json()['name']
        else:
            spec_class = None
            
        if "YSE" in transient['name']:
            tns_name = None
        else:
            tns_name = transient['name']
            
        return tns_name, spec_class, transient['redshift'], transient['ra'], transient['dec']
    
    
    def retrieve_candidates(self, filt, max_num):
        """Main loop to retrieve YSE transients from 
        the night, get light curves, quality check, and return df.
        """
        self.check_watch()
        processed_dicts = []
        
        if os.path.exists(self._prost_path):
            full_table = pd.read_csv(self._prost_path)
        else:
            full_table = None
            
        nightly_transients = self.query_all_yse_recent()
        ts_dict = {}

        for i, transient in enumerate(nightly_transients):
            
            if i % 10 == 0:
                print(f"Pre-processed {i} transients...")
                                 
            ts = self.retrieve_yse_photometry(transient)
            if ts is None:
                continue
                
            detections = ts.loc[ts['ant_magerr'] < 0.362]
            
            if self._mjd - detections['ant_mjd'].max() > self._lookback:
                continue
                
            if not self.quality_check(detections):
                continue
                                                
            if (full_table is not None) and (transient in full_table['name'].to_numpy()):
                locus_dict = full_table.loc[
                    full_table['name'] == transient,
                    ['name', 'ra', 'dec', 'tns_name', 'tns_class', 'tns_redshift', 'nuclear']
                ].iloc[0].to_dict()
                
            else:
                tns_name, tns_cls, tns_redshift, ra, dec = self.get_tns_info(transient)

                # Is it nuclear?
                nuclear_flag = self.is_it_nuclear(ra, dec)

                locus_dict = {
                    'name': transient,
                    'ra': ra,
                    'dec': dec,
                    'tns_name': tns_name,
                    'tns_class': tns_cls,
                    'tns_redshift': tns_redshift,
                    'nuclear': nuclear_flag,
                }
                
            ts_dict[transient] = detections
            locus_dict['brightest_alert_magnitude'] =(detections['ant_mag'] + detections['ant_magerr']).min()
            locus_dict['peak_phase'] = self._mjd - self.brightest_time(detections)
            """
            if filt._filt is not None:
                for p in filt._filt.input_properties:
                    if (p not in locus_dict) and (p not in self.host_properties):
                        locus_dict[p] = locus.properties[p]
            """
                
            processed_dicts.append(locus_dict)
            
        if len(processed_dicts) == 0:
            return None, None
            
        locus_df = pd.DataFrame.from_records(processed_dicts)
        return locus_df, ts_dict

    
class RelaxedANTARESRetriever(ANTARESRetriever):
    def __init__(self, lookback_days=1.0):
        """Retriever with relaxed constraints. Used currently
        to find precursor emission candidates.
        """
        super().__init__(lookback_days=lookback_days)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST_relaxed.csv"
        )
        
        
    def star_catalog_check(self, locus):
        """Check whether locus is in star or AGN catalog.
        """
        var_catalogs = [
            "bright_guide_star_cat",
        ]
        for cat in locus.catalog_objects:
            if cat in var_catalogs:
                return False
            
            info = locus.catalog_objects[cat][0]
            if cat == 'linear_ll':
                if (info['dist'] is not None) and ~np.isnan(info['dist']):
                    if info['dist'] < 20_000:
                        return False
            
            if cat in ['veron_agn_qso', 'milliquas']:
                if (info['z'] is not None) and ~np.isnan(info['z']):
                    if info['z'] > 0.03:
                        return False
            
            if cat == 'gaia_dr3_gaia_source':
                info = locus.catalog_objects[cat][0]
                if (info['parallax'] is not None) and ~np.isnan(info['parallax']):
                    return False
                if (info['distance_gspphot'] is not None) and ~np.isnan(info['distance_gspphot']):
                    if info['distance_gspphot'] < 20_000: # in milky way
                        return False
            
        return True
    
    def quality_check(self, ts):
        return relaxed_quality_check(ts)
    
    

class ArchivalYSERetriever(YSERetriever):
    def query_all_yse_recent(self):
        """Query all YSE transients with last detection in a certain time range."""
        updated_transients = set()
            
        # get transients within that range
        transient_url = os.path.join(
            self._base_url,
            f"transients/?tag_in=YSE&limit=1000"
        )

        response = requests.get(transient_url, auth=self._auth).json()
        
        try:
            transients = response['results']
        except:
            transients = []

        transient_names = []

        while len(transients) > 0:
            for t in transients:
                if t['name'][:4].isnumeric(): # TNS
                    print(t['name'])
                    continue
                if self._yse_tag not in t['tags']:
                    continue
                #if self.parse_modified_mjd(t['modified_date']) < self._mjd - self._lookback:      
                #    print(t['modified_date'])
                #    continue
                if t['status'] in self._ignore_status:
                    continue
                if t["TNS_spec_class"] == 'SN Ia':
                    continue

                transient_names.append(t['name'])
            
            print(response['next'])
            if response['next'] is not None:
                response = requests.get(transient_url, auth=self._auth).json()
                try:
                    transients = response['results']
                except:
                    transients = []
            
        print(len(transient_names))

        #updated_transients.update(transient_names)
            
        return transient_names
    
    
    
class TestANTARESRetriever(RelaxedANTARESRetriever):
    def __init__(self):
        """Retriever with relaxed constraints. Used currently
        to find precursor emission candidates.
        """
        super().__init__(lookback_days=1.0)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST_test.csv"
        )
        
    def retrieve_candidates(self, filt, loci):
        """Generate candidate df for the ranker. MUST return a dataframe
        with set columns.
        """
        print("Running ANTARESRanker for filter "+filt.name)
        filt.setup()
        self.reset_query()
        self.check_watch()
        out1, out2 = self.process_query_results(loci, filt)
        return out1, out2
    
    
if __name__ == "__main__":
    retriever = YSERetriever(lookback_days=1.)
    df = retriever.run(10)