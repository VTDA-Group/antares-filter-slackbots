# Functions for querying YSE and converting to ANTARES loci for processing.

import os
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path
import multiprocessing as mp
import zipfile
import time

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
from snapi import Transient
from snapi.query_agents import TNSQueryAgent, ATLASQueryAgent

from antares_filter_slackbots.antares_ranker import RankingFilter
from antares_filter_slackbots.slack_formatters import YSESlackPoster
from antares_filter_slackbots.quality_filters import (
    standard_quality_check, yse_quality_check, relaxed_quality_check, atlas_quality_check
)
from antares_filter_slackbots.auth import login_ysepz, password_ysepz, tns_id, tns_name, tns_key


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

        self.var_catalogs = [
            "sdss_stars",
            'gaia_edr3_distances_bailer_jones',
            "vsx",
            "linear_ll",
            "veron_agn_qso", # agn/qso
            "milliquas", # qso
        ]
        
        
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
        if 'bright_guide_star_cat' in locus.catalog_objects:
            cls = locus.catalog_objects['bright_guide_star_cat'][0]['classification']
            if int(cls) == 0:
                return False
        if 'gaia_dr3_gaia_source' in locus.catalog_objects:
            info = locus.catalog_objects['gaia_dr3_gaia_source'][0]
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
                            "must": self._query,
                            "must_not":  [{
                                "terms": {
                                    "catalogs": self.var_catalogs
                                }
                            },],
                        }
                    }
                }
            }
        }
        return antares_client.search.search(full_query)
    

    def _process_single_locus(self, args):
        """
        Worker function to process one locus. 
        Receives a tuple (locus, full_table, host_properties, _mjd, filt_input_props).
        Returns (locus_id, new_lc, locus_dict) or None if skipped.
        """
        locus_dict, ts, properties, catalog_objects, filt_input_props = args

        if 'nuclear' not in locus_dict: # not already saved
            if not self.quality_check(ts):
                return None
            locus_dict['nuclear'] = None

        try:
            new_lc = self.add_alerce_phot(ts, properties['ztf_object_id'])
        except:
            new_lc = ts

        # 4) Extract TNS information
        if 'tns_public_objects' in catalog_objects:
            tns = catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls, tns_redshift = tns['name'], tns['type'], tns['redshift']
            if tns_cls == '':
                tns_cls = '---'
            if tns_redshift is None:
                tns_redshift = np.nan
        else:
            tns_name, tns_cls, tns_redshift = '---', '---', np.nan

        locus_dict['tns_name'] = tns_name
        if not isinstance(tns_cls, str):
            tns_cls = '---'
        locus_dict['tns_class'] = tns_cls
        locus_dict['tns_redshift'] = tns_redshift
        locus_dict['brightest_alert_magnitude'] = properties['brightest_alert_magnitude']
        locus_dict['peak_phase'] = self._mjd - properties['brightest_alert_observation_time']

        # 5) Add any other filter‐specific properties
        for p in filt_input_props:
            if p not in locus_dict and p not in self.host_properties:
                locus_dict[p] = properties.get(p, None)

        print(locus_dict['name'])
        return (locus_dict['name'], new_lc, locus_dict)

        
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
        worker_args = []

        # initial locus filter
        for i, locus in enumerate(loci):
            if i % 100 == 0:
                print(f"Processed {i} loci...")
                
            try:
                # 1) Check if locus in full_table
                if full_table is not None and (locus.locus_id in full_table['name'].to_numpy()):
                    row = full_table.loc[full_table['name'] == locus.locus_id, ['name', 'ra', 'dec', 'nuclear']].iloc[0]
                    locus_dict = row.to_dict()
                else:
                    # 2) Check star catalog
                    if not self.star_catalog_check(locus):
                        continue

                    locus_dict = {
                        'name': locus.locus_id,
                        'ra': locus.ra,
                        'dec': locus.dec,
                    }

                try:
                    ts = locus.lightcurve[[
                        'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                    ]].to_pandas()

                except:
                    ts = locus.lightcurve[[
                        'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
                    ]]

            except:
                continue
                        
            ts.dropna(inplace=True, axis=0, ignore_index=True)
            worker_args.append((locus_dict, ts, locus.properties, locus.catalog_objects, filt._filt.input_properties))

        # 4) Use 'spawn' start method to avoid fork issues with JAX
        ts_dict = {}
        processed_dicts = []
        parallelize = True # works better not on a distributed cluster
            
        if parallelize:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=10) as pool:
                # Use imap, which yields results one by one in order:
                for i, out in enumerate(pool.imap_unordered(self._process_single_locus, worker_args), start=1):
                    if i % 10 == 0 or i == len(worker_args):
                        print(f"  → main: completed {i}/{len(worker_args)} loci")

                    if out is None:
                        continue

                    locus_id, new_lc, locus_dict = out
                    ts_dict[locus_id] = new_lc
                    processed_dicts.append(locus_dict)
        
        else:
            for i, w in enumerate(worker_args):
                if i % 10 == 0 or i == len(worker_args):
                    print(f"  → main: completed {i}/{len(worker_args)} loci")
                out = self._process_single_locus(w)
                if out is None:
                    continue

                locus_id, new_lc, locus_dict = out
                ts_dict[locus_id] = new_lc
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
        transient_url += "&tag_in=YSE&limit=10_000"
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

            tns_name, tns_cls, tns_redshift, ra, dec = self.get_tns_info(transient)
            detections['ant_ra'] = ra
            detections['ant_dec'] = dec
                                                
            if (full_table is not None) and (transient in full_table['name'].to_numpy()):
                locus_dict = full_table.loc[
                    full_table['name'] == transient,
                    ['name', 'ra', 'dec', 'nuclear']
                ].iloc[0].to_dict()
                
            else:
                # Is it nuclear?
                locus_dict = {
                    'name': transient,
                    'ra': ra,
                    'dec': dec,
                    'nuclear': None,
                }
                
            ts_dict[transient] = detections

            locus_dict['tns_name'] = tns_name
            if not isinstance(tns_cls, str):
                tns_cls = '---'

            if not isinstance(tns_redshift, float):
                print(tns_redshift)
                tns_redshift = np.nan
                
            locus_dict['tns_class'] = tns_cls
            locus_dict['tns_redshift'] = tns_redshift
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
        self.var_catalogs = []
        
        
    def star_catalog_check(self, locus):
        """Check whether locus is in star or AGN catalog.
        """
        if "bright_guide_star_cat" in locus.catalog_objects:
            cls = locus.catalog_objects['bright_guide_star_cat'][0]['classification']
            if int(cls) == 0:
                return False

        if 'linear_ll' in locus.catalog_objects:
            info = locus.catalog_objects['linear_ll'][0]
            if (info['dist'] is not None) and ~np.isnan(info['dist']):
                if info['dist'] < 20_000:
                    return False
            
        if 'veron_agn_qso' in locus.catalog_objects:
            info = locus.catalog_objects['veron_agn_qso'][0]
            if (info['z'] is not None) and ~np.isnan(info['z']):
                if info['z'] > 0.03:
                    return False
                
        if 'milliquas' in locus.catalog_objects:
            info = locus.catalog_objects['milliquas'][0]
            if (info['z'] is not None) and ~np.isnan(info['z']):
                if info['z'] > 0.03:
                    return False
                    
        if 'gaia_dr3_gaia_source' in locus.catalog_objects:
            info = locus.catalog_objects['gaia_dr3_gaia_source'][0]
            if (info['parallax'] is not None) and ~np.isnan(info['parallax']):
                return False
            if (info['distance_gspphot'] is not None) and ~np.isnan(info['distance_gspphot']):
                if info['distance_gspphot'] < 20_000: # in milky way
                    return False
            
        return True
    
    def quality_check(self, ts):
        return relaxed_quality_check(ts)
    
    

class ArchivalYSERetriever(YSERetriever):
    def __init__(self, lookback_days=1.0):
        super().__init__(lookback_days)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST_archival.csv"
        )

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
                
            if not self.quality_check(detections):
                continue

            tns_name, tns_cls, tns_redshift, ra, dec = self.get_tns_info(transient)
            detections['ant_ra'] = ra
            detections['ant_dec'] = dec
                                                
            if (full_table is not None) and (transient in full_table['name'].to_numpy()):
                locus_dict = full_table.loc[
                    full_table['name'] == transient,
                    ['name', 'ra', 'dec', 'nuclear']
                ].iloc[0].to_dict()
                
            else:
                # Is it nuclear?
                nuclear_flag = None

                locus_dict = {
                    'name': transient,
                    'ra': ra,
                    'dec': dec,
                    'nuclear': nuclear_flag,
                }
                
            ts_dict[transient] = detections

            locus_dict['tns_name'] = tns_name
            if not isinstance(tns_cls, str):
                tns_cls = '---'
            if not isinstance(tns_redshift, float):
                print(tns_redshift)
                tns_redshift = np.nan
            locus_dict['tns_class'] = tns_cls
            locus_dict['tns_redshift'] = tns_redshift
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

    def query_all_yse_recent(self):
        """Query all YSE transients with last detection in a certain time range."""
        updated_transients = set()
            
        # get transients within that range
        transient_url = os.path.join(
            self._base_url,
            f"transients/?limit=1000"
        )

        response = requests.get(transient_url, auth=self._auth).json()
        
        try:
            transients = response['results']
        except:
            transients = []

        transient_names = []

        while len(transients) > 0:
            for t in transients:
                if 'YSE' not in t['name']: # TNS
                    print(t['name'])
                    continue
                transient_names.append(t['name'])
            
            if response['next'] is not None:
                print(response['next'])
                try:
                    response = requests.get(response['next'], auth=self._auth).json()
                    transients = response['results']
                except:
                    transients = []

            else:
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


class TNSRetriever(Retriever):
    """Grabs all new TNS events without ZTF names, adds ATLAS forced photometry (if possible),
    and applies quality cuts.
    """
    def __init__(self, lookback_days=1.):
        """All URLs go here."""
        super().__init__(lookback_days)
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST_TNS.csv"
        )
        
        self._base_url  = "https://ziggy.ucolick.org/yse/api"
        self._current_time = datetime.datetime.utcnow()
        self._current_mjd = Time.now().mjd
                
        self.save_prefix = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            ) # src
        ), "data", "tns")

        self._data_path = os.path.join(self.save_prefix, "tmp_retrieval.csv")
        os.makedirs(self.save_prefix, exist_ok=True)

        self._ignore_groups = [
            "ZTF",
            "ALeRCE",
            "ANTARES",
            "YSE",
        ]
        
        self._tns_bot_id = tns_id
        self._tns_api_key = tns_key
        self._tns_bot_name = tns_name

        self._tns_path_prefix = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects"
        self._agent = TNSQueryAgent()
        self._atlas_agent = ATLASQueryAgent()
        
        
    def quality_check(self, ts):
        return relaxed_quality_check(ts)
        
    
    def retrieve_all_names(self):
        """Retrieve all names for events modified in 
        last lookback_days days."""
        earliest_mjd = self._mjd - self._lookback
        latest_mjd = self._mjd
        all_mjds = np.arange(earliest_mjd, latest_mjd+1., 1.)

        all_names = []
        for mjd in all_mjds:
            t = Time(mjd, format='mjd')
            datestr = t.strftime('%Y%m%d')
            print(datestr)
            os_prompt = f'''curl -X POST -H 'user-agent: tns_marker{{"tns_id":{self._tns_bot_id},'''
            os_prompt += f'''"type": "bot", "name":"{self._tns_bot_name}"}}' -d '''
            os_prompt += f'''"api_key={self._tns_api_key}" '''
            os_prompt += f'''{self._tns_path_prefix}_{datestr}.csv.zip > {self._data_path}.zip'''
            os.system(os_prompt)

            try:
                # unzip the resulting file
                with zipfile.ZipFile(f"{self._data_path}.zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(self.save_prefix, "tmp"))
                tmp_path = os.path.join(
                    self.save_prefix, f'tmp/tns_public_objects_{datestr}.csv'
                )
                os.rename(tmp_path, self._data_path)
                os.remove(f"{self._data_path}.zip")

                # weird extra top row
                tmp_df = pd.read_csv(self._data_path, header=1)
                filtered_names = tmp_df.loc[~tmp_df.reporting_group.isin(self._ignore_groups), 'name']
                all_names.extend(filtered_names)
                os.remove(self._data_path)
            except:
                continue

        return list(set(all_names))
    
        
    def retrieve_tns_info(self, transient_name):
        """Retrieve YSE photometry for a given transient name.
        """
        transient = Transient(iid=transient_name)
        ntries = 0
        success = False
        while (not success) and (ntries < 3):
            try:
                results, success = self._agent.query_transient(transient)
            except:
                success = False
            if not success:
                time.sleep(30.)
            ntries += 1
            
        if success:
            for r in results:
                rdict = r.to_dict()
                rdict['spectra'] = []
                transient.ingest_query_info(rdict)
        else:
            return None

        if (transient.photometry is None) or (len(transient.photometry.detections.mag.dropna()) == 0):
            min_mjd = Time.now().mjd - 200. # 200 day from now
            try:
                atlas_results, success = self._atlas_agent.query_transient(transient, min_mjd=min_mjd)
            except:
                success = False
            if success:
                for r in atlas_results:
                    rdict = r.to_dict()
                    rdict['spectra'] = []
                    transient.ingest_query_info(rdict)
                    
        else:
            peak_mag = transient.photometry.detections.mag.dropna().min()
            earliest_time = np.min(transient.photometry.detections.index)
            if peak_mag < 20.0: # then add ATLAS data (5-sig is 19.7, so a little dimmer for lower SNR)
                min_mjd = Time(earliest_time).mjd - 200. # 200 day buffer for baseline calculation
                try:
                    atlas_results, success = self._atlas_agent.query_transient(transient, min_mjd=min_mjd)
                except:
                    success = False
                if success:
                    for r in atlas_results:
                        rdict = r.to_dict()
                        rdict['spectra'] = []
                        transient.ingest_query_info(rdict)
        if transient.photometry is None:
            return None
        df = transient.photometry.detections
        df['ant_mjd'] = Time(df.index).mjd
        ts = df[['ant_mjd', 'mag', 'mag_error', 'filter']]
        ts.rename(
            columns={
                'mag': 'ant_mag',
                'mag_error': 'ant_magerr',
                'filter': 'ant_passband',
            },
            inplace=True
        )
        # if magerr == 5.0 or is negative, not a valid datapoint
        mask = (ts['ant_magerr'] >= 5.0) | (ts['ant_magerr'] <= 0.0)
        ts = ts.loc[~mask]
        # removes rows with nan values (aka upper limits)
        ts.dropna(inplace=True, axis=0, ignore_index=True)

        tns_name = transient_name
        spec_class = transient.spec_class
        redshift = transient.redshift
        coord = transient.coordinates
        ra = coord.ra.to(u.deg).value
        dec = coord.dec.to(u.deg).value

        ts['ant_ra'] = ra
        ts['ant_dec'] = dec

        return ts, tns_name, spec_class, redshift, ra, dec
    
    
    def brightest_time(self, ts):
        """Find brightest time."""
        bright_idx = (ts['ant_mag'] + ts['ant_magerr']).idxmin()
        return ts.loc[bright_idx, 'ant_mjd']

    
    def retrieve_candidates(self, filt, max_num):
        """Main loop to retrieve YSE transients from 
        the night, get light curves, quality check, and return df.
        """
        filt.setup()
        self.check_watch()
        processed_dicts = []
        
        if os.path.exists(self._prost_path):
            full_table = pd.read_csv(self._prost_path)
        else:
            full_table = None
            
        nightly_transients = self.retrieve_all_names()
        ts_dict = {}

        for i, transient in enumerate(nightly_transients):
            if i % 10 == 0:
                print(f"Pre-processed {i} transients...")
                                 
            out = self.retrieve_tns_info(transient)
            if out is None:
                continue

            ts, tns_name, tns_cls, tns_redshift, ra, dec = out
            ts.loc[:, 'ant_ra'] = ra
            ts.loc[:, 'ant_dec'] = dec
                            
            detections = ts.loc[ts['ant_magerr'] < 0.362]

            if not self.quality_check(detections):
                continue

            if (full_table is not None) and (transient in full_table['name'].to_numpy()):
                locus_dict = full_table.loc[
                    full_table['name'] == transient,
                    ['name', 'ra', 'dec', 'nuclear']
                ].iloc[0].to_dict()
                
            else:
                # Is it nuclear?
                nuclear_flag = None

                locus_dict = {
                    'name': transient,
                    'ra': ra,
                    'dec': dec,
                    'nuclear': nuclear_flag,
                }
                
            ts_dict[transient] = detections

            locus_dict['tns_name'] = tns_name
            if not isinstance(tns_cls, str):
                tns_cls = '---'

            if not isinstance(tns_redshift, float):
                print(tns_redshift)
                tns_redshift = np.nan
                
            locus_dict['tns_class'] = tns_cls
            locus_dict['tns_redshift'] = tns_redshift
            locus_dict['brightest_alert_magnitude'] = (detections['ant_mag'] + detections['ant_magerr']).min()
            locus_dict['oldest_alert_observation_time'] = detections.ant_mjd.min()
            locus_dict['newest_alert_observation_time'] = detections.ant_mjd.max()
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
    

class ATLASRetriever(TNSRetriever):
    def quality_check(self, ts):
        return atlas_quality_check(ts)



if __name__ == "__main__":
    retriever = YSERetriever(lookback_days=1.)
    df = retriever.run(10)

