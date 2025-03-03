import os
from pathlib import Path
import subprocess
import glob
from typing import Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import uniform, gamma, halfnorm
from dustmaps.config import config as dustmaps_config
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astro_prost.associate import associate_sample, prepare_catalog
from astro_prost.helpers import SnRateAbsmag
import antares_client

from .slack_formatters import SlackPoster
from .slack_requests import *


import warnings
warnings.filterwarnings('ignore')

# https://stackoverflow.com/questions/1528237/how-to-handle-exceptions-in-a-list-comprehensions
def catch(func, *args, handle=lambda e : e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)

def generate_coordinates(ras, decs):
    """Generate series of SkyCoords from ras + decs
    """
    return [
        SkyCoord(ra=ra*u.deg, dec=dec*u.deg) for (ra, dec) in zip(ras, decs)
    ]

class RankingFilter:
    """Filter along with which
    outputs to save and rank by.

    Parameters
    ----------
    filter_obj: dk.Filter
        filter to apply to each locus
    ranking_property: str
        property in resulting locus to rank by (must be numeric)
    filter_tags: Optional[list]
        locus must have all of these tags before filter is applied
    filter_properties: Optional[dict]
        locus must match all of the property's stated value ranges before filter is applied
    groupby_properties: Optional[dict]
        groups rankings by each value per key
    """
    def __init__(
        self,
        name,
        filter_obj,
        channel,
        ranking_property,
        save_properties: Optional[list] = None,
        pre_filter_tags: Optional[list] = None,
        pre_filter_catalogs: Optional[list] = None,
        post_filter_tags: Optional[list] = None,
        pre_filter_properties: Optional[dict[str,Any]] = None,
        post_filter_properties: Optional[dict[str,Any]] = None,
        groupby_properties: Optional[dict[str,Any]] = None,
    ):
        self.save_prefix = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            ) # src
        ), "data", name)
        os.makedirs(self.save_prefix, exist_ok=True)

        self.name = name
        self._filt = filter_obj
        self.channel = channel
        self.ranked_property = ranking_property

        if groupby_properties is None:
            self._group_props = {}
        else:
            self._group_props = dict(groupby_properties)

        if save_properties is None:
            self.save_properties = []
        else:
            self.save_properties = save_properties

        if self.ranked_property in self.save_properties:
            self.save_properties.remove(self.ranked_property)
        for k in self._group_props:
            if k in self.save_properties:
                self.save_properties.remove(k)

        if pre_filter_tags is None:
            self._pre_filt_tags = []
        else:
            self._pre_filt_tags = list(pre_filter_tags)

        if pre_filter_catalogs is None:
            self._catalogs = []
        else:
            self._catalogs = list(pre_filter_catalogs)

        if post_filter_tags is None:
            self._post_filt_tags = []
        else:
            self._post_filt_tags = list(post_filter_tags)

        if pre_filter_properties is None:
            self._pre_filt_props = {}
        else:
            self._pre_filt_props = dict(pre_filter_properties)
        
        if post_filter_properties is None:
            self._post_filt_props = {}
        else:
            self._post_filt_props = dict(post_filter_properties)

    def modify_query(self, query):
        """Modify query based on tags and property
        filters.
        """
        for t in self._pre_filt_tags:
            query.append(
                {
                    "terms": {
                        "tags": [t,]
                    }
                }
            )
        
        for c in self._catalogs:
            query.append(
                {
                    "terms": {
                        "catalogs": [c,]
                    }
                }
            )
            
        for (k,v) in self._pre_filt_props.items():
            query.append(
                {
                    "range": {
                        f"properties.{k}": {
                            'gte': v[0],
                            'lte': v[1]
                        }
                    }
                }
            )
        return query

    def validate_result(self, locus):
        """Validate whether resulting locus
        meets criteria for being saved/ranked.
        """
        for t in self._post_filt_tags:
            if t not in locus.tags:
                return False
            
        for (k,v) in self._post_filt_props.items():
            if (locus.properties[k] < v[0]) or (locus.properties[k] > v[1]):
                print(k, locus.properties[k])
                return False
                        
        for (k,v) in self._group_props.items():
            if locus.properties[k] not in v:
                return False
                        
        return True
    
    def setup(self):
        """Run setup for that filter.
        """
        if self._filt is not None:
            self._filt.setup()
    
    def apply(self, locus):
        """Apply filter to locus. Modifies locus in place.
        """
        if self._filt is not None:
            self._filt.run(locus)


class ANTARESRanker:
    """Ranks objects of interest from nightly ANTARES batches."""
    def __init__(
            self,
            lookback_days=1.,
        ):
        dustmaps_config.reset()
        dustmaps_config['data_dir'] = os.path.join(
            Path(__file__).parent.parent.parent.absolute(),
            "data/dustmaps"
        )
        self._prost_path = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/PROST.csv"
        )
        self._lookback = lookback_days
        self._query = []

        # initialize prost distributions
        priorfunc_offset = uniform(loc=0, scale=10)
        likefunc_offset = gamma(a=0.75)
        likefunc_absmag = SnRateAbsmag(a=-30, b=-10)
        priorfunc_absmag = uniform(loc=-30, scale=20)
        priorfunc_z = halfnorm(loc=0.0001, scale=0.1)

        self.priors_z = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "redshift": priorfunc_z}
        self.likes_z = {"offset": likefunc_offset, "absmag": likefunc_absmag}
        self.priors_noz = {"offset": priorfunc_offset}
        self.likes_noz = {"offset": likefunc_offset}


    def reset_query(self):
        """Reset query."""
        self._query = []

    def check_watch(self):
        """Update current time, in MJD.
        """
        self._mjd = Time.now().mjd

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
            
        return True
    
            
    def process_query_results(self, loci, filt: RankingFilter):
        """Validate post-query loci and save as dataframe.
        """
        processed_loci = []
        names = []
        ras = []
        decs = []
        redshifts = []
        for i, locus in enumerate(loci):
            if i % 100 == 0:
                print(f"Pre-processed {i} loci...")
            if not self.star_catalog_check(locus):
                continue

            if 'tns_public_objects' in locus.catalog_objects:
                tns = locus.catalog_objects['tns_public_objects'][0]
                tns_name, tns_cls, tns_redshift = tns['name'], tns['type'], tns['redshift']
                if tns_cls == '':
                    tns_cls = '---'
                if tns_redshift is None:
                    tns_redshift = np.nan
            else:
                tns_name = '---'
                tns_cls = '---'
                tns_redshift = np.nan
                
            locus.extra_properties = {
                'tns_name': tns_name,
                'tns_class': tns_cls,
                'peak_phase': self._mjd - locus.properties['brightest_alert_observation_time'],
            }
            processed_loci.append(locus)
            
            ras.append(locus.ra)
            decs.append(locus.dec)
            names.append(locus.locus_id)
            redshifts.append(tns_redshift)
            
        if len(names) == 0:
            return None
            
        locus_df = pd.DataFrame({
            'name': names,
            'ra': ras,
            'dec': decs,
            'tns_redshift': redshifts
        })
        host_df = self.add_host_galaxy_info(locus_df)

        locus_dicts = []
        for i, locus in enumerate(processed_loci):
            if i % 100 == 0:
                print(f"Processed {i} loci...")
                
            locus_host_info = host_df.loc[
                host_df.index == locus.locus_id
            ].iloc[0].to_dict()
            locus.extra_properties.update(locus_host_info)
            
            filt.apply(locus)
            
            if not filt.validate_result(locus):
                continue
                
            locus_dict = {
                'name': locus.locus_id,
                filt.ranked_property: locus.properties[filt.ranked_property],
                'peak_mag': locus.properties['brightest_alert_magnitude'],
                'peak_abs_mag': locus.properties['peak_abs_mag'],
            }
            
            for p in filt._group_props:
                if p in locus.properties:
                    locus_dict[p] = locus.properties[p]

            for p in filt.save_properties:
                if p in locus.properties:
                    locus_dict[p] = locus.properties[p]
                    
            locus_dict.update(locus.extra_properties)
            
            print(locus_dict)
            locus_dicts.append(locus_dict)
            

        if len(locus_dicts) == 0:
            return None

        full_locus_df = pd.DataFrame.from_records(locus_dicts)
        full_locus_df.set_index('name', inplace=True)
        return full_locus_df
    
    
    def add_host_galaxy_info(self, df):
        """Retrieve host galaxy info for bunch of loci.
        """
        merged_hosts = None

        for attempt in range(5):
            print(f"Attempt {attempt+1} out of 5")
            
            try:
                noz_mask = df.tns_redshift.isna()

                if len(df[noz_mask]) == 0: # all redshift associated
                    df_prep = prepare_catalog(
                        df, transient_name_col="name", transient_coord_cols=("ra", "dec")
                    )
                    merged_hosts = associate_sample(
                        df_prep,
                        priors=self.priors_z,
                        likes=self.likes_z,
                        catalogs=['glade', 'decals', 'panstarrs'],
                        parallel=False,
                        verbose=0,
                        save=False,
                        progress_bar=False,
                        cat_cols=True,
                    )

                elif len(df[~noz_mask]) == 0: # no redshifts
                    df_prep = prepare_catalog(
                        df, transient_name_col="name", transient_coord_cols=("ra", "dec")
                    )
                    merged_hosts = associate_sample(
                        df_prep,
                        priors=self.priors_noz,
                        likes=self.likes_noz,
                        catalogs=['glade', 'decals', 'panstarrs'],
                        parallel=False,
                        verbose=0,
                        save=False,
                        progress_bar=False,
                        cat_cols=True,
                    )

                else:
                    df_prep_z = prepare_catalog(
                        df[~noz_mask], transient_name_col="name", transient_coord_cols=("ra", "dec")
                    )
                    df_prep_noz = prepare_catalog(
                        df[noz_mask], transient_name_col="name", transient_coord_cols=("ra", "dec")
                    )

                    hosts_z = associate_sample(
                        df_prep_z,
                        priors=self.priors_z,
                        likes=self.likes_z,
                        catalogs=['glade', 'decals', 'panstarrs'],
                        parallel=False,
                        verbose=0,
                        save=False,
                        progress_bar=False,
                        cat_cols=True,
                    )
                    hosts_noz = associate_sample(
                        df_prep_noz,
                        priors=self.priors_z,
                        likes=self.likes_z,
                        catalogs=['glade', 'decals', 'panstarrs'],
                        parallel=False,
                        verbose=0,
                        save=False,
                        progress_bar=False,
                        cat_cols=True,
                    )
                    merged_hosts = pd.concat([hosts_z, hosts_noz], ignore_index=True)

                break
            except:
                pass

        if merged_hosts is None:
            raise ConnectionError("Could not retrieve host info")
            
        # add to existing prost table
        if os.path.exists(self._prost_path):
            full_table = pd.read_csv(self._prost_path)
            orig_cols = np.intersect1d(full_table.columns, merged_hosts.columns)
            full_table = pd.concat([full_table, merged_hosts[orig_cols]], ignore_index=True).drop_duplicates(subset=['name'])
            full_table.to_csv(self._prost_path, index=False)
        else:
            merged_hosts.to_csv(self._prost_path, index=False)
            
        merged_hosts.host_name.mask(merged_hosts.host_name.isna(), merged_hosts.host_objID, inplace=True)
        merged_hosts.host_name.mask(merged_hosts.host_name.eq(""), merged_hosts.host_objID, inplace=True)
        merged_hosts.host_name.mask(merged_hosts.host_name.eq("nan"), merged_hosts.host_objID, inplace=True)
        merged_hosts.host_name = merged_hosts.host_name.astype(str)
        
        merged_hosts['host_prob_ratio'] = merged_hosts.host_total_posterior / merged_hosts.host_2_total_posterior
        merged_hosts['host_any_non_ratio'] = merged_hosts.any_posterior / merged_hosts.none_posterior
        merged_hosts['high_host_confidence'] = ((merged_hosts.host_prob_ratio >= 10.) & (merged_hosts.host_any_non_ratio >= 10.)).astype(bool)
        merged_df = merged_hosts.loc[:, [*df.columns,
            'host_name', 'best_cat', 'host_redshift_mean', 'host_offset_mean',
            'host_absmag_mean', 'host_total_posterior', 'high_host_confidence'
        ]]

        merged_df.rename(columns={
            'host_redshift_mean': 'host_redshift',
            'host_total_posterior': 'host_prob',
            'host_offset_mean': 'host_sep_arcsec',
            'best_cat': 'best_host_catalog',
            'host_absmag_mean': 'host_absmag'
        }, inplace=True)
                
        merged_df.set_index('name', inplace=True)
        merged_df.index.name = 'antid'

        return merged_df
    

    def rank(self, df, filt, max_num=10):
        """
        Rank dataframe by filter's ranking_property, and keep top max_num.
        Does so for each property value in groupby_properties.
        """
        df['eff_rank'] = df[filt.ranked_property] - np.where(df.host_sep_arcsec > 0.5, 0., 1000.)
        df.sort_values(by="eff_rank", ascending=False, inplace=True)
        if len(filt._group_props.keys()) > 0:
            df_pruned = df.groupby(list(filt._group_props.keys())[0]).head(max_num)
        else:
            df_pruned = df.head(max_num)
        df_pruned.drop("eff_rank", axis=1, inplace=True)

        return df_pruned
    
    
    def get_last_posted(self, channel, num_days=5):
        """Get all antares IDs already posted in channel in last num_days days."""
        print(f"Collecting names of events already posted in the last {num_days} days.")
        response_data = get_conversation_history(channel)

        posted_names = []
        posted_ts = []

        for message in response_data['messages']:
            try:
                names = [x['title'].split(" ")[1] for x in message['attachments']] 
                ts = [message['ts']]*len(names)
                posted_names.append(names)
                posted_ts.append(ts)
            except:
                continue
        df = pd.DataFrame(
            {'timestamp': np.concatenate(posted_ts)},
            index = np.concatenate(posted_names)
        )
        # Subtract N days from current time
        min_datetime = datetime.now() - timedelta(days=num_days)
        df = df.loc[df.timestamp > min_datetime.timestamp()]

        df.sort_values('timestamp', ascending=False, inplace=True)
        df = df.groupby(df.index).first() # no repeats
        print(f"Number of total posted transients: {len(df)}")
        return df.index
        
        #the names of the transients posted <numDays ago
        #include the events in the locally-stored db 
        #df_posted = pd.read_csv(os.path.join(data_dir, "anomalies_db.csv"))
        #df_posted.rename(columns={"ANTID":'Transient'}, inplace=True)
        #df_comb = pd.concat([df_posted, df], ignore_index=True)

    def run(self, filt, max_num):
        """Run full cycle of the ANTARESRanker.
        """
        print("Running ANTARESRanker for filter "+filt.name)
        filt.setup()
        self.reset_query()
        self.check_watch()
        self.constrain_query_mjd()
        self.apply_filter(filt)
        loci = self.run_query()
        df = self.process_query_results(loci, filt)
        if df is None:
            slack_loci = SlackPoster(None, {}, filt.save_prefix)
            slack_loci.post_empty(filt.channel)
            return

        df_pruned = self.rank(df, filt, max_num)
        df_pruned["posted_before"] = False
        #posted_before_names = self.get_last_posted(filt.channel, 10_000_000)
        #df_pruned.posted_before = df_pruned.index.isin(posted_before_names)

        filt_meta = {
            'max_num': max_num,
            'ranking_property': filt.ranked_property,
            'groupby': list(filt._group_props.keys())[0] if len(filt._group_props.keys()) > 0 else None,
        }

        for (k, vals) in filt._group_props.items():
            for v in vals:
                filt_meta[f'overflow_{v}'] = (len(df[df[k] == v]) < len(df_pruned[df_pruned[k] == v]))

        slack_loci = SlackPoster(df_pruned, filt_meta, filt.save_prefix)
        slack_loci.post(filt.channel)

        # save to local df
        self.save_objects(df_pruned, filt.save_prefix)
        return

    def save_objects(self, df, save_prefix, append=True):
        save_fn = os.path.join(save_prefix, "loci.csv")
        if (not os.path.exists(save_fn)) or (not append):
            df.to_csv(save_fn)

        else:
            orig_df = pd.read_csv(save_fn, index_col=0)
            merged_df = pd.concat([orig_df, df])
            # keep the last occurrence of each duplicate row -- (most up to date)
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df.to_csv(save_fn) # overwrite the old file with unique new + old objects
