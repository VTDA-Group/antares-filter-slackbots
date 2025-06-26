import os
import gc
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPBasicAuth

import numpy as np
import pandas as pd
from scipy.stats import uniform, gamma, halfnorm
from dustmaps.config import config as dustmaps_config
import astropy.units as u
from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module
from astropy.coordinates import SkyCoord, Angle
from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag
from iinuclear.utils import check_nuclear

from antares_filter_slackbots.slack_formatters import SlackPoster
from antares_filter_slackbots.slack_requests import *
from antares_filter_slackbots.auth import login_ysepz, password_ysepz
from antares_filter_slackbots.webpage import create_html_tab

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
        retriever,
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
        ascending: bool = False,
    ):
        self.save_prefix = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            ) # src
        ), "data", name)
        os.makedirs(self.save_prefix, exist_ok=True)

        self.name = name
        self._filt = filter_obj
        self.retriever = retriever
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
            
        self.ascending = ascending

            
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

    def validate_result(self, meta_dict):
        """Validate whether resulting locus
        meets criteria for being saved/ranked.
        """
        for t in self._post_filt_tags:
            if (t not in meta_dict) or (not meta_dict[t]):
                return False
            
        for (k,v) in self._post_filt_props.items():
            if (k not in meta_dict) or (meta_dict[k] < v[0]) or (meta_dict[k] > v[1]):
                return False
                        
        for (k,v) in self._group_props.items():
            if (k not in meta_dict) or (meta_dict[k] not in v):
                return False
                        
        return True
    
    def setup(self):
        """Run setup for that filter.
        """
        if self._filt is not None:
            self._filt.setup()
    
    def apply(self, meta_dict, ts):
        """Apply filter to locus. Modifies locus in place.
        """
        if self._filt is not None:
            new_meta_dict = self._filt._run(meta_dict, ts)
            return new_meta_dict
        return meta_dict


            
class ANTARESRanker:
    """Ranks objects of interest from nightly ANTARES batches."""
    def __init__(self):
        dustmaps_config.reset()
        dustmaps_config['data_dir'] = os.path.join(
            Path(__file__).parent.parent.parent.absolute(),
            "data/dustmaps"
        )
        self._auth = HTTPBasicAuth(login_ysepz, password_ysepz)

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
        
        self._ztf_yse_tag = "https://ziggy.ucolick.org/yse/api/transienttags/39/"
        self._slackbot_tag = 'https://ziggy.ucolick.org/yse/api/transienttags/123/'
        self._save_properties = [
            'name', 'ra', 'dec', 'nuclear'
        ]
        self._host_properties = [
            'ra',
            'dec',
            'tns_name',
            'tns_redshift',
            'tns_class',
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
            'peak_phase',
        ]
        self._laiss_properties = [
            "gKronMagCorrected",
            "gKronRad",
            "gExtNSigma",
            "rKronMagCorrected",
            "rKronRad",
            "rExtNSigma",
            "iKronMagCorrected",
            "iKronRad",
            "iExtNSigma",
            "zKronMagCorrected",
            "zKronRad",
            "zExtNSigma",
            "gminusrKronMag",
            "rminusiKronMag",
            "iminuszKronMag",
            "rmomentXX",
            "rmomentXY",
            "rmomentYY",
        ]
        
        self.get_yse_fields()
        
    
    def get_yse_fields(self):
        """Retrieve active YSE fields."""
        yse_fields = requests.get(
            'http://ziggy.ucolick.org/yse/api/surveyfields/?obs_group=YSE',
            auth = self._auth
        ).json()['results']
        
        self._yse_fields = []
        
        for f in yse_fields:
            d = f['dec_cen']*np.pi/180
            width_corr = 3.4 / np.abs(np.cos(d)) # to match website? #f['width_deg'] / np.abs(np.cos(d))
            # Define the tile offsets:
            ra_offset = Angle(width_corr / 2., unit=u.deg).value
            #dec_offset = Angle(f['height_deg'] / 2., unit=u.deg).value
            dec_offset = Angle(3.4 / 2., unit=u.deg).value
            self._yse_fields.append(
                [
                    f['ra_cen'] - ra_offset,
                    f['ra_cen'] + ra_offset,
                    f['dec_cen'] - dec_offset,
                    f['dec_cen'] + dec_offset,
                    f['field_id']
                ]
            )
    
    
    def apply_filter_to_df(self, meta_df, ts_dict, filt: RankingFilter):
        """Validate post-query loci and save as dataframe.
        """
        locus_dicts = []
        
        for i, row in meta_df.iterrows():
            if i % 10 == 0:
                print(f"Filtered {i} events...")
                
            event_dict = row.to_dict()
            
            # extract redshift
            tns_redshift = event_dict['tns_redshift']
            if ~np.isnan(tns_redshift) and (tns_redshift > 0.):
                redshift = tns_redshift
            elif event_dict['high_host_confidence'] and (
                ~np.isnan(event_dict['host_redshift'])
            ) and (event_dict['host_redshift'] > 0.):
                redshift = event_dict['host_redshift']
            else:
                redshift = np.nan
                
            #redshift = 0.0362 # MANUAL OVERRIDE
                
            event_dict['best_redshift'] = redshift
            
            
            if ~np.isnan(redshift):
                app_mag = event_dict['brightest_alert_magnitude']
                k_corr = 2.5 * np.log10(1.0 + redshift)
                distmod = Planck15.distmod(redshift).value
                abs_mag = app_mag - distmod + k_corr
                event_dict['peak_abs_mag'] = abs_mag
            else:
                event_dict['peak_abs_mag'] = np.nan
            
            print(event_dict['name'])
            new_dict = filt.apply(event_dict, ts_dict[event_dict['name']])
            
            if not filt.validate_result(new_dict):
                continue
                                                
            locus_dict = {
                'name': new_dict['name'],
                filt.ranked_property: new_dict[filt.ranked_property],
                'peak_mag': new_dict['brightest_alert_magnitude'],
            }
            
            tns_name = new_dict['tns_name']
            if (tns_name is not None) and str(tns_name)[:4].isnumeric():
                yse_search_url = f"https://ziggy.ucolick.org/yse/api/transients/?name={tns_name}"
            else:
                ra_lower = new_dict['ra'] - 0.0001
                ra_upper = new_dict['ra'] + 0.0001
                dec_lower = new_dict['dec'] - 0.0001
                dec_upper = new_dict['dec'] + 0.0001
                yse_search_url = f"https://ziggy.ucolick.org/yse/api/transients/?ra_gte={ra_lower}"
                yse_search_url += f"&ra_lte={ra_upper}&dec_gte={dec_lower}&dec_lte={dec_upper}"
                
            yse_results = requests.get(yse_search_url, auth=self._auth).json()['results']

            if len(yse_results) > 0:
                yse_result = yse_results[0]
                name = yse_result['name']
                url = yse_result['url']

                yse_result['tags'].append(self._slackbot_tag)
                yse_result['tags'] = list(set(yse_result['tags']))
                url = yse_result['url']

                requests.put(
                    url,
                    json=yse_result,
                    auth=self._auth
                )
                locus_dict['yse_pz'] = f"<https://ziggy.ucolick.org/yse/transient_detail/{name}|*YSE-PZ*>"
            else:
                locus_dict['yse_pz'] = None
            
            locus_dict['yse_field'] = None
            # check if in yse field
            for f in self._yse_fields:
                if (
                    (new_dict['ra'] > f[0]) and (new_dict['ra'] < f[1])
                ) or ((new_dict['dec'] > f[2]) and (new_dict['dec'] < f[3])):
                    locus_dict['yse_field'] = f[4]
                    
            for p in self._host_properties:
                if p in new_dict:
                    locus_dict[p] = new_dict[p]
                    
            for p in filt._group_props:
                if p in new_dict:
                    locus_dict[p] = new_dict[p]

            for p in filt.save_properties:
                if p in new_dict:
                    locus_dict[p] = new_dict[p]
                    
            print(locus_dict)
            locus_dicts.append(locus_dict)
            
        if len(locus_dicts) == 0:
            return None

        full_locus_df = pd.DataFrame.from_records(locus_dicts)
        full_locus_df.set_index('name', inplace=True)
        return full_locus_df
    
    
    def associate_sample_prost(self, df, with_redshift=False):
        max_size = 20 # only associate 10 at a time or it freezes
        list_df = [df[i:i + max_size] for i in range(0, len(df), max_size)]
        merged_hosts = []
        
        t = time.time()        
        for i, df_i in enumerate(list_df):
            merged_hosts_i = None
            #try:
            print("STARTING NOW")
            if with_redshift:
                merged_hosts_i = associate_sample(
                    df_i,
                    priors=self.priors_z,
                    likes=self.likes_z,
                    catalogs=['glade', 'decals',],
                    parallel=False,
                    verbose=2,
                    save=False,
                    progress_bar=True,
                    cat_cols=True,
                    name_col='name',
                    coord_cols=('ra', 'dec'),
                    redshift_col='tns_redshift',
                )
            else:
                merged_hosts_i = associate_sample(
                    df_i,
                    priors=self.priors_noz,
                    likes=self.likes_noz,
                    catalogs=['glade', 'decals',],
                    parallel=False,
                    verbose=2,
                    save=False,
                    progress_bar=True,
                    cat_cols=True,
                    name_col='name',
                    coord_cols=('ra', 'dec'),
                )
            # add back in df_i without hosts
            df_remaining_i = df_i.loc[~df_i.name.isin(merged_hosts_i.name)]
            merged_hosts_i = pd.concat([merged_hosts_i, df_remaining_i], ignore_index=True)
            #break
            #except:
            #    continue
            
            print(len(merged_hosts_i))
            merged_hosts.append(merged_hosts_i)
            del merged_hosts_i
            gc.collect()
            print(f"Time after chunk {i}: {time.time() - t}")
            t = time.time()
            
        merged_hosts_concat = pd.concat(merged_hosts, ignore_index=True)
                
        return merged_hosts_concat
    
    
    def is_it_nuclear(self, event_dict, ts):
        """Applies isitnuclear package to determine
        whether loci are likely nuclear.
        """
        coord = (event_dict['ra'], event_dict['dec'])
        ras = ts['ant_ra']
        decs = ts['ant_dec']
        ra_galaxy = event_dict['host_ra']
        dec_galaxy = event_dict['host_dec']
        error_arcsec = event_dict['host_offset_std']
        _, _, _, nuclear_bool = check_nuclear(
            ras, decs, ra_galaxy, dec_galaxy, error_arcsec,
            p_threshold=0.05
        )
        if (nuclear_bool is None) or np.isnan(nuclear_bool):
            return False
        
        return nuclear_bool
    
    def add_host_galaxy_info(self, orig_df, ts_dict, retriever):
        """Retrieve host galaxy info for bunch of loci.
        """
        df = orig_df.drop_duplicates(subset=['name'], keep='last')
        df.loc[df.tns_redshift <= 0.0, 'tns_redshift'] = np.nan
        merged_hosts = None
        
        prost_path = retriever._prost_path
        
        if os.path.exists(prost_path):
            full_table = pd.read_csv(prost_path)
            all_names = full_table['name']
            existing_names = np.intersect1d(all_names, df['name'])
            if len(existing_names) > 0:
                existing_table = full_table.loc[full_table['name'].isin(existing_names)]
                existing_merged_df = pd.merge(
                    df.loc[df['name'].isin(existing_names)],
                    existing_table,
                    on='name',
                )
                rename_cols = [c[:-2] for c in existing_merged_df.columns if '_x' in c]
                existing_merged_df.rename(
                    columns={
                        c+'_x': c for c in rename_cols
                    },
                    inplace=True
                )
        else:
            existing_names = []
            full_table = None
            
        keep_mask = ~df['name'].isin(existing_names)
        if keep_mask.any():
            df = df.loc[keep_mask]
            print("KEEP DF", df)
            for attempt in range(5):
                print(f"Attempt {attempt+1} out of 5")

                #try:
                noz_mask = df.tns_redshift.isna()

                if len(df[noz_mask]) == 0: # all redshift associated
                    merged_hosts = self.associate_sample_prost(df, with_redshift=True)

                elif len(df[~noz_mask]) == 0: # no redshifts
                    merged_hosts = self.associate_sample_prost(df, with_redshift=False)

                else:
                    hosts_z = self.associate_sample_prost(
                        df[~noz_mask], with_redshift=True
                    )
                    hosts_noz = self.associate_sample_prost(
                        df[noz_mask], with_redshift=False
                    )
                    merged_hosts = pd.concat([hosts_z, hosts_noz], ignore_index=True)

                break
                #except:
                #    pass

            if merged_hosts is None:
                raise ConnectionError("Could not retrieve host info")
                
            start_time = time.time()
            print("DONE", len(merged_hosts), time.time() - start_time)
            
            no_nuclear = merged_hosts.loc[merged_hosts.nuclear.isna()]
            for i, row in no_nuclear.iterrows():
                row_dict = row.to_dict()
                if 'host_name' in row_dict:
                    nuclear_flag = self.is_it_nuclear(row_dict, ts_dict[row_dict['name']])
                else:
                    nuclear_flag = False
                merged_hosts.loc[i, 'nuclear'] = nuclear_flag
            
            if 'host_name' not in merged_hosts:
                merged_hosts.loc[:,[
                    'host_name',
                    'host_objID',
                    'host_prob_ratio',
                    'host_any_non_ratio',
                    'best_cat',
                    'host_redshift_mean',
                    'host_offset_mean',
                    'host_absmag_mean', 
                    'host_total_posterior',
                    'host_2_total_posterior',
                    'any_posterior',
                    'none_posterior',
                ]] = pd.NA
                
            merged_hosts.host_name.mask(merged_hosts.host_name.isna(), merged_hosts.host_objID, inplace=True)
            merged_hosts.host_name.mask(merged_hosts.host_name.eq(""), merged_hosts.host_objID, inplace=True)
            merged_hosts.host_name.mask(merged_hosts.host_name.eq("nan"), merged_hosts.host_objID, inplace=True)
            merged_hosts.host_name = merged_hosts.host_name.astype(str)

            merged_hosts['host_prob_ratio'] = merged_hosts.host_total_posterior / merged_hosts.host_2_total_posterior
            merged_hosts['host_any_non_ratio'] = merged_hosts.any_posterior / merged_hosts.none_posterior
            merged_hosts['high_host_confidence'] = (
                (
                    merged_hosts.host_prob_ratio >= 10.
                ) & (
                    merged_hosts.host_any_non_ratio >= 10.
                ) #& (
                #   merged_hosts.best_cat != 'panstarrs' # removing PS1 from the beginning
                #)
            ).astype(bool)
            
            host_cols = np.setdiff1d(merged_hosts.columns, df.columns)
            save_cols = [*self._save_properties, *host_cols]
            
            # add to existing prost table
            if len(existing_names) > 0:
                merged_table = pd.concat(
                    [existing_merged_df, merged_hosts], ignore_index=True
                ).drop_duplicates(subset=['name'], keep='last')
                save_df = merged_table[save_cols]
                
            else:
                save_df = merged_hosts[save_cols]
                merged_table = merged_hosts
                
            if full_table is None:
                total_save_df = save_df
            else:
                total_save_df = pd.concat(
                    [full_table, save_df], ignore_index=True
                ).drop_duplicates(subset=['name'], keep='last')
                
            total_save_df.to_csv(prost_path, index=False)
                
        else:
            merged_table = existing_merged_df
            
        
        merged_df = merged_table.loc[:, [*df.columns,
            'host_name', 'best_cat', 'host_redshift_mean', 'host_offset_mean',
            'host_absmag_mean', 'host_total_posterior', 'high_host_confidence',
            #*self._laiss_properties,
        ]]

        merged_df.rename(columns={
            'host_redshift_mean': 'host_redshift',
            'host_total_posterior': 'host_prob',
            'host_offset_mean': 'host_sep_arcsec',
            'best_cat': 'best_host_catalog',
            'host_absmag_mean': 'host_absmag'
        }, inplace=True)
        
        print("MERGED DF", merged_df)
        return merged_df
    

    def rank(self, df, filt, max_num=10):
        """
        Rank dataframe by filter's ranking_property, and keep top max_num.
        Does so for each property value in groupby_properties.
        """
        df.sort_values(by=filt.ranked_property, ascending=filt.ascending, inplace=True)
        if len(filt._group_props.keys()) > 0:
            df_pruned = df.groupby(list(filt._group_props.keys())[0]).head(max_num)
        else:
            df_pruned = df.head(max_num)

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
        df, ts_dict = filt.retriever.retrieve_candidates(filt, max_num)
        
        if df is None:
            slack_loci = SlackPoster(None, {}, filt.save_prefix)
            slack_loci.post_empty(filt.channel)
            return
        
        host_df = self.add_host_galaxy_info(df, ts_dict, filt.retriever)
        if host_df is None:
            slack_loci = SlackPoster(None, {}, filt.save_prefix)
            slack_loci.post_empty(filt.channel)
            return
        
        final_df = self.apply_filter_to_df(host_df, ts_dict, filt)
        if final_df is None:
            slack_loci = SlackPoster(None, {}, filt.save_prefix)
            slack_loci.post_empty(filt.channel)
            return
        
        df_pruned = self.rank(final_df, filt, max_num)
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
                filt_meta[f'overflow_{v}'] = (len(final_df[final_df[k] == v]) < len(df_pruned[df_pruned[k] == v]))

        slack_loci = SlackPoster(df_pruned, filt_meta, filt.save_prefix)
        slack_loci.post(filt.channel)
        create_html_tab(df_pruned, slack_loci)

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
