from astropy.time import Time
import pandas as pd
import os
from pathlib import Path
from antares_filter_slackbots.antares_ranker import ANTARESRanker, RankingFilter
from antares_filter_slackbots.filters import SuperphotPlusZTF, ShapleyPlotLAISS, PrecursorEmission
from antares_filter_slackbots.slack_formatters import SlackPoster
from antares_filter_slackbots.retrievers import (
    ANTARESRetriever, YSERetriever, RelaxedANTARESRetriever,
    TNSRetriever, ATLASRetriever
)

def all_current_filters():
    """Where all filters to run are defined.
    """
    # lookback of 1 day
    ant_retriever = ANTARESRetriever(1.0)
    yse_retriever = YSERetriever(2.0)
    tns_retriever = TNSRetriever(2.0)
    atlas_retriever = ATLASRetriever(2.0)
    relaxed_ant_retriever = RelaxedANTARESRetriever(1.0)
    
    current_time = Time.now().mjd
    nuclear = RankingFilter(
        "nuclear_transients",
        ant_retriever,
        PrecursorEmission(),
        "#nuclear-transients",
        "brightest_alert_magnitude",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-1000., 99_999_999,),
            "num_mag_values": (5, 2000),
        },
        save_properties = ['peak_abs_mag', 'duration', 'duration_z0', 'lum_dist'],
        post_filter_tags = ["valid_nuclear",],
    )
    precursor = RankingFilter(
        "precursor_emission",
        relaxed_ant_retriever,
        PrecursorEmission(),
        "#precursor-emission",
        "peak_abs_mag",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-1000., 99_999_999,),
            "num_mag_values": (5, 2000),
            "brightest_alert_magnitude": (10., 21.)
        },
        save_properties = ['duration', 'duration_z0', 'long_lived'],
        post_filter_tags = ["valid_precursor",],
        post_filter_properties = {
            "peak_abs_mag": (-16, -11), "lum_dist": (0., 100.)
        },
    )
    laiss = RankingFilter(
        "LAISS_anomalies",
        ant_retriever,
        ShapleyPlotLAISS(),
        "#anomaly-detection",
        "LAISS_RFC_anomaly_score",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-1000., 99_999_999,),
            "num_mag_values": (4, 2000),
            "LAISS_RFC_anomaly_score": (30., 100.),
        },
        pre_filter_tags = ["LAISS_RFC_AD_filter",],
        save_properties = ["shap_url", "nuclear"]
    )
    sp_bright = RankingFilter(
        "superphot-plus_bright",
        ant_retriever,
        SuperphotPlusZTF(),
        "#superphot-plus-bright-followup", # change
        "superphot_plus_prob",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-20., 99_999_999,),
            "num_mag_values": (4, 50),
            "newest_alert_magnitude": (10, 18.5)
        },
        save_properties = [
            "superphot_plus_class_without_redshift", "superphot_plus_prob_without_redshift",    
            "superphot_plus_classifier", "superphot_plus_sampler", "superphot_non_Ia_prob",
        ],
        post_filter_tags = ["superphot_plus_classified",],
        post_filter_properties = {"superphot_plus_valid": (1,1), "superphot_non_Ia_prob": (0.8, 1.0)},
        groupby_properties={'superphot_plus_class': ('SN II', 'SLSN-I', 'SN IIn', 'SN Ibc')}
    )
    sp = RankingFilter(
        "superphot-plus",
        ant_retriever,
        SuperphotPlusZTF(),
        "#superphotplus",
        "superphot_plus_prob",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-200., 99_999_999,),
            "num_mag_values": (5, 2000),
        },
        save_properties = [
            "superphot_plus_class_without_redshift", "superphot_plus_prob_without_redshift",    
            "superphot_plus_classifier", "superphot_plus_sampler", "superphot_non_Ia_prob",
        ],
        post_filter_tags = ["superphot_plus_classified",],
        post_filter_properties = {
            "superphot_plus_valid": (1,1), "superphot_plus_prob": (0.5, 1.0), "superphot_non_Ia_prob": (0.8, 1.0)
        },
        groupby_properties={'superphot_plus_class': ('SN IIn', 'SN Ibc', 'SLSN-I',)}
    )
    yse_query = RankingFilter(
        "yse_candidates",
        yse_retriever,
        None,
        "#yse-daily-candidates",
        "peak_phase",
        ascending=True,
    )
    tns_query = RankingFilter(
        "tns_precursor",
        tns_retriever,
        PrecursorEmission(),
        "#precursor-emission",
        "peak_abs_mag",
        save_properties = ['duration', 'duration_z0', 'long_lived'],
        post_filter_tags = ["valid_precursor",],
        post_filter_properties = {
            "peak_abs_mag": (-16, -11), "lum_dist": (0., 100.)
        },
    )
    atlas_query = RankingFilter(
        "atlas_superphot-plus",
        atlas_retriever,
        SuperphotPlusZTF(),
        "#superphotplus",
        "superphot_plus_prob",
        save_properties = [
            "superphot_plus_class_without_redshift", "superphot_plus_prob_without_redshift",    
            "superphot_plus_classifier", "superphot_plus_sampler", "superphot_non_Ia_prob",
        ],
        post_filter_tags = ["superphot_plus_classified",],
        post_filter_properties = {
            "superphot_plus_valid": (1,1), "superphot_plus_prob": (0.5, 1.0), "superphot_non_Ia_prob": (0.8, 1.0)
        },
        groupby_properties={'superphot_plus_class': ('SN IIn', 'SN Ibc', 'SLSN-I',)}
    )
    all_filters = [
        laiss,
        tns_query,
        sp_bright,
        sp,
        atlas_query,
        yse_query,
        precursor,
        nuclear,
    ]
    return all_filters


def run():
    current_time = Time.now().mjd
    ranker = ANTARESRanker() # lookback of 1 day
    
    for filt in all_current_filters():
        for _ in range(3): # three attempts
            #try:
            ranker.run(filt, 10) # max_num
            break
            #except:
            #    pass

if __name__ == '__main__':
    run()
    
