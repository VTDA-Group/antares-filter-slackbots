from astropy.time import Time
from antares_filter_slackbots.antares_ranker import ANTARESRanker, RankingFilter
from antares_filter_slackbots.filters import SuperphotPlusZTF, ShapleyPlotLAISS

def all_current_filters():
    """Where all filters to run are defined.
    """
    current_time = Time.now().mjd
    all_filters = [
        # superphot-plus filter
        RankingFilter(
            "superphot-plus-bright",
            SuperphotPlusZTF(),
            "#slackbot-test", #"#superphot-plus-bright-followup", # change
            "superphot_plus_prob",
            pre_filter_properties = {
                "oldest_alert_observation_time": (current_time-20., 99_999_999,),
                "num_mag_values": (4, 50),
                "newest_alert_magnitude": (10, 18.5)
            },
            save_properties = [
                "superphot_plus_class_without_redshift", "superphot_plus_prob_without_redshift",    
                "superphot_plus_classifier", "superphot_plus_sampler",
            ],
            post_filter_tags = ["superphot_plus_classified",],
            post_filter_properties = {"superphot_plus_valid": (1,1), "superphot_plus_prob": (0.4, 1.0)},
            groupby_properties={'superphot_plus_class': ('SN II', 'SLSN-I', 'SN IIn', 'SN Ibc')}
        ),
        RankingFilter(
            "superphot-plus",
            SuperphotPlusZTF(),
            "#slackbot-test", #"#superphotplus",
            "superphot_plus_prob",
            pre_filter_properties = {
                "oldest_alert_observation_time": (current_time-100., 99_999_999,),
                "num_mag_values": (4, 500),
            },
            save_properties = [
                "superphot_plus_class_without_redshift", "superphot_plus_prob_without_redshift",    
                "superphot_plus_classifier", "superphot_plus_sampler",
            ],
            post_filter_tags = ["superphot_plus_classified",],
            post_filter_properties = {"superphot_plus_valid": (1,1), "superphot_plus_prob": (0.4, 1.0)},
            groupby_properties={'superphot_plus_class': ('SN IIn', 'SN Ibc')}
        ),
        # anomaly detection filter: currently just uses ANTARES version of filter
    ]
    """
    RankingFilter(
        "LAISS_anomalies",
        ShapleyPlotLAISS(),
        "#anomaly-detection",
        "LAISS_RFC_anomaly_score",
        pre_filter_properties = {
            "oldest_alert_observation_time": (current_time-1000., 99_999_999,),
            "num_mag_values": (3, 500),
            "LAISS_RFC_anomaly_score": (30., 100.),
        },
        pre_filter_tags = ["LAISS_RFC_AD_filter",],
        save_properties = ["shap_url"]
    ),
    """
        
    return all_filters


def run():
    ranker = ANTARESRanker(1.0) # lookback of 1 day

    for filt in all_current_filters():
        ranker.run(filt, 10) # max_num

if __name__ == '__main__':
    run()