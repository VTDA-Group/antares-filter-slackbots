from astropy.time import Time
from superphot_plus_antares.antares_ranker import ANTARESRanker, RankingFilter
from superphot_plus_antares.filters import SuperphotPlusZTF

def all_current_filters():
    """Where all filters to run are defined.
    """
    current_time = Time.now().mjd
    all_filters = [
        # anomaly detection filter: currently just uses ANTARES version of filter
        RankingFilter(
            "LAISS_anomalies",
            None,
            "#slackbot-test",
            "LAISS_RFC_anomaly_score",
            pre_filter_properties = {
                "oldest_alert_observation_time": (current_time-200., 99_999_999,),
                "num_mag_values": (3, 500),
                "LAISS_RFC_anomaly_score": (50., 100.)
            },
            pre_filter_tags = ["LAISS_RFC_AD_filter",],
        ),

        # superphot-plus filter
        RankingFilter(
            "superphot-plus",
            SuperphotPlusZTF(),
            "#slackbot-test",
            "superphot_plus_class_prob",
            pre_filter_properties = {
                "oldest_alert_observation_time": (current_time-200., 99_999_999,),
                "num_mag_values": (3, 500),
            },
            save_properties = ["superphot_plus_classifier", "superphot_plus_sampler",],
            post_filter_tags = ["superphot_plus_classified",],
            post_filter_properties = {"superphot_plus_valid": (1,1), "superphot_plus_class_prob": (0.5, 1.0)},
            groupby_properties={'superphot_plus_class': ('SN IIn', 'SN Ibc')}
        )
    ]
    return all_filters

def run():
    ranker = ANTARESRanker(1.0) # lookback of 1 day

    for filt in all_current_filters():
        ranker.run(filt, 5) # max_num

if __name__ == '__main__':
    run()