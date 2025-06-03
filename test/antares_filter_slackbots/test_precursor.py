import pytest
import os
from astropy.time import Time
from antares_filter_slackbots import (
    ANTARESRanker,
    TestANTARESRetriever,
    RankingFilter,
    generate_locus_from_file
)
from antares_filter_slackbots.filters import PrecursorEmission


@pytest.mark.parametrize(
    "filename", ["2019fmb", "2025fsm",]
)
def test_precursor(filename) -> None:
    """Verify the output of the `greetings` function"""
    alerce_fn = os.path.join(os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        ) # test
    ), "data", "test_loci", "test_precursors", f"{filename}.csv")
    locus = generate_locus_from_file(alerce_fn)
    
    current_time = Time.now().mjd
    test_ant_retriever = TestANTARESRetriever() 
    filt = RankingFilter(
        "precursor-emission",
        test_ant_retriever,
        PrecursorEmission(),
        "#slackbot-test", #"#precursor-emission",
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
    
    ranker = ANTARESRanker()
    df, ts_dict = test_ant_retriever.retrieve_candidates(filt, [locus,])
    assert len(df) == 1
    
    host_df = ranker.add_host_galaxy_info(df, test_ant_retriever)
    assert len(host_df) == 1

    final_df = ranker.apply_filter_to_df(host_df, ts_dict, filt)
    assert len(final_df) == 1
