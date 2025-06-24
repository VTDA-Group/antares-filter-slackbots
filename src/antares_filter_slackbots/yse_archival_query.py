from astropy.time import Time
import pandas as pd
import os
from pathlib import Path

from antares_filter_slackbots.retrievers import ArchivalYSERetriever
from antares_filter_slackbots.antares_ranker import ANTARESRanker, RankingFilter


def prune_unknown_yse():
    """Return DF of all potential YSE SNe previously ignored."""
    yse_retriever = ArchivalYSERetriever(4.0)
    ranker = ANTARESRanker()
    filt = RankingFilter(
        "archival-yse",
        yse_retriever,
        None,
        None,
        "peak_phase",
        ascending=True,
    )
    
    df, ts_dict = yse_retriever.retrieve_candidates(filt, None)
        
    if df is None:
        print("NO PREPROCESSED")
        return

    host_df = ranker.add_host_galaxy_info(df, yse_retriever)
    if host_df is None:
        print("NO HOSTS")
        return

    final_df = ranker.apply_filter_to_df(host_df, ts_dict, filt)
    
    if final_df is None:
        print("NO FINAL")
        return
    
    # save to local df
    save_path = os.path.join(os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        ) # src
    ), "data", 'archival_yse_candidates.csv')
    print(save_path)
    final_df.to_csv(save_path)


if __name__ == "__main__":
    prune_unknown_yse()
