import os
import pandas as pd
import numpy as np
from pathlib import Path
from snapi.query_agents import ANTARESQueryAgent
from dustmaps.config import config as dustmaps_config
from astro_ghost.photoz_helper import calc_photoz

def data_dir():
    return os.path.join(
        Path(__file__).parent.parent.parent.absolute(),
        "data"
    )

def merge_ghost_dfs(*dfs):
    """Merge GHOST dataframes with archival data.
    """
    merged_df = dfs[0]

    for df in dfs[1:]:
        intersect_cols = np.intersect1d(merged_df.columns, df.columns)
        merged_df = merged_df.merge(df, on=list(intersect_cols), how='outer')
        grouped = merged_df.groupby('objName')['TransientName'].nunique()
        unique_transient_names = grouped[grouped > 1].index
        merged_df = merged_df[merged_df['objName'].isin(unique_transient_names) | ~merged_df.duplicated(subset='objName', keep='first')]

    return merged_df

def extract_new_hosts(df):
    """Extract hosts not from the original catalog.
    """
    new_transients_mask = df.TransientName.map(lambda x: x[:3] in ['ZTF', 'ANT'])
    masked_df = df.loc[new_transients_mask, :]
    masked_df.loc[:,'TransientName'] = masked_df.TransientName.map(lambda x: x.split("\n")[0])
    return masked_df

def convert_transients_to_tns(df):
    """Convert TransientName to TNS names.
    """
    qa = ANTARESQueryAgent()
    tns_names = []
    for row in df.itertuples():
        qr, _ = qa.query_by_name(row.TransientName)
        for query_result in qr:
            tns_name = query_result.objname
            int_names = query_result.internal_names
            for i in int_names:
                if i[:4].isnumeric():
                    tns_name = i
                    break
        tns_names.append(tns_name)

    df.TransientName = tns_names
    print(tns_names)
    return df

def calculate_all_redshifts(df):
    """Populate dataframe with NED/redshift information.
    """
    pass

def add_redshift_info(df):
    """Query GHOST for redshift information.
    """
    dustmaps_config.reset()
    dustmaps_config['data_dir'] = os.path.join(
        Path(__file__).parent.parent.parent.absolute(),
        "data/dustmaps"
    )
    photo_z_hosts = calc_photoz(
        df.loc[df.NED_redshift.isna(), :],
        dust_path=dustmaps_config['data_dir'],
        model_path=os.path.join(
            Path(__file__).parent.parent.parent.absolute(),
            "data/MLP_lupton.hdf5"
        )
    )
    merged_df = pd.concat([df, photo_z_hosts], join='outer')
    # remove duplicates, keep latest instance with same TransientName
    merged_df.drop_duplicates(subset='TransientName', keep='last', inplace=True)
    return merged_df

def merge_voting_histories(current_df, archival_df, save_prefix):
    """Merge current and archival voting histories.
    """
    archival_df.rename(columns={'User': 'UserName'}, inplace=True)
    archival_df.replace(
        {
            'Anomalous': 'anomalous',
            'Not Anomalous': 'not_anomalous',
            'Very Anomalous': 'very_anomalous',
            'AGN': 'agn',
            'Not an AGN': 'not_agn',
        }, inplace=True
    )
    current_df.replace(
        {
            'upvote': 'anomalous',
            'downvote': 'not_anomalous',
            'strong_upvote': 'very_anomalous',
            'AGN': 'agn',
        }, inplace=True
    )
    merged_df = pd.concat([archival_df, current_df], join='outer')
    reduced_df1 = merged_df[['Transient', 'UserID', 'UserName', 'Response', 'TimeStamp']]
    reduced_df2 = merged_df[['Transient', 'UserID', 'UserName', 'Response']].drop_duplicates(subset=['Transient', 'UserName'], keep='last')
    reduced_df1.to_csv(save_prefix + "_with_repeats.csv", index=False)
    reduced_df2.to_csv(save_prefix + "_no_repeats.csv", index=False)


if __name__ == '__main__':
    data_path = data_dir()
    ghost_path = os.path.join(data_path, "ghost")
    """
    orig_df = pd.read_csv(os.path.join(ghost_path, "database", "GHOST.csv"))
    archival_df1 = pd.read_csv(os.path.join(ghost_path, "ghost_archival1.csv"))
    archival_df2 = pd.read_csv(os.path.join(ghost_path, "ghost_archival2.csv"))

    merged_df = merge_ghost_dfs(orig_df, archival_df1, archival_df2)
    new_df = extract_new_hosts(merged_df)
    tns_df = convert_transients_to_tns(new_df)
    redshift_df = add_redshift_info(tns_df)
    redshift_df.to_csv(os.path.join(ghost_path, "all_new_hosts.csv"), index=False)
    """
    archival_votes = os.path.join(data_path, "archival", "slack_reactions.csv")
    current_votes = os.path.join(data_path, "LAISS_anomalies", "voting_history.csv")
    archival_df = pd.read_csv(archival_votes)
    current_df = pd.read_csv(current_votes)
    merge_voting_histories(current_df, archival_df, os.path.join(data_path, "LAISS_anomalies", "merged_votes"))