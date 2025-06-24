import os
import pandas as pd
from devkit2_poc.models import DevKitLocus

def generate_alerts_from_file(alerce_fn):
    """Generate alert dictionary from ALeRCE downloaded
    file.
    """
    df = pd.read_csv(alerce_fn)
    ra = df['ra'].dropna().median()
    dec = df['dec'].dropna().median()
    name = df['oid'].iloc[0]
    
    timeseries = df[['candid', 'mjd', 'mag', 'e_mag', 'diffmaglim', 'fid', 'ra', 'dec']]
    
    brightest_mag = df.mag.dropna().min()
    brightest_time = df.loc[df.mag.dropna().idxmin(), 'mjd']
    
    properties = {
        'brightest_alert_magnitude': brightest_mag,
        'brightest_alert_observation_time': brightest_time,
        'oldest_alert_observation_time': df.mjd.dropna().min(),
        'newest_alert_observation_time': df.mjd.dropna().max(),
    }
    
    alerts = []
    for row in timeseries.itertuples():
        alerts.append(
            {
                'alert_id': row.candid,
                'locus_id': name,
                'mjd': row.mjd,
                'properties': {
                    'ant_mjd': row.mjd,
                    'ant_mag': row.mag,
                    'ant_magerr': row.e_mag,
                    'ant_maglim': row.diffmaglim,
                    'ant_survey': 1,
                    'ant_passband': ('g' if row.fid == 1 else 'r'),
                    'ant_ra': row.ra,
                    'ant_dec': row.dec,
                },
            }
        )
    
    return name, ra, dec, alerts, properties

    
def generate_locus_from_file(alert_fn):
    """From file with detections, generate locus.
    """
    name, ra, dec, alerts, properties = generate_alerts_from_file(alert_fn)
        
    locus_dict = {
        'locus_id': name,
        'ra': ra,
        'dec': dec,
        'properties': {
            **properties,
            'num_alerts': len(alerts),
            'num_mag_values': len(alerts),
            'ztf_object_id': name,
        },
        'tags': [],
        'watch_list_ids': [],
        'watch_object_ids': [],
        'catalog_objects': {},
        'alerts': alerts,
    }

    locus = DevKitLocus.model_validate(locus_dict)
    return locus

    
if __name__ == "__main__":
    alerce_fn = os.path.join(os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        ) # src
    ), "data", "test_loci", "test_precursors", "2019fmb.csv")
    
    generate_locus_from_file(alerce_fn)