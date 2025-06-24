from devkit2_poc.models import BaseFilter
import os
import warnings
from astropy.cosmology import Planck15 as cosmo
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np


class PrecursorEmission(BaseFilter):
    NAME = "Precursor emission filter"
    ERROR_SLACK_CHANNEL = "U03QP2KEK1V"  # Put your Slack user ID here
    INPUT_LOCUS_PROPERTIES = []
    INPUT_ALERT_PROPERTIES = []
    
    OUTPUT_LOCUS_PROPERTIES = [
        {
            'name': 'peak_mag',
            'type': 'float',
            'description': 'Peak absolute mag of event.'
        },
        {
            'name': 'lum_dist',
            'type': 'float',
            'description': 'Luminosity distance to event.'
        },
        {
            'name': 'duration',
            'type': 'float',
            'description': 'Duration of event.'
        },
        {
            'name': 'duration_z0',
            'type': 'float',
            'description': 'Duration of event when redshift corrected.'
        },
        {
            'name': 'long_lived',
            'type': 'int',
            'description': 'True (1) if duration > 50 days.'
        },
    ]
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = [
        {
            'name': 'valid_precursor',
            'description': 'Not nuclear and with valid redshift'
        },
    ]
    
    REQUIRES_FILES = []
        
    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        self.data_dir = os.path.join(
            Path(__file__).parent.parent.parent.parent.absolute(), "data/precursor"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.input_properties = [
            'best_redshift',
            'nuclear',
            'tns_class',
            'oldest_alert_observation_time',
            'newest_alert_observation_time',
            'peak_abs_mag',
        ]
        

    def _run(self, event_dict, ts):
        """
        Function applies to each locus.
        """
        redshift = event_dict['best_redshift']
        print("REDSHIFT", redshift)
        #redshift = 0.0362
        
        print(event_dict['tns_class'])
        
        if isinstance(event_dict['tns_class'], float):
            event_dict['tns_class'] = '---'
        
        event_dict['valid_nuclear'] = (event_dict['nuclear']) and ((
            event_dict['tns_class'] is None
        ) or (event_dict['tns_class'][:2] != "SN"))
        
        event_dict['valid_precursor'] = ((
            event_dict['tns_class'] is None
        ) or (event_dict['tns_class'][:2] != "SN")) and (not event_dict['nuclear']) and ~np.isnan(redshift)
        
        oldest_alert = event_dict['oldest_alert_observation_time']
        newest_alert = event_dict['newest_alert_observation_time']
        duration = newest_alert - oldest_alert
        event_dict['peak_abs_mag'] = event_dict['peak_abs_mag']
        event_dict['lum_dist'] = cosmo.luminosity_distance(redshift).value        
        event_dict['duration'] = duration
        event_dict['duration_z0'] = duration / (1. + redshift)
        event_dict['long_lived'] = int(duration > 50.)
        
        print(event_dict['name'], event_dict['valid_precursor'], event_dict['peak_abs_mag'], event_dict['lum_dist'])
        
        return event_dict
