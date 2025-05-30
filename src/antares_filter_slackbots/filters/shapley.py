import antares.devkit as dk
import os
import requests
import warnings
import pickle
import yaml
from pathlib import Path
warnings.filterwarnings("ignore")

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..auth import toku
from ..slack_requests import setup_client

def dict_merge(dict_list):
    dict1 = dict_list[0]
    for dict in dict_list[1:]:
        dict1.update(dict)
    return(dict1)

class ShapleyPlotLAISS(dk.Filter):
    NAME = "Shapley Plot Generator for LAISS"
    ERROR_SLACK_CHANNEL = "U03QP2KEK1V"  # Put your Slack user ID here
    INPUT_LOCUS_PROPERTIES = []
    INPUT_ALERT_PROPERTIES = []
    
    OUTPUT_LOCUS_PROPERTIES = [
        {
            'name': 'shap_url',
            'type': 'str',
            'description': f'URL for shapley force plot to identify potential cause for anomaly flag.'
        }
    ]
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = []
    
    REQUIRES_FILES = []
        
    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        self.data_dir = os.path.join(
            Path(__file__).parent.parent.parent.parent.absolute(), "data/shapley"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "force_plots"), exist_ok=True)

        # set up shapley features
        # lc features
        with open(os.path.join(self.data_dir, 'shapley_descriptions.yaml')) as stream:
            descript_dict = yaml.safe_load(stream)

        lc_descripts = descript_dict['lightcurve']

        #add all g and r-band features here
        lc_descripts_bands = {}

        for key, val in lc_descripts.items():
            for band in 'gr':
                lc_descripts_bands[key + f'_{band}'] = val + f", in {band}"

        #host features
        host_descripts = descript_dict['host']

        #add all g and r-band features here
        host_descripts_bands = {}
        for key, val in host_descripts.items():
            for band in 'grizy':
                host_descripts_bands[band + key] = val + f", in {band}"

        host_descripts_nonGeneric = descript_dict['host_nongeneric']

        self.shapley_descriptions = dict_merge([lc_descripts_bands, host_descripts_bands, host_descripts_nonGeneric])
        self.shapley_features = self.shapley_descriptions.keys()
        self.input_properties = list(self.shapley_features)
        self.input_properties.extend(['name', 'LAISS_RFC_anomaly_score'])

        self.channel_id = "C078CJZE3K5"
        self._client = setup_client()

        # import RF info
        with open(os.path.join(self.data_dir, 'ad_random_forest.pkl'), 'rb') as f:
            clf = pickle.load(f)

        RFdata = pd.read_csv(
            os.path.join(self.data_dir, "ad_random_forest_train_data.csv.gz")
        )
        self.explainer = shap.TreeExplainer(clf, data=RFdata[self.shapley_features])


    def run(self, event_dict, ts):
        """
        Function applies to each locus.
        """
        #first, store the shapley values 
        shap_features = {}
        for feat in self.shapley_features:
            try:
                shap_features[feat] = event_dict[feat]
            except:
                shap_features[feat] = np.nan

        plotpath, length = self.plot_shap(event_dict['name'], shap_features)
        filename = plotpath.split("/")[-1]
        file_id = self.upload_file(toku, event_dict['name'], plotpath, length)
        self.make_file_public(toku, file_id)
        initial_url = self.share_public_link(toku, file_id)
        final_url = self.format_url(initial_url, filename)
        event_dict["shap_url"] = final_url
        return event_dict
        
    def plot_shap(self, antares_id, shap_features):
        """Generate Shapley force plot from object ID and
        feature dictionary."""
        lc_and_hosts_df = pd.DataFrame(shap_features, index=[0])
        chosen_instance = lc_and_hosts_df[self.shapley_features].iloc[[-1]]
        shap_values = self.explainer.shap_values(chosen_instance)
        fig = shap.force_plot(
            self.explainer.expected_value[1],
            shap_values[0][:, 1],
            chosen_instance,
            matplotlib=True, show=False,
            text_rotation=15, feature_names=self.shapley_features
        );

        filepath = os.path.join(self.data_dir, f"force_plots/{antares_id}_ForcePlot.png")
        plt.title(f"Force Plot for {antares_id}\n\n\n\n", fontsize=16, fontweight='bold')
        fig.patch.set_edgecolor('k')
        plt.savefig(
            filepath, dpi=200, bbox_inches='tight',
            facecolor='white', pad_inches=0.3,
            transparent=False,
            edgecolor=fig.get_edgecolor()
        );
        print(f"File successfully saved at {filepath}.")
        length = os.path.getsize(filepath)
        return filepath, length

    def upload_file(self, token, antares_id, file_path, length):
        """Upload file to Slack.
        """
        response = self._client.files_getUploadURLExternal(
            filename=file_path,
            length=length + 16,
            initial_comment=f"Here is the SHAP plot for {antares_id}",
        )
        
        upload_url = response.get('upload_url')
        file_id = response.get('file_id')
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
            put_response = requests.put(upload_url, data=file_data)
            print(put_response)
            
       # confirm upload
        response = self._client.files_completeUploadExternal(
            files=[{'id': file_id,},],
            channels=self.channel_id,
        )
        return file_id

    def make_file_public(self, token, file_id):
        """Make file publicly accessible by all members.
        """
        url = "https://slack.com/api/files.sharedPublicURL"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        data = {
            'file': file_id
        }
        response = requests.put(url, headers=headers, data=data)
        return response

    def share_public_link(self, token, file_id):
        """Get the public URL from the file response.
        """
        response = self._client.files_info(file=file_id)
        public_url = response.get('file')['permalink_public']
        return public_url

    def format_url(self, url, filename):
        """Correct URL formatting.
        """
        team_id = url.split("/")[-1].split("-")[0]
        file_id = url.split("/")[-1].split("-")[1]
        pub_secret = url.split("/")[-1].split("-")[2]

        formatted_url = f"https://files.slack.com/files-pri/{team_id}-{file_id}/{filename.lower()}?pub_secret={pub_secret}"
        return formatted_url        
