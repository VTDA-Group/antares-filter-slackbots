import antares.devkit as dk
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import truncnorm

import io
import torch
import torch.nn as nn
import torch.nn.functional as F

from jax import random, lax, jit
from numpyro.infer import SVI, Trace_ELBO

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.surveys.surveys import Survey
from superphot_plus.import_utils import clip_lightcurve_end
from top_level_utils import (
    get_meta_features,
    get_galactic_coordinates,
    estimate_period
)
    
    
def preprocess_lightcurve(lc_arr, name, *, survey: Survey):
    """
    Imports all relevant data relevant to one lightcurve/locus,
    removes points with NaN errors, converts to flux units, and 
    performs data clipping/pre-processing for better fitting.
    
    Parameters
    ----------
    lc_arr : numpy array
        lightcurve t, f, ferr, b
    name : str
        Target name
    survey : Survey
        Survey to use, e.g. LSST
    
    Returns
    ----------
    preprocessed_lc : Lightcurve
        reformatted light curve
    """
    # remove invalid points
    nan_entries = np.any(np.isnan(lc_arr[:4]), axis=0)
    lc_arr_pruned = lc_arr[:, ~nan_entries]
    
    ra, dec, mwebv = np.nanmean(lc_arr_pruned[4:], axis=1)
    
    # convert to Lightcurve object
    lc = Lightcurve(
        times = lc_arr_pruned[0],
        fluxes = lc_arr_pruned[1],
        flux_errors = lc_arr_pruned[2],
        bands = lc_arr_pruned[3],
        name = name
    )
    
    ext_dict = survey.get_band_extinctions(ra, dec, mwebv=mwebv)
    for unique_b in ext_dict:
        lc.fluxes[lc.bands == unique_b] *= 10**(0.4 * ext_dict[unique_b])
    
    # sort lightcurve
    lc.sort_by_time()
    
    # clip both ends
    clipped_lc = clip_lightcurve_end(
        lc.times, lc.fluxes, lc.flux_errors, lc.bands
    )
    reversed_clipped_lc = clip_lightcurve_end(
        *clipped_lc[:,::-1]
    )
    lc.times, lc.fluxes, lc.flux_errors, lc.bands = reversed_clipped_lc[:,::-1]
    
    if len(lc.times) < 3:
        raise ValueError
        
    return lc, ra, dec

    
class Superphot_Plus_ELASTICC2_v1(dk.Filter):
    NAME = "Superphot+ for ELASTICC2 (v1)"
    ERROR_SLACK_CHANNEL = "U03QP2KEK1V"  # Put your Slack user ID here
    INPUT_LOCUS_PROPERTIES = []
    
    INPUT_ALERT_PROPERTIES = [
        'ant_mjd',
        'rubin_source_psFlux',
        'rubin_source_psFluxErr',
        'rubin_source_filterName',
        'rubin_source_ra',
        'rubin_source_decl',
        'rubin_object_mwebv'
    ]

    OUTPUT_LOCUS_PROPERTIES = [
        {
            'name': 'classifications',
            'type': 'elasticc_classification',
            'description': 'classifications of a model for elasticc data'
        },
    ]

    OUTPUT_LOCUS_PROPERTIES = []
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = [
        {
            'name': 'superphot_plus_classified',
            'description': 'Classified for the ELASTICC challenge with the Superphot+ pipeline.'
        }
    ]

    REQUIRES_FILES = [
        'superphot_elasticc2_recurring_model_v1.pt',
        'superphot_elasticc2_nonrecurring_model_v1.pt',
        'superphot_elasticc2_top_level_model_v1.pt'
    ]

    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        self.name = "Superphot+_ELASTICC2_v1"
        self.parameters = "v1"
        
        self.nonrecurring_classes = [
            110, 111, 112, 113, 114,
            115, 121, 122, 123, 124,
            131, 132, 133, 134, 135
        ]
        self.recurring_classes = [
            211, 212, 213, 214, 221
        ]
        
        self.nonrecurring_classes_to_labels = {
            e: x for e, x in enumerate(self.nonrecurring_classes)
        } 
        self.recurring_classes_to_labels = {
            e: x for e, x in enumerate(self.recurring_classes)
        } 
        
        self.recurring_prob = 0.
        self.nonrecurring_prob = 0.
        
        # load model files
        recurring_model_fn = self.files[
            'superphot_elasticc2_recurring_model_v1.pt'
        ]
        recurring_config_fn = self.files[''] # placeholder
        nonrecurring_model_fn = self.files[
            'superphot_elasticc2_nonrecurring_model_v1.pt'
        ]
        nonrecurring_config_fn = self.files[''] # placeholder

        top_level_model_fn = self.files[
            'superphot_elasticc2_top_level_model_v1.pt'
        ]
        top_level_config_fn = self.files[''] # placeholder

        
        stream = io.BytesIO(recurring_model_fn)
        stream2 = io.BytesIO(recurring_config_fn)
        self.top_level_model, _ = RecurringClassifier.load(
            stream,
            stream2
        )
        
        stream = io.BytesIO(nonrecurring_model_fn)
        stream2 = io.BytesIO(nonrecurring_config_fn)
        self.nonrecurring_model, _ = SuperphotClassifier.load(
            stream,
            stream2
        )
        
        stream = io.BytesIO(top_level_model_fn)
        stream2 = io.BytesIO(top_level_config_fn)
        self.top_level_model, _ = TopLevelClassifier.load(
            stream,
            stream2
        )

        self.sampler = DynestySampler()
        self.survey = Survey.LSST()
                
        optimizer = numpyro.optim.Adam(step_size=0.001)
        self.svi = SVI(jax_model, jax_guide, optimizer, loss=Trace_ELBO())
        self.svi_state = None
        self.num_iter = 10_000
        self.lax_jit = jit(lax_helper_function, static_argnums=(0, 2))
        
        
        
    def add_classification(self, class_id, prob):
        """Helper function to add classification result.
        """
        self.classifications.append(
            {
            "classifierName": self.name,
            "classifierParams": self.parameters,
            "classId": class_id,
            "probability": prob,
            }
        )
        
    def distribute_prob_evenly(self, recurring):
        """Distributes prob evenly among less likely class.
        """
        if recurring:
            N = len(self.recurring_classes)
            for class_id in self.recurring_classes:
                self.add_classification(
                    int(class_id),
                    self.recurring_prob / N,
                )
        else:
            N = len(self.nonrecurring_classes)
            for class_id in self.nonrecurring_classes:
                self.add_classification(
                    int(class_id),
                    self.nonrecurring_prob / N,
                )
                
    def run(self, locus):
        """
        Runs a filter that first applies a top-level classifier, and then fits all suspected non-recurring
        transients tagged with supernovae-like properties to the model described by de Soto et al, TBD. Saves the 
        median model parameters found using nested sampling to later be input into the classifier of choice. Then, 
        inputs all posterior sample model parameters in a pre-trained classifier to estimate the most-likely SN 
        class and confidence score.
        
        Parameters
        ----------
        locus : Locus object
            the SN-like transient to be fit
        """
        self.classifications = []
        
        # gets listed properties of each alert in Locus, sorted by MJD
        ts = locus.timeseries[['ant_mjd', 'rubin_source_psFlux', 'rubin_source_psFluxErr', 'rubin_source_filterName', 'rubin_source_ra', 'rubin_source_decl', 'rubin_object_mwebv']]   
        arr = ts.to_pandas().to_numpy().T

        # preprocessing + calculating meta features
        try:
            lc, ra, dec = preprocess_lightcurve(arr, locus.locus_id, survey=self.survey)
        except:
            return
        
        # apply top-level classifier
        meta_features = get_meta_features(lc, ra, dec)
        (
            self.recurring_prob,
            self.nonrecurring_prob
        ) = self.top_level_model.classify_from_fit_param(meta_features)
        
        self.add_classification(2, self.recurring_prob)
        self.add_classification(1, self.nonrecurring_prob)
        
        recurring = self.recurring_prob > self.nonrecurring_prob
        
        self.distribute_prob_evenly(~recurring)
        
        if recurring:
            # normed_recurring_params = (meta_features - self.recurring_means) / self.recurring_stddevs
            # probs_recurring = get_predictions(self.recurring_model, torch.Tensor(np.array([normed_recurring_params,])), 'cpu').numpy()
            self.recurring_model_probs = self.recurring_model.classify_from_fit_param(meta_features)
            for e, class_id in enumerate(self.recurring_classes):
                self.add_classification(
                    int(class_id),
                    self.recurring_prob * self.recurring_model_probs[e]
                )
            
        else:  # non-recurring
            #print("starting run nested sampling")
            gri_lc = lc.filter_by_band(["g", "r", "i"], in_place=False)
            max_flux_r = gri_lc.find_max_flux(band="r")
            
            gri_samples, red_neg_chisq, self.svi_state = _svi_helper_no_recompile(
                gri_lc,
                max_flux_r,
                self.survey.priors,
                self.svi,
                self.svi_state,
                self.lax_jit,
                self.num_iter,
            )

            if gri_samples is None:
                self.distribute_prob_evenly(False)
                
            mean_params = gri_samples.sample_mean() # TODO: only get included bands
            mean_params = np.append(mean_params, np.mean(red_neg_chisq))
            
            """
            ref_band_idx = np.where(
                self.priors.ordered_bands == self.priors.reference_band
            )[0][0]
            
            mean_r = mean_params[7*ref_band_idx:7*(ref_band_idx+1)]
            
            for aux_b in ["u", "z", "Y"]:
                aux_lc = lc.filter_by_band(aux_b, in_place=False)
                
                aux_priors = self.survey.priors.filter_by_band(
                    [self.priors.reference_band, aux_b]
                )
                
                aux_samples, aux_neg_chisq, self.svi_state = _svi_helper_no_recompile(
                    aux_lc,
                    max_flux_r,
                    aux_priors,
                    self.svi,
                    self.svi_state,
                    self.lax_jit,
                    self.num_iter,
                )

                if samples_aux is None:
                    self.distribute_prob_evenly(False)

                if aux_b == 0:
                    eq_samples = np.hstack((eq_samples_aux, eq_samples))
                else:
                    eq_samples = np.hstack((eq_samples, eq_samples_aux))
            """
            probs_all = []
            for eq_s in eq_samples:
                post = np.append(eq_s, meta_features)
                
                probs_all.append(probs_single)

            probs_avg = np.mean(np.array(probs_all), axis=0)
            for e, class_id in enumerate(self.nonrecurring_classes):
                self.add_classification(
                        int(class_id),
                        probs[1].item() * probs_avg[e].item(),
                    )
                #print(int(class_id), probs[1].item() * probs_avg[e].item())
            #print(probs[1].item(), np.sum(probs_avg))
        
        locus.set_classifications(self.name, self.classifications)
        locus.tags.add('superphot_plus_classified')
        
        
"""
def main():
    # Fetch 2 example Locus IDs from the test dataset (classification known)
    locus_ids = ['ANT2021ocix4', 'ANT2021cg25w']
    report = dk.run_many(DesotoSNFit, locus_ids=locus_ids)
"""    
        