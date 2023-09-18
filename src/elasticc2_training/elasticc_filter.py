import antares.devkit as dk
import numpy as np
import extinction
from astropy.coordinates import SkyCoord
from dynesty import NestedSampler
from scipy.stats import truncnorm
from dynesty import utils as dyfunc
import P4J

import io
import torch
import torch.nn as nn
import torch.nn.functional as F

from superphot_plus.lightcurve import Lightcurve


def get_meta_features(lc, ra, dec):
    """
    Calculate the meta features used in the top level and recurring classifier.
    
    TO ADD:
    - MWEBV
    - HOST/SOURCE OFFSET (both in alert file)
    - HOST GAL MAG
    """
    t, f, ferr, b = lc.times, lc.fluxes, lc.flux_errors, lc.bands
    lin_slopes = []
    
    for b_opt in range(6):
        try:
            if len(f[b == b_opt]) >= 2:
                lin_slopes.append(
                    curve_fit(
                        lambda x, *p: p[0]*x + p[1],
                        t[b == b_opt],
                        f[b == b_opt],
                        [0, 0],
                        sigma=ferr[b == b_opt]
                    )[0][0]
                )
            else:
                lin_slopes.append(0.)
        except:
            lin_slopes.append(0.)

    u_r_max = 0
    g_r_max = 0
    i_r_max = 0
    z_r_max = 0
    Y_r_max = 0

    u_r_mean = 0
    g_r_mean = 0
    i_r_mean = 0
    z_r_mean = 0
    Y_r_mean = 0

    try:
        u_r_max = np.abs(np.max(f[b == 0]) / np.max(f[b == 2]))
        u_r_mean =  np.abs(np.mean(f[b == 0]) / np.mean(f[b == 2]))
    except:
        pass

    try:
        g_r_max = np.abs(np.max(f[b == 1]) / np.max(f[b == 2]))
        g_r_mean = np.abs(np.mean(f[b == 1]) / np.mean(f[b == 2]))
    except:
        pass

    try:
        i_r_max = np.abs(np.max(f[b == 3]) / np.max(f[b == 2]))
        i_r_mean = np.abs(np.mean(f[b == 3]) / np.mean(f[b == 2]))
    except:
        pass

    try:
        z_r_max = np.abs(np.max(f[b == 4]) / np.max(f[b == 2]))
        z_r_mean = np.abs(np.mean(f[b == 4]) / np.mean(f[b == 2]))
    except:
        pass

    try:
        Y_r_max = np.abs(np.max(f[b == 5]) / np.max(f[b == 2]))
        Y_r_mean = np.abs(np.mean(f[b == 5]) / np.mean(f[b == 2]))
    except:
        pass

    positive_fraction = len(f[f > 0]) / len(f)
    best_period = estimate_period(t, f, ferr)
    best_period_long = estimate_period(t, f, ferr, 0.2) # only 5 days or longer
    gal_b, gal_l = get_galactic_coordinates(ra, dec)

    return np.array([gal_b, gal_l, positive_fraction, best_period, best_period_long, *lin_slopes, u_r_max, u_r_mean, g_r_max, g_r_mean, i_r_max, i_r_mean, z_r_max, z_r_mean, Y_r_max, Y_r_mean])


def get_galactic_coordinates(ra, dec):
    """
    Get galactic coordinates corresponding to RA and Dec
    """
    coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
    g_coords = coords.galactic
    return g_coords.b.degree, g_coords.l.degree


def estimate_period(t, f, ferr, fmax=50.):
    """
    Use MHAOV to estimate the best period of an assumed periodic signal.
    """
    #freqs = np.linspace(0., fmax, num=1e5)
    #pgram = lombscargle(t, f, freqs, precenter=True)
    #fbest = freqs[np.argmax(pgram)]
    my_per = P4J.periodogram(method='MHAOV')
    my_per.set_data(t, f - np.mean(f), ferr, 6) # shift to be centered vertically around 0
    my_per.frequency_grid_evaluation(fmin=0.0, fmax=fmax, fresolution=1e-3)  # frequency sweep parameters
    my_per.finetune_best_frequencies(fresolution=1e-4, n_local_optima=1)
    fbest, pbest = my_per.get_best_frequencies()
    #return 1. / fbest
    return 1. / fbest[0]
    
    
def preprocess_lightcurve(lc_arr, name, mwebv):
    """
    Imports all relevant data relevant to one lightcurve/locus,
    removes points with NaN errors, converts to flux units, and 
    performs data clipping/pre-processing for better fitting.
    
    Parameters
    ----------
    lc_arr : numpy array
        lightcurve t, f, ferr, b
    mwebv : float
        Milky Way E(B-V)
    
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
        flux_errs = lc_arr_pruned[2],
        bands = lc_arr_prunced[3],
        name = name
    )
    
    survey = Survey.LSST()
    ext_dict = survey.get_band_extinctions(ra, dec, mwebv=mwebv)
    for unique_b in ext_dict:
        lc.fluxes[lc.bands == unique_b] *= 10**(0.4 * ext_dict[unique_b])
    
    # sort lightcurve
    lc.sort_by_time()
    
    # clip both ends
    clipped_lc = clip_lightcurve_end(
        lc.times, lc.fluxes, lc.flux_errs, lc.bands
    )
    reversed_clipped_lc = clip_lightcurve_end(
        clipped_lc[:,::-1]
    )
    lc.times, lc.fluxes, lc.flux_errors, lc.bands = reversed_clipped_lc[:,::-1]
    
    if len(lc.times) < 3:
        raise ValueError
        
    return lc, ra, dec
    
    
def adjust_log_dists(features):
    """
    Some parameters distributions more closely follow a Gaussian
    in log space, so these features are converted before being
    normalized.
    
    Parameters
    ----------
    features : numpy float array
        the input parameters
    
    Returns
    ----------
    features : numpy float array
        the modified original array
    """
    features[18:21] = np.log10(features[18:21])
    features[16] = np.log10(features[16])
    return np.delete(features, [14,17])

def get_predictions(model, input_features, device='cpu'):
    """
    Given a trained model, returns the probabilities of the
    given object being each supernovae type, according to
    the class-to-label conversion dictionary.
    
    Parameters
    ----------
    model : MLP object
        the trained SN classifier
    input_features : torch float tensor
        the normalized model parameters
    device : string
        device to evaluate model on. By default CPU
        
    Returns
    ----------
    probabilities : torch float tensor
        the probability of the input object being
        each of the five SN types
    """

    model.eval()

    with torch.no_grad():

        x = input_features.to(device)
        # applies trained model to input features
        y_pred, _ = model(x)
        y_prob = F.softmax(y_pred, dim=-1)
        return y_prob[0]

    
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
        
        # load model files
        recurring_model_fn = self.files[
            'superphot_elasticc2_recurring_model_v1.pt'
        ]
        nonrecurring_model_fn = self.files[
            'superphot_elasticc2_nonrecurring_model_v1.pt'
        ]
        top_level_model_fn = self.files[
            'superphot_elasticc2_top_level_model_v1.pt'
        ]
        
        stream = io.BytesIO(recurring_model_fn)
        self.recurring_model = MLP(21, len(self.recurring_classes), 256, 4) # set up empty multi-layer perceptron
        self.recurring_model.load_state_dict(torch.load(stream)) # load trained state dict to the MLP
        
        stream = io.BytesIO(nonrecurring_model_fn)
        self.nonrecurring_model, _ = SuperphotClassifier.load(
        )
        
        stream = io.BytesIO(top_level_model_fn)
        self.top_level_model = MLP(21, 2, 256, 4) # set up empty multi-layer perceptron
        self.top_level_model.load_state_dict(torch.load(stream)) # load trained state dict to the MLP
        
        #print("Loaded model")
        
        # to normalize the input features, specific to model version (v2)
        self.top_level_means = FILLIN               
        self.top_level_stddevs = FILLIN
        
        self.recurring_means = FILLIN
        self.recurring_stddevs = FILLIN
        
        self.nonrecurring_means = FILLIN
        self.nonrecurring_stddevs = FILLIN
        
        self.band_names_to_numbers = {
            "u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5
        }
        
    def add_classification(self, class_id, prob):
        """Helper function to add classification result.
        """
        self.classifications.append(
            {
            "classifierName": self.name,
            "classifierParams": self.parameters,
            "classId": 2,
            "probability": probs[0].item(),
            }
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
            lc, ra, dec = preprocess_lightcurve(arr)
        except:
            return
        
        meta_features = get_meta_features(lc, ra, dec)
        
        #first apply top level sampler
        normed_top_level_params = (meta_features - self.top_level_means) / self.top_level_stddevs
        probs = get_predictions(self.top_level_model, torch.Tensor(np.array([normed_top_level_params,])), 'cpu').numpy() # uses model to output SN type probabilities

        self.add_classification(2, probs[0].item())
        self.add_classification(1, probs[1].item())
        
        if probs[0] < 0.5:
            recurring = False
            for class_id in self.recurring_classes:
                self.add_classification(
                    int(class_id),
                    probs[0].item() / 5.,
                )
        else:
            recurring = True
            for class_id in self.nonrecurring_classes:
                self.add_classification(
                    int(class_id),
                    probs[1].item() / 15.,
                )
                #print(int(class_id), probs[1].item() / 15.)
        
        if recurring:
            normed_recurring_params = (meta_features - self.recurring_means) / self.recurring_stddevs
            probs_recurring = get_predictions(self.recurring_model, torch.Tensor(np.array([normed_recurring_params,])), 'cpu').numpy()
            for e, class_id in enumerate(self.recurring_classes):
                self.add_classification(
                    int(class_id),
                    probs[0].item() * probs_recurring[e].item()
                )
            
        else:
            #print("starting run nested sampling")
            base_band_i = 1 # second of g, r band base fit
            initial_bands = np.array([1,2,3])
            init_idx = np.isin(b, initial_bands)
            eq_samples, max_flux = run_nested_sampling(mjd[init_idx], f[init_idx], ferr[init_idx], b[init_idx], [1,2,3], base_band_i)
            if eq_samples is None:
                for class_id in self.nonrecurring_classes:
                    self.add_classification(
                        int(class_id),
                        probs[1].item() / 15.,
                    )
            median_params = np.median(eq_samples, axis=0)
            median_red = median_params[7 * base_band_i: (base_band_i + 1) * 7]
            for aux_b in [0, 4, 5]:
                #if len(b[b == aux_b]) == 0:
                #    eq_samples_aux = np.array([
                #    max_flux_aux = max_flux
                #else:
                eq_samples_aux, max_flux_aux = run_nested_sampling(mjd[b == aux_b], f[b == aux_b], ferr[b == aux_b], b[b == aux_b], [aux_b,], None, median_red, max_flux)
                
                if eq_samples_aux is None:
                    for class_id in self.nonrecurring_classes:
                        self.add_classification(
                            int(class_id),
                            probs[1].item() / 15.,
                        )
                if aux_b == 0:
                    eq_samples = np.hstack((eq_samples_aux, eq_samples))
                else:
                    eq_samples = np.hstack((eq_samples, eq_samples_aux))
            
            if eq_samples is None: # max_flux < 0
                for class_id in self.nonrecurring_classes:
                    self.add_classification(
                        int(class_id),
                        probs[1].item() / 15.,
                    )
                    #print(int(class_id), probs[1].item() / 15.)

            
            probs_all = []
            for eq_s in eq_samples:
                post = np.append(eq_s, meta_features)
                logL = calc_logL(eq_s, mjd, f, ferr, b)
                post = np.append(post, logL)
                adjusted_params = adjust_log_dists(post) # converts some params to log space
                normed_params = (adjusted_params - self.nonrecurring_means) / self.nonrecurring_stddevs # normalizes by mean and stddev used in training normalization

                probs_single = get_predictions(self.nonrecurring_model, torch.Tensor(np.array([normed_params,])), 'cpu').numpy() # uses model to output SN type probabilities
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
        