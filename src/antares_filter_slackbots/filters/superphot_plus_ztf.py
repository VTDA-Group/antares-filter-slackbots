from devkit2_poc.models import BaseFilter
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from pathlib import Path
import pandas as pd

from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import SVISampler
from superphot_plus.priors import SuperphotPrior
from superphot_plus.model import SuperphotLightGBM
from snapi import Photometry, Transient

class SuperphotPlusZTF(BaseFilter):
    NAME = "Superphot+ Supernovae Classification for ZTF"
    ERROR_SLACK_CHANNEL = "U03QP2KEK1V"  # Put your Slack user ID here
    INPUT_LOCUS_PROPERTIES = [
        'ztf_object_id',
    ]
    
    INPUT_ALERT_PROPERTIES = [
        'ant_mjd',
        'ant_passband',
        'ant_mag',
        'ant_magerr',
    ]
    
    PARAM_NAMES = SuperphotPrior.load(
        os.path.join(
            Path(__file__).parent.parent.parent.parent.absolute(),
            "data/superphot-plus/global_priors_hier_svi"
        )
    ).dataframe['param']

    OUTPUT_LOCUS_PROPERTIES = [
        {
            'name': f'superphot_plus_{feat}',
            'type': 'float',
            'description': f'Median {feat} from the Villar fit, as described in de Soto et al. 2024.',
        } for feat in PARAM_NAMES
    ] + [
        {
            'name': 'superphot_plus_score',
            'type': 'float',
            'description': 'Median reduced chi-squared value of our Villar fit, as described in de Soto et al. 2024. Golf rules (lower score is better)',
        },
        {
            'name': 'superphot_plus_class',
            'type': 'str',
            'description': 'Type of SN according to a LightGBM classifier detailed in de Soto et al. 2024. One of SNIa, SNIbc, SNII, SNIIn, or SLSN-I.',
        },
        {
            'name': 'superphot_plus_class_prob',
            'type': 'float',
            'description': 'Probability associated with the most-likely SN type from the classifier detailed in de Soto et al. 2024.',
        },
        {
            'name': 'superphot_non_Ia_prob',
            'type': 'float',
            'description': 'Probability that event is not a Type Ia.',
        },
        {
            'name': 'superphot_plus_classifier',
            'type': 'str',
            'description': 'Classifier type (early or full, LightGBM or MLP, plus version) from de Soto et al. 2024 used for classification.',
        },
        {
            'name': 'superphot_plus_sampler',
            'type': 'str',
            'description': 'Sampler used for this locus in the Superphot+ ANTARES ZTF filter. First tries stochastic variational inference, then switches to dynesty nested sampling if the reduced chi-squared is too high.',
        },
        {
            'name': 'superphot_plus_valid',
            'type': 'int',
            'description': 'Whether locus passes the catalog and light curve quality checks to be fit by the Superphot+ ANTARES ZTF filter. 1 if True and 0 if False. Loci previously tagged by Superphot+ that are no longer considered valid should be ignored by downstream tasks.',
        },
    ]
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = [
        {
            'name': 'superphot_plus_classified',
            'description': 'Successfully classified by the superphot_plus filter.',
        },
    ]
    
    REQUIRES_FILES = [
        'superphot_plus_lightgbm_02_2025_v4.pt',
        'superphot_plus_early_lightgbm_02_2025_v4.pt'
        'superphot_plus_lightgbm_02_2025_v4.pt',
        'superphot_plus_early_lightgbm_02_2025_v4.pt'
    ]
        
    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        self.input_properties = [
            'ztf_object_id',
            'nuclear',
            'best_redshift',
            'ra',
            'dec',
            'name'
        ]
            
        self.data_dir = os.path.join(
            Path(__file__).parent.parent.parent.parent.absolute(),
            "data"
        )
        
        # Loads SFDQuery object once to lower overhead
        from dustmaps.config import config
        config.reset()
        config['data_dir'] = os.path.join(
            self.data_dir, "dustmaps"
        )
        import dustmaps.sfd
        dustmaps.sfd.fetch()
        
        # generate sampling priors
        self.priors = SuperphotPrior.load(
            os.path.join(self.data_dir, 'superphot-plus/global_priors_hier_svi')
        )
        self.random_seed = 42

        # initialize dynesty sampler object
        self.dynesty_sampler = DynestySampler(
            priors=self.priors,
            random_state=self.random_seed
        )

        # for LC padding
        self.fill = {'phase': 1000., 'flux': 0.1, 'flux_error': 1000., 'zeropoint': 23.90, 'upper_limit': False}

        # initialize SVI sampler object
        self.svi_sampler = SVISampler(
            priors=self.priors,
            num_iter=10_000,
            random_state=self.random_seed
        )
        # loads trained model files (for actual filter)
        """
        full_model_fn = self.files['superphot_plus_lightgbm_12_2024_v3.pt'] 
        early_model_fn = self.files['superphot_plus_early_lightgbm_12_2024_v3.pt']
        """
        # replacement for local filter call
        full_model_fn = os.path.join(
            self.data_dir,
            "superphot-plus/model_superphot_full.pt"
        )
        early_model_fn = os.path.join(
            self.data_dir,
            "superphot-plus/model_superphot_early.pt"
        )
        self.full_model = SuperphotLightGBM.load(full_model_fn)
        self.early_model = SuperphotLightGBM.load(early_model_fn)
        
        # redshift-inclusive models
        full_model_fn = os.path.join(
            self.data_dir,
            "superphot-plus/model_superphot_redshift.pt"
        )
        early_model_fn = os.path.join(
            self.data_dir,
            "superphot-plus/model_superphot_early_redshift.pt"
        )
        self.full_model_z = SuperphotLightGBM.load(full_model_fn)
        self.early_model_z = SuperphotLightGBM.load(early_model_fn)
        
        self.score_cutoff = 1.2 # fits with reduced chisq above this are ignored
        self._allowed_types = ['SLSN-I', 'SN Ia', 'SN Ibc', 'SN II', 'SN IIn']

    
    def generate_antares_phot(self, ts):
        """Generate SNAPI photometry from ANTARES time series.
        """
        # renames dataframe to be SNAPI-compatible
        ts.rename(columns={
            'ant_mjd': 'mjd',
            'ant_mag': 'mag',
            'ant_magerr': 'mag_error',
            'ant_passband': 'filter'
        }, inplace=True)

        ts = ts.loc[ts['filter'].isin(['R', 'g', 'ATLAS_o', 'ATLAS_c'])]

        ts.loc[ts['filter'] == 'ATLAS_o', 'filter'] = 'R' # basically the same
        ts.loc[ts['filter'] == 'ATLAS_c', 'filter'] = 'g' # basically the same

        ts['filt_center'] = np.where(
            ts['filter'] == 'R',
            6366.38, 4746.48
        )
        ts['filt_width'] = np.where(
            ts['filter'] == 'R',
            1553.43, 1317.15
        )
        ts['filter'] = np.where(
            ts['filter'] == 'R',
            'ZTF_r', 'ZTF_g'
        )
        ts['zeropoint'] = 23.90 # AB mag
    
        # phases, normalizes, and extinction-corrects photometry
        phot = Photometry(ts) # SNAPI photometry object
        
        new_lcs = []
        for lc in phot.light_curves:
            lc.merge_close_times(inplace=True)
            new_lcs.append(lc)
        
        phot = Photometry.from_light_curves(new_lcs)
        return phot
    
    
    def evaluate_cal_probs(self, model, orig_features):
        input_features = model.best_model.feature_name_
        test_features = model.normalize(orig_features[input_features])
        
        probabilities = pd.DataFrame(
            model.best_model.predict_proba(test_features),
            index=test_features.index
        )
        
        probabilities.columns = np.sort(self._allowed_types)
        best_classes = probabilities.idxmax(axis=1)
        # "probability" = number of fits where that class is the best class
        # more frequentist in interpretation = better calibrated? idk
        probs = best_classes.value_counts() / best_classes.count()
        try:
            ia_prob = probs[probs.index == "SN Ia"].iloc[0]
        except:
            ia_prob = 0.
        return probs.idxmax(), probs.max(), ia_prob
        
    
    def _run(self, event_dict, ts):
        """
        Runs a filter that fits all transients tagged with supernovae-like properties to the model
        described by de Soto et al, 2024. Saves the median model parameters found using SVI or nested sampling
        to later be input into the classifier of choice. Then, inputs all posterior sample model parameters
        in a pre-trained classifier to estimate the most-likely SN class and confidence score.
        
        Parameters
        ----------
        locus : Locus object
            the SN-like transient to be fit
        """
        # skip if nuclear
        if event_dict['nuclear']:
            event_dict['superphot_plus_valid'] = 0
            return event_dict
        
        # removes rows with nan values
        ts.dropna(inplace=True, axis=0, ignore_index=True)

        if 'ant_ra' not in ts.columns:
            print(ts)
            print(ts.columns)

        if ts['ant_ra'].std() > 0.5 / 3600.: # arcsec
            event_dict['superphot_plus_valid'] = 0
            return event_dict # marked as variable star or AGN
        
        if ts['ant_dec'].std() > 0.5 / 3600.: # arcsec
            event_dict['superphot_plus_valid'] = 0
            return event_dict # marked as variable star or AGN
        
        ts.drop(columns=['ant_ra', 'ant_dec'], inplace=True)

        phot = None
        phot = self.generate_antares_phot(ts)
            
        phot.phase(inplace=True)
        
        # adjust for time dilation
        redshift = event_dict['best_redshift']
        
        if ~np.isnan(redshift):
            phot.times /= (1. + redshift)
            
        phot.truncate(min_t=-50., max_t=100.)

        if len(phot) < 2: # removed a filter
            event_dict['superphot_plus_valid'] = 0
            return event_dict
        
        event_dict['superphot_plus_valid'] = 1
        
        phot.correct_extinction(
            coordinates=SkyCoord(ra = event_dict['ra'] * u.deg, dec = event_dict['dec'] * u.deg),
            inplace=True
        )
        # let's recalculate peak_app_mag of extincted LC
        phot_abs = phot.absolute(redshift=redshift)
        peak_abs_mag = phot_abs.detections.mag.dropna().min()
        
        phot.normalize(inplace=True)

        # create padded photometry for use in SVI
        try:
            padded_lcs = []
            orig_size = len(phot.detections)
            num_pad = int(2**np.ceil(np.log2(orig_size)))
            for lc in phot.light_curves:
                padded_lc=lc.pad(self.fill, num_pad - len(lc.detections))
                padded_lcs.append(padded_lc)
            padded_phot = Photometry.from_light_curves(padded_lcs)
        except IndexError: # TODO: fix this in snapi
            return event_dict

        # fit with SVI and extract result
        self.svi_sampler.reset() # reinitialize for future SVI fits
        self.svi_sampler.fit_photometry(padded_phot, orig_num_times=orig_size)
        res = self.svi_sampler.result

        # only check the reduced chisq if there are at least 8 datapoints,
        # because calculation breaks down for small number of observations
        if (orig_size >= 8) and (np.median(res.score) > self.score_cutoff):
            # poor fit = reset SVI and use dynesty instead
            self.dynesty_sampler.fit_photometry(phot)
            res_dynesty = self.dynesty_sampler.result

            # check whether SVI or dynesty performed better
            if np.median(res_dynesty.score[:100]) < np.median(res.score[:100]):
                sampler, results = 'dynesty', res_dynesty            
            else:
                sampler, results = 'svi', res
        else:
            # SVI fit is fine
            sampler, results = 'svi', res

        for param in results.fit_parameters.columns:
            event_dict[f'superphot_plus_{param}'] = results.fit_parameters[param].median()

        # save sampler and fit score to output properties
        event_dict['superphot_plus_sampler'] = sampler
        event_dict['superphot_plus_score'] = np.median(results.score)

        # remove fits with reduced chisq above score cutoff
        if orig_size >= 8:
            valid_fits = results.fit_parameters[results.score <= self.score_cutoff]
            if len(valid_fits) == 0:
                return event_dict
        else:
            valid_fits = results.fit_parameters

        # check which fits place all observations before piecewise transition
        early_fit_mask = valid_fits['gamma_ZTF_r'] + valid_fits['t_0_ZTF_r'] > np.max(phot.times)

        # convert fit parameters back to uncorrelated Gaussian draws
        uncorr_fits = self.priors.reverse_transform(valid_fits)

        # fix index for groupby() operations within model.evaluate()
        uncorr_fits.index = [event_dict['name']] * len(uncorr_fits)
        
        
        if ~np.isnan(redshift):
            # add magnitude to uncorr_fits
            
            uncorr_fits['peak_abs_mag'] = peak_abs_mag
            
            if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
                class_z, prob_z, Ia_prob_z = self.evaluate_cal_probs(self.early_model_z, uncorr_fits)
            else:
                class_z, prob_z, Ia_prob_z = self.evaluate_cal_probs(self.full_model_z, uncorr_fits)
                
            event_dict['superphot_plus_class'] = class_z
            event_dict['superphot_plus_prob'] = np.round(prob_z, 3)
            event_dict['superphot_non_Ia_prob'] = 1. - np.round(Ia_prob_z, 3)

        
        if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
            # use early-phase classifier
            class_noz, prob_noz, Ia_prob_noz = self.evaluate_cal_probs(self.early_model, uncorr_fits)
            event_dict['superphot_plus_classifier'] = 'early_lightgbm_02_2025'
        else:
            # use full-phase classifier
            class_noz, prob_noz, Ia_prob_noz = self.evaluate_cal_probs(self.full_model, uncorr_fits)
            event_dict['superphot_plus_classifier'] = 'full_lightgbm_02_2025'
            
            
        # set predicted SN class and output probability of that classification
        if ~np.isnan(redshift):
            event_dict['superphot_plus_class_without_redshift'] = class_noz
            event_dict['superphot_plus_prob_without_redshift'] = np.round(prob_noz, 3)
        else:
            event_dict['superphot_plus_class'] = class_noz
            event_dict['superphot_plus_prob'] = np.round(prob_noz, 3)
            event_dict['superphot_non_Ia_prob'] = 1. - np.round(Ia_prob_noz, 3)
            
        event_dict['superphot_plus_classified'] = True
        return event_dict
