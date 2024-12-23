import antares.devkit as dk
import numpy as np
import os
from tempfile import TemporaryDirectory
import pickle
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import SVISampler
from superphot_plus.priors import generate_priors
from superphot_plus.model import SuperphotLightGBM
from snapi import Photometry

class SuperphotPlusZTF(dk.Filter):
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
    
    PARAM_NAMES = generate_priors(["ZTF_r", "ZTF_g"]).dataframe['param']

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
        'superphot_plus_lightgbm_12_2024_v3.pt',
        'superphot_plus_early_lightgbm_12_2024_v3.pt'
    ]
        
    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        # Loads SFDQuery object once to lower overhead
        from dustmaps.config import config
        config.reset()
        self.tempdir = TemporaryDirectory(prefix='superphot_plus_sfds_')
        #config['data_dir'] = self.tempdir.name
        config['data_dir'] = '/tmp/'  # datalab
        import dustmaps.sfd
        dustmaps.sfd.fetch()
        from dustmaps.sfd import SFDQuery
        self.dustmap = SFDQuery()
        
        # for initial pruning
        self.variable_catalogs = [
            "gaia_dr3_variability",
            "sdss_stars",
            "bright_guide_star_cat",
            "asassn_variable_catalog_v2_20190802",
            "vsx",
            "linear_ll",
            "veron_agn_qso", # agn/qso
            "milliquas", # qso
        ]
        
        # generate sampling priors
        self.priors = generate_priors(["ZTF_r", "ZTF_g"])
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
        full_model_fn = os.path.dirname(__file__) + "/data/model_superphot_full.pt"
        early_model_fn = os.path.dirname(__file__) + "/data/model_superphot_early.pt"
        self.full_model = SuperphotLightGBM.load(full_model_fn)
        self.early_model = SuperphotLightGBM.load(early_model_fn)

        # subset of features for early-phase classifier
        self.early_input_features = self.PARAM_NAMES[~self.PARAM_NAMES.isin([
            "gamma_ZTF_r", "gamma_ZTF_g", "tau_fall_ZTF_r", "tau_fall_ZTF_g"
        ])]

        self.score_cutoff = 1.2 # fits with reduced chisq above this are ignored
        self._allowed_types = ['SLSN-I', 'SN Ia', 'SN Ibc', 'SN II', 'SN IIn']

    
    def quality_check(self, ts):
        """Return True if all quality checks are passed, else False."""
        if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
            return False
        
        # ignore any events lasting longer than 200 days
        # note, this will inevitably exclude some SNe, but that's fine
        if np.ptp(ts['ant_mjd'].quantile([0.1, 0.9])) >= 200.:
            return False
        
        if len(ts) < 5:
            return True # don't do variability checks if < 5 points
                
        # first variability cut
        if np.ptp(ts['ant_mag']) < 3 * ts['ant_magerr'].mean():
            return False
        
        # second variability cut
        if ts['ant_mag'].std() < ts['ant_magerr'].mean():
            return False

        return True

    def run(self, locus):
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
        # checks associated catalogs
        cats = locus.catalog_objects
        for cat in cats:
            if cat in self.variable_catalogs:
                locus.properties['superphot_plus_valid'] = 0
                return None # marked as variable star or AGN
        
        # gets dataframe with locus alerts, ordered by mjd
        ts = locus.lightcurve[[
            'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband'
        ]]#.to_pandas() # may have to turn this back on for actual submission

        # removes rows with nan values
        ts.dropna(inplace=True)

        # drops i-band data
        ts = ts[ts['ant_passband'] != 'i']

        if not self.quality_check(ts):
            locus.properties['superphot_plus_valid'] = 0
            return None

        # renames dataframe to be SNAPI-compatible
        ts.rename(columns={
            'ant_mjd': 'mjd',
            'ant_mag': 'mag',
            'ant_magerr': 'mag_error',
            'ant_passband': 'filter'
        }, inplace=True)

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
        phot.phase(inplace=True)
        phot.truncate(min_t=-50., max_t=100.)

        if len(phot) < 2: # removed a filter
            locus.properties['superphot_plus_valid'] = 0
            return None
        
        locus.properties['superphot_plus_valid'] = 1
        
        phot.correct_extinction(
            coordinates=SkyCoord(ra = locus.ra * u.deg, dec = locus.dec * u.deg),
            inplace=True
        )
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
            return None

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
            locus.properties[f'superphot_plus_{param}'] = results.fit_parameters[param].median()

        # save sampler and fit score to output properties
        locus.properties['superphot_plus_sampler'] = sampler
        locus.properties['superphot_plus_score'] = np.median(results.score)

        # remove fits with reduced chisq above score cutoff
        if orig_size >= 8:
            valid_fits = results.fit_parameters[results.score <= self.score_cutoff]
            if len(valid_fits) == 0:
                return None
        else:
            valid_fits = results.fit_parameters

        # check which fits place all observations before piecewise transition
        early_fit_mask = valid_fits['gamma_ZTF_r'] + valid_fits['t_0_ZTF_r'] > np.max(phot.times)

        # convert fit parameters back to uncorrelated Gaussian draws
        uncorr_fits = self.priors.reverse_transform(valid_fits)

        # fix index for groupby() operations within model.evaluate()
        uncorr_fits.index = [locus.locus_id,] * len(uncorr_fits)

        if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
            # use early-phase classifier
            probs_avg = self.early_model.evaluate(uncorr_fits[self.early_input_features])
            locus.properties['superphot_plus_classifier'] = 'early_lightgbm_12_2024'
        else:
            # use full-phase classifier
            probs_avg = self.full_model.evaluate(uncorr_fits)
            locus.properties['superphot_plus_classifier'] = 'full_lightgbm_12_2024'
            
        probs_avg.columns = np.sort(self._allowed_types)

        # set predicted SN class and output probability of that classification
        locus.properties['superphot_plus_class'] = probs_avg.idxmax(axis=1).iloc[0]
        locus.properties['superphot_plus_class_prob'] = np.round(probs_avg.max(axis=1).iloc[0], 3)
        locus.tags.append('superphot_plus_classified')