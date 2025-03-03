import antares.devkit as dk
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from pathlib import Path
from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module

from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import SVISampler
from superphot_plus.priors import SuperphotPrior
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


        # subset of features for each classifier
        self.full_input_features = self.PARAM_NAMES[~self.PARAM_NAMES.isin([
            "A_ZTF_r", "t_0_ZTF_r",
        ])]
        self.early_input_features = self.PARAM_NAMES[~self.PARAM_NAMES.isin([
            "A_ZTF_r", "t_0_ZTF_r", "gamma_ZTF_r", "gamma_ZTF_g", "tau_fall_ZTF_r", "tau_fall_ZTF_g"
        ])]
        self.early_redshift_input_features = self.PARAM_NAMES[~self.PARAM_NAMES.isin([
            "t_0_ZTF_r", "gamma_ZTF_r", "gamma_ZTF_g", "tau_fall_ZTF_r", "tau_fall_ZTF_g"
        ])]
        self.full_redshift_input_features = self.PARAM_NAMES[~self.PARAM_NAMES.isin([
            "t_0_ZTF_r",
        ])]
        
        self.score_cutoff = 1.2 # fits with reduced chisq above this are ignored
        self._allowed_types = ['SLSN-I', 'SN Ia', 'SN Ibc', 'SN II', 'SN IIn']

    
    def quality_check(self, ts):
        """Return True if all quality checks are passed, else False."""
        if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
            return False
        
        if np.ptp(ts['ant_mjd']) < 4.: # at least 4 days of data
            return False
        
        # ignore any events lasting longer than 200 days
        # note, this will inevitably exclude some SNe, but that's fine
        if np.ptp(ts['ant_mjd'].quantile([0.1, 0.9])) >= 200.:
            return False
        
        
        for b in ts['ant_passband'].unique():
            sub_ts = ts.loc[ts['ant_passband'] == b,:]
            if len(sub_ts) < 5:
                continue # don't do variability checks if < 5 points
            
            # first variability cut
            if np.ptp(sub_ts['ant_mag']) < 3 * sub_ts['ant_magerr'].mean():
                return False
            
            # second variability cut
            if sub_ts['ant_mag'].std() < sub_ts['ant_magerr'].mean():
                return False
            
            # third variability cut
            if np.ptp(sub_ts['ant_mag']) < 0.5: # < 0.5 mag spread
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
            'ant_mjd', 'ant_mag', 'ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec'
        ]]#.to_pandas() # may have to turn this back on for actual submission

        if ts['ant_ra'].std() > 0.5 / 3600.: # arcsec
            locus.properties['superphot_plus_valid'] = 0
            return None # marked as variable star or AGN
        
        if ts['ant_dec'].std() > 0.5 / 3600.: # arcsec
            locus.properties['superphot_plus_valid'] = 0
            return None # marked as variable star or AGN
        
        ts.drop(columns=['ant_ra', 'ant_dec'], inplace=True)

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
        
        # extract redshift
        tns_redshift = locus.extra_properties['tns_redshift']
        if ~np.isnan(tns_redshift) and (tns_redshift > 0.):
            redshift = tns_redshift
        elif locus.extra_properties['high_host_confidence'] and (
            ~np.isnan(locus.extra_properties['host_redshift'])
        ) and (locus.extra_properties['host_redshift'] > 0.):
            redshift = locus.extra_properties['host_redshift']
        else:
            redshift = np.nan
    
        # phases, normalizes, and extinction-corrects photometry
        phot = Photometry(ts) # SNAPI photometry object
        
        new_lcs = []
        for lc in phot.light_curves:
            lc.merge_close_times(inplace=True)
            new_lcs.append(lc)
        
        phot = Photometry.from_light_curves(new_lcs)
        phot.phase(inplace=True)
        
        # adjust for time dilation
        if ~np.isnan(redshift):
            phot.times /= (1. + redshift)
            
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
        
        if ~np.isnan(redshift):
            # add magnitude to uncorr_fits
            app_mag = locus.properties['brightest_alert_magnitude']
            k_corr = 2.5 * np.log10(1.0 + redshift)
            distmod = Planck15.distmod(redshift).value
            abs_mag = app_mag - distmod + k_corr
            uncorr_fits['peak_abs_mag'] = abs_mag
            locus.properties['peak_abs_mag'] = abs_mag
            
            if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
                probs_avg_z = self.early_model_z.evaluate(
                    uncorr_fits[self.early_redshift_input_features]
                )
            else:
                probs_avg_z = self.full_model_z.evaluate(
                    uncorr_fits[self.full_redshift_input_features]
                )
            probs_avg_z.columns = np.sort(self._allowed_types)
            locus.properties['superphot_plus_class'] = probs_avg_z.idxmax(axis=1).iloc[0]
            locus.properties['superphot_plus_prob'] = np.round(probs_avg_z.max(axis=1).iloc[0], 3)

        
        if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
            # use early-phase classifier
            probs_avg = self.early_model.evaluate(uncorr_fits[self.early_input_features])
            locus.properties['superphot_plus_classifier'] = 'early_lightgbm_02_2025'
        else:
            # use full-phase classifier
            probs_avg = self.full_model.evaluate(uncorr_fits[self.full_input_features])
            locus.properties['superphot_plus_classifier'] = 'full_lightgbm_02_2025'
            
            
        # set predicted SN class and output probability of that classification
        probs_avg.columns = np.sort(self._allowed_types)
        sp_class = probs_avg.idxmax(axis=1).iloc[0]
        sp_prob = np.round(probs_avg.max(axis=1).iloc[0], 3)
        
        if ~np.isnan(redshift):
            locus.properties['superphot_plus_class_without_redshift'] = sp_class
            locus.properties['superphot_plus_prob_without_redshift'] = sp_prob
        else:
            locus.properties['superphot_plus_class'] = sp_class
            locus.properties['superphot_plus_prob'] = sp_prob
            
        locus.tags.append('superphot_plus_classified')
