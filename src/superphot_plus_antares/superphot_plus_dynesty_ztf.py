import antares.devkit as dk
import numpy as np
import extinction
from astropy.coordinates import SkyCoord
from tempfile import TemporaryDirectory
import pickle
import time

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.model.lightgbm import SuperphotLightGBM
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import (
    get_band_extinctions_from_mwebv,
    convert_mags_to_flux
)


class SuperphotPlusZTF(dk.Filter):
    NAME = "Superphot+ Supernovae Classification for ZTF"
    ERROR_SLACK_CHANNEL = "U03QP2KEK1V"  # Put your Slack user ID here
    INPUT_LOCUS_PROPERTIES = [
        'ztf_object_id',
    ]
    
    INPUT_ALERT_PROPERTIES = [
        'ant_mjd',
        'ztf_magpsf',
        'ztf_fid', # 1=g, 2=r
        'ztf_sigmapsf',
        'ztf_magzpsci',
        'ant_ra',
        'ant_dec',
    ]
    
    OUTPUT_LOCUS_PROPERTIES = [
        {
            'name': 'superphot_plus_peak_amplitude',
            'type': 'float',
            'description': 'Median peak amplitude (in flux units) from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_rel_plateau_slope',
            'type': 'float',
            'description': 'Median plateau slope (relative to maximum amplitude) from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_plateau_duration',
            'type': 'float',
            'description': 'Median plateau duration (in days) from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.', 
        },
        
        {
            'name': 'superphot_plus_center_phase',
            'type': 'float',
            'description': 'Median "center"-ish phase (in days) from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_rise_timescale',
            'type': 'float',
            'description': 'Median rise timescale from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_fall_timescale',
            'type': 'float',
            'description': 'Median fall timescale from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_extra_sigma',
            'type': 'float',
            'description': 'Median extra uncertainty component from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_peak_amplitude_ratio',
            'type': 'float',
            'description': 'Median g-r amplitude ratio (color) from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_plateau_slope_ratio',
            'type': 'float',
            'description': 'Median g-r plateau slope ratio from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_plateau_duration_ratio',
            'type': 'float',
            'description': 'Median g-r plateau duration ratio from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_center_phase_offset',
            'type': 'float',
            'description': 'Median g-r center phase offset from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_rise_timescale_ratio',
            'type': 'float',
            'description': 'Median g-r rise timescale ratio from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_fall_timescale_ratio',
            'type': 'float',
            'description': 'Median g-r fall timescale ratio from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_extra_sigma_ratio',
            'type': 'float',
            'description': 'Median g-r extra uncertainty ratio from the Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        {
            'name': 'superphot_plus_reduced_chisquared',
            'type': 'float',
            'description': 'Median reduced chi-squared value of our Villar fit, as described in de Soto et al. 2024, calculated using dynesty nested sampling.',
        },
        {
            'name': 'superphot_plus_class',
            'type': 'str',
            'description': 'Type of SN according to a multi-layer perceptron classifier detailed in de Soto et al. 2024. One of SNIa, SNIbc, SNII, SNIIn, or SLSN-I.',
        },
        {
            'name': 'superphot_plus_class_prob',
            'type': 'float',
            'description': 'Probability associated with the most-likely SN type from the classifier detailed in de Soto et al. 2024.',
        },
        {
            'name': 'superphot_plus_dynesty_runtime',
            'type': 'float',
            'description': 'Nested sampling runtime of the Superphot+ ANTARES ZTF filter for this locus',
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
        'superphot_plus_lightgbm_v1.pt',
    ]

    
    def import_lightcurve(self, mjd, m, merr, b, ra, dec):
        """Create Superphot+ Lightcurve object and 
        perform pre-processing.
        """
        coords = SkyCoord(ra, dec, frame="icrs", unit="deg")
        mwebv = self.dustmap(coords)
        ext_dict = get_band_extinctions_from_mwebv(
            mwebv, self.survey.get_ordered_wavelengths()
        )
        b_str = np.where(b == 1, 'g', 'r')
        m = np.where(b_str == 'g', m - ext_dict[1], m - ext_dict[0])
        # convert mags to fluxes
        f, ferr = convert_mags_to_flux(m, merr, self.zpt)

        # make Superphot+ Lightcurve object
        lc = Lightcurve(
            mjd, f, ferr, b_str
        )
        lc.sort_by_time()
        max_flux, max_flux_loc = lc.find_max_flux(band='r')
        lc.times -= max_flux_loc # phase LC
        return lc, max_flux
    
    
    def reformat_features(self, posteriors, max_flux):
        """Reformat features to map Villar fit parameters (aka de-log them).
        """
        idxs_exponentiate = np.delete(np.arange(14), [1, 3, 10,])
        post_reformatted = np.median(posteriors, axis=0)
        post_reformatted[idxs_exponentiate] = 10**post_reformatted[idxs_exponentiate]
        post_reformatted[[0, 6]] *= max_flux
        round_values = (-1 * np.floor(np.log10(np.abs(post_reformatted)))).astype(int)
        return np.asarray([
            np.round(post_reformatted[i], round_values[i] + 2) for i in range(len(post_reformatted))
        ]) # 3 sig figs
        
        
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
            "asassn_variable_catalog_v2_20190802",
            "vsx",
            "linear_ll"
        ]
        
        # create ZTF survey object
        self.survey = Survey.ZTF()
        self.zpt = 26.3 # approximate for ZTF
        
        # load Dynesty sampler object
        self.dynesty_sampler = DynestySampler()
        
        # for property enumeration
        self.output_properties = [
            'peak_amplitude',
            'rel_plateau_slope',
            'plateau_duration',
            'center_phase',
            'rise_timescale',
            'fall_timescale',
            'extra_sigma',
            'peak_amplitude_ratio',
            'plateau_slope_ratio',
            'plateau_duration_ratio',
            'center_phase_offset',
            'rise_timescale_ratio',
            'fall_timescale_ratio',
            'extra_sigma_ratio',
            'reduced_chisquared'
        ]
        
        self.classes_to_labels = SnClass.get_type_maps()[1]
        # load classification model information
        model_fn = self.files['superphot_plus_lightgbm_v1.pt'] # loads trained model file
        #config_fn = self.files['superphot_plus_lightgbm_v1.yaml'] # loads trained model file
        #stream = io.BytesIO(model_fn)

        self.model = pickle.loads(model_fn)
        
        """
        # initialize SVI
        self.optimizer = numpyro.optim.Adam(step_size=0.001)
        self.svi = SVI(jax_model, jax_guide, self.optimizer, loss=Trace_ELBO())
        self.svi_state = None
        self.num_iter = 10000
        
        self.lax_jit = jit(lax_helper_function, static_argnums=(0,2))
        """
    
    def run(self, locus):
        """
        Runs a filter that fits all transients tagged with supernovae-like properties to the model
        described by de Soto et al, TBD. Saves the median model parameters found using nested sampling
        to later be input into the classifier of choice. Then, inputs all posterior sample model parameters
        in a pre-trained classifier to estimate the most-likely SN class and confidence score.
        
        Parameters
        ----------
        locus : Locus object
            the SN-like transient to be fit
        """
        cats = locus.catalog_objects

        for cat in cats:
            if cat in self.variable_catalogs:
                return None # marked as variable star
        
        # gets listed properties of each alert in Locus, sorted by MJD
        ts = locus.timeseries[[
            'ant_mjd', 'ztf_magpsf', 'ztf_sigmapsf',
            'ztf_fid', 'ant_ra', 'ant_dec', 'ztf_magzpsci'
        ]]   
        ts_table = ts.to_pandas().to_numpy()
        # filter out NaNs
        skip_idx = np.any(np.isnan(ts_table[:,:4]), axis=1)
        mjd, m, merr, b, ra, dec, zp = ts_table[~skip_idx].T

        # if LC longer than 200 days, probably not a transient (so skip)
        try:
            if ( np.nanmax(mjd) - np.nanmin(mjd) >= 200 ):
                return None
            ra = np.median(ra[~np.isnan(ra)])
            dec = np.median(dec[~np.isnan(dec)])
        except:
            return None
        
        if len(mjd[b == 1]) < 2: # need more data to fit
            return None
        if len(mjd[b == 2]) < 2:
            return None
        
        #try:
        lc, max_flux = self.import_lightcurve(mjd, m, merr, b, ra, dec)

        start_time = time.time()
        dynesty_post = self.dynesty_sampler.run_single_curve(
            lc, self.survey.priors
        ).samples
        
        if dynesty_post is None:
            return None
        locus.properties['superphot_plus_dynesty_runtime'] = time.time() - start_time
        #post = self.run_svi(tdata, fdata, ferrdata, bdata)
        #except:
        #    return None

        dynesty_post_reformatted = self.reformat_features(dynesty_post, max_flux)
        # save each mean model parameter in locus properties
        for i in range(len(dynesty_post_reformatted)):
            locus.properties[
                f"superphot_plus_{self.output_properties[i]}"
            ] = dynesty_post_reformatted[i]
        
        training_params = np.delete(dynesty_post, [0, 3], axis=1)
        probs = self.model.classify_from_fit_params(training_params)
        probs_avg = np.mean(probs, axis=0)
        pred_class = np.argmax(probs_avg)
        class_confidence = np.max(probs_avg)
        pred_sn_type = self.classes_to_labels[pred_class]
        
        # set predicted SN class and output probability of that classification
        locus.properties['superphot_plus_class'] = pred_sn_type
        locus.properties['superphot_plus_class_prob'] = class_confidence.item()
        
        locus.tag('superphot_plus_classified')
        