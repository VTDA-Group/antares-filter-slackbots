import antares.devkit as dk
import numpy as np
import extinction
from astropy.coordinates import SkyCoord
from tempfile import TemporaryDirectory

import io
import torch
import torch.nn as nn
import torch.nn.functional as F

#import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from jax import random, lax, jit
from jax.config import config as jaxconfig

jaxconfig.update("jax_enable_x64", True)

from numpyro.infer import SVI, Trace_ELBO

class MLP(nn.Module):
    """
    A pytorch multi-layer perceptron object.
    """
    def __init__(self, input_dim, output_dim, neurons_per_layer, num_hidden_layers):
        """
        Initialize the MLP. Uses only linear/dense/fully connected layers.
        
        Parameters
        ----------
        input_dim : int
            the number of input features. For this classifier, input_dim = 13.
        output_dim : int
            the number of output classes. For us, we differentiate between 5 supernovae classes.
        neurons_per_layer : int
            the number of neurons in each layer of the classifier. While we could use a different
            neuron count per layer (for example, a "bottleneck" architecture), we assume here that
            the number of neurons is constant across layers. Too many neurons leads to overfitting.
        num_hidden_layers : int
            number of layers between the input and output layers. Should be at least 1. Too many
            layers leads to overfitting.
        """
        super().__init__() # draws from torch's neural network module
        
        n_neurons = neurons_per_layer
        self.input_fc = nn.Linear(input_dim, n_neurons)
        self.hidden_layers = nn.ModuleList()
        
        # num layers = 1 (input) + n hidden + 1 (output)
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons))
                                      
        self.output_fc = nn.Linear(n_neurons, output_dim)

    def forward(self, x):
        """
        Forward propagates a training feature set through
        the MLP.
        
        Parameters
        ----------
        x : Torch tensor
            input feature set
            
        Returns
        ----------
        y_pred : Torch tensor
            output classification probabilities
        h_hidden : Torch tensor
            state values of the layer prior to the output layer
        """
        
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))
        
        h_hidden = h_1
        for i in range(len(self.hidden_layers)):
            h_hidden = F.relu(self.hidden_layers[i](h_hidden))

        y_pred = self.output_fc(h_hidden)
        return y_pred, h_hidden

@jit
def trunc_norm(low, high, loc, scale):
    """
    Helper function for dist.TruncatedNormal()
    """
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


def jax_model(t = None, obsflux = None, uncertainties = None, max_flux = None):
    """
    Model that jax uses for numpyro's stochastic variational inference.
    Establishes both prior and loglikelihood calculation.
    """
    PAD_SIZE = 30
    inc_band_ix = np.arange(PAD_SIZE)
    
    # Nested sampling priors
    PRIOR_A = [-0.2, 1.2, 0., 0.5]
    PRIOR_BETA = [0., 0.02, 0.0052, 1.5 * 0.000336]
    PRIOR_GAMMA = [-2., 2.5, 1.1391, 1.5 * .1719]
    PRIOR_T0 = [-100., 200., 0., 50.]
    PRIOR_TAU_RISE = [-1.0, 3., 0.5990, 1.5 * 0.2073]
    PRIOR_TAU_FALL = [0.5, 4., 1.4296, 1.5 * 0.1003]
    PRIOR_EXTRA_SIGMA = [-5., -0.5, -1.5364, 0.2691]

    PRIOR_A_g = [0., 5., 1.0607, 1.5 * 0.1544]
    PRIOR_BETA_g = [1., 1.07, 1.0424, 0.0026]
    PRIOR_GAMMA_g = [0.8, 1.2, 1.0075, 0.0139]
    PRIOR_T0_g = [1. - 0.0006, 1.0006, 0.9999 + 8.9289e-5, 1.5 * 4.5055e-05]
    PRIOR_TAU_RISE_g = [0.5, 2., 0.9663, 0.0128]
    PRIOR_TAU_FALL_g = [0.1, 3., 0.5488, 0.0553]
    PRIOR_EXTRA_SIGMA_g = [0.2, 2., 0.8606, 0.0388]

    A = max_flux * 10**numpyro.sample("logA", trunc_norm(*PRIOR_A))
    beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
    gamma = 10**numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
    t0 = numpyro.sample("t0", trunc_norm(*PRIOR_T0))
    tau_rise = 10**numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
    tau_fall = 10**numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
    extra_sigma = 10**numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))

    A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
    beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
    gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
    t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
    tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
    tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
    extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))

    A_b = A * A_g
    beta_b = beta * beta_g
    gamma_b = gamma * gamma_g
    t0_b = t0 * t0_g
    tau_rise_b = tau_rise * tau_rise_g
    tau_fall_b = tau_fall * tau_fall_g

    phase = t - t0
    flux_const = A / (1. + jnp.exp(-phase / tau_rise))
    sigmoid = 1 / (1 + jnp.exp(10.*(gamma - phase)))

    flux = flux_const * ( (1-sigmoid) * (1 - beta*phase) + sigmoid * (1 - beta*gamma) * jnp.exp(-(phase-gamma)/tau_fall) )

    # g band
    phase_b = (t - t0_b)[inc_band_ix]
    flux_const_b = A_b / (1. + jnp.exp(-phase_b / tau_rise_b))
    sigmoid_b = 1 / (1 + jnp.exp(10.*(gamma_b - phase_b)))

    flux = flux.at[inc_band_ix].set(flux_const_b * ( (1-sigmoid_b) * (1 - beta_b*phase_b) + sigmoid_b * (1 - beta_b*gamma_b) * jnp.exp(-(phase_b-gamma_b)/tau_fall_b) ))

    sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
    sigma_tot = sigma_tot.at[inc_band_ix].set(jnp.sqrt(uncertainties[inc_band_ix]**2 + extra_sigma_g**2 * extra_sigma**2))

    obs = numpyro.sample("obs",dist.Normal(flux, sigma_tot),obs=obsflux)
                   
def jax_guide(t = None, obsflux = None, uncertainties = None, max_flux = None):
    """
    Guide that jax uses for numpyro's stochastic variational inference.
    Tells numpyro to approximate every marginal dist as a Gaussian.
    """

    # Nested sampling priors
    PRIOR_A = [-0.2, 1.2, 0., 0.5]
    PRIOR_BETA = [0., 0.02, 0.0052, 1.5 * 0.000336]
    PRIOR_GAMMA = [-2., 2.5, 1.1391, 1.5 * .1719]
    PRIOR_T0 = [-100., 200., 0., 50.]
    PRIOR_TAU_RISE = [-1.0, 3., 0.5990, 1.5 * 0.2073]
    PRIOR_TAU_FALL = [0.5, 4., 1.4296, 1.5 * 0.1003]
    PRIOR_EXTRA_SIGMA = [-5., -0.5, -1.5364, 0.2691]

    PRIOR_A_g = [0., 5., 1.0607, 1.5 * 0.1544]
    PRIOR_BETA_g = [1., 1.07, 1.0424, 0.0026]
    PRIOR_GAMMA_g = [0.8, 1.2, 1.0075, 0.0139]
    PRIOR_T0_g = [1. - 0.0006, 1.0006, 0.9999 + 8.9289e-5, 1.5 * 4.5055e-05]
    PRIOR_TAU_RISE_g = [0.5, 2., 0.9663, 0.0128]
    PRIOR_TAU_FALL_g = [0.1, 3., 0.5488, 0.0553]
    PRIOR_EXTRA_SIGMA_g = [0.2, 2., 0.8606, 0.0388]
    
    logA_mu = numpyro.param("logA_mu", PRIOR_A[2], constraint=constraints.interval(PRIOR_A[0], PRIOR_A[1]))
    logA_sigma = numpyro.param("logA_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("logA", dist.Normal(logA_mu, logA_sigma))

    beta_mu = numpyro.param("beta_mu", PRIOR_BETA[2], constraint=constraints.interval(PRIOR_BETA[0], PRIOR_BETA[1]))
    beta_sigma = numpyro.param("beta_sigma", 1e-5, constraint=constraints.positive)
    numpyro.sample("beta", dist.Normal(beta_mu, beta_sigma))

    log_gamma_mu = numpyro.param("log_gamma_mu", PRIOR_GAMMA[2], constraint=constraints.interval(PRIOR_GAMMA[0], PRIOR_GAMMA[1]))
    log_gamma_sigma = numpyro.param("log_gamma_sigma",1e-3, constraint=constraints.positive)
    numpyro.sample("log_gamma", dist.Normal(log_gamma_mu, log_gamma_sigma))

    t0_mu = numpyro.param("t0_mu", PRIOR_T0[2], constraint=constraints.interval(PRIOR_T0[0], PRIOR_T0[1]))
    t0_sigma = numpyro.param("t0_sigma",1e-3, constraint=constraints.positive)
    numpyro.sample("t0", dist.Normal(t0_mu, t0_sigma))

    log_tau_rise_mu = numpyro.param("log_tau_rise_mu", PRIOR_TAU_RISE[2], constraint=constraints.interval(PRIOR_TAU_RISE[0], PRIOR_TAU_RISE[1]))
    log_tau_rise_sigma = numpyro.param("log_tau_rise_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("log_tau_rise", dist.Normal(log_tau_rise_mu, log_tau_rise_sigma))

    log_tau_fall_mu = numpyro.param("log_tau_fall_mu", PRIOR_TAU_FALL[2], constraint=constraints.interval(PRIOR_TAU_FALL[0], PRIOR_TAU_FALL[1]))
    log_tau_fall_sigma = numpyro.param("log_tau_fall_sigma",1e-3, constraint=constraints.positive)
    numpyro.sample("log_tau_fall", dist.Normal(log_tau_fall_mu, log_tau_fall_sigma))

    log_extra_sigma_mu = numpyro.param("log_extra_sigma_mu", PRIOR_EXTRA_SIGMA[2], constraint=constraints.interval(PRIOR_EXTRA_SIGMA[0], PRIOR_EXTRA_SIGMA[1]))
    log_extra_sigma_sigma = numpyro.param("log_extra_sigma_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("log_extra_sigma", dist.Normal(log_extra_sigma_mu, log_extra_sigma_sigma))

    # aux bands

    Ag_mu = numpyro.param("A_g_mu", PRIOR_A_g[2], constraint=constraints.interval(PRIOR_A_g[0], PRIOR_A_g[1]))
    Ag_sigma = numpyro.param("A_g_sigma", 1e-3,constraint=constraints.positive)
    numpyro.sample("A_g", dist.Normal(Ag_mu, Ag_sigma))

    beta_g_mu = numpyro.param("beta_g_mu", PRIOR_BETA_g[2], constraint=constraints.interval(PRIOR_BETA_g[0], PRIOR_BETA_g[1]))
    beta_g_sigma = numpyro.param("beta_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("beta_g", dist.Normal(beta_g_mu, beta_g_sigma))

    gamma_g_mu = numpyro.param("gamma_g_mu", PRIOR_GAMMA_g[2], constraint=constraints.interval(PRIOR_GAMMA_g[0], PRIOR_GAMMA_g[1]))
    gamma_g_sigma = numpyro.param("gamma_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("gamma_g", dist.Normal(gamma_g_mu, gamma_g_sigma))

    t0_g_mu = numpyro.param("t0_g_mu", PRIOR_T0_g[2], constraint=constraints.interval(PRIOR_T0_g[0], PRIOR_T0_g[1]))
    t0_g_sigma = numpyro.param("t0_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("t0_g", dist.Normal(t0_g_mu, t0_g_sigma))

    tau_rise_g_mu = numpyro.param("tau_rise_g_mu", PRIOR_TAU_RISE_g[2], constraint=constraints.interval(PRIOR_TAU_RISE_g[0], PRIOR_TAU_RISE_g[1]))
    tau_rise_g_sigma = numpyro.param("tau_rise_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("tau_rise_g", dist.Normal(tau_rise_g_mu, tau_rise_g_sigma))

    tau_fall_g_mu = numpyro.param("tau_fall_g_mu", PRIOR_TAU_FALL_g[2], constraint=constraints.interval(PRIOR_TAU_FALL_g[0], PRIOR_TAU_FALL_g[1]))
    tau_fall_g_sigma = numpyro.param("tau_fall_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("tau_fall_g", dist.Normal(tau_fall_g_mu, tau_fall_g_sigma))

    extra_sigma_g_mu = numpyro.param("extra_sigma_g_mu", PRIOR_EXTRA_SIGMA_g[2], constraint=constraints.interval(PRIOR_EXTRA_SIGMA_g[0], PRIOR_EXTRA_SIGMA_g[1]))
    extra_sigma_g_sigma = numpyro.param("extra_sigma_g_sigma", 1e-3, constraint=constraints.positive)
    numpyro.sample("extra_sigma_g", dist.Normal(extra_sigma_g_mu, extra_sigma_g_sigma))


def import_data(mjd, m, merr, b, ra, dec, zp, sfd):
    """
    Imports all relevant data relevant to one lightcurve/locus,
    removes points with NaN errors, converts to flux units, and 
    performs data clipping/pre-processing for better fitting.
    
    Parameters
    ----------
    mjd : float numpy array
        the modified Julian date (time) values
    m : float numpy array
        the measured magnitude datapoints of the transient, not
        adjusted for extinction
    merr : float numpy array
        the errors associated with the magnitude array m
    b : int numpy array
        bands associated with each measurement (2 = red, 1 = green).
        ignores infrared (b = 0) measurements
    ra : float
        right ascension of transient
    dec : float
        declination of transient
    sfd : SFDQuery object
        used to query the dust E(B-V) value at (ra, dec)
    
    Returns
    ----------
    t : float numpy array
        pre-processed mjd values
    f : float numpy array
        pre-processed flux values
    ferr : float numpy array
        errors associated with f
    b : int numpy array
        bands associated with f
    """
    PAD_SIZE = 30
    g_ext, r_ext = get_band_extinctions(ra, dec, sfd)

    valid_idxs = ~np.isnan(merr) & ~np.isnan(mjd)
    valid_idxs = valid_idxs & ~np.isnan(m)
    valid_idxs = valid_idxs & ~np.isnan(zp)
    valid_idxs = valid_idxs & ~np.isnan(b)

    t = mjd[valid_idxs]
    m = m[valid_idxs]
    b = b[valid_idxs].astype(np.int64)
    zp = zp[valid_idxs]
    merr = merr[valid_idxs]
    
    # get rid of repeat times
    uniq, uniq_idxs = np.unique(t, return_index=True)
    t, m, b, zp, merr = t[uniq_idxs], m[uniq_idxs], b[uniq_idxs], zp[uniq_idxs], merr[uniq_idxs]
    
    # don't desync the redshift corrections made from flux -> mag
    # use mean zp instead
    zp_mean = np.mean(zp)
    
    f, ferr = convert_mags_to_flux(m, merr, zp_mean)
    
    #print("mags converted to flux")
    f[b == 1] = f[b == 1] - g_ext
    f[b == 2] = f[b == 2] - r_ext

    t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
    #print("lightcurves clipped")
    
    snr = np.abs(f / ferr)
    for band in [1, 2]:
        if len(snr[(snr > 5.) & (b == band)]) < 3: # not enough good datapoints
            return None, None, None, None
        if (np.max(f[b == band]) - np.min(f[b == band])) < 5. * np.mean(ferr[b == band]): # data is too uncertainty-dominated
            return None, None, None, None
            
    max_flux_loc = t[b == 2][np.argmax(f[b == 2] - np.abs(ferr[b == 2]))]
    t -= max_flux_loc # make relative
    
    # pad data
    t_padded, f_padded, ferr_padded, b_padded = np.array([]), np.array([]), np.array([]), np.array([])
    
    for b_int in [1,2]:
        len_b = len(b[b == b_int])
        t_s = t[b == b_int]
        f_s = f[b == b_int]
        ferr_s = ferr[b == b_int]
        b_s = b[b == b_int]
        
        if len_b > PAD_SIZE:
            t_padded = np.append(t_padded, t_s[:PAD_SIZE])
            f_padded = np.append(f_padded, f_s[:PAD_SIZE])
            ferr_padded = np.append(ferr_padded, ferr_s[:PAD_SIZE])
            b_padded = np.append(b_padded, b_s[:PAD_SIZE])
        else:
            t_padded = np.append(t_padded, t_s)
            f_padded = np.append(f_padded, f_s)
            ferr_padded = np.append(ferr_padded, ferr_s)
            b_padded = np.append(b_padded, b_s)
            
            t_padded = np.append(t_padded, [5000.] * (PAD_SIZE - len_b))
            f_padded = np.append(f_padded, [1e-5] * (PAD_SIZE - len_b))
            ferr_padded = np.append(ferr_padded, [1e7] * (PAD_SIZE - len_b))
            b_padded = np.append(b_padded, [b_int] * (PAD_SIZE - len_b) )
    
    return t_padded, f_padded, ferr_padded, b_padded

    #return t, f, ferr, b


def convert_mags_to_flux(m, merr, zp):
    """
    Converts magnitude to flux values given a zeropoint.
    Note that, for a constant zeropoint choice, flux values
    will all be vertically scaled by the same factor, so should
    not impact the other final parameter values.
    
    Parameters
    ----------
    m : numpy float array
        magnitude values
    merr : numpy float array
        errors associated with m values
    zp : float
        zero-point magnitude (corresponds to flux = 1)
        
    Returns
    ----------
    fluxes : numpy float array
        the fluxes converted from m
    flux_unc : numpy float array
        flux uncertainties associated with fluxes
    """
    fluxes = 10. ** (-1. * ( m - zp ) / 2.5)
    flux_unc = np.log(10.)/2.5 * fluxes * merr
    return fluxes, flux_unc


def clip_lightcurve_end(times, fluxes, fluxerrs, bands):
    """
    Clips the far end of lightcurve with approx. 0 slope.
    Checks from most recent points to the argmax of lightcurve.
    Improves fitting in case where end of transient event is not
    properly recognized or there's an unrelated underlying signal.
    
    Parameters
    ----------
    times : numpy float array
        unclipped mjd values
    fluxes: numpy float array
        unclipped flux values
    fluxerr: numpy float array
        uncertainties associated with 'fluxes'
    bands : numpy int array
        bands associated with each measurement
        
    Returns
    ----------
    t_clip : numpy float array
        clipped mjd values
    flux_clip : numpy float array
        clipped flux values
    ferr_clip: numpy float array
        clipped flux uncertainties
    b_clip : numpy float array
        bands of all clipped measurements
    """
    def line_fit(x, a, b):
        return a*x + b
    
    t_clip, flux_clip, ferr_clip, b_clip = [], [], [], []
    for b in [1, 2]: # clips each band separately
        idx_b = (bands == b)
        t_b, f_b, ferr_b = times[idx_b], fluxes[idx_b], fluxerrs[idx_b]
        
        if len(f_b) == 0: # does not clip if no data for one band
            continue
        
        if np.argmax(f_b) == -1:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            b_clip.extend([b] * len(f_b))
            continue
        
        end_i = len(t_b) - np.argmax(f_b)
        num_to_cut = 0
        
        # calculates cutoff slope (anything more shallow is considered after transient ended)
        m_cutoff = 0.1 * (f_b[-1] - np.amax(f_b)) / (t_b[-1] - t_b[np.argmax(f_b)])

        for i in range(2, end_i):
            # checks if cumulative absolute slope from back of lightcurve dips below m_cutoff
            cut_idx = -1*i
            m = (f_b[-1] - f_b[cut_idx]) / (t_b[-1] - t_b[cut_idx])

            if m > m_cutoff:
                num_to_cut = i
        
        # adds clipped or unclipped band data to overall t, f arrays
        if num_to_cut > 0:
            # clips lightcurve if end of transient detected
            t_clip.extend(t_b[:-num_to_cut])
            flux_clip.extend(f_b[:-num_to_cut])
            ferr_clip.extend(ferr_b[:-num_to_cut])
            b_clip.extend([b] * len(f_b[:-num_to_cut]))
        else:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            b_clip.extend([b] * len(f_b))
            
    return np.array(t_clip), np.array(flux_clip), np.array(ferr_clip), np.array(b_clip)


def get_band_extinctions(ra, dec, sfd):
    """
    Get green and red band extinctions in magnitudes for
    a single transient LC based on RA and DEC.
    
    Parameters
    ----------
    ra : float
        right ascension of transient
    dec : float
        declination of transient
    sfd : SFDQuery object
        used to get E(B-V) for certain coordinate
        
    Returns
    ----------
    ext_list: 2-element float numpy array
        g magnitude, r magnitude that is subtracted by dust
    """
    #First look up the amount of mw dust at this location
    coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
    Av_sfd = 2.742 * sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html
    # for g and r ZTF bands, the wavelengths are:
    band_wvs = 1./ (0.0001 * np.asarray([4741.64, 6173.23])) # in inverse microns
    
    #Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit='invum') # in magnitudes
    return ext_list
    


def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    
    @jit
    def update_svi(s, _):
        return svi.update(s, *args, **kwargs)

    u = svi_state
    u, _ = lax.scan(update_svi, svi_state, None, length=num_iters)
    return u

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
    features[4:7] = np.log10(features[4:7])
    features[2] = np.log10(features[2])
    return np.delete(features, [0,3])

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

def calc_logL(cube, mjd, flux, flux_err, bands):
    """
    Calculates logL from model parameters and observations
    """
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]

    phase = mjd - t0    
    f_model = A / (1. + np.exp(-phase / tau_rise)) * (1. - beta * gamma) * np.exp((gamma - phase) / tau_fall)
    f_model[phase < gamma] = A / (1. + np.exp(-phase[phase < gamma] / tau_rise)) * (1. - beta * phase[phase < gamma])

    start_idx = 7
    A_b = A * cube[start_idx]
    beta_b = beta * cube[start_idx + 1]
    gamma_b = gamma * cube[start_idx + 2]
    t0_b = t0 * cube[start_idx + 3]
    tau_rise_b = tau_rise * cube[start_idx + 4]
    tau_fall_b = tau_fall * cube[start_idx + 5]

    inc_band_ix = (bands == 1)
    phase_b = (mjd - t0_b)[inc_band_ix]
    phase_b2 = (mjd - t0_b)[inc_band_ix & (mjd - t0_b < gamma_b)]

    f_model[inc_band_ix] = A_b / (1. + np.exp(-phase_b / tau_rise_b)) \
        * (1. - beta_b * gamma_b) * np.exp((gamma_b - phase_b) / tau_fall_b)
    f_model[inc_band_ix & (mjd - t0_b < gamma_b)] = A_b / (1. + np.exp(-phase_b2 / tau_rise_b)) \
        * (1. - phase_b2 * beta_b)
    
    extra_sigma_arr = np.ones(len(mjd)) * np.max(flux[bands == 2]) * cube[6]
    extra_sigma_arr[bands == 1] *= cube[-1]
    
    sigma_sq = extra_sigma_arr**2 + flux_err**2
    logL = np.sum(np.log(1. / np.sqrt(2. * np.pi * sigma_sq)) - 0.5 * (flux - f_model)**2 / sigma_sq) / len(mjd)
    return logL

class SuperphotPlusSNClassification(dk.Filter):
    NAME = "Superphot+ Supernovae Classification"
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
            'name': 'superphot_plus_param_1',
            'type': 'float',
            'description': 'Median fitting parameter 1 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_2',
            'type': 'float',
            'description': 'Median fitting parameter 2 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_3',
            'type': 'float',
            'description': 'Median fitting parameter 3 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_4',
            'type': 'float',
            'description': 'Median fitting parameter 4 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_5',
            'type': 'float',
            'description': 'Median fitting parameter 5 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_6',
            'type': 'float',
            'description': 'Median fitting parameter 6 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_7',
            'type': 'float',
            'description': 'Median fitting parameter 7 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_8',
            'type': 'float',
            'description': 'Median fitting parameter 8 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_9',
            'type': 'float',
            'description': 'Median fitting parameter 9 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_10',
            'type': 'float',
            'description': 'Median fitting parameter 10 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_11',
            'type': 'float',
            'description': 'Median fitting parameter 11 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_12',
            'type': 'float',
            'description': 'Median fitting parameter 12 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_13',
            'type': 'float',
            'description': 'Median fitting parameter 13 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        
        {
            'name': 'superphot_plus_param_14',
            'type': 'float',
            'description': 'Median fitting parameter 14 of the model described in de Soto et al. (TBD) using dynesty nested sampling.',
        },
        {
            'name': 'superphot_plus_class',
            'type': 'str',
            'description': 'Type of SN according to a multi-layer perceptron classifier detailed in de Soto et al. (TBD). One of SNIa, SNIbc, SNII, SNIIn, or SLSN-I.',
        },
        {
            'name': 'superphot_plus_class_prob',
            'type': 'float',
            'description': 'Probability associated with the most-likely SN type from the classifier detailed in de Soto et al. (TBD).',
        },
        
    ]
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = [
        {
            'name': 'superphot_plus_classified',
            'description': 'Successfully classified by the superphot_plus filter.',
        },
    ]
    
    REQUIRES_FILES = ['desoto_snclassifier_v4.pt',]

    def setup(self):
        """
        ANTARES will call this function once at the beginning of each night
        when filters are loaded.
        """
        # Loads SFDQuery object once to lower overhead
        
        from dustmaps.config import config
        config.reset()
        self.tempdir = TemporaryDirectory(prefix='superphot_plus_sfds_')
        config['data_dir'] = self.tempdir.name
        #config['data_dir'] = '/tmp/'  # datalab
        import dustmaps.sfd
        dustmaps.sfd.fetch()
        from dustmaps.sfd import SFDQuery
        self.sfd = SFDQuery()
        
        # below this line, won't need for anomaly detection
        model_fn = self.files['desoto_snclassifier_v4.pt'] # loads trained model file
        stream = io.BytesIO(model_fn)
        self.model = MLP(13, 5, 32, 2) # set up empty multi-layer perceptron
        self.model.load_state_dict(torch.load(stream)) # load trained state dict to the MLP
        
        #print("Loaded model")
        self.classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"} #converts the MLP classes to types
        
        # to normalize the input features, specific to model version (v2)
        self.means = np.array([5.21746163e-03, 1.25438494e+00, 7.47701555e-01, 1.56897030e+00, \
                               -1.50431711e+00, 9.64408639e-01, 1.04243307e+00, 1.00746242e+00, \
                               9.99978398e-01, 9.66099513e-01, 5.56633324e-01, 8.60829954e-01, \
                               -7.76704607e+00])
        self.stddevs = np.array([4.83968372e-04, 2.99370937e-01, 4.12001107e-01, 2.40517501e-01, \
                                 3.39083527e-01, 2.11565819e-01, 2.44746067e-03, 1.30120910e-02, \
                                 4.76878524e-05, 1.19610975e-02, 5.84996669e-02, 3.56732795e-02, \
                                 2.05751971e+01])
        
        self.variable_catalogs = ["gaia_dr3_variability", "sdss_stars", "asassn_variable_catalog_v2_20190802", "vsx", "linear_ll"]
        
        
        # initialize SVI
        self.optimizer = numpyro.optim.Adam(step_size=0.001)
        self.svi = SVI(jax_model, jax_guide, self.optimizer, loss=Trace_ELBO())
        self.svi_state = None
        self.num_iter = 10000
        
        self.lax_jit = jit(lax_helper_function, static_argnums=(0,2))

    def run_svi(self, tdata, fdata, ferrdata, bdata):
        """
        Run stochastic variational inference with numpyro.
        """
        max_flux = np.max(fdata[bdata == 2] - np.abs(ferrdata[bdata == 2]))

        if self.svi_state is None:
            self.svi_state = self.svi.init(random.PRNGKey(1), obsflux = fdata, t=tdata, uncertainties=ferrdata, max_flux=max_flux)

        self.svi_state = self.lax_jit(self.svi, self.svi_state, self.num_iter, obsflux = fdata, t=tdata, uncertainties=ferrdata, max_flux=max_flux)
        
        params = self.svi.get_params(self.svi_state)
        """
        posterior_samples = {}
        for p in params:
            if p[-2:] == "mu":
                posterior_samples[p[:-3]] = np.random.normal(loc=params[p], scale=params[p[:-2] + "sigma"], size=30)
        """
        param_list = ["logA", "beta", "log_gamma", "t0", "log_tau_rise",
                      "log_tau_fall", "log_extra_sigma", "A_g", "beta_g", 
                      "gamma_g", "t0_g", "tau_rise_g", "tau_fall_g", "extra_sigma_g"]

        post_reformatted_for_save = []
        
        for p in param_list:
            if p == "logA":
                post_reformatted_for_save.append(max_flux * 10**(params[p+"_mu"]))
            elif p[:3] == "log":
                post_reformatted_for_save.append(10**params[p+"_mu"])
            else:
                post_reformatted_for_save.append(params[p+"_mu"])

        return np.array(post_reformatted_for_save)

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
        ts = locus.timeseries[['ant_mjd','ztf_magpsf','ztf_sigmapsf', 'ztf_fid', 'ant_ra', 'ant_dec', 'ztf_magzpsci']]   
        mjd, m, merr, b, ra, dec, zp = ts.to_pandas().to_numpy().T
        
        # if LC longer than 200 days, probably not a transient (so skip)
        if ( np.nanmax(mjd) - np.nanmin(mjd) >= 200 ):
            return None
        
        # RA and dec expected to stay approximately constant across alerts
        # just take first non-nan ra and dec value, if there is one
        try:
            ra = ra[~np.isnan(ra)][0]
            dec = dec[~np.isnan(dec)][0]
        except:
            return None
        
        if len(mjd[b == 1]) < 3: # need more data to fit
            return None
        if len(mjd[b == 2]) < 3:
            return None
        
        #print("starting run nested sampling")
        try:
            tdata, fdata, ferrdata, bdata = import_data(mjd, m, merr, b, ra, dec, zp, self.sfd)
            if tdata is None: # if not enough data, import_data will return None, None, None, None
                return None

            post = self.run_svi(tdata, fdata, ferrdata, bdata)
        
        except:
            return None
            
        if post is None:
            return None
          
        if np.any(np.isnan(post)) or np.any(np.isinf(post)):
        	return None
                    
        # save each mean model parameter in locus properties
        for i in range(len(post)):
            locus.properties['superphot_plus_param_%d' % (i+1)] = post[i]
        
        try:
            logL = calc_logL(post, tdata, fdata, ferrdata, bdata)
            adjusted_params = adjust_log_dists(post) # converts some params to log space
            adjusted_params = np.append(adjusted_params, logL)
            normed_params = (adjusted_params - self.means) / self.stddevs # normalizes by mean and stddev used in training normalization

            probs_avg = get_predictions(self.model, torch.Tensor(normed_params[np.newaxis, :]), 'cpu').numpy() # uses model to output SN type probabilities
            
            pred_class = np.argmax(probs_avg)
            class_confidence = np.max(probs_avg)
            pred_sn_type = self.classes_to_labels[pred_class]

        except:
            return None
        
        # set predicted SN class and output probability of that classification
        locus.properties['superphot_plus_class'] = pred_sn_type
        locus.properties['superphot_plus_class_prob'] = class_confidence.item()
        locus.tag('superphot_plus_classified')
        
        
