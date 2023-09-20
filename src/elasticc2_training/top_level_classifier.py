"""NOTE: THIS WILL BE MOVED INTO SUPERPHOT+ EVENTUALLY"""
import numpy as np
from scipy.optimize import curve_fit
import P4J
from astropy.coordinates import SkyCoord


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


def get_meta_features(
    lc,
    priors,
                     
):
    """
    Calculate the meta features used in the top level and recurring classifier.
    
    TO ADD:
    - MWEBV
    - HOST/SOURCE OFFSET (both in alert file)
    - HOST GAL MAG
    """
    extra_info = lc.extras
    ra, dec = extra_info['ra'], extra_info['dec'] #TODO: add coords and mwebv fields
    mwebv = extra_info['mwebv']
    host_sep = extra_info['host_sep']
    host_mag = extra_info['host_mag']
    
    t, f, ferr, b = lc.times, lc.fluxes, lc.flux_errors, lc.bands
    lin_slopes = []
    
    for unique_b in priors.ordered_bands:
        if len(f[b == unique_b]) >= 2:
            lin_slopes.append(
                curve_fit(
                    lambda x, *p: p[0]*x + p[1],
                    t[b == unique_b],
                    f[b == unique_b],
                    [0, 0],
                    sigma=ferr[unique_b]
                )[0][0]
            )
        else:
            lin_slopes.append(0.)


    N = len(priors.bands)
    max_ratios = np.zeros(N-1)
    mean_ratios = np.zeros(N-1)
    
    b_ref = priors.reference_band
    max_ref = np.max(f[b == b_ref])
    mean_ref = np.mean(f[b == b_ref])
    
    for i, b_i in enumerate(priors.aux_bands):
        f_b = f[b == b_i]
        if len(f_b) < 2:
            continue
        max_ratios[i] = np.abs(np.max(f_b) / max_ref)
        mean_ratios[i] = np.abs(np.mean(f_b) / mean_ref)

    positive_fraction = len(f[f > 0]) / len(f)
    best_period = estimate_period(t, f, ferr)
    best_period_long = estimate_period(t, f, ferr, 0.2) # only 5 days or longer
    gal_b, gal_l = get_galactic_coordinates(ra, dec)

    return np.array([
        gal_b, gal_l, mwebv, host_sep,
        host_mag, positive_fraction,
        best_period, best_period_long,
        *lin_slopes, *max_ratios, *mean_ratios              
    ])


def train_top_level_model():
    """Generate dataset and train top-level model."""
    pass

