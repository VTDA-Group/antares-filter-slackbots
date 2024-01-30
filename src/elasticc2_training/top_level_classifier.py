"""NOTE: THIS WILL BE MOVED INTO SUPERPHOT+ EVENTUALLY"""
import numpy as np
from scipy.optimize import curve_fit
from gatspy.periodic import LombScargleMultibandFast
from astropy.coordinates import SkyCoord
import warnings
from sklearn.model_selection import train_test_split
import os

from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TrainData
from superphot_plus.utils import create_dataset


CLASS_TO_ID: dict = {
    "Meta": 0,
    "Meta/Other": 100,
    "Residual": 200,
    "NotClassified": 300,
    "Static": 1000,
    "Static/Other": 1100,
    "Variable": 2000,
    "Variable/Other": 2100,
    "Non-Recurring": 2200,
    "Non-Recurring/Other": 2210,
    "SN-like": 2220,
    "SN-like/Other": 2221,
    "Ia": 2222,
    "Ibc": 2223,
    "II": 2224,
    "Iax": 2225,
    "91bg": 2226,
    "Fast": 2230,
    "Fast/Other": 2231,
    "KN": 2232,
    "M-dwarf-Flare": 2233,
    "Dwarf-Novae": 2234,
    "uLens": 2235,
    "Long": 2240,
    "Long/Other": 2241,
    "SLSN": 2242,
    "TDE": 2243,
    "ILOT": 2244,
    "CART": 2245,
    "PISN": 2246,
    "Recurring": 2300,
    "Recurring/Other": 2310,
    "Periodic": 2320,
    "Periodic/Other": 2321,
    "Cepheid": 2322,
    "RR-Lyrae": 2323,
    "Delta-Scuti": 2324,
    "EB": 2325,
    "LPV/Mira": 2326,
    "Non-Periodic": 2330,
    "Non-Periodic/Other": 2331,
    "AGN": 2332,
}

ORIG_ID_TO_CLASS = {
    10: "Ia",
    11: "91bg",
    12: "Iax",
    20: "Ibc",
    21: "Ibc",
    25: "Ibc",
    26: "Ibc",
    27: "Ibc",
    30: "II",
    31: "II",
    32: "II",
    35: "II",
    36: "II",
    37: "II",
    40: "SLSN",
    42: "TDE",
    45: "ILOT",
    46: "CART",
    50: "KN",
    51: "KN",
    59: "PISN",
    60: "AGN",
    80: "RR-Lyrae",
    82: "M-dwarf-Flare",
    83: "EB",
    84: "Dwarf-Novae",
    87: "uLens",
    88: "uLens",
    89: "uLens",
    90: "Cepheid",
    91: "Delta-Scuti"
}

def get_galactic_coordinates(ra, dec):
    """
    Get galactic coordinates corresponding to RA and Dec
    """
    coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
    g_coords = coords.galactic
    return g_coords.b.degree, g_coords.l.degree


def estimate_period(t, f, ferr, b, min_period=0.1):
    """
    Use MHAOV to estimate the best period of an assumed periodic signal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_period = np.max(t) - np.min(t)
        period_range = (min_period, max_period)
        optimizer_kwds = {
            "period_range": period_range,
            "quiet": True
        }
        per = LombScargleMultibandFast(
            fit_period=True,
            optimizer_kwds=optimizer_kwds
        )
        per.fit(t, f, ferr, b)
        return per.best_period


def get_meta_features(
    lc,
    priors,
):
    """
    Calculate the meta features used in the top level and recurring classifier.
    """
    extra_info = lc.extra_properties
    ra, dec = extra_info['ra'], extra_info['dec']
    mwebv = extra_info['mwebv']
    host_sep = extra_info['host_sep']
    host_mag = [extra_info[f'host_mag_{b}'] for b in priors.ordered_bands]
    
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
                    sigma=ferr[b == unique_b]
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
    best_period = extra_info['period']
    best_period_long = extra_info['period_long'] # only 5 days or longer
    gal_b, gal_l = get_galactic_coordinates(ra, dec)

    return np.array([
        gal_b, gal_l, mwebv, host_sep, *host_mag,
        positive_fraction,
        best_period, best_period_long,
        *lin_slopes, *max_ratios, *mean_ratios              
    ])


def convert_to_train_class(class_id):
    """Convert class name to enumeration."""
    class_name = ORIG_ID_TO_CLASS[class_id]
    new_id = CLASS_TO_ID[class_name]
    return new_id


def train_top_level_model(lcs, priors):
    """Generate dataset and train top-level model."""
    y = np.array([])
    X = []
    

    for i, lc in enumerate(lcs):
        if i % 1000 == 0:
            print(i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                label = convert_to_train_class(int(lc.sn_class))
                if len(y[y == label]) == 500: # cap at 500 per class
                    continue
                X.append(
                    get_meta_features(lc, priors)
                )
                y = np.append(y, label)
            except:
                continue
        
        
    X = np.array(X)
    
    # change to enumeration
    y_unique, y = np.unique(y, return_inverse=True)
    
    N_classes = len(y_unique)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, shuffle=True, test_size=0.1
    )
    
    X_normed, mean, stddev = normalize_features(X_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_normed, y_train, stratify=y_train, shuffle=True, test_size=0.1
    )
    
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
        
    model_config = ModelConfig(
        input_dim = X_normed.shape[1],
        output_dim = N_classes,
        normalization_means = mean,
        normalization_stddevs = stddev,
        neurons_per_layer = 64,
        num_hidden_layers = 3,
        num_folds = 10,
        num_epochs = 1000,
        batch_size = 128,
        learning_rate = 1e-3,
    )
    
    model_config.write_to_file("top_level_config.yaml")
    model = SuperphotClassifier.create(model_config)
    
    
    metrics = model.train_and_validate(
        TrainData(train_dataset, val_dataset),
        num_epochs=model.config.num_epochs
    )
    os.makedirs("top_level_model", exist_ok=True)
    model.save("top_level_model")
    
    
    y_probs = model.classify_from_fit_params(X_test)
    
    return X_test, y_test, y_probs, metrics
        
        

