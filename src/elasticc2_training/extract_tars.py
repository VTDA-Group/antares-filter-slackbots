"""From online directory structure, go through and extract all tarred FITS files into SNANA ASCII. Will enlarge folder size."""
import numpy as np
import gzip
import shutil
from astropy.io import fits
from astropy.table import Table
import os, glob
from superphot_plus.surveys.surveys import Survey
from superphot_plus.import_utils import clip_lightcurve_end
from superphot_plus.lightcurve import Lightcurve


def adjust_lc(mjd, flux, flux_err, bands, ra, dec, survey=Survey.LSST()):
    bands = [b.decode("utf-8").strip() for b in bands]
    ext_dict = survey.get_extinctions(ra, dec)
    sort_idx = np.argsort(np.array(mjd))
    t = np.array(mjd)[sort_idx].astype(float)
    f = np.array(flux)[sort_idx].astype(float)
    ferr = np.array(flux_err)[sort_idx].astype(float)
    b = np.array(bands)[sort_idx]

    no_nan_mask = ferr != np.nan
    t = t[no_nan_mask]
    f = f[no_nan_mask]
    b = b[no_nan_mask]
    ferr = ferr[no_nan_mask]

    unique_b = np.unique(b)

    for ub in unique_b:
        if ub in survey.wavelengths:
            f[b == ub] *= 10 ** (0.4 * ext_dict[ub])
        else:
            t = t[b != ub]
            f = f[b != ub]
            ferr = ferr[b != ub]
            b = b[b != ub]

    t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
    
    # clip other end
    t_rev, f_rev, ferr_rev, b_rev = clip_lightcurve_end(t[::-1], f[::-1], ferr[::-1], b[::-1])
    
    t = t_rev[::-1]
    f = f_rev[::-1]
    ferr = ferr_rev[::-1]
    b = b_rev[::-1]

    snr = np.abs(f / ferr)
    #for band in survey.wavelengths:
    if len(snr[snr > 3.0]) < 10:  # pragma: no cover
        print("low SNR")
        return [None] * 6
    if (np.max(f) - np.min(f)) < 3.0 * np.mean(ferr):  # pragma: no cover
        print("low amp")
        return [None] * 6

    # look for some keywords used in LightCurve object, move rest to kwargs

    return t, f, ferr, b, ra, dec


def extract_fits_gz(fn, save_dir):
    """Extract a .FITS.gz file and return contents."""
    prefix = fn.split('/')[-1][:-3] # remove .gz
    save_fn = os.path.join(save_dir, prefix)
    with gzip.open(fn, 'rb') as f_in:
        with open(save_fn, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return save_fn
            
            
def convert_fits_to_lc_obj(prefix, save_dir):
    """Parse photometric and header FITS files. Pre-processes
    corresponding light curve.
    """
    survey = Survey.ZTF()
    phot_fn = prefix + "_PHOT.FITS"
    head_fn = prefix + "_HEAD.FITS"
    
    dat = Table.read(phot_fn, format='fits')
    df = dat.to_pandas()
    
    # extract mjd, flux, flux_err, and bands arrays
    flags = df['PHOTFLAG']
    start_idxs = np.where(flags == 6144)[0]
    
    header = Table.read(head_fn, format='fits')
    head_df = header.to_pandas()
    
    ra_all = head_df["RA"].to_numpy()
    dec_all = head_df["DEC"].to_numpy()
    mwebv_all = head_df["MWEBV"].to_numpy()
    host_sep_all = head_df["HOSTGAL_SNSEP"].to_numpy()
    
    host_mag_all = []
    for b in "ugrizY":
        host_mag_all.append(head_df[f"HOSTGAL_MAG_{b}"].to_numpy())
        
    names_all = head_df["SNID"].to_numpy()
    
    try:
        labels_all = head_df["SIM_TYPE_INDEX"].to_numpy()
    except:
        labels_all = -1 * np.ones(len(names_all))
        
    for i, start_idx in enumerate(start_idxs):
        try:
            df_concat = df.iloc[start_idx:start_idxs[i+1]]
        except:
            df_concat = df.iloc[start_idx:]
        
        mjd = df_concat['MJD'].to_numpy()
        flux = df_concat['FLUXCAL'].to_numpy()
        flux_err = df_concat['FLUXCALERR'].to_numpy()
        band = df_concat['BAND'].to_numpy()
        
        ra = ra_all[i]
        dec = dec_all[i]
        mwebv = mwebv_all[i]
        host_sep = host_sep_all[i]
        
        sn_name = names_all[i].decode("utf-8").strip()
        t, f, ferr, b, ra, dec = adjust_lc(mjd, flux, flux_err, band, ra, dec)
        
        host_mag_dict = {
            "host_mag_u": host_mag_all[0][i],
            "host_mag_g": host_mag_all[1][i],
            "host_mag_r": host_mag_all[2][i],
            "host_mag_i": host_mag_all[3][i],
            "host_mag_z": host_mag_all[4][i],
            "host_mag_Y": host_mag_all[5][i],
        }
        
        if t is not None:
            lc = Lightcurve(
                t, f, ferr, b,
                name=sn_name,
                sn_class=labels_all[i],
                ra=ra,
                dec=dec,
                mwebv=mwebv,
                host_sep=host_sep,
                **host_mag_dict
                
            )
            save_fn = os.path.join(save_dir, sn_name + ".npz")
            lc.save_to_file(save_fn, overwrite=True)
    
    
def unfits_entire_directory(fits_dir, new_dir, lc_dir):
    """Un-FITS entire subdirectory structure into identical
    directory structure.
    """
    all_fits = glob.glob(os.path.join(fits_dir, "*", "*-0001_*.FITS.gz"))
    print(len(all_fits))
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(lc_dir, exist_ok=True)

    for fits_fn in all_fits:
        save_dir = os.path.join(new_dir, fits_fn.split('/')[-2])
        save_dir2 = os.path.join(lc_dir, fits_fn.split('/')[-2])
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir2, exist_ok=True)
        #try:
        save_fn = extract_fits_gz(fits_fn, save_dir)
        convert_fits_to_lc_obj(save_fn[:-10], save_dir2)
        #except:
        #    continue
        
if __name__ == "__main__":
    fits_dir = "../../../../elasticc2_data/elasticc2_dataset/"
    save_dir = "../../../../elasticc2_data/elasticc2_dataset_unzipped"
    lc_dir = "../../../../elasticc2_data/elasticc2_dataset_preprocessed"
    unfits_entire_directory(fits_dir, save_dir, lc_dir)
    
        
        
    