import numpy as np

    
def standard_quality_check(ts):
    """Return True if all quality checks are passed, else False."""
    if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
        return False

    # check enough datapoints
    times = ts.ant_mjd
    rounded_times = times.round().unique()
    if len(rounded_times) < 3:
        return False
    
    if ts['ant_ra'].std() > 0.5 / 3600.: # arcsec
        return False

    if ts['ant_dec'].std() > 0.5 / 3600.: # arcsec
        return False

    for b in ts['ant_passband'].unique():

        sub_ts = ts.loc[ts['ant_passband'] == b,:]

        # cut even when fewer points
        if (len(sub_ts) > 1) and (np.ptp(sub_ts['ant_mag']) < 0.2): # < 0.2 mag spread
            return False

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



def yse_quality_check(ts):
    """Return True if all quality checks are passed, else False."""        
    if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
        return False

    # check enough datapoints
    times = ts.ant_mjd
    rounded_times = times.round().unique()
    if len(rounded_times) < 3:
        return False

    if np.ptp(times.quantile([0.1, 0.9])) >= 100.:
        return False

    # require 2 good bands
    good_band1 = False
    good_band2 = False

    for b in ts['ant_passband'].unique():

        sub_ts = ts.loc[ts['ant_passband'] == b,:]

        if len(sub_ts) < 2:
            continue

        # cut even when fewer points
        if np.ptp(sub_ts['ant_mag']) < 0.2: # < 0.5 mag spread
            continue

        if len(sub_ts) < 5:
            if not good_band1:
                good_band1 = True
            else:
                good_band2 = True
            continue # don't do variability checks if < 5 points

        # first variability cut
        if np.ptp(sub_ts['ant_mag']) < 3 * sub_ts['ant_magerr'].mean():
            continue

        # second variability cut
        if sub_ts['ant_mag'].std() < sub_ts['ant_magerr'].mean():
            continue
            
        # third variability cut
        if np.ptp(sub_ts['ant_mag']) < 0.5: # < 0.5 mag spread
            return False
            
        if not good_band1:
            good_band1 = True
        else:
            good_band2 = True

    return (good_band1 and good_band2)


def atlas_quality_check(ts):
    """Same as YSE but without the duration check."""        
    if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
        return False

    # check enough datapoints
    times = ts.ant_mjd
    rounded_times = times.round().unique()
    if len(rounded_times) < 3:
        return False
    
    # require 2 good bands
    good_band1 = False
    good_band2 = False

    for b in ts['ant_passband'].unique():

        sub_ts = ts.loc[ts['ant_passband'] == b,:]

        if len(sub_ts) < 2:
            continue

        # cut even when fewer points
        if np.ptp(sub_ts['ant_mag']) < 0.2: # < 0.5 mag spread
            continue

        if len(sub_ts) < 5:
            if not good_band1:
                good_band1 = True
            else:
                good_band2 = True
            continue # don't do variability checks if < 5 points

        # first variability cut
        if np.ptp(sub_ts['ant_mag']) < 3 * sub_ts['ant_magerr'].mean():
            continue

        # second variability cut
        if sub_ts['ant_mag'].std() < sub_ts['ant_magerr'].mean():
            continue
            
        # third variability cut
        if np.ptp(sub_ts['ant_mag']) < 0.5: # < 0.5 mag spread
            return False
            
        if not good_band1:
            good_band1 = True
        else:
            good_band2 = True

    return (good_band1 and good_band2)


def relaxed_quality_check(ts):
    """Return True if all quality checks are passed, else False."""
    if len(ts['ant_passband'].unique()) < 2: # at least 1 point in each band
        return False

    # check enough datapoints
    times = ts.ant_mjd
    rounded_times = times.round().unique()
    if len(rounded_times) < 5:
        return False

    one_band_spread = False
    for b in ts['ant_passband'].unique():

        sub_ts = ts.loc[ts['ant_passband'] == b,:]

        # cut even when fewer points
        if len(sub_ts) < 2:
            continue
            
        if np.ptp(sub_ts['ant_mag']) < 0.2: # < 0.2 mag spread
            continue

        if len(sub_ts) > 4:
            # second variability cut
            if (sub_ts['ant_mag'].max() - sub_ts['ant_mag'].min()) >= sub_ts['ant_magerr'].median():
                one_band_spread = True
        else:
            one_band_spread = True
        
    return one_band_spread