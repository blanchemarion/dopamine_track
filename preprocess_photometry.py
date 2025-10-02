"""
Fiber photometry preprocessing and movement-artifact correction

Pipeline:
1) Load doric files (isosbestic, functional Ca2+, tdTomato, DIO)
2) Interpolate signals to a common time base and compute a causal baseline (F0)
3) Convert to ΔF/F and estimate motion contamination
4) Regress out motion (isosbestic and/or tdTomato) per ROI and save results
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import h5py
import statsmodels.api as sm
from scipy.signal.windows import gaussian
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import h5py


# CONSTANTS

FS_DEFAULT: float = 20.0 # camera sampling rate
LOWPASS_CUTOFF: float = 2.0  
BASELINE_PCTL: float = 6.0   # percentile for F0
GAUSS_WIN: int = 21   # Gaussian conv window length (samples)
GAUSS_ALPHA: float = 5.0   # Gaussian shape parameter


FILE_MAP: Dict[str, List[str]] = {
    'Eva': [
        'dopamine/Data/Expert_mice/Day1/Dat_LL_Eva_0002.doric',
        'dopamine/Data/Expert_mice/Day2/Dat_LL_Eva_0000.doric',
        'dopamine/Data/Expert_mice/Day3/Dat_LL_Eva_0000.doric',
    ],
    'Private': [
        'dopamine/Data/Expert_mice/Day1/Dat_LL_Private_0001.doric',
        'dopamine/Data/Expert_mice/Day2/Dat_LL_Private_0001.doric',
        'dopamine/Data/Expert_mice/Day3/Dat_LL_Private_0001.doric',
    ],
    'Rico': [
        'dopamine/Data/Expert_mice/Day1/Dat_R_Rico_0000.doric',
        'dopamine/Data/Expert_mice/Day2/Dat_R_Rico_0004.doric',
        'dopamine/Data/Expert_mice/Day3/Dat_R_Rico_0000.doric',
    ],
    'Skip': [
        'dopamine/Data/Expert_mice/Day1/Dat_L_Skip_0001.doric',
        # '../dopamine/Data/Expert_mice/Day2/Dat_L_Skip_0003.doric',
        'dopamine/Data/Expert_mice/Day3/Dat_L_Skip5_0000.doric',
    ],
    'Sanity': [
        'dopamine/Data/Expert_mice/Sanity/Dat_L_Skippe2_0004.doric',
        'dopamine/Data/Expert_mice/Sanity/Dat_LL_Eva2_0000.doric',
        'dopamine/Data/Expert_mice/Sanity/Dat_R_Rico2_0000.doric',
    ],
}

# keys: either animals or days
DEFAULT_SESSION_KEYS = ['Skip', 'Eva', 'Rico'] #['Day1', 'Day2', 'Day3']

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Optional base directory prefixed to FILE_MAP relative paths"
    )
    parser.add_argument(
        "--mouse-name",
        type=str,
        choices=sorted(FILE_MAP.keys()),
        default="Sanity",
        help="Which mouse/session set to process"
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default="dopamine/Data/Expert_mice/all_processed_Sanity.pkl",
        help=""
    )

    parser.add_argument(
        "--filttype",
        type=str,
        choices=["filtmove", "filtgau"],
        default="filtgau",
        help="pre-regression smoothing: moving average ('filtmove') or Gaussian ('filtgau')"
    )
    parser.add_argument(
        "--remove-small",
        action="store_true",
        help="mask near-zero points before regression"
    )
    parser.add_argument(
        "--crop-seconds",
        type=float,
        default=0.0,
        help="crop this many seconds from the start"
    )
    parser.add_argument(
        "--common-crop",
        action="store_true",
        help="if set, align to a common start time and then crop"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=FS_DEFAULT,
        help="sampling rate "
    )

    parser.add_argument(
        "--session-keys",
        nargs=3,
        default=DEFAULT_SESSION_KEYS,
        metavar=("S1", "S2", "S3"),
        help="Labels for the three sessions/files (default: %(default)s)."
    )

    parser.add_argument("--only_test", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)

    return params


################ Helper functions 

def running_percentile(x, win, p, nan_threshold=None):
    """
    Running percentile (with interpolation) over a sliding window
    """
    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    
    if win < 1 or win > N or not float(win).is_integer():
        raise ValueError("win must be integer between 1 and length(x)")
    win = int(win)
    if p < 0 or p > 100:
        raise ValueError("p must be between 0 and 100")
    if nan_threshold is None:
        nan_threshold = win // 2
    
    left_len= int(np.ceil(win/2) - 1)
    right_len = win - left_len - 1
    
    left = x[left_len-1::-1]
    right = x[:-right_len-1:-1]
    xpad= np.concatenate([left, x, right, [np.nan]])
    
    tmp = np.sort(xpad[:win])
    y = np.full(N, np.nan)
    numnans = np.sum(np.isnan(tmp))
    
    offset = left_len + win//2
    
    for i in range(N):
        valid = win - numnans
        pt    = p * valid / 100.0 + 0.5
        
        if numnans > nan_threshold:
            # leave y[i] as NaN
            pass
        elif pt < 1:
            y[i] = tmp[0]
        elif pt > valid:
            y[i] = tmp[valid-1]
        elif pt.is_integer():
            idx = int(pt) - 1
            y[i] = tmp[idx]
        else:
            lo = int(np.floor(pt))
            x0 = 100*(lo - 0.5)/valid
            x1 = 100*(lo + 0.5)/valid
            frac = (p - x0)/(x1 - x0)
            y[i] = tmp[lo-1] + (tmp[lo] - tmp[lo-1]) * frac
        
        # remove the oldest sample from 'tmp'
        oldest = xpad[i]
        if np.isnan(oldest):
            ix = win - 1
            numnans -= 1
        else:
            ix = np.where(tmp == oldest)[0][0]
        
        # bring in the next new sample
        new = xpad[offset + i + 1]
        tmp[ix] = new
        if np.isnan(new):
            numnans += 1
        
        tmp.sort()
    
    return y



def lowpass(data, fs, cutoff = LOWPASS_CUTOFF, order = 3):
    b, a = butter(order, cutoff/(fs/2), btype='low', analog=False)
    return filtfilt(b, a, data, axis=-1)


def fix_isolated_drops(sig, threshold):
    """Replace any point more than threshold below either neighbor with that neighbor"""
    clean = sig.copy()
    n = clean.size
    for i in range(n):
        curr = clean[i]
        if i > 0:
            prev = clean[i-1]
            if curr < prev - threshold:
                clean[i] = prev
                continue
        if i < n-1:
            nxt = clean[i+1]
            if curr < nxt - threshold:
                clean[i] = nxt
    return clean


def filter_peaks(x, k = 50, t0 = 4):
    """Median/MAD outlier suppression (symmetric window of size 2k)"""
    n = len(x)
    new_x = x.copy()
    for i in range(n):
        start, end = max(0, i-k), min(n, i+k)
        window = x[start:end]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if x[i] < med - t0 * mad:
            new_x[i] = med
    return new_x


################ Core pipeline 


def analysis_movement_artifact_arrays(time_iso, time_ca, time_td,time_dio, iso_raw, ca_raw, td_raw, dio, fs, 
                                      filttype = 'filtmove', remove_small = False, crop_seconds = 20.0, common_crop = True, 
                                      baseline_pctl = BASELINE_PCTL, gauss_win = GAUSS_WIN, gauss_alpha = GAUSS_ALPHA,
) :
    """
    Interpolate, baseline, ΔF/F, and motion regression per ROI
      - 'time_interp' : (T, R) times per ROI on the isosbestic time
      - 'fluo_interp' : (T, 3, R) raw interpolated (iso, ca, td)
      - 'fluo_interp_detrend' : trend reduced version
      - 'fluo_F0' : (T, 3, R) running-percentile baseline
      - 'fluo_dFoF' : (T, 3, R) ΔF/F per channel
      - 'corrected_data_iso' : (T, R) 
      - 'corrected_data_td'  : (T, R) 
      - 'corrected_data_best': (T, R) 
    """
    def _crop_to_t0(t: np.ndarray, arr: np.ndarray, t0: float) -> Tuple[np.ndarray, np.ndarray]:
        mask = t >= t0
        return t[mask], (arr[mask] if arr.ndim == 1 else arr[:, mask])

    if crop_seconds and crop_seconds > 0:
        if common_crop:
            # Align all streams to a common (latest start + crop_seconds)
            t0_common = max(time_iso[0], time_ca[0], time_td[0], time_dio[0]) + crop_seconds
            time_iso, iso_raw = _crop_to_t0(time_iso, iso_raw, t0_common)
            time_ca,  ca_raw  = _crop_to_t0(time_ca,  ca_raw,  t0_common)
            time_td,  td_raw  = _crop_to_t0(time_td,  td_raw,  t0_common)
            time_dio, dio     = _crop_to_t0(time_dio, dio,     t0_common)
        else:
            # Crop each stream independently by its own first + crop_seconds
            for t_arr, a_arr in [(time_iso, iso_raw), (time_ca, ca_raw),
                                 (time_td, td_raw), (time_dio, dio)]:
                t0 = t_arr[0] + crop_seconds
                tt, aa = _crop_to_t0(t_arr, a_arr, t0)
                if a_arr is iso_raw:
                    time_iso, iso_raw = tt, aa
                elif a_arr is ca_raw:
                    time_ca, ca_raw = tt, aa
                elif a_arr is td_raw:
                    time_td, td_raw = tt, aa
                else:
                    time_dio, dio = tt, aa

    t_ref = time_iso.copy()
    T = len(t_ref)

    g1 = gaussian(gauss_win, gauss_alpha)
    g1 /= g1.sum()

    num_ROI, time_len = iso_raw.shape
    num_EXC = 3  # iso, ca, td

    fluo_values = [iso_raw, ca_raw, td_raw]
    time_values = [time_iso, time_ca, time_td]

    time_interp = np.zeros((time_len, num_ROI))
    fluo_interp = np.zeros((time_len, num_EXC, num_ROI))
    fluo_interp_detrend = np.zeros_like(fluo_interp)
    fluo_F0 = np.zeros_like(fluo_interp)
    corrected_data_iso = np.zeros((time_len, num_ROI))
    corrected_data_td = np.zeros((time_len, num_ROI))
    corrected_data_best = np.zeros((time_len, num_ROI))
    corrChannel = np.full(num_ROI, np.nan)  # 1 for iso, 3 for td, NaN if none

    # Interpolate + detrend + baseline
    for roi in range(num_ROI):
        time_interp[:, roi] = t_ref

        for exc, (f_raw, t_raw) in enumerate(zip(fluo_values, time_values)):
            v_orig = f_raw[roi]
            t_orig = t_raw

            fluo_interp[:, exc, roi] = np.interp(t_ref, t_orig, v_orig)
            
            # Useful only when there are large artefacts in the raw signals 
            #fluo_interp[:, exc, roi] = fix_isolated_drops(fluo_interp[:, exc, roi], 8)
            #fluo_interp[:, exc, roi] = filter_peaks(fluo_interp[:, exc, roi])
            
            # Drift removal 
            runmin_win = max(1, int(round(fs * 40)))
            runmin = minimum_filter1d(fluo_interp[:, exc, roi], size=runmin_win)
            runmax = maximum_filter1d(runmin, size=runmin_win * 2)
            fluo_interp_detrend[:, exc, roi] = (
                fluo_interp[:, exc, roi] - runmax + np.mean(runmax[:min(40, len(runmax))])
            )

            # Moving-percentile baseline F0 (percentile of raw signal)
            runpr_win = max(1, int(round(fs * 20)))
            fluo_F0[:, exc, roi] = running_percentile(fluo_interp[:, exc, roi], runpr_win, baseline_pctl)

    # Interpolate DIO to reference time (not used further here, but preserved)
    DIO = np.interp(t_ref, time_dio, dio)

    #  ΔF/F 
    fluo_dFoF = (fluo_interp - fluo_F0) / fluo_F0

    #  Motion regression per ROI 
    for roi in range(num_ROI):
        if filttype == 'filtmove':
            roll = lambda v: pd.Series(v).rolling(20, min_periods=1).mean().values
            x_iso = roll(fluo_dFoF[:, 0, roi])
            y_ca1 = roll(fluo_dFoF[:, 1, roi])
            y_ca2 = y_ca1.copy()
            x_td  = roll(fluo_dFoF[:, 2, roi])
        else:  # 'filtgau'
            conv = lambda v: np.convolve(v, g1, mode='same')
            x_iso = conv(fluo_dFoF[:, 0, roi])
            y_ca1 = conv(fluo_dFoF[:, 1, roi])
            y_ca2 = y_ca1.copy()
            x_td  = conv(fluo_dFoF[:, 2, roi])

        # masking of near-zero points
        if remove_small:
            m1 = np.abs(x_iso) > np.nanmean(np.abs(x_iso))
            m2 = np.abs(x_td)  > np.nanmean(np.abs(x_td))
            x_iso_s, y1_s = x_iso[m1], y_ca1[m1]
            x_td_s,  y2_s = x_td[m2],  y_ca2[m2]
        else:
            x_iso_s, y1_s = x_iso, y_ca1
            x_td_s,  y2_s = x_td,  y_ca2

        # Correlations (for choosing the best motion regressor)
        R1, _ = pearsonr(x_iso_s, y1_s)
        R2, _ = pearsonr(x_td_s,  y2_s)

        # GLM (Gaussian / OLS) regressions
        beta1 = sm.GLM(y1_s, sm.add_constant(x_iso_s), family=sm.families.Gaussian()).fit()
        beta2 = sm.GLM(y2_s, sm.add_constant(x_td_s),  family=sm.families.Gaussian()).fit()
        corr_isosb = int(beta1.pvalues[1] < 0.05)
        corr_tdT   = int(beta2.pvalues[1] < 0.05)

        # Decide which channel to use
        if corr_isosb and not corr_tdT:
            corrChannel[roi] = 1
            beta_corr = beta1.params
        elif (not corr_isosb) and corr_tdT:
            corrChannel[roi] = 3
            beta_corr = beta2.params
        elif corr_isosb and corr_tdT:
            if abs(R1) > abs(R2):
                corrChannel[roi] = 1
                beta_corr = beta1.params
            else:
                corrChannel[roi] = 3
                beta_corr = beta2.params
        else:
            corrChannel[roi] = np.nan
            beta_corr = np.array([0.0, 0.0])

        # Apply corrections as differences (ΔF/F units)
        fluo_tocorr = fluo_dFoF[:, 1, roi]

        forced_iso = fluo_tocorr - (beta1.params[0] + beta1.params[1] * fluo_dFoF[:, 0, roi])
        forced_td  = fluo_tocorr - (beta2.params[0] + beta2.params[1] * fluo_dFoF[:, 2, roi])

        if np.isnan(corrChannel[roi]):
            fluo_corrected_best = fluo_tocorr.copy()
        elif corr_isosb and corr_tdT:
            fluo_corrected_best  = fluo_tocorr - (beta1.params[0] + beta1.params[1]*fluo_dFoF[:, 0, roi])
            fluo_corrected_best -= (beta2.params[0] + beta2.params[1]*fluo_dFoF[:, 2, roi])
        else:
            if corrChannel[roi] == 1:
                beta = beta1.params
                ref  = fluo_dFoF[:, 0, roi]
            else:
                beta = beta2.params
                ref  = fluo_dFoF[:, 2, roi]
            fluo_corrected_best = fluo_tocorr - (beta[0] + beta[1]*ref)

        corrected_data_iso[:, roi]  = forced_iso
        corrected_data_td[:,  roi]  = forced_td
        corrected_data_best[:, roi] = fluo_corrected_best

    return {
        'time_interp': time_interp,
        'fluo_interp': fluo_interp,
        'fluo_interp_detrend': fluo_interp_detrend,
        'fluo_F0': fluo_F0,
        'fluo_dFoF': fluo_dFoF,
        'corrected_data_iso': corrected_data_iso,
        'corrected_data_td': corrected_data_td,
        'corrected_data_best': corrected_data_best,
        # 'corrChannel': corrChannel,
        # 'DIO_interp': DIO,
    }


# doric utils

def unroll_CAM_data(cam_group):
    """Collect ROI datasets under a CAM group into an array of shape (R, T)."""
    keys = list(cam_group.keys())
    rois = [k for k in keys if 'ROI' in k]
    data = np.zeros((len(rois), len(cam_group['Time'])))
    for i, roi in enumerate(rois):
        data[i, :] = np.array(cam_group[roi])
    return data


def unroll_CAM_time(cam_group) :
    """Return the 'Time' dataset as a 1D numpy array."""
    return np.array(cam_group['Time'])


def load_doric_triplet(fname):
    """
    Load the three excitation streams and DIO from a Doric HDF5 file.
    Returns dict with time_* and *_raw arrays.
    """
    with h5py.File(fname, 'r') as data:
        base = data['DataAcquisition']['BBC300']
        cam_iso = base['ROISignals']['Series0001']['CAM1EXC1']
        cam_ca  = base['ROISignals']['Series0001']['CAM1EXC2']
        cam_td  = base['ROISignals']['Series0001']['CAM2EXC3']
        dio_ds  = base['Signals']['Series0001']['DigitalIO']

        time_iso = unroll_CAM_time(cam_iso)
        time_ca  = unroll_CAM_time(cam_ca)
        time_td  = unroll_CAM_time(cam_td)
        time_dio = np.array(dio_ds['Time'])

        iso_raw = unroll_CAM_data(cam_iso)
        ca_raw  = unroll_CAM_data(cam_ca)
        td_raw  = unroll_CAM_data(cam_td)
        dio     = np.array(dio_ds['Camera1'])

    return {
        'time_iso': time_iso, 'time_ca': time_ca, 'time_td': time_td, 'time_dio': time_dio,
        'iso_raw': iso_raw, 'ca_raw': ca_raw, 'td_raw': td_raw, 'dio': dio,
    }

# plotting functions 

def _roi_trace(x, roi):
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return x[roi, :]

def plot_iso_ca_td(all_data, day='Day1', roi=0, align_to_ca=False, title=None):

    d = all_data[day]
    t_iso, t_ca, t_td = d['time_iso'], d['time_ca'], d['time_td']
    iso  = _roi_trace(d['iso_raw'], roi)
    ca   = _roi_trace(d['ca_raw'],  roi)
    td   = _roi_trace(d['td_raw'],  roi)

    plt.figure(figsize=(12, 5))

    if not align_to_ca:
        # Plot each trace against its native time vector
        plt.plot(t_iso, iso, label='iso (CAM1EXC1)', alpha=0.8)
        plt.plot(t_ca,  ca,  label='ca  (CAM1EXC2)', alpha=0.8)
        plt.plot(t_td,  td,  label='td  (CAM2EXC3)', alpha=0.8)
        plt.xlabel('Time (s)')
    else:
        # Align iso & td to the calcium time vector before plotting
        x = t_ca
        # Use np.interp with NaN outside bounds so we don’t extrapolate silently
        iso_al = np.interp(x, t_iso, iso, left=np.nan, right=np.nan)
        td_al  = np.interp(x, t_td,  td,  left=np.nan, right=np.nan)

        plt.plot(x, iso_al, label='iso → aligned to ca', alpha=0.8)
        plt.plot(x, ca,     label='ca (reference)',     alpha=0.8)
        plt.plot(x, td_al,  label='td  → aligned to ca',alpha=0.8)
        plt.xlabel('Time (s)')

    plt.ylabel('Fluorescence (a.u.)')
    plt.title(title or f'{day} — ROI {roi}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def detect_edges(ttl, t, kind="rising"):
    """Return times of TTL edges (rising/falling)."""
    ttl = np.asarray(ttl).astype(float)
    t   = np.asarray(t).astype(float)
    thr = 0.5 * (np.nanmin(ttl) + np.nanmax(ttl))  # midpoint threshold
    b   = ttl > thr
    if kind == "rising":
        idx = np.flatnonzero(~b[:-1] &  b[1:]) + 1
    else:
        idx = np.flatnonzero( b[:-1] & ~b[1:]) + 1
    return t[idx], idx

def nearest_edge_indices(frame_t, edge_t):
    """For each frame time, find index of nearest edge time."""
    frame_t = np.asarray(frame_t)
    edge_t  = np.asarray(edge_t)
    pos = np.searchsorted(edge_t, frame_t)
    pos = np.clip(pos, 1, len(edge_t)-1)
    left  = edge_t[pos-1]
    right = edge_t[pos]
    # choose nearest of left/right
    use_left = (frame_t - left) <= (right - frame_t)
    idx = np.where(use_left, pos-1, pos)
    return idx

def summarize_alignment(frame_t, edge_t, name, tol_ms=10.0):
    """Print alignment stats: residuals, % within tolerance, parity, drops."""
    if len(edge_t) < 3 or len(frame_t) == 0:
        print(f"[{name}] Not enough edges/frames for summary.")
        return

    idx = nearest_edge_indices(frame_t, edge_t)
    residual = (frame_t - edge_t[idx]) * 1000.0  # ms

    # how many frames align within tolerance?
    pct_within = np.mean(np.abs(residual) <= tol_ms) * 100.0

    # TTL gaps (to spot possible dropped frames): look for abnormally large gaps
    edge_dt = np.diff(edge_t)
    med_gap = np.median(edge_dt) if len(edge_dt) else np.nan
    big_gap_thr = 2.5 * med_gap if np.isfinite(med_gap) else np.inf
    n_big_gaps = int(np.sum(edge_dt > big_gap_thr))

    print(f"[{name}] frames: {len(frame_t)}, ttl edges: {len(edge_t)}")
    print(f"  residuals (ms): median={np.nanmedian(residual):.2f}, "
          f"IQR≈[{np.nanpercentile(residual,25):.2f}, {np.nanpercentile(residual,75):.2f}]")
    print(f"  TTL big gaps (>~{big_gap_thr:.3f}s): {n_big_gaps}")
    print("")

def check_alignment_with_dio(doric_path):
    with h5py.File(doric_path, "r") as f:
        # ROI frame time vectors (pick any ROI; timestamps are identical across ROIs within an EXC)
        t_iso = np.array(f["DataAcquisition"]["BBC300"]["ROISignals"]["Series0001"]["CAM1EXC1"]["Time"])
        t_ca  = np.array(f["DataAcquisition"]["BBC300"]["ROISignals"]["Series0001"]["CAM1EXC2"]["Time"])
        t_td  = np.array(f["DataAcquisition"]["BBC300"]["ROISignals"]["Series0001"]["CAM2EXC3"]["Time"])

        # Digital I/O (high-rate)
        dio_grp   = f["DataAcquisition"]["BBC300"]["Signals"]["Series0001"]["DigitalIO"]
        t_dio     = np.array(dio_grp["Time"])
        cam1_ttl  = np.array(dio_grp["Camera1"])  # 0/1 TTL samples
        cam2_ttl  = np.array(dio_grp["Camera2"])  # 0/1 TTL samples

    # Detect rising edges (you can switch to 'falling' if that matches your rig)
    cam1_edges, _ = detect_edges(cam1_ttl, t_dio, kind="rising")
    cam2_edges, _ = detect_edges(cam2_ttl, t_dio, kind="rising")

    # Summaries:
    # Camera 1 usually alternates EXC1/EXC2 on successive frames → expect iso/ca to map to opposite parity.
    summarize_alignment(t_iso, cam1_edges, "CAM1EXC1 vs DIO/Camera1")
    summarize_alignment(t_ca,  cam1_edges, "CAM1EXC2 vs DIO/Camera1")

    # Camera 2 typically hosts EXC3 (e.g., tdTomato)
    summarize_alignment(t_td,  cam2_edges, "CAM2EXC3 vs DIO/Camera2")


def main():
    params = init_params()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logging.info("Selected mouse: %s", params['mouse_name'])

    # file list for the chosen mouse 
    try:
        file_list = FILE_MAP[params['mouse_name']]
    except KeyError as e:
        raise ValueError(f"Unknown mouse name: {params['mouse_name']!r}. "
                         f"Valid options: {sorted(FILE_MAP.keys())}") from e

    # Load 
    all_data= {}
    for session_key, fname in zip(params["session_keys"], file_list):
        logging.info("Loading %s (%s)", session_key, fname)
        all_data[session_key] = load_doric_triplet(fname)

    # Process 
    all_processed = {}
    for session_key, dd in all_data.items():
        logging.info("Processing %s ...", session_key)
        result = analysis_movement_artifact_arrays(
            dd['time_iso'], dd['time_ca'], dd['time_td'], dd['time_dio'],
            dd['iso_raw'], dd['ca_raw'], dd['td_raw'], dd['dio'],
            fs=params["fs"],
            filttype=params["filttype"],
            remove_small=params["remove_small"],
            crop_seconds=params["crop_seconds"],
            common_crop=params["common_crop"],
        )
        all_processed[session_key] = result

    # Save 
    logging.info("Saving results to %s", params["output_pkl"])
    with open(params["output_pkl"], 'wb') as f:
        pickle.dump(all_processed, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Done.")
    
    
    # Optionnally plot:
    plot_iso_ca_td(all_data, day='Rico', roi=10, align_to_ca=False)
    for animal in FILE_MAP:
        for f in FILE_MAP[animal]:
            print(f)
            check_alignment_with_dio(f)


if __name__ == "__main__":
    main()
