# ---------- helpers ----------
def _safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.median(x) if x.size else np.nan

def compute_photometry_dff(sig, ctrl):
    """
    ΔF/F0 with control regression:
      sig ≈ a*ctrl + b
      residual = sig - pred
      F0 = median(pred)
      dff = residual / |F0|
    """
    sig = np.asarray(sig, dtype=float).ravel()
    ctrl = np.asarray(ctrl, dtype=float).ravel()
    A = np.c_[ctrl, np.ones_like(ctrl)]
    coef, _, _, _ = np.linalg.lstsq(A, sig, rcond=None)
    pred = A @ coef
    resid = sig - pred
    F0 = _safe_median(pred)
    if not np.isfinite(F0) or np.isclose(F0, 0):
        F0 = _safe_median(sig)
    if not np.isfinite(F0) or np.isclose(F0, 0):
        F0 = 1.0
    return resid / abs(F0)

def zscore_1d(arr):
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return np.full_like(arr, np.nan)
    return (arr - mu) / sd

def frame_times(n_frames, fps=30.0):
    return np.arange(n_frames, dtype=float) / float(fps)

def resample_to_frames(t_src, y_src, t_frames):
    t_src = np.asarray(t_src, dtype=float).ravel()
    y_src = np.asarray(y_src, dtype=float).ravel()
    t_frames = np.asarray(t_frames, dtype=float).ravel()
    out = np.full_like(t_frames, np.nan, dtype=float)
    if t_src.size == 0:
        return out
    mask = (t_frames >= t_src[0]) & (t_frames <= t_src[-1])
    out[mask] = np.interp(t_frames[mask], t_src, y_src)
    return out

def local_intensity_from_flags(flags, fps=30.0, half_window_s=0.25):
    flags = np.asarray(flags, dtype=bool)
    w = max(1, int(round(2 * half_window_s * fps)))
    kern = np.ones(w, dtype=float)
    dens = convolve(flags.astype(float), kern, mode="same")
    peak = np.nanmax(dens) if dens.size else 0.0
    return dens / peak if peak > 0 else np.zeros_like(dens)

# ---------- Doric multi-ROI loading ----------
def load_doric_rois(
    filename,
    roi_names,           # list like ["ROI09","ROI10"]
    signal_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM1EXC2",
    control_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM2EXC3",
):
    """
    Returns:
      t_doric: (T,) seconds, anchored at 0
      sigs: dict roi_name -> (T,) signal
      ctrls: dict roi_name -> (T,) control
    """
    sigs, ctrls = {}, {}
    with h5py.File(filename, "r") as f:
        time_key = f"{signal_path}/Time" if f"{signal_path}/Time" in f else f"{signal_path}/Time(s)"
        t = np.array(f[time_key], dtype=float)
        t = t - t[0]  # start at 0 s
        for roi in roi_names:
            sigs[roi]  = np.array(f[f"{signal_path}/{roi}"], dtype=float).ravel()
            ctrls[roi] = np.array(f[f"{control_path}/{roi}"], dtype=float).ravel()
    return t, sigs, ctrls

# ---------- multi-ROI alignment ----------
def align_photometry_to_frames_multi(
    result,                 # your per-trial dict with tracking arrays
    doric_path,
    rois=("ROI09",),        # tuple/list of ROI names
    fps=30.0,
    z_threshold=1.0,
    intensity_half_window_s=0.25,
    signal_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM1EXC2",
    control_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM2EXC3",
):
    """
    For each ROI:
      compute ΔF/F0, z-score on Doric grid, resample z to frame grid,
      threshold, and compute local intensity.
    Returns a dict:
      {
        'meta': {...},
        'per_roi': {
            'ROI09': {'z_frame':..., 'above_thresh_mask':..., 'intensity':..., 'z_threshold':...},
            ...
        }
      }
    """
    # Step 1: load requested ROIs from Doric
    t_doric, sigs, ctrls = load_doric_rois(doric_path, list(rois), signal_path, control_path)

    # Step 2: video frame grid
    n_frames = len(result["snout_x"])
    t_frames = frame_times(n_frames, fps=fps)

    # Step 3–4 per ROI
    per_roi = {}
    for roi in rois:
        dff = compute_photometry_dff(sigs[roi], ctrls[roi])
        z_doric = zscore_1d(dff)
        z_frame = resample_to_frames(t_doric, z_doric, t_frames)
        above   = np.isfinite(z_frame) & (z_frame >= z_threshold)
        intensity = local_intensity_from_flags(above, fps=fps, half_window_s=intensity_half_window_s)

        per_roi[roi] = {
            "z_frame": z_frame,
            "above_thresh_mask": above,
            "intensity": intensity,
            "z_threshold": float(z_threshold),
        }

    return {
        "meta": {
            "doric_path": doric_path,
            "fps": float(fps),
            "t_frames": t_frames,
            "signal_path": signal_path,
            "control_path": control_path,
            "rois": list(rois),
        },
        "per_roi": per_roi
    }

# ---------- wrapper to keep your loader unchanged ----------
def attach_photometry_multi(
    result,     # dict from load_mouse_data
    doric_path,
    rois=("ROI09",),        # list/tuple of ROI names
    fps=30.0,
    z_threshold=1.0,
    intensity_half_window_s=0.25,
    signal_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM1EXC2",
    control_path="DataAcquisition/BBC300/ROISignals/Series0001/CAM2EXC3",
):
    phot = align_photometry_to_frames_multi(
        result,
        doric_path=doric_path,
        rois=rois,
        fps=fps,
        z_threshold=z_threshold,
        intensity_half_window_s=intensity_half_window_s,
        signal_path=signal_path,
        control_path=control_path,
    )
    out = dict(result)
    out["photometry"] = phot
    return out

# ---------- combine multiple ROIs for a single overlay ----------
def _combine_overlay_from_rois(photometry, rois, combine_mode="max"):
    """
    photometry: result["photometry"] dict from attach_photometry_multi
    rois: str or list/tuple of ROI names
    combine_mode: 'max' or 'mean' for intensity across ROIs
    Returns (zmask_combined, intensity_combined)
    """
    if isinstance(rois, str):
        rois = [rois]
    per_roi = photometry.get("per_roi", {})
    chosen = [r for r in rois if r in per_roi]
    if not chosen:
        return None, None

    # stack masks and intensities
    masks = np.vstack([per_roi[r]["above_thresh_mask"].astype(bool) for r in chosen])
    intens = np.vstack([per_roi[r]["intensity"].astype(float) for r in chosen])

    # combined mask: any ROI crosses threshold at a frame
    zmask = np.any(masks, axis=0)

    # combined intensity: max or mean across ROIs, then re-normalize to [0,1]
    if combine_mode == "mean":
        intensity = np.nanmean(intens, axis=0)
    else:
        intensity = np.nanmax(intens, axis=0)

    peak = np.nanmax(intensity) if intensity.size else 0.0
    if peak > 0:
        intensity = intensity / peak
    else:
        intensity = np.zeros_like(intensity)

    return zmask, intensity

# ---------- your trail plot with single or multi-ROI overlay ----------
def plot_specific_data_2trails_with_overlay(
    result, scaling_factor=0.83, subtle_disp=0, x_min=None, x_max=None,
    second_trail_color='navy', no_scale=False, save_filename=None,
    overlay_rois=None,          # None, "ROI09", or ["ROI09","ROI10"]
    combine_mode="max"          # 'max' or 'mean' when overlay_rois is multiple
):
    trail_x_px_data_refined = np.array(result['corrected_trail_x'])
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Coordinates from your code
    x_snout = (np.flip(trail_x_px_data_refined[:]) * scaling_factor + result['snout_x']) / 36
    y_snout = result['snout_y'] / 37

    x_shoulder = (np.flip(trail_x_px_data_refined[:]) * scaling_factor + result['shoulder_x']) / 36
    y_shoulder = result['shoulder_y'] / 37

    x_tailbase = (np.flip(trail_x_px_data_refined[:]) * scaling_factor + result['tailbase_x']) / 36
    y_tailbase = result['tailbase_y'] / 37

    x_leftear = (np.flip(trail_x_px_data_refined[:]) * scaling_factor + result['leftear_x']) / 36
    y_leftear = result['leftear_y'] / 37

    x_rightear = (np.flip(trail_x_px_data_refined[:]) * scaling_factor + result['rightear_x']) / 36
    y_rightear = result['rightear_y'] / 37

    x_trail = (np.flip(trail_x_px_data_refined[:]) * scaling_factor) / 36 + subtle_disp
    y_trail = result['closest_trail_point_y'] / 37
    ax.plot(x_trail, y_trail, label='Corrected Dragging Sections Y', color='black', linewidth=5, alpha=0.9)

    if 'closest_trail_point_y2' in result:
        x_second_trail = (np.flip(trail_x_px_data_refined[:]) * scaling_factor) / 36 + subtle_disp
        y_second_trail = result['closest_trail_point_y2'] / 37
        ax.plot(x_second_trail, y_second_trail, label='Second Trail', color=second_trail_color, linewidth=5, alpha=0.9)

    x_points = [x_snout, x_leftear, x_tailbase, x_rightear, x_snout]
    y_points = [y_snout, y_leftear, y_tailbase, y_rightear, y_snout]
    ax.plot(x_points, y_points, label='Connecting Line', color='plum', alpha=0.25)
    ax.plot(x_points[0], y_points[0], 'darkred', marker='o', markersize=3)

    # ---- Optional activity overlay: one ROI or multiple ROIs combined ----
    if overlay_rois is not None and "photometry" in result:
        zmask, inten = _combine_overlay_from_rois(result["photometry"], overlay_rois, combine_mode=combine_mode)
        if zmask is not None and inten is not None:
            sc = ax.scatter(x_snout, y_snout, c=inten, s=18, cmap="viridis", vmin=0, vmax=1, alpha=0.9,
                            label=f"Overlay {overlay_rois} ({combine_mode})")
            hi = np.where(zmask & np.isfinite(inten))[0]
            if hi.size:
                ax.scatter(x_snout[hi], y_snout[hi], facecolors='none', edgecolors='k', s=40, linewidths=0.8,
                           label="z ≥ thr")
            cb = fig.colorbar(sc, ax=ax, pad=0.01)
            cb.set_label("Local activity density")

    ax.set_ylim(0, 35)
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(np.max(x_trail) + 50, np.min(x_trail) - 150)

    # Clean fallback if your style helpers are not in scope
    try:
        if no_scale:
            style_axis_no_axes_no_scale(ax)
        else:
            style_axis_no_axes(ax)
    except NameError:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    if save_filename:
        plt.savefig(f"{save_filename}.svg", dpi=600, bbox_inches='tight', pad_inches=0.1)

    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

# ---------- Example usage ----------
# 1) Load your trial as usual
trial_result = load_mouse_data(...)

# 2) Attach multi-ROI photometry (timestamps handle 1000 Hz automatically)
trial_with_phot = attach_photometry_multi(
    trial_result,
    doric_path=r"...\Console_Acq_0000.doric",
    rois=("ROI09","ROI10","ROI14"),
    fps=30.0,
    z_threshold=1.0
)

# 3a) Overlay a single ROI (ROI + Trail)
plot_specific_data_2trails_with_overlay(
    trial_with_phot,
    overlay_rois="ROI10"
)

# 3b) Overlay multiple ROIs, combined 
plot_specific_data_2trails_with_overlay(
    trial_with_phot,
    overlay_rois=("ROI09","ROI10"),
    combine_mode="max"     # or "mean"
)
