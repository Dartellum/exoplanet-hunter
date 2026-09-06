import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from lightkurve.periodogram import BoxLeastSquaresPeriodogram
from astropy.time import Time

def run_bls_analysis(light_curve, object_id, display_name, sde_threshold, mask_diurnal=False):
    """
    Core BLS analysis with explicit grid, diurnal alias filtering, and true SDE calculation.
    Safe for both space missions (TESS) and ground-based surveys (SuperWASP).
    """
    try:
        processed_lc = light_curve.flatten(window_length=401).remove_outliers(sigma=5)

        if len(processed_lc) == 0:
            return None

        # Optional: downsample if huge (WASP multi-season)
        if len(processed_lc) > 500_000:
            processed_lc = processed_lc.bin(time_bin_size=0.01)

        periods = np.linspace(0.5, 20.0, 30_000)
        durations = np.array([0.02, 0.05, 0.08, 0.12])

        # Setting frequency_factor=500 silences the lightkurve/astropy period point count warning
        bls_model = processed_lc.to_periodogram(method='bls', period=periods, duration=durations, frequency_factor=500)

    except Exception as bls_error:
        print(f"  !!! Error processing {display_name}: {bls_error} !!!")
        return None

    # Calculate true SDE: (power - mean) / std across all searched periods
    powers = bls_model.power.value
    mean_power = np.mean(powers)
    std_power = np.std(powers)
    if std_power <= 0:
        return None

    sde_array = (powers - mean_power) / std_power
    periods_val = bls_model.period.value

    # Diurnal alias masking for ground-based surveys:
    # Masks 1-day diurnal harmonics (sidereal 0.99727 d and solar 1.00000 d) and half-day boundary
    if mask_diurnal:
        alias_mask = ((periods_val >= 0.982) & (periods_val <= 1.018)) | (periods_val <= 0.515)
        sde_search = np.where(~alias_mask, sde_array, 0.0)
    else:
        sde_search = sde_array

    index = int(np.argmax(sde_search))
    sde_val = float(sde_search[index])

    if sde_val < sde_threshold:
        return None

    planet_period_val = float(periods_val[index])
    planet_t0_val = float(bls_model.transit_time[index].value)
    planet_depth_val = float(bls_model.depth[index].value)
    planet_duration_val = float(bls_model.duration[index].value)
    depth_ppm = planet_depth_val * 1e6

    # Physical plausibility check on transit depth:
    # Valid astrophysical transits/eclipses: 500 ppm (0.05%) to 750,000 ppm (75%)
    # Depths > 75% or negative are background noise / uncalibrated artifacts
    if depth_ppm < 500 or depth_ppm > 750_000:
        return None

    # Multi-transit and phase coverage verification
    time_vals = processed_lc.time.value if hasattr(processed_lc.time, "value") else np.array(processed_lc.time)
    phase = ((time_vals - planet_t0_val + 0.5 * planet_period_val) % planet_period_val) / planet_period_val - 0.5
    in_transit = np.abs(phase) < (planet_duration_val / (2.0 * planet_period_val))
    n_in_transit = int(np.sum(in_transit))
    if n_in_transit < 15:
        return None

    # Require transits observed across at least 3 distinct epochs
    transit_epochs = np.round((time_vals[in_transit] - planet_t0_val) / planet_period_val)
    n_transits = len(np.unique(transit_epochs))
    if n_transits < 3:
        return None

    candidate_type = "Exoplanet" if depth_ppm <= 30000 else "Eclipsing Binary"

    candidate_data = {
        "tic_id": object_id,
        "obj_name": display_name,
        "period_days": planet_period_val,
        "t0_bjd": planet_t0_val,
        "depth_ppm": depth_ppm,
        "sde": sde_val,
        "n_transits": n_transits,
        "n_in_transit": n_in_transit,
        "candidate_type": candidate_type,
        "status": "Unvetted Candidate",
    }
    return (candidate_data, processed_lc, bls_model)

# In utils.py

def save_plots(plots_subdir, candidate_data, processed_lc, bls_model):
    """
    Generates and saves standard plots for photometric (TESS/WASP) or RV candidates.
    Uses manual folding for WASP to avoid TimeDelta errors.
    """
    display_name = candidate_data['obj_name']
    clean_name = (
        display_name.replace(" ", "_")
        .replace("/", "_").replace("\\", "_")
        .replace(".", "_").replace("-", "_").strip()
    )

    os.makedirs(plots_subdir, exist_ok=True)

    # Determine if RV
    is_rv = 'rv' in candidate_data.get('obj_name', '').lower() or 'rv_amplitude' in candidate_data

    # ---------------------------------------------------------------------
    # PHOTOMETRIC (TESS / WASP)
    # ---------------------------------------------------------------------
    if not is_rv:
        plot_lc = processed_lc.copy()
        plot_lc.flux = (plot_lc.flux / np.median(plot_lc.flux) - 1.0) * 1e6  # normalize & convert to ppm

        # --- 1. Flattened LC ---
        fig_lc, ax_lc = plt.subplots(figsize=(12, 6))
        time_vals = plot_lc.time.value if hasattr(plot_lc.time, "value") else np.array(plot_lc.time)
        flux_vals = plot_lc.flux.value if hasattr(plot_lc.flux, "value") else np.array(plot_lc.flux)
        ax_lc.plot(time_vals, flux_vals, 'k.', markersize=1, alpha=0.5)
        ax_lc.set_title(f'{display_name} Light Curve')
        ax_lc.set_xlabel('Time (BJD)')
        ax_lc.set_ylabel('Normalized Flux (ppm)')
        fig_lc.savefig(os.path.join(plots_subdir, f'{clean_name}_flattened_lc.png'))
        plt.close(fig_lc)

        # --- 2. BLS Periodogram ---
        fig_bls, ax_bls = plt.subplots(figsize=(12, 6))
        powers = bls_model.power.value
        mean_p = np.mean(powers)
        std_p = np.std(powers)
        sde_curve = (powers - mean_p) / std_p if std_p > 0 else powers
        ax_bls.plot(bls_model.period.value, sde_curve, color='blue', lw=1)
        ax_bls.axvline(candidate_data['period_days'], color='red', linestyle='--', label=f"P = {candidate_data['period_days']:.4f} d (SDE {candidate_data['sde']:.1f})")
        ax_bls.set_title(f"BLS Periodogram for {display_name}")
        ax_bls.set_xlabel('Period (days)')
        ax_bls.set_ylabel('Signal Detection Efficiency (SDE)')
        ax_bls.legend(loc='upper right')
        fig_bls.savefig(os.path.join(plots_subdir, f'{clean_name}_bls_periodogram.png'))
        plt.close(fig_bls)

        # --- 3. Folded LC ---
        period = candidate_data['period_days']
        t0 = float(candidate_data['t0_bjd'])

        # Ensure time is numeric JD array, no astropy.Time object left
        time = plot_lc.time
        if hasattr(time, "jd"):          # astropy.Time
            time = time.jd
        elif hasattr(time, "value"):     # Quantity or similar
            time = time.value
        time = np.array(time, dtype=float)

        flux = np.array(plot_lc.flux)

        # Manual folding (works with pure numeric JD)
        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        sort_idx = np.argsort(phase)
        folded_time = phase[sort_idx]
        folded_flux = flux[sort_idx]

        # Simple binning
        bins = 200
        bin_means, bin_edges = np.histogram(folded_time, bins=bins, weights=folded_flux)
        bin_counts, _ = np.histogram(folded_time, bins=bins)
        binned_flux = bin_means / np.maximum(bin_counts, 1)
        binned_time = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Plot folded LC
        fig_fold, ax_fold = plt.subplots(figsize=(10, 6))
        ax_fold.plot(folded_time, folded_flux, 'k.', markersize=2, alpha=0.4)
        ax_fold.plot(binned_time, binned_flux, color='red', lw=2)
        c_type = candidate_data.get('candidate_type', 'Candidate')
        depth_pct = candidate_data['depth_ppm'] / 10000.0
        ax_fold.set_title(f"{c_type}: {display_name} (P={period:.4f} d, Depth={candidate_data['depth_ppm']:.0f} ppm / {depth_pct:.2f}%)")
        ax_fold.set_xlabel('Phase')
        ax_fold.set_ylabel('Normalized Flux (ppm)')
        if candidate_data['depth_ppm'] > 30000:
            ax_fold.set_xlim(-0.5, 0.5)
        else:
            ax_fold.set_xlim(-0.15, 0.15)
        fig_fold.savefig(os.path.join(plots_subdir, f'{clean_name}_folded_lc.png'))
        plt.close(fig_fold)


    # ---------------------------------------------------------------------
    # RV placeholder
    # ---------------------------------------------------------------------
    else:
        rv_times = candidate_data.get('rv_times', [])
        rv_values = candidate_data.get('rv_values', [])
        rv_errors = candidate_data.get('rv_errors', [])
        if len(rv_times) == 0 or len(rv_values) == 0:
            print(f"  !!! No RV data for {display_name}. Skipping RV plot.")
            return
        fig_rv, ax_rv = plt.subplots(figsize=(10, 6))
        ax_rv.errorbar(rv_times, rv_values, yerr=rv_errors, fmt='o', color='blue', ecolor='gray', alpha=0.7)
        ax_rv.set_title(f'Radial Velocity Curve for {display_name}')
        ax_rv.set_xlabel('Time (JD)')
        ax_rv.set_ylabel('RV (m/s)')
        fig_rv.savefig(os.path.join(plots_subdir, f'{clean_name}_rv_curve.png'))
        plt.close(fig_rv)
