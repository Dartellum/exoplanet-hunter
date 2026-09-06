import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from lightkurve.periodogram import BoxLeastSquaresPeriodogram
from astropy.time import Time

def run_bls_analysis(light_curve, object_id, display_name, sde_threshold):
    """
    Core BLS analysis with explicit grid, safe for WASP & TESS.
    """
    try:
        processed_lc = light_curve.flatten(window_length=401).remove_outliers(sigma=5)

        if len(processed_lc) == 0:
            print(f"  !!! No data remaining for {display_name} after cleaning. Skipping.")
            return None

        # Optional: downsample if huge (WASP multi-season)
        if len(processed_lc) > 500_000:
            print(f"  -> Downsampling {display_name} to reduce grid size...")
            processed_lc = processed_lc.bin(time_bin_size=0.01)

        periods = np.linspace(0.5, 20.0, 30_000)
        durations = np.array([0.02, 0.05, 0.08, 0.12])

        bls_model = processed_lc.to_periodogram(method='bls', period=periods, duration=durations)

    except Exception as bls_error:
        print(f"  !!! Error processing {display_name}: {bls_error} !!!")
        return None

    # Extract strongest candidate
    index = np.argmax(bls_model.power)
    planet_period_val = float(bls_model.period[index].value)
    planet_t0_val = float(bls_model.transit_time[index].value)
    planet_depth_val = float(bls_model.depth[index].value)
    sde_val = float(bls_model.power[index].value)

    if sde_val >= sde_threshold:
        print("\n" + "=" * 78)
        print(f"\033[1;42;30m 🌟 EXOPLANET / BINARY CANDIDATE DETECTED! 🌟 \033[0m")
        print(f"\033[1;33m  -> Star:   {display_name}\033[0m")
        print(f"\033[1;32m  -> SDE:    {sde_val:.2f} (Signal Detection Efficiency)\033[0m")
        print(f"\033[1;36m  -> Period: {planet_period_val:.5f} days | Epoch: {planet_t0_val:.4f} BJD\033[0m")
        print(f"\033[1;35m  -> Depth:  {planet_depth_val * 1e6:.0f} ppm\033[0m")
        print("=" * 78 + "\n")
        candidate_data = {
            "tic_id": object_id,
            "obj_name": display_name,
            "period_days": planet_period_val,
            "t0_bjd": planet_t0_val,
            "depth_ppm": planet_depth_val * 1e6,
            "sde": sde_val,
            "status": "Unvetted Candidate",
        }
        return (candidate_data, processed_lc, bls_model)

    return None

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
        ax_bls.plot(bls_model.period.value, bls_model.power.value, color='blue')
        ax_bls.axvline(candidate_data['period_days'], color='red', linestyle='--')
        ax_bls.set_title(f"BLS Periodogram for {display_name}")
        ax_bls.set_xlabel('Period (days)')
        ax_bls.set_ylabel('Power (SDE)')
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
        ax_fold.set_title(f"Folded at P={period:.4f} d")
        ax_fold.set_xlabel('Phase')
        ax_fold.set_ylabel('Normalized Flux (ppm)')
        ax_fold.set_xlim(-0.1, 0.1)
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
