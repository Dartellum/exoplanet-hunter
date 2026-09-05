import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy
import warnings
import os
from astroquery.mast import Catalogs # Keep Catalogs for potential future use
from astropy.utils.data import conf as data_conf
import pandas as pd

# Suppress common Astropy Units warnings and general UserWarnings for cleaner output
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning)


print(f"lightkurve version: {lk.__version__}")

# --- Configuration for the Hunt ---
RESULTS_DIR = "exoplanet_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
SDE_THRESHOLD = 8.0 # Signal Detection Efficiency threshold for a promising candidate

# Create results directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_SUBDIR, exist_ok=True)

# Set a higher timeout for astroquery data downloads (e.g., 5 minutes = 300 seconds)
data_conf.timeout = 300 # Setting a global timeout for astroquery


# --- Step 0: Directly Define the List of TIC IDs to Search ---
print("--- Using Manually Defined Target List for Search ---")
# This list is hardcoded due to persistent issues with dynamic target list generation.
# For a real hunt, you would manually prepare a list of TICs (e.g., from a downloaded catalog, or from ExoFOP).
tic_ids_to_search = [
    278892590, # TRAPPIST-1 (Known system, for pipeline validation)
    261136679, # KELT-11 (Known system, for pipeline validation)
    100100438  # WASP-18 (Known system - might have no data as seen before)
]
print(f"Prepared {len(tic_ids_to_search)} targets for search.")
print(f"First 5 targets: {tic_ids_to_search[:5]}")


candidate_results = [] # List to store promising candidate details

# --- Loop through each target star ---
for tic_id in tic_ids_to_search:
    print(f"\n--- Processing TIC ID: {tic_id} ---")

    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS',
                                              author='SPOC',
                                              exptime="short",
                                              cadence='lc')

        if len(search_result) == 0:
            print(f"No SPOC short cadence data found for TIC {tic_id}. Trying long cadence.")
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS',
                                                  author='SPOC',
                                                  exptime="long",
                                                  cadence='lc')

        if len(search_result) == 0:
            print(f"No TESS SPOC light curve data found for TIC {tic_id}. Skipping.")
            plt.close('all')
            continue
        else:
            print(f"Found {len(search_result)} SPOC light curve files for TIC {tic_id}.")
            print(search_result) # Display search results for the current star

        print("Downloading and stitching light curves...")
        lc_collection = search_result.download_all()

        if isinstance(lc_collection, lk.LightCurve):
            lc_collection = lk.LightCurveCollection([lc_collection])

        stitched_time_values = np.array([])
        stitched_flux_values = np.array([])
        stitched_flux_err_values = np.array([])
        stitched_cadenceno_values = np.array([])
        stitched_quality_values = np.array([])

        for lc in lc_collection:
            cleaned_lc = lc.remove_nans()
            stitched_time_values = np.append(stitched_time_values, cleaned_lc.time.value)
            stitched_flux_values = np.append(stitched_flux_values, cleaned_lc.flux.value)
            stitched_flux_err_values = np.append(stitched_flux_err_values, cleaned_lc.flux_err.value)
            stitched_cadenceno_values = np.append(stitched_cadenceno_values, cleaned_lc.cadenceno.value)
            stitched_quality_values = np.append(stitched_quality_values, cleaned_lc.quality.value)


        median_flux_value = np.nanmedian(stitched_flux_values)
        if median_flux_value != 0:
            normalized_flux_values = stitched_flux_values / median_flux_value
            normalized_flux_err_values = stitched_flux_err_values / median_flux_value
        else:
            print(f"Warning: Median flux is zero for TIC {tic_id}. Cannot normalize light curve. Skipping.")
            plt.close('all')
            continue

        stitched_lc = lk.LightCurve(time=astropy.time.Time(stitched_time_values, format='jd'),
                                     flux=normalized_flux_values * u.dimensionless_unscaled,
                                     flux_err=normalized_flux_err_values * u.dimensionless_unscaled,
                                     cadenceno=stitched_cadenceno_values,
                                     quality=stitched_quality_values,
                                     targetid=tic_id)


        stitched_lc = stitched_lc.flatten(window_length=401).remove_outliers(sigma=5)

        # Store flux as Quantity, but use its .value for calculations and plotting
        stitched_lc.flux = (stitched_lc.flux.value - 1.0) * 1e6 * u.dimensionless_unscaled
        stitched_lc.flux_err = stitched_lc.flux_err.value * 1e6 * u.dimensionless_unscaled

        # --- Diagnostics: Print properties (using float() and .value for robust printing) ---
        print("\n--- Diagnostic: stitched_lc properties before BLS ---")
        print(f"Number of data points: {len(stitched_lc.flux)}")
        # Use .value and convert to float for robust printing
        print(f"Flux min: {float(np.nanmin(stitched_lc.flux.value)):.2f} (ppm)")
        print(f"Flux max: {float(np.nanmax(stitched_lc.flux.value)):.2f} (ppm)")
        print(f"Flux median: {float(np.nanmedian(stitched_lc.flux.value)):.2f} (ppm)")
        print(f"Contains NaNs in flux: {np.any(np.isnan(stitched_lc.flux))}")
        print(f"Contains Infs in flux: {np.any(np.isinf(stitched_lc.flux))}")
        print(f"--- End Diagnostic ---")

        # --- Plotting Light Curve (using matplotlib directly for robustness) ---
        print("Plotting the stitched light curve...")
        fig_lc, ax_lc = plt.subplots(figsize=(12, 6))
        # Plot using .value for flux, .value for time
        ax_lc.plot(stitched_lc.time.value, stitched_lc.flux.value, linewidth=0.5, alpha=0.8, color='blue', label='Flattened Light Curve')
        ax_lc.set_title(f'TIC {tic_id} TESS Light Curve (Flattened)')
        ax_lc.set_xlabel('Time (BJD - 2457000)')
        ax_lc.set_ylabel('Normalized Flux (ppm)')
        plt.tight_layout()


        # --- Perform BLS transit search ---
        print("\nPerforming BLS transit search...")
        min_period_search = 0.5
        max_period_search = 20.0
        durations = np.linspace(0.01, 0.1, 10)

        bls_model = stitched_lc.to_periodogram(method='bls',
                                                minimum_period=min_period_search,
                                                maximum_period=max_period_search,
                                                duration=durations,
                                                frequency_factor=2000.0)

        # --- Find the strongest transit signal ---
        # Find the index of the single highest power peak in the periodogram
        index = np.argmax(bls_model.power)

        # Extract the parameters for that single best signal
        planet_period_val = float(bls_model.period[index].value)
        planet_t0_val = float(bls_model.transit_time[index].value)
        planet_depth_val = float(bls_model.depth[index].value)
        sde_val = float(bls_model.power[index].value)

        # Print with values directly, not Quantity objects
        print(f"Strongest transit candidate found for TIC {tic_id}:")
        print(f"  Period: {planet_period_val:.4f} d")
        print(f"  Transit Time (t0): {planet_t0_val:.4f} BJD")
        print(f"  Transit Depth: {planet_depth_val:.2e} ppm")
        print(f"  Signal Detection Efficiency (SDE): {sde_val:.2f}")

        # --- Plot the BLS periodogram (explicitly using matplotlib) ---
        fig_bls, ax_bls = plt.subplots(figsize=(12, 6))
        ax_bls.plot(bls_model.period.value, bls_model.power.value, linewidth=0.5, alpha=0.8, color='blue')
        ax_bls.axvline(planet_period_val, color='red', linestyle='--', label='Detected Period')
        ax_bls.set_title(f"BLS Periodogram for TIC {tic_id} (Strongest: {planet_period_val:.4f} d)")
        ax_bls.set_xlabel('Period (days)')
        ax_bls.set_ylabel('Power (SDE)')
        ax_bls.legend()
        plt.tight_layout()

        # --- Fold the light curve (explicitly using matplotlib) ---
        # Note: folded_lc object itself is still an astropy.units.Quantity internally
        folded_lc = stitched_lc.fold(period=bls_model.period_at_max_power, epoch_time=bls_model.transit_time_at_max_power)

        fig_fold, ax_fold = plt.subplots(figsize=(10, 6))
        ax_fold.scatter(folded_lc.time.value, folded_lc.flux.value, s=5, alpha=0.5, label='Folded Data', color='blue')

        # Create a temporary LightCurve with phase as numerical time for binning
        temp_lc_for_binning = lk.LightCurve(time=folded_lc.time.value, flux=folded_lc.flux.value, flux_err=folded_lc.flux_err.value)
        binned_folded_lc = temp_lc_for_binning.bin(bins=200)
        ax_fold.plot(binned_folded_lc.time.value, binned_folded_lc.flux.value, color='red', linewidth=2, label='Binned Data')

        ax_fold.set_title(f"TIC {tic_id} Light Curve Folded at P={planet_period_val:.4f} d")
        ax_fold.set_xlabel('Phase')
        ax_fold.set_ylabel('Normalized Flux (ppm)')
        ax_fold.set_xlim(-0.1, 0.1)
        ax_fold.legend()
        plt.tight_layout()

        # --- Save results if it's a promising candidate ---
        if sde_val >= SDE_THRESHOLD:
            print(f"  >>> Candidate for TIC {tic_id} meets SDE threshold! Saving plots and data. <<<")
            # Filename for plots will be TICID_flattened_lc.png etc.
            fig_lc.savefig(os.path.join(PLOTS_SUBDIR, f'TIC{tic_id}_flattened_lc.png'))
            fig_bls.savefig(os.path.join(PLOTS_SUBDIR, f'TIC{tic_id}_bls_periodogram.png'))
            fig_fold.savefig(os.path.join(PLOTS_SUBDIR, f'TIC{tic_id}_folded_lc.png'))

            candidate_results.append({
                'tic_id': tic_id,
                'obj_name': 'N/A', # Common name not extracted in this version
                'period_days': planet_period_val,
                't0_bjd': planet_t0_val,
                'depth_ppm': planet_depth_val,
                'sde': sde_val,
                'status': 'Unvetted Candidate'
            })
        else:
            print(f"  Candidate for TIC {tic_id} did NOT meet SDE threshold ({sde_val:.2f} < {SDE_THRESHOLD:.1f}).")

        plt.close('all')

        print(f"\nBLS search and folded light curve plot complete for TIC {tic_id}.")

    except Exception as e:
        print(f"An error occurred while processing TIC {tic_id}: {e}")
        print(f"Skipping TIC {tic_id} and moving to the next target.")
        plt.close('all')


# --- After loop: Save all candidate results to CSV ---
if candidate_results:
    results_df = pd.DataFrame(candidate_results)
    results_df.to_csv(CANDIDATES_CSV, index=False)
    print(f"\n--- Saved {len(candidate_results)} promising candidates to {CANDIDATES_CSV} ---")
else:
    print("\n--- No promising candidates found based on the SDE threshold. ---")

print("\n--- All specified targets processed. ---")