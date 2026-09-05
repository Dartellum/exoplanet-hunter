import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy
import warnings
import os
import datetime
import pandas as pd
import sqlite3 # MODIFICATION: Import sqlite3
from multiprocessing import Pool

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="the tpfmodel submodule is not available*")
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- Configuration ---
RESULTS_DIR = "exoplanet_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
FAILED_PATCHES_LOG = os.path.join(RESULTS_DIR, "failed_patches.log")
DATABASE_FILE = "tic_catalog.db" # MODIFICATION: Path to the new database
SDE_THRESHOLD = 8.0
NUM_PROCESSES = 8 # Changed back to 8 due to memory usage. Will change back to 16 when memory increased to 192GB
SKY_SURVEY_STEP_DEG = 4.0
SKY_PATCH_RADIUS_DEG = SKY_SURVEY_STEP_DEG / 2.0

#==============================================================================
# --- Worker Function for a Single Star (no changes) ---
#==============================================================================
def process_star(args):
    star_info, sde_threshold, plots_subdir = args
    tic_id, obj_name, _ = star_info
    display_name = obj_name if obj_name else f"TIC {tic_id}"
    clean_name = display_name.replace(" ", "_").replace("/", "_").replace("\\", "_").strip()

    print(f"  -> Starting analysis for TIC {tic_id}")
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC', exptime='short')
        if not search_result: search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC', exptime='long')
        if not search_result: return None
        lc_collection = search_result.download_all(download_dir=os.path.join(RESULTS_DIR, "cache"))
        stitched_lc = lc_collection.stitch().remove_nans()
        if len(stitched_lc) == 0: return None
        processed_lc = stitched_lc.flatten(window_length=401).remove_outliers(sigma=5)
        try:
            n_points = len(processed_lc)
            if n_points > 400000: bls_frequency_factor = 25.0
            elif n_points > 150000: bls_frequency_factor = 15.0
            else: bls_frequency_factor = 5.0
            bls_model = processed_lc.to_periodogram(method='bls', minimum_period=0.5, maximum_period=20.0, duration=np.linspace(0.01, 0.1, 10), frequency_factor=bls_frequency_factor)
        except Exception as bls_error:
            if "Periodogram is too large" in str(bls_error):
                print(f"  BLS failed for TIC {tic_id}, retrying with emergency factor..."); bls_model = processed_lc.to_periodogram(method='bls', minimum_period=0.5, maximum_period=20.0, duration=np.linspace(0.01, 0.1, 10), frequency_factor=150.0)
            else: raise bls_error
        index = np.argmax(bls_model.power); planet_period_val = float(bls_model.period[index].value); planet_t0_val = float(bls_model.transit_time[index].value)
        planet_depth_val = float(bls_model.depth[index].value); sde_val = float(bls_model.power[index].value)
        if sde_val >= sde_threshold:
            print(f"  >>> FOUND CANDIDATE in TIC {tic_id} with SDE={sde_val:.2f}! Saving plots... <<<")
            candidate_data = {'tic_id': tic_id, 'obj_name': obj_name, 'period_days': planet_period_val, 't0_bjd': planet_t0_val, 'depth_ppm': planet_depth_val * 1e6, 'sde': sde_val, 'status': 'Unvetted Candidate'}
            plot_lc = processed_lc.copy(); plot_lc.flux = (plot_lc.flux - 1.0) * 1e6
            fig_lc, ax_lc = plt.subplots(figsize=(12, 6)); ax_lc.plot(plot_lc.time.value, plot_lc.flux.value, 'k.', markersize=1, alpha=0.5); ax_lc.set_title(f'TIC {tic_id} ({display_name}) TESS Light Curve'); ax_lc.set_xlabel('Time (BJD)'); ax_lc.set_ylabel('Normalized Flux (ppm)'); fig_lc.savefig(os.path.join(plots_subdir, f'{clean_name}_flattened_lc.png')); plt.close(fig_lc)
            fig_bls, ax_bls = plt.subplots(figsize=(12, 6)); ax_bls.plot(bls_model.period.value, bls_model.power.value, color='blue'); ax_bls.axvline(candidate_data['period_days'], color='red', linestyle='--'); ax_bls.set_title(f"BLS Periodogram for TIC {tic_id}"); ax_bls.set_xlabel('Period (days)'); ax_bls.set_ylabel('Power (SDE)'); fig_bls.savefig(os.path.join(plots_subdir, f'{clean_name}_bls_periodogram.png')); plt.close(fig_bls)
            folded_lc = plot_lc.fold(period=candidate_data['period_days'], epoch_time=candidate_data['t0_bjd']); fig_fold, ax_fold = plt.subplots(figsize=(10, 6)); ax_fold.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', markersize=2, alpha=0.5); binned_folded_lc = folded_lc.bin(bins=200); ax_fold.plot(binned_folded_lc.time.value, binned_folded_lc.flux.value, color='red', lw=2); ax_fold.set_title(f"Folded at P={candidate_data['period_days']:.4f} d"); ax_fold.set_xlabel('Phase'); ax_fold.set_ylabel('Normalized Flux (ppm)'); ax_fold.set_xlim(-0.1, 0.1); fig_fold.savefig(os.path.join(plots_subdir, f'{clean_name}_folded_lc.png')); plt.close(fig_fold)
            return candidate_data
        return None
    except Exception as e:
        print(f"  !!! Error processing TIC {tic_id}: {e} !!!"); return None

#==============================================================================
# --- Main Execution Block ---
#==============================================================================
if __name__ == '__main__':
    print(f"lightkurve version: {lk.__version__}")

    # Pre-computation
    try:
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        toi_df = pd.read_csv(url); confirmed_df = toi_df[toi_df['TFOPWG Disposition'] == 'CP']; known_planet_tids = confirmed_df['TIC ID'].dropna().unique()
        print(f"Successfully downloaded {len(known_planet_tids)} confirmed exoplanet host TICs.")
    except Exception as e:
        print(f"Could not download TOI catalog. Will proceed without filtering known hosts. Error: {e}"); known_planet_tids = []

    # Load previously processed TICs
    processed_tics = set()
    if os.path.isfile(CANDIDATES_CSV):
        print(f"\nFound existing candidates file. Loading processed TICs..."); processed_df = pd.read_csv(CANDIDATES_CSV); processed_tics = set(processed_df['tic_id'].unique()); print(f"Loaded {len(processed_tics)} already processed TIC IDs.")

    # Setup
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PLOTS_SUBDIR, exist_ok=True); os.makedirs(os.path.join(RESULTS_DIR, "cache"), exist_ok=True)
    total_candidates_found_this_run = 0

    if not os.path.isfile(DATABASE_FILE):
        print(f"Error: Database file '{DATABASE_FILE}' not found. Please run the `build_tic_database.py` script first."); exit()

    # All-Sky Loop
    print("\n--- Starting Full All-Sky Survey using LOCAL DATABASE ---")
    for dec_center in np.arange(-90.0, 90.0 + SKY_SURVEY_STEP_DEG, SKY_SURVEY_STEP_DEG):
        for ra_center in np.arange(0.0, 360.0, SKY_SURVEY_STEP_DEG):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n====================================================================="); print(f"--- Processing Sky Patch: RA={ra_center}, Dec={dec_center} ---"); print(f"--- Timestamp: {timestamp} ---"); print(f"=====================================================================")

            # --- MODIFICATION: Target generation is now a fast database query ---
            try:
                ra_min, ra_max = ra_center - SKY_PATCH_RADIUS_DEG, ra_center + SKY_PATCH_RADIUS_DEG
                dec_min, dec_max = dec_center - SKY_PATCH_RADIUS_DEG, dec_center + SKY_PATCH_RADIUS_DEG
                tmag_min, tmag_max = 8, 13

                # Connect to the database and perform the query
                conn = sqlite3.connect(DATABASE_FILE)
                query = f"SELECT ID, RA, Dec, Tmag FROM stars WHERE RA >= {ra_min} AND RA < {ra_max} AND Dec >= {dec_min} AND Dec < {dec_max} AND Tmag >= {tmag_min} AND Tmag <= {tmag_max}"

                print("Querying local database...")
                patch_df = pd.read_sql_query(query, conn)
                conn.close()
                print(f"Found {len(patch_df)} stars in patch from database.")

                if len(patch_df) == 0: continue

                candidate_df = patch_df[~patch_df['ID'].isin(known_planet_tids)]
                if len(candidate_df) > 50: candidate_df = candidate_df.head(50)

                tic_ids_and_names_to_search = [[row['ID'], '', ''] for _, row in candidate_df.iterrows()]
                tic_ids_and_names_to_search = [star for star in tic_ids_and_names_to_search if star[0] not in processed_tics]

                if not tic_ids_and_names_to_search: print("No new targets to process in this patch."); continue
                print(f"Prepared {len(tic_ids_and_names_to_search)} new targets for parallel processing.")
            except Exception as e:
                print(f"An error occurred while generating the target list from the database: {e}"); continue

            # Parallel Processing Block
            tasks = [(star_info, SDE_THRESHOLD, PLOTS_SUBDIR) for star_info in tic_ids_and_names_to_search]
            with Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(process_star, tasks)

            # Process results in the main thread
            new_finds_in_patch = 0
            for result in results:
                if result is not None:
                    candidate_data = result; new_finds_in_patch += 1
                    candidate_df = pd.DataFrame([candidate_data]); file_exists = os.path.isfile(CANDIDATES_CSV)
                    candidate_df.to_csv(CANDIDATES_CSV, mode='a', header=not file_exists, index=False)
                    processed_tics.add(candidate_data['tic_id'])
            if new_finds_in_patch > 0:
                total_candidates_found_this_run += new_finds_in_patch
                print(f"\n>>> Found and saved {new_finds_in_patch} new candidates in this patch! <<<")

    # Final summary message
    timestamp_final = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n====================================================================="); print("--- All-Sky Survey Complete ---"); print(f"--- Time completed: {timestamp_final} ---"); print(f"Found a total of {total_candidates_found_this_run} new promising candidates during this run."); print(f"All candidates have been saved to {CANDIDATES_CSV}"); print("--- Script Finished ---")