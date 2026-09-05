import lightkurve as lk
import pandas as pd
import numpy as np
import os
import datetime
import sqlite3
import re
from multiprocessing import Pool
import subprocess
import time
import warnings
import astropy
import utils # MODIFICATION: Import our new utils module

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="the tpfmodel submodule is not available*")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- Configuration ---
RESULTS_DIR = "exoplanet_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
TIC_CATALOG_DIR = "TIC_CATALOG"
DATABASE_FILE = "tic_catalog.db"
SDE_THRESHOLD = 8.0
NUM_PROCESSES = 16
SKY_SURVEY_STEP_DEG = 4.0
SKY_PATCH_RADIUS_DEG = SKY_SURVEY_STEP_DEG / 2.0

#==============================================================================
# --- TESS Sub-Menu ---
#==============================================================================
def menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n--- TESS Survey Menu ---")
        print("1. Download TESS Input Catalog (TIC) files")
        print("2. Build Local TIC Database from CSVs")
        print("3. Run All-Sky Survey using Local Database")
        print("4. Back to Main Menu")
        choice = input("\nSelect an option: ")
        if choice == '1': download_tic_data()
        elif choice == '2': build_database()
        elif choice == '3': run_survey()
        elif choice == '4': break

#==============================================================================
# --- Data Preparation Functions ---
#==============================================================================
def download_tic_data():
    print("\n--- Downloading and Unzipping TESS Input Catalog ---")
    try:
        print("Running download script..."); subprocess.run(['bash', 'download_tic.sh'], check=True)
        print("\nDownload finished. Running unzip script..."); subprocess.run(['bash', 'unzip_tic.sh'], check=True)
        print("\nData preparation complete.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    input("Press Enter to return to the menu...")

def build_database():
    if not os.path.isdir(TIC_CATALOG_DIR): print(f"Error: '{TIC_CATALOG_DIR}' not found."); return
    all_files = [os.path.join(TIC_CATALOG_DIR, f) for f in os.listdir(TIC_CATALOG_DIR) if f.endswith('.csv')]
    if not all_files: print(f"Error: No .csv files found in '{TIC_CATALOG_DIR}'."); return
    if os.path.exists(DATABASE_FILE): os.remove(DATABASE_FILE)
    conn = sqlite3.connect(DATABASE_FILE); print(f"Created new database: {DATABASE_FILE}"); start_time = time.time()
    for i, f in enumerate(all_files):
        print(f"  Processing file {i+1}/{len(all_files)}: {os.path.basename(f)}...")
        chunk_iter = pd.read_csv(f, header=None, chunksize=1_000_000, low_memory=False)
        for chunk in chunk_iter:
            if chunk.shape[1] < 8: continue
            required_cols = chunk[[0, 5, 6, 7]]; required_cols.columns = ['ID', 'RA', 'Dec', 'Tmag']
            required_cols['ID'] = pd.to_numeric(required_cols['ID'], errors='coerce')
            required_cols.dropna(subset=['ID'], inplace=True); required_cols['ID'] = required_cols['ID'].astype(np.int64)
            required_cols.to_sql('stars', conn, if_exists='append', index=False)
    print("\nCreating indexes..."); cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_ra ON stars (RA)"); cursor.execute("CREATE INDEX idx_dec ON stars (Dec)"); cursor.execute("CREATE INDEX idx_id ON stars (ID)")
    conn.commit(); print("Indexes created."); conn.close(); end_time = time.time()
    print(f"\n--- Database Build Complete in {(end_time - start_time) / 60:.2f} minutes. ---")
    input("Press Enter to return to the menu...")

#==============================================================================
# --- Main Survey and Worker Functions ---
#==============================================================================
def run_survey():
    try:
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        toi_df = pd.read_csv(url); confirmed_df = toi_df[toi_df['TFOPWG Disposition'] == 'CP']; known_planet_tids = confirmed_df['TIC ID'].dropna().unique()
        print(f"Successfully downloaded {len(known_planet_tids)} confirmed exoplanet host TICs.")
    except Exception as e:
        print(f"Could not download TOI catalog. Error: {e}"); known_planet_tids = []
    processed_tics = set()
    if os.path.isfile(CANDIDATES_CSV):
        print(f"\nLoading processed TICs..."); processed_df = pd.read_csv(CANDIDATES_CSV); processed_tics = set(processed_df['tic_id'].unique()); print(f"Loaded {len(processed_tics)} IDs.")
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PLOTS_SUBDIR, exist_ok=True); os.makedirs(os.path.join(RESULTS_DIR, "cache"), exist_ok=True)
    total_candidates_found_this_run = 0
    if not os.path.isfile(DATABASE_FILE):
        print(f"Error: Database file '{DATABASE_FILE}' not found."); input("Press Enter..."); return
    print("\n--- Starting TESS All-Sky Survey ---")
    for dec_center in np.arange(-90.0, 90.0 + SKY_SURVEY_STEP_DEG, SKY_SURVEY_STEP_DEG):
        for ra_center in np.arange(0.0, 360.0, SKY_SURVEY_STEP_DEG):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n====================================================================="); print(f"--- Processing Sky Patch: RA={ra_center}, Dec={dec_center} --- {timestamp} ---"); print(f"=====================================================================")
            try:
                if -20 <= dec_center <= 20: tmag_range = [11, 13]
                elif -40 <= dec_center <= 40: tmag_range = [10, 13]
                else: tmag_range = [8, 13]
                ra_min, ra_max = ra_center - SKY_PATCH_RADIUS_DEG, ra_center + SKY_PATCH_RADIUS_DEG
                dec_min, dec_max = dec_center - SKY_PATCH_RADIUS_DEG, dec_center + SKY_PATCH_RADIUS_DEG
                conn = sqlite3.connect(DATABASE_FILE)
                query = f"SELECT ID, RA, Dec, Tmag FROM stars WHERE RA >= {ra_min} AND RA < {ra_max} AND Dec >= {dec_min} AND Dec < {dec_max} AND Tmag >= {tmag_range[0]} AND Tmag <= {tmag_range[1]}"
                patch_df = pd.read_sql_query(query, conn); conn.close()
                if len(patch_df) == 0: continue
                candidate_df = patch_df[~patch_df['ID'].isin(known_planet_tids)]
                if len(candidate_df) > 50: candidate_df = candidate_df.head(50)
                tic_ids_and_names_to_search = [[row['ID'], '', ''] for _, row in candidate_df.iterrows()]
                tic_ids_and_names_to_search = [star for star in tic_ids_and_names_to_search if star[0] not in processed_tics]
                if not tic_ids_and_names_to_search: print("No new targets to process in this patch."); continue
                print(f"Prepared {len(tic_ids_and_names_to_search)} new targets for parallel processing.")
            except Exception as e:
                print(f"An error occurred while generating the target list: {e}"); continue
            tasks = [(star_info, SDE_THRESHOLD) for star_info in tic_ids_and_names_to_search]
            with Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(process_tess_star, tasks)
            new_finds_in_patch = 0
            for result in results:
                if result is not None:
                    candidate_data, processed_lc, bls_model = result; new_finds_in_patch += 1
                    utils.save_plots(PLOTS_SUBDIR, candidate_data, processed_lc, bls_model)
                    candidate_df = pd.DataFrame([candidate_data]); file_exists = os.path.isfile(CANDIDATES_CSV)
                    candidate_df.to_csv(CANDIDATES_CSV, mode='a', header=not file_exists, index=False)
                    processed_tics.add(candidate_data['tic_id'])
            if new_finds_in_patch > 0:
                total_candidates_found_this_run += new_finds_in_patch
                print(f"\n>>> Found and saved {new_finds_in_patch} new candidates in this patch! <<<")
    timestamp_final = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n====================================================================="); print(f"--- TESS Survey Complete --- {timestamp_final} ---"); print(f"Found {total_candidates_found_this_run} new candidates."); print(f"Results saved to {CANDIDATES_CSV}"); print("--- Script Finished ---")
    input("Press Enter to return to the menu...")

def process_tess_star(args):
    """Worker function to process a single TESS star."""
    star_info, sde_threshold = args
    tic_id, obj_name, _ = star_info
    display_name = obj_name if obj_name else f"TIC {tic_id}"
    print(f"  -> Starting TESS analysis for TIC {tic_id}")
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')
        if not search_result: return None
        lc_collection = search_result.download_all(download_dir=os.path.join(RESULTS_DIR, "cache"))
        stitched_lc = lc_collection.stitch().remove_nans()
        if len(stitched_lc) == 0: return None
        # --- MODIFICATION: Call the generic analysis function ---
        return utils.run_bls_analysis(stitched_lc, tic_id, display_name, sde_threshold)
    except Exception as e:
        print(f"  !!! Error processing TIC {tic_id}: {e} !!!"); return None
