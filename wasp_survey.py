import lightkurve as lk
import pandas as pd
import os
import shutil
import subprocess
from astropy.io import fits
import warnings
import astropy
import glob
import re
import sys
import utils
import numpy as np
from astropy.time import Time, TimeDelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
RESULTS_DIR = "wasp_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
LOCAL_WASP_DIR = "SuperWASP_data"
SDE_THRESHOLD = 8.0
WGET_SCRIPTS_DIR = "/home/wes/scripts/astronomy/SuperWASP_wget"

#==============================================================================
# --- Main WASP Survey and Worker Functions ---
#==============================================================================

def process_wget_script(wget_script_path, limit=None):
    """
    Downloads and processes data from a wget script line by line.
    A limit can be set on the number of FITS files to process.
    """
    print(f"\n--- Processing wget script: {os.path.basename(wget_script_path)} ---")
    os.makedirs(LOCAL_WASP_DIR, exist_ok=True)

    with open(wget_script_path, 'r') as f:
        lines = f.readlines()

    i = 0
    fits_processed_count = 0
    while i < len(lines):
        if limit is not None and fits_processed_count >= limit:
            print(f"Reached processing limit of {limit} files.")
            break

        line = lines[i].strip()
        i += 1
        if not line.startswith('wget') or '.fits' not in line:
            continue

        fits_processed_count += 1
        fits_line = line
        tbl_line = None
        if i < len(lines) and 'wget' in lines[i] and '_lc.tbl' in lines[i]:
            tbl_line = lines[i].strip()
            i += 1

        fits_match = re.search(r"-O '([^']+\.fits)'", fits_line)
        if not fits_match:
            continue
        fits_filename = fits_match.group(1)
        fits_filepath = os.path.join(LOCAL_WASP_DIR, fits_filename)
        wasp_id = fits_filename.replace('.fits', '')

        tbl_filename = None
        if tbl_line:
            tbl_match = re.search(r"-O '([^']+)'", tbl_line)
            if tbl_match:
                tbl_filename = tbl_match.group(1)

        processed_files = set()
        if os.path.isfile(CANDIDATES_CSV):
            try:
                processed_df = pd.read_csv(CANDIDATES_CSV)
                if 'obj_name' in processed_df.columns:
                    processed_files = set(processed_df['obj_name'].dropna().unique())
            except pd.errors.EmptyDataError:
                pass

        if wasp_id in processed_files:
            print(f"  -> Skipping {wasp_id} (already processed).")
            continue

        try:
            print(f"  -> Downloading and processing {wasp_id} ({fits_processed_count}/{limit or 'all'})...")
            subprocess.run(fits_line, shell=True, check=True, cwd=LOCAL_WASP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tbl_line:
                subprocess.run(tbl_line, shell=True, check=True, cwd=LOCAL_WASP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            result = process_local_wasp_star((fits_filepath, SDE_THRESHOLD))
            if result:
                candidate_data, processed_lc, bls_model = result

                # Save candidate data
                candidate_df = pd.DataFrame([candidate_data])
                file_exists = os.path.isfile(CANDIDATES_CSV)
                candidate_df.to_csv(CANDIDATES_CSV, mode='a', header=not file_exists, index=False)
                print(f"  -> Found candidate in {wasp_id}")

                # Save plots
                utils.save_plots(PLOTS_SUBDIR, candidate_data, processed_lc, bls_model)

        except subprocess.CalledProcessError as e:
            print(f"  !!! Download failed for {wasp_id}: {e} !!!")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Cleaning up and exiting.")
            raise
        except Exception as e:
            print(f"  !!! Processing failed for {wasp_id}: {e} !!!")
        finally:
            if os.path.exists(fits_filepath):
                os.remove(fits_filepath)
            if tbl_filename:
                tbl_filepath = os.path.join(LOCAL_WASP_DIR, tbl_filename)
                if os.path.exists(tbl_filepath):
                    os.remove(tbl_filepath)

def process_local_wasp_star(args):
    """
    Worker function to process a single WASP star from a local FITS file.
    """
    file_path, sde_threshold = args
    wasp_id = os.path.basename(file_path).replace('.fits','')

    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data

            # SuperWASP reference time is in the header, in JD.
            jd_ref = hdul[0].header['JD_REF']

            # Time for each observation is in the 'TMID' column, in seconds from JD_REF.
            time_seconds = data['TMID']

            # --- START: MODIFIED CODE ---
            #
            # Convert seconds to days and add to the reference Julian Date.
            # This creates a simple NumPy array of JD values. This is a more robust
            # way to create the time series for Lightkurve, avoiding potential
            # object type issues with Time/TimeDelta in downstream functions.
            # 86400 seconds in a day.
            time_jd = jd_ref + (time_seconds / 86400.0)

            # Extract flux and normalize
            flux = data['FLUX2']
            normalized_flux = flux / np.median(flux)

            # Create LightCurve object using the JD numpy array.
            # Lightkurve understands JD as the default time format.
            lc = lk.LightCurve(time=time_jd, flux=normalized_flux)
            #
            # --- END: MODIFIED CODE ---

        if len(lc) == 0:
            return None

        numeric_id_placeholder = int(''.join(filter(str.isdigit, wasp_id)))
        return utils.run_bls_analysis(lc, numeric_id_placeholder, wasp_id, sde_threshold)

    except Exception as e:
        print(f"  !!! Error processing {wasp_id}: {e} !!!")
        return None

def process_all_wget_scripts():
    all_wget_scripts = sorted(glob.glob(os.path.join(WGET_SCRIPTS_DIR, "*.bat")))
    if not all_wget_scripts:
        print(f"No wget scripts found in {WGET_SCRIPTS_DIR}")
    else:
        print(f"Found {len(all_wget_scripts)} wget scripts to process.")
        try:
            for script_path in all_wget_scripts:
                process_wget_script(script_path)
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Exiting.")
            sys.exit(0)


#==============================================================================
# --- WASP Sub-Menu ---
#==============================================================================
def menu():
    """Displays the WASP survey sub-menu."""
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n--- WASP Survey Menu ---")
            print("1. Process all WASP tile scripts")
            print("2. Back to Main Menu")

            choice = input("\nSelect an option: ")

            if choice == '1':
                process_all_wget_scripts()
            elif choice == '2':
                break
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)