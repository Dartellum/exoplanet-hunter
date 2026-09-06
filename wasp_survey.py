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
import itertools
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
RESULTS_DIR = "wasp_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
COMPLETED_TILES_FILE = os.path.join(RESULTS_DIR, "completed_tiles.txt")
PROCESSED_STARS_FILE = os.path.join(RESULTS_DIR, "processed_stars.log")
LOCAL_WASP_DIR = "SuperWASP_data"
SDE_THRESHOLD = 8.0
NUM_WORKERS = 8
WGET_SCRIPTS_DIR = "/home/wes/scripts/astronomy/SuperWASP_wget"

#==============================================================================
# --- Main WASP Survey and Worker Functions ---
#==============================================================================

def apply_system_protections():
    """
    Guarantees Technitium DNS and Jellyfin run with zero performance impact:
    1. Sets CPU nice level to 19 (lowest CPU priority).
    2. Pins WASP process to cores 8-31 (reserving cores 0-7 exclusively for DNS/Jellyfin/OS).
    3. Sets disk I/O scheduling to Idle priority (ionice class 3).
    """
    try:
        os.nice(19)
    except Exception:
        pass

    try:
        total_cores = os.cpu_count() or 32
        if total_cores > 8:
            safe_cores = set(range(8, total_cores))
            os.sched_setaffinity(0, safe_cores)
    except Exception:
        pass

    try:
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

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

            # Convert seconds to days and add to the reference Julian Date.
            time_jd = jd_ref + (time_seconds / 86400.0)

            # Extract flux and normalize
            flux = data['FLUX2']
            normalized_flux = flux / np.median(flux)

            # Create LightCurve object using the JD numpy array.
            lc = lk.LightCurve(time=time_jd, flux=normalized_flux)

        if len(lc) == 0:
            return None

        numeric_id_placeholder = int(''.join(filter(str.isdigit, wasp_id)))
        return utils.run_bls_analysis(lc, numeric_id_placeholder, wasp_id, sde_threshold)

    except Exception as e:
        print(f"  !!! Error processing {wasp_id}: {e} !!!")
        return None

def process_single_star_worker(task):
    """
    Worker function to download and analyze a single star.
    Runs inside the parallel worker pool with system protections active.
    """
    fits_line, tbl_line, wasp_id, fits_filename, tbl_filename, sde_threshold = task
    fits_filepath = os.path.join(LOCAL_WASP_DIR, fits_filename)
    tbl_filepath = os.path.join(LOCAL_WASP_DIR, tbl_filename) if tbl_filename else None

    candidate_result = None
    err_msg = None

    try:
        subprocess.run(fits_line, shell=True, check=True, cwd=LOCAL_WASP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if tbl_line:
            subprocess.run(tbl_line, shell=True, check=True, cwd=LOCAL_WASP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        candidate_result = process_local_wasp_star((fits_filepath, sde_threshold))
    except Exception as e:
        err_msg = str(e)
    finally:
        if os.path.exists(fits_filepath):
            try:
                os.remove(fits_filepath)
            except Exception:
                pass
        if tbl_filepath and os.path.exists(tbl_filepath):
            try:
                os.remove(tbl_filepath)
            except Exception:
                pass

    return (wasp_id, candidate_result, err_msg)

def process_wget_script(wget_script_path, processed_stars=None, limit=None, num_workers=NUM_WORKERS):
    """
    Downloads and processes data from a wget script using parallel CPU workers.
    Uses a sliding window queue to keep disk and memory footprints minimal.
    """
    script_name = os.path.basename(wget_script_path)
    print(f"\n--- Processing wget script: {script_name} ---")
    os.makedirs(LOCAL_WASP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if processed_stars is None:
        processed_stars = set()

    with open(wget_script_path, 'r') as f:
        lines = f.readlines()

    # Parse all star tasks from the script
    tasks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line.startswith('wget') or '.fits' not in line:
            continue

        fits_line = line
        tbl_line = None
        if i < len(lines) and 'wget' in lines[i] and '_lc.tbl' in lines[i]:
            tbl_line = lines[i].strip()
            i += 1

        fits_match = re.search(r"-O '([^']+\.fits)'", fits_line)
        if not fits_match:
            continue
        fits_filename = fits_match.group(1)
        wasp_id = fits_filename.replace('.fits', '')

        tbl_filename = None
        if tbl_line:
            tbl_match = re.search(r"-O '([^']+)'", tbl_line)
            if tbl_match:
                tbl_filename = tbl_match.group(1)

        if wasp_id in processed_stars:
            continue

        tasks.append((fits_line, tbl_line, wasp_id, fits_filename, tbl_filename, SDE_THRESHOLD))

    if not tasks:
        print(f"All stars in {script_name} already processed.")
        return True

    if limit is not None:
        tasks = tasks[:limit]

    print(f"  -> {len(tasks)} unprocessed stars in {script_name}. Processing with {num_workers} parallel workers on cores 8-31...")

    fits_processed_count = 0
    max_queue = num_workers * 2

    try:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=apply_system_protections) as executor:
            futures = {}
            task_iter = iter(tasks)

            # Prime the queue
            for star_task in itertools.islice(task_iter, max_queue):
                fut = executor.submit(process_single_star_worker, star_task)
                futures[fut] = star_task[2]

            while futures:
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    wasp_id = futures.pop(fut)
                    fits_processed_count += 1
                    try:
                        wid, cand_result, err = fut.result()
                        if err:
                            print(f"  !!! Download/analysis failed for {wid}: {err} !!!")
                        elif cand_result:
                            candidate_data, processed_lc, bls_model = cand_result
                            candidate_df = pd.DataFrame([candidate_data])
                            file_exists = os.path.isfile(CANDIDATES_CSV)
                            candidate_df.to_csv(CANDIDATES_CSV, mode='a', header=not file_exists, index=False)
                            print("\n" + "=" * 78)
                            print(f"\033[1;42;30m 🌟 EXOPLANET / BINARY CANDIDATE DETECTED! 🌟 \033[0m")
                            print(f"\033[1;33m  -> Star:   {wid} ({fits_processed_count}/{len(tasks)})\033[0m")
                            print(f"\033[1;32m  -> SDE:    {candidate_data['sde']:.2f} (Signal Detection Efficiency)\033[0m")
                            print(f"\033[1;36m  -> Period: {candidate_data['period_days']:.5f} days | Epoch: {candidate_data['t0_bjd']:.4f} BJD\033[0m")
                            print(f"\033[1;35m  -> Depth:  {candidate_data['depth_ppm']:.0f} ppm\033[0m")
                            print("=" * 78 + "\n")
                            utils.save_plots(PLOTS_SUBDIR, candidate_data, processed_lc, bls_model)
                        else:
                            if fits_processed_count % 5 == 0 or fits_processed_count == len(tasks):
                                print(f"  -> Processed {wid} ({fits_processed_count}/{len(tasks)})...")

                        processed_stars.add(wid)
                        with open(PROCESSED_STARS_FILE, 'a') as pf:
                            pf.write(f"{wid}\n")

                    except Exception as ex:
                        print(f"  !!! Error on {wasp_id}: {ex} !!!")

                    # Submit next star
                    try:
                        next_task = next(task_iter)
                        next_fut = executor.submit(process_single_star_worker, next_task)
                        futures[next_fut] = next_task[2]
                    except StopIteration:
                        pass

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Cleaning up and exiting.")
        raise

    return limit is None

def process_all_wget_scripts(script_limit=None, file_limit=None):
    apply_system_protections()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_wget_scripts = sorted(glob.glob(os.path.join(WGET_SCRIPTS_DIR, "*.bat")))
    if not all_wget_scripts:
        print(f"No wget scripts found in {WGET_SCRIPTS_DIR}")
        return

    # Load completed tiles
    completed_tiles = set()
    if os.path.isfile(COMPLETED_TILES_FILE):
        with open(COMPLETED_TILES_FILE, 'r') as f:
            completed_tiles = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(completed_tiles)} completed tile scripts to skip.")

    # Load processed stars
    processed_stars = set()
    if os.path.isfile(PROCESSED_STARS_FILE):
        with open(PROCESSED_STARS_FILE, 'r') as f:
            processed_stars = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(processed_stars)} previously processed stars.")

    # Also load from candidates.csv if available
    if os.path.isfile(CANDIDATES_CSV):
        try:
            candidates_df = pd.read_csv(CANDIDATES_CSV)
            if 'obj_name' in candidates_df.columns:
                cand_stars = set(candidates_df['obj_name'].dropna().unique())
                processed_stars.update(cand_stars)
        except Exception:
            pass

    print(f"Found {len(all_wget_scripts)} total wget scripts.")
    scripts_to_process = [s for s in all_wget_scripts if os.path.basename(s) not in completed_tiles]
    print(f"{len(scripts_to_process)} scripts remaining to process.")

    if script_limit is not None:
        scripts_to_process = scripts_to_process[:script_limit]

    try:
        for script_path in scripts_to_process:
            script_name = os.path.basename(script_path)
            tile_completed = process_wget_script(script_path, processed_stars=processed_stars, limit=file_limit)
            if tile_completed and file_limit is None:
                completed_tiles.add(script_name)
                with open(COMPLETED_TILES_FILE, 'a') as f:
                    f.write(f"{script_name}\n")
                print(f"✓ Tile {script_name} fully completed and logged.")
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