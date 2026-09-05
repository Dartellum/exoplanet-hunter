import pandas as pd
import requests
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import utils

# --- Configuration ---
RESULTS_DIR = "rv_candidates_results"
PLOTS_SUBDIR = os.path.join(RESULTS_DIR, "plots")
CANDIDATES_CSV = os.path.join(RESULTS_DIR, "candidates.csv")
RV_TARGET_LIST_FILE = "rv_target_list.txt"

#==============================================================================
# --- RV Data Acquisition ---
#==============================================================================
def download_rv_target_list():
    """
    Downloads and cleans a list of confirmed exoplanet host stars from the
    NASA Exoplanet Archive to use as a target list for RV data search.
    """
    # ... (function body remains the same) ...
    print("\n--- Downloading Confirmed Exoplanet Host List ---")
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+hostname+from+ps&format=csv"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        rv_df = pd.read_csv(io.StringIO(response.text))
        unique_stars = rv_df['hostname'].dropna().unique()

        with open(RV_TARGET_LIST_FILE, 'w') as f:
            for star in unique_stars:
                f.write(f"{star}\n")

        print(f"\nSuccessfully created master target list with {len(unique_stars):,} unique stars.")

    except Exception as e:
        print(f"\nAn error occurred while downloading the list: {e}")

    input("Press Enter to return to the menu...")

def get_rv_data(star_name):
    """
    Downloads and parses radial velocity data for a given star from the
    NASA Exoplanet Archive.
    """
    print(f"  -> Attempting to download RV data for {star_name}...")
    base_url = "https://exoplanetarchive.ipac.caltech.edu/data/RV/published_rv_curves/"
    url = f"{base_url}{star_name.lower().replace(' ', '_')}.csv"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        rv_df = pd.read_csv(io.StringIO(response.text))

        # We need to standardize column names. The archive uses 'time', 'vel', 'vel_err'
        if all(col in rv_df.columns for col in ['time', 'vel', 'vel_err']):
            rv_data = {
                'time': rv_df['time'],
                'vel': rv_df['vel'],
                'vel_err': rv_df['vel_err']
            }
            print(f"  -> Successfully downloaded data for {star_name}.")
            return rv_data
        else:
            print(f"  -> Data for {star_name} has unexpected column names.")
            return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  -> No published RV data found for {star_name}.")
        else:
            print(f"  !!! HTTP Error for {star_name}: {e} !!!")
        return None
    except Exception as e:
        print(f"  !!! Error processing {star_name}: {e} !!!")
        return None

#==============================================================================
# --- RV Analysis Function ---
#==============================================================================
def run_rv_analysis(rv_data, object_id, display_name):
    """
    Performs Lomb-Scargle periodogram analysis on RV data.
    """
    # Use the astropy LombScargle class to compute the periodogram
    # We pass time, velocity, and uncertainty for a more robust analysis
    ls = LombScargle(rv_data['time'], rv_data['vel'], rv_data['vel_err'])
    frequency, power = ls.autopower(minimum_frequency=1/1000.0, maximum_frequency=1/0.1) # Looking for periods between 0.1 and 1000 days

    # Find the period with the highest power
    best_frequency = frequency[np.argmax(power)]
    best_period = 1.0 / best_frequency

    # We need a metric to determine if the signal is significant
    # For now, we'll use a simple threshold
    # A more robust approach would be to calculate the False Alarm Probability (FAP)
    # The highest power value can serve as a simple proxy for significance
    highest_power = np.max(power)

    # We will use the same SDE threshold for consistency, but this is a simplification
    # In a real RV survey, you would use a False Alarm Probability (FAP)
    if highest_power >= 0.2: # This threshold is for demonstration and may need to be adjusted
        print(f"  >>> FOUND CANDIDATE in {display_name} with Power={highest_power:.2f}! <<<")
        candidate_data = {
            'star_id': object_id,
            'star_name': display_name,
            'period_days': best_period.value,
            'power': highest_power,
            'status': 'Unvetted Candidate'
        }
        return (candidate_data, power, frequency, best_period)
    return None

#==============================================================================
# --- Main RV Survey Logic ---
#==============================================================================
def run_survey():
    """
    The main survey logic for Radial Velocity.
    """
    if not os.path.isfile(RV_TARGET_LIST_FILE):
        print(f"Error: Master target list '{RV_TARGET_LIST_FILE}' not found.")
        print("Please run option 1 to download and prepare the list first.")
        input("Press Enter to continue...")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_SUBDIR, exist_ok=True)

    with open(RV_TARGET_LIST_FILE, 'r') as f:
        all_rv_targets = [line.strip() for line in f if line.strip()]

    print(f"\n--- Starting Radial Velocity Survey on {len(all_rv_targets):,} targets ---")

    # For now, we will process stars sequentially for simplicity
    for star_name in all_rv_targets:
        rv_data = get_rv_data(star_name)
        if rv_data:
            # We'll use the star name as a unique ID for simplicity
            candidate_result = run_rv_analysis(rv_data, star_name, star_name)

            if candidate_result:
                candidate_data, power, frequency, best_period = candidate_result

                # Plot the periodogram
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(1.0/frequency, power)
                ax.axvline(best_period.value, color='red', linestyle='--')
                ax.set_title(f"Lomb-Scargle Periodogram for {star_name}")
                ax.set_xlabel("Period (days)")
                ax.set_ylabel("Power")
                fig.savefig(os.path.join(PLOTS_SUBDIR, f'{star_name.replace(" ", "_")}_periodogram.png'))
                plt.close(fig)

                # Save the candidate data to CSV
                candidate_df = pd.DataFrame([candidate_data])
                file_exists = os.path.isfile(CANDIDATES_CSV)
                candidate_df.to_csv(CANDIDATES_CSV, mode='a', header=not file_exists, index=False)


    print("\n--- RV Survey Complete ---")
    print(f"Candidates have been saved to {CANDIDATES_CSV}")
    print("--- Script Finished ---")
    input("Press Enter to return to the menu...")

#==============================================================================
# --- RV Survey Menu ---
#==============================================================================
def menu():
    """Displays the RV survey sub-menu."""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n--- Radial Velocity Survey Menu ---")
        print("1. Download and Prepare RV Target List")
        print("2. Run RV All-Sky Survey")
        print("3. Back to Main Menu")

        choice = input("\nSelect an option: ")

        if choice == '1':
            download_rv_target_list()
        elif choice == '2':
            run_survey()
        elif choice == '3':
            break