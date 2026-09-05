import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import astropy

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
# Point this to the list of good candidates you are vetting
CANDIDATES_TO_VET_CSV = "exoplanet_candidates_results/good_candidates.csv"

# A new folder to save the vetting plots
VETTING_PLOTS_DIR = "vetting_results"
os.makedirs(VETTING_PLOTS_DIR, exist_ok=True)


# --- Main Vetting Loop ---
print(f"Reading candidates from: {CANDIDATES_TO_VET_CSV}")
try:
    candidates_df = pd.read_csv(CANDIDATES_TO_VET_CSV)
except FileNotFoundError:
    print(f"Error: Candidate file not found at '{CANDIDATES_TO_VET_CSV}'")
    exit()

print(f"Found {len(candidates_df)} candidates to vet.")

# Loop through each candidate from your file
for index, candidate in candidates_df.iterrows():
    tic_id = candidate['tic_id']
    period = candidate['period_days']
    t0 = candidate['t0_bjd']

    print(f"\n--- Vetting TIC {tic_id} (Period={period:.4f} d) ---")

    try:
        # Download the light curve data for the specific TIC ID
        print("  Downloading light curve data...")
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')
        if not search_result:
            print(f"  No SPOC data found for TIC {tic_id}. Skipping.")
            continue

        lc_collection = search_result.download_all()
        stitched_lc = lc_collection.stitch().remove_nans()
        if len(stitched_lc) == 0:
            print(f"  Light curve is empty for TIC {tic_id}. Skipping.")
            continue

        # Flatten the light curve to remove stellar variability
        flat_lc = stitched_lc.flatten(window_length=401)

        # --- The Key Vetting Step ---
        # Fold the light curve at TWICE the detected period
        print(f"  Folding at 2x period ({2*period:.4f} d) to check for a secondary eclipse...")
        folded_lc = flat_lc.fold(period=2*period, epoch_time=t0)

        # Bin the folded light curve to make the signal clearer
        binned_lc = folded_lc.bin(bins=200)

        # --- Plotting the Vetting Result ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the folded data points
        ax.scatter(folded_lc.time.value, folded_lc.flux.value, s=5, alpha=0.3, c='k', label='Folded Data')

        # Plot the binned average
        ax.plot(binned_lc.time.value, binned_lc.flux.value, color='red', linewidth=2, label='Binned Average')

        ax.set_title(f'TIC {tic_id} - Odd/Even Transit Vet (Folded at P={2*period:.4f} d)')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        plt.tight_layout()

        # Save the figure
        output_filename = os.path.join(VETTING_PLOTS_DIR, f"TIC_{tic_id}_odd_even_vet.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"  Saved vetting plot to: {output_filename}")

    except Exception as e:
        print(f"  An error occurred while processing TIC {tic_id}: {e}")
        continue

print("\n--- Vetting Complete ---")