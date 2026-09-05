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
# Point this to your refined list of candidates that passed the odd/even test
CANDIDATES_TO_VET_CSV = "exoplanet_candidates_results/vetted1_candidates.csv"

# Use the same folder for the vetting plots
VETTING_PLOTS_DIR = "vetting_results"
os.makedirs(VETTING_PLOTS_DIR, exist_ok=True)


# --- Main Vetting Loop ---
print(f"Reading candidates from: {CANDIDATES_TO_VET_CSV}")
try:
    candidates_df = pd.read_csv(CANDIDATES_TO_VET_CSV)
except FileNotFoundError:
    print(f"Error: Candidate file not found at '{CANDIDATES_TO_VET_CSV}'")
    exit()

print(f"Found {len(candidates_df)} candidates to vet for contamination.")

# Loop through each candidate from your file
for index, candidate in candidates_df.iterrows():
    tic_id = candidate['tic_id']
    period = candidate['period_days']
    t0 = candidate['t0_bjd']

    print(f"\n--- Vetting TIC {tic_id} (Period={period:.4f} d) ---")

    try:
        # Download the Target Pixel File (TPF) data for the specific TIC ID
        print("  Downloading pixel data...")
        tpf_search = lk.search_targetpixelfile(f"TIC {tic_id}", mission='TESS', author='SPOC')
        if not tpf_search:
            print(f"  No TPF data found for TIC {tic_id}. Skipping.")
            continue

        tpf_collection = tpf_search.download_all()

        # --- MODIFICATION: Process each TPF individually, then stitch the light curves ---
        all_target_lcs = []
        all_background_lcs = []

        for tpf in tpf_collection:
            # Create an aperture mask for the target star
            target_mask = tpf.create_threshold_mask(threshold=3.0)
            # Create a background aperture mask
            background_mask = ~target_mask

            # Create the two light curves for this sector
            target_lc_sector = tpf.to_lightcurve(aperture_mask=target_mask)
            background_lc_sector = tpf.to_lightcurve(aperture_mask=background_mask)

            # Add the light curves for this sector to our lists
            all_target_lcs.append(target_lc_sector)
            all_background_lcs.append(background_lc_sector)

        # Now, create LightCurveCollections and stitch them
        stitched_target_lc = lk.LightCurveCollection(all_target_lcs).stitch().remove_nans()
        stitched_background_lc = lk.LightCurveCollection(all_background_lcs).stitch().remove_nans()

        if len(stitched_target_lc) == 0:
            print(f"  Stitched target light curve is empty for TIC {tic_id}. Skipping.")
            continue

        # Flatten both light curves
        flat_target_lc = stitched_target_lc.flatten(window_length=401)
        flat_background_lc = stitched_background_lc.flatten(window_length=401)

        # Fold both light curves at the known period
        folded_target_lc = flat_target_lc.fold(period=period, epoch_time=t0)
        folded_background_lc = flat_background_lc.fold(period=period, epoch_time=t0)

        # --- Plotting the Vetting Result ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the folded background data and its binned average
        ax.scatter(folded_background_lc.time.value, folded_background_lc.flux.value, s=5, alpha=0.3, c='gray', label='Background Pixels')
        binned_bg_lc = folded_background_lc.bin(bins=100)
        ax.plot(binned_bg_lc.time.value, binned_bg_lc.flux.value, color='blue', linewidth=2, label='Binned Background')

        # Plot the folded target data and its binned average
        ax.scatter(folded_target_lc.time.value, folded_target_lc.flux.value, s=5, alpha=0.3, c='k', label='Target Pixels')
        binned_target_lc = folded_target_lc.bin(bins=100)
        ax.plot(binned_target_lc.time.value, binned_target_lc.flux.value, color='red', linewidth=2, label='Binned Target')

        ax.set_title(f'TIC {tic_id} - Contamination Vet (Folded at P={period:.4f} d)')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_xlim(-0.1, 0.1) # Zoom in on the transit
        ax.legend()
        plt.tight_layout()

        # Save the figure
        output_filename = os.path.join(VETTING_PLOTS_DIR, f"TIC_{tic_id}_contamination_vet.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"  Saved vetting plot to: {output_filename}")

    except Exception as e:
        print(f"  An error occurred while processing TIC {tic_id}: {e}")
        continue

print("\n--- Vetting Complete ---")