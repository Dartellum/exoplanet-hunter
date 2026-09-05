import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# We will *not* define u.ppm here, as it's causing issues.
# Instead, we'll manually scale by 1e6 * u.dimensionless_unscaled.

print(f"lightkurve version: {lk.__version__}")

# --- Step 1: Search for TRAPPIST-1 data ---
print("Searching for TRAPPIST-1 data...")
search_result = lk.search_lightcurve("TRAPPIST-1", exptime="short")

if len(search_result) == 0:
    print("No short cadence data found for TRAPPIST-1. Trying long cadence.")
    search_result = lk.search_lightcurve("TRAPPIST-1")

if len(search_result) == 0:
    print("No TESS light curve data found for TRAPPIST-1. Exiting.")
    exit()
else:
    print(f"Found {len(search_result)} light curve files for TRAPPIST-1.")
    print(search_result)

# --- Step 2: Download and stitch the light curves ---
print("Downloading and stitching light curves...")
lc_collection = search_result.download_all()

if isinstance(lc_collection, lk.LightCurve):
    lc_collection = lk.LightCurveCollection([lc_collection])

# Stitching and pre-processing steps:
# 1. Stitch the raw light curves (just remove NaNs).
# 2. Manually normalize the entire stitched light curve by dividing by its median, making it relative flux (median 1.0).
# 3. Flatten to remove long-term trends.
# 4. Remove outliers.
# 5. Convert to parts-per-million (ppm) by manual scaling and assigning dimensionless_unscaled unit.

# Step 1: Stitch the light curves (without initial normalization in stitch)
stitched_lc = lc_collection.stitch(lambda x: x.remove_nans()) # Stitch, just remove NaNs

# Step 2: Manually normalize the entire stitched light curve
# This converts the raw counts to relative flux (median 1.0)
median_flux = np.nanmedian(stitched_lc.flux)
if median_flux != 0: # Avoid division by zero
    stitched_lc.flux = (stitched_lc.flux / median_flux)
    stitched_lc.flux_err = (stitched_lc.flux_err / median_flux)
else:
    print("Warning: Median flux is zero. Cannot normalize light curve. Exiting.")
    exit() # Exit if normalization fails

# Step 3: Flatten to remove long-term trends
stitched_lc = stitched_lc.flatten(window_length=401)

# Step 4: Remove outliers
stitched_lc = stitched_lc.remove_outliers(sigma=5)

# Step 5: Convert to ppm: (relative flux - 1) * 1e6 ppm
stitched_lc.flux = (stitched_lc.flux - 1.0) * 1e6 * u.dimensionless_unscaled
stitched_lc.flux_err = stitched_lc.flux_err * 1e6 * u.dimensionless_unscaled # Scale error too


# --- Diagnostics: Print properties of stitched_lc before BLS ---
print("\n--- Diagnostic: stitched_lc properties before BLS ---")
print(f"Number of data points: {len(stitched_lc.flux)}")
print(f"Flux min: {np.nanmin(stitched_lc.flux):.2f} {stitched_lc.flux.unit}")
print(f"Flux max: {np.nanmax(stitched_lc.flux):.2f} {stitched_lc.flux.unit}")
print(f"Flux median: {np.nanmedian(stitched_lc.flux):.2f} {stitched_lc.flux.unit}")
print(f"Contains NaNs in flux: {np.any(np.isnan(stitched_lc.flux))}")
print(f"Contains Infs in flux: {np.any(np.isinf(stitched_lc.flux))}")
print(f"--- End Diagnostic ---")


# --- Step 3: Plot the light curve ---
print("Plotting the stitched light curve...")
fig, ax = plt.subplots(figsize=(12, 6))
stitched_lc.plot(ax=ax, linewidth=0.5, alpha=0.8, color='blue', label='Flattened Light Curve')
ax.set_title('TRAPPIST-1 TESS Light Curve (Flattened)')
ax.set_xlabel('Time (BJD - 2457000)') # BJD: Barycentric Julian Date
ax.set_ylabel('Normalized Flux (ppm)')
plt.tight_layout()

# -----------------------------------------------------------------------------
# --- NEW STEP 4: Perform a Box-Least Squares (BLS) search ---
print("\nPerforming BLS transit search...")

min_period_search = 0.5  # days
max_period_search = 20.0 # days
durations = np.linspace(0.01, 0.1, 10) # Reduced duration samples further (from 20 to 10)

bls_model = stitched_lc.to_periodogram(method='bls',
                                        minimum_period=min_period_search,
                                        maximum_period=max_period_search,
                                        duration=durations,
                                        frequency_factor=2000.0)

# --- Step 5: Find the strongest transit signal ---
planet_period = bls_model.period_at_max_power
# FIX for deprecation: Use epoch_time instead of t0
planet_t0 = bls_model.transit_time_at_max_power
planet_depth = bls_model.depth_at_max_power

print(f"Strongest transit candidate found:")
print(f"  Period: {planet_period:.4f}")
print(f"  Transit Time (t0): {np.round(planet_t0.value, 4)} BJD")
print(f"  Transit Depth: {planet_depth:.2e}")

# --- Step 6: Plot the BLS periodogram ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
bls_model.plot(ax=ax2)
ax2.set_title(f"BLS Periodogram for TRAPPIST-1 (Strongest: {planet_period:.4f})")
plt.tight_layout()

# --- Step 7: Fold the light curve at the detected period ---
# FIX for deprecation: Use epoch_time instead of t0
folded_lc = stitched_lc.fold(period=planet_period, epoch_time=planet_t0) # <--- Using epoch_time

fig3, ax3 = plt.subplots(figsize=(10, 6))
folded_lc.scatter(ax=ax3, s=5, alpha=0.5, label='Folded Data')
# --- FIX FOR BINNING ERROR ---
# Create a temporary LightCurve with phase as numerical time for binning
temp_lc_for_binning = lk.LightCurve(time=folded_lc.time.value, flux=folded_lc.flux, flux_err=folded_lc.flux_err)
binned_folded_lc = temp_lc_for_binning.bin(bins=200) # Now bin on this temp object
# --- END FIX ---
binned_folded_lc.plot(ax=ax3, color='red', linewidth=2, label='Binned Data')
ax3.set_title(f"TRAPPIST-1 Light Curve Folded at P={planet_period:.4f} days")
ax3.set_xlabel('Phase')
ax3.set_ylabel('Normalized Flux (ppm)')
ax3.set_xlim(-0.1, 0.1)
ax3.legend()
plt.tight_layout()

# --- Show all plots ---
plt.show()

print("\nBLS search and folded light curve plot complete.")
print("The deepest transit you see is likely TRAPPIST-1b or 1c, as they are the innermost and cause frequent, relatively deep transits.")