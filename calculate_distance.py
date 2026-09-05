from astroquery.mast import Catalogs
import astropy
import astropy.units as u
import warnings

# Suppress common Astropy Units warnings for cleaner output
warnings.filterwarnings('ignore', category=astropy.units.core.UnitsWarning)

# --- Your Top 3 Vetted Candidates ---
CANDIDATE_TICS = [
    31313288,
    403265752,
    314888045
]

print("--- Querying TIC Catalog for Parallax Data ---")

try:
    # Query the Tic catalog by ID
    # MODIFICATION: Removed the invalid 'columns' parameter.
    results_table = Catalogs.query_criteria(
        catalog='Tic',
        ID=CANDIDATE_TICS
    )

    if len(results_table) == 0:
        print("Could not retrieve data for the given TIC IDs.")
    else:
        print("\n--- Calculated Distances ---")
        for star in results_table:
            tic_id = star['ID']
            parallax_mas = star['plx']

            print(f"\n## Candidate: TIC {tic_id}")
            print(f"   TESS Magnitude: {star['Tmag']:.2f}")

            # A valid parallax must be a positive number
            if parallax_mas is not None and parallax_mas > 0:
                # Convert parallax from milliarcseconds to arcseconds
                parallax_arcsec = parallax_mas / 1000.0

                # Calculate distance in parsecs
                distance_pc = 1 / parallax_arcsec

                # Convert parsecs to light-years (1 pc = 3.26156 ly)
                distance_ly = distance_pc * 3.26156

                print(f"   Parallax: {parallax_mas:.4f} mas")
                print(f"   Distance: {distance_pc:.2f} parsecs")
                print(f"   Distance: {distance_ly:.2f} light-years")
            else:
                print(f"   Parallax: {parallax_mas} (invalid or unavailable)")
                print(f"   Distance: Cannot be calculated.")

except Exception as e:
    print(f"An error occurred during the query: {e}")