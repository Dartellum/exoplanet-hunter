import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord, get_constellation
from astroquery.mast import Catalogs

# --- Configuration ---
BINARY_LIST_FILE = "binary_candidates_list.txt"
CANDIDATES_CSV = "exoplanet_candidates_results/candidates.csv"
OUTPUT_DIR = "binary_submissions"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "aavso_vsx_submission.csv")

# Suppress minor Astropy / Lightkurve warnings
warnings.filterwarnings('ignore')

def manual_fold(time_arr, flux_arr, period, t0, n_bins=200):
    """
    Folds time and flux using pure NumPy to completely avoid the Lightkurve
    TimeDelta.mjd bug. Returns folded data and binned mean across [-0.5, 0.5].
    """
    phase = ((time_arr - t0 + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux_arr[sort_idx]

    # Binning with numpy histogram
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_means, bin_edges = np.histogram(phase_sorted, bins=bins, weights=flux_sorted)
    bin_counts, _ = np.histogram(phase_sorted, bins=bins)
    valid = bin_counts > 0
    binned_flux = np.where(valid, bin_means / np.maximum(bin_counts, 1), np.nan)
    binned_phase = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return phase_sorted, flux_sorted, binned_phase[valid], binned_flux[valid]

def analyze_eclipses(binned_phase, binned_flux):
    """
    Measures primary depth (around phase 0.0) and secondary depth (around phase ±0.5).
    """
    # Primary eclipse window: |phase| <= 0.1
    pri_mask = np.abs(binned_phase) <= 0.1
    pri_min_flux = float(np.nanmin(binned_flux[pri_mask])) if np.any(pri_mask) else 1.0
    pri_phase_min = float(binned_phase[pri_mask][np.nanargmin(binned_flux[pri_mask])]) if np.any(pri_mask) else 0.0
    pri_depth = max(0.0, 1.0 - pri_min_flux)

    # Secondary eclipse window: 0.35 <= |phase| <= 0.5
    sec_mask = np.abs(binned_phase) >= 0.35
    sec_min_flux = float(np.nanmin(binned_flux[sec_mask])) if np.any(sec_mask) else 1.0
    sec_phase_min = float(binned_phase[sec_mask][np.nanargmin(binned_flux[sec_mask])]) if np.any(sec_mask) else 0.5
    sec_depth = max(0.0, 1.0 - sec_min_flux)

    # Out-of-eclipse baseline variation (0.15 <= |phase| <= 0.35)
    ooe_mask = (np.abs(binned_phase) >= 0.15) & (np.abs(binned_phase) <= 0.35)
    ooe_flux = binned_flux[ooe_mask] if np.any(ooe_mask) else np.array([1.0])
    ooe_var = float(np.nanmax(ooe_flux) - np.nanmin(ooe_flux)) if len(ooe_flux) > 0 else 0.0

    return pri_depth, pri_phase_min, sec_depth, sec_phase_min, ooe_var

def classify_binary(period_days, pri_depth, sec_depth, ooe_var):
    """
    Classifies the eclipsing binary into AAVSO VSX variability types:
    - EA: Algol-type (detached, flat out-of-eclipse, distinct eclipses)
    - EB: Beta Lyrae-type (semi-detached, continuous ellipsoidal curvature, unequal depths)
    - EW: W UMa-type (contact, period < 1.0 d, continuous light variation, nearly equal depths)
    """
    depth_ratio = min(pri_depth, sec_depth) / max(pri_depth, sec_depth) if max(pri_depth, sec_depth) > 0 else 0.0

    if period_days < 1.0 and depth_ratio > 0.75 and ooe_var > 0.005:
        return "EW"
    elif ooe_var > 0.01 or (sec_depth > 0.005 and depth_ratio < 0.75 and ooe_var > 0.003):
        return "EB"
    else:
        return "EA"

def process_all_binaries():
    print("=" * 75)
    print("🔭 AAVSO VSX ECLIPSING BINARY CHARACTERIZATION PIPELINE (SNY V01-V33) 🔭")
    print("=" * 75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load the 33 TIC IDs
    if not os.path.isfile(BINARY_LIST_FILE):
        print(f"❌ Target list file '{BINARY_LIST_FILE}' not found.")
        return

    with open(BINARY_LIST_FILE, 'r') as f:
        target_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

    print(f"Loaded {len(target_ids)} binary candidate TIC IDs.")

    # 2. Batch query MAST TIC for 2MASS and Gaia cross-identifications
    print("📡 Querying MAST TIC for official 2MASS and Gaia DR2 cross-identifications...")
    cross_id_map = {}
    try:
        res = Catalogs.query_criteria(catalog='Tic', ID=target_ids)
        tdf = res.to_pandas()
        for _, r in tdf.iterrows():
            cid = int(r['ID'])
            twomass = str(r['TWOMASS']).strip() if pd.notna(r['TWOMASS']) else ""
            gaia = str(r['GAIA']).strip() if pd.notna(r['GAIA']) else ""
            cross_id_map[cid] = {
                'twomass': f"2MASS J{twomass}" if twomass else "",
                'gaia': f"Gaia DR2 {gaia}" if gaia else ""
            }
        print(f"✅ Successfully retrieved cross-identifications for {len(cross_id_map)} stars.")
    except Exception as e:
        print(f"⚠️ Could not batch query MAST for cross-IDs: {e}")

    # 3. Load prior candidates table for initial BLS period and t0 estimates
    candidates_info = {}
    if os.path.isfile(CANDIDATES_CSV):
        df_cand = pd.read_csv(CANDIDATES_CSV)
        for _, row in df_cand.iterrows():
            cid = int(row['tic_id'])
            if cid in target_ids:
                candidates_info[cid] = {
                    'period': float(row['period_days']),
                    't0': float(row['t0_bjd']),
                    'depth_ppm': float(row['depth_ppm'])
                }

    submission_rows = []

    for idx, tic_id in enumerate(target_ids, 1):
        sny_id = f"SNY V{idx:02d}"
        print(f"\n[{idx}/{len(target_ids)}] Processing {sny_id} (TIC {tic_id})...")

        # A. Download TESS Light Curve
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')
            if len(search) == 0:
                print(f"  ⚠️ No SPOC light curves found for TIC {tic_id}, trying all authors...")
                search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            if len(search) == 0:
                print(f"  ❌ No TESS data found for TIC {tic_id}. Skipping.")
                continue

            lc_col = search.download_all()
            stitched_lc = lc_col.stitch().remove_nans()
            if len(stitched_lc) == 0:
                print(f"  ❌ Stitched light curve is empty for TIC {tic_id}. Skipping.")
                continue

            flat_lc = stitched_lc.flatten(window_length=401)
            time_btjd = np.array(flat_lc.time.value, dtype=float)
            flux = np.array(flat_lc.flux.value, dtype=float)

            # Metadata from FITS header
            first_lc = lc_col[0]
            ra_deg = float(first_lc.meta.get('RA_OBJ', 0.0))
            dec_deg = float(first_lc.meta.get('DEC_OBJ', 0.0))
            tmag = float(first_lc.meta.get('TESSMAG', 12.0))

            # Astropy Coordinates & Constellation
            coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
            constellation = get_constellation(coord)
            ra_hms = coord.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
            dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True)

            # Initial period & epoch
            init_p = candidates_info.get(tic_id, {}).get('period', 1.0)
            init_t0 = candidates_info.get(tic_id, {}).get('t0', time_btjd[0])

            # B. Test period P vs 2P to resolve BLS harmonic aliasing
            # Test 1: Fold at initial period P
            _, _, bp1, bf1 = manual_fold(time_btjd, flux, init_p, init_t0)
            pri_d1, pri_ph1, sec_d1, sec_ph1, ooe1 = analyze_eclipses(bp1, bf1)

            # Test 2: Fold at 2*P
            _, _, bp2, bf2 = manual_fold(time_btjd, flux, 2 * init_p, init_t0)
            pri_d2, pri_ph2, sec_d2, sec_ph2, ooe2 = analyze_eclipses(bp2, bf2)

            # Decide whether true orbital period is P or 2P
            if (sec_d2 > 0.002 and abs(pri_d2 - sec_d2) > 0.001) or (init_p < 0.6 and pri_d2 > 0.01):
                orbital_period = 2 * init_p
                best_t0 = init_t0 + pri_ph2 * orbital_period
                pri_depth, sec_depth, ooe_var = pri_d2, sec_d2, ooe2
                used_phase, used_flux, bp, bf = manual_fold(time_btjd, flux, orbital_period, best_t0)
            else:
                orbital_period = init_p
                best_t0 = init_t0 + pri_ph1 * orbital_period
                pri_depth, sec_depth, ooe_var = pri_d1, sec_d1, ooe1
                used_phase, used_flux, bp, bf = manual_fold(time_btjd, flux, orbital_period, best_t0)

            # Epoch in full BJD (TESS BTJD + 2457000.0)
            epoch_bjd = 2457000.0 + best_t0

            # C. Magnitude calculations
            mag_max = tmag
            flux_drop_pri = min(0.999, max(0.0001, pri_depth))
            delta_mag_pri = -2.5 * np.log10(1.0 - flux_drop_pri)
            mag_min_pri = mag_max + delta_mag_pri

            if sec_depth > 0.0005:
                flux_drop_sec = min(0.999, sec_depth)
                delta_mag_sec = -2.5 * np.log10(1.0 - flux_drop_sec)
                mag_min_sec = mag_max + delta_mag_sec
            else:
                delta_mag_sec = 0.0
                mag_min_sec = np.nan

            # D. Classify variability
            var_type = classify_binary(orbital_period, pri_depth, sec_depth, ooe_var)

            # E. Cross-identifications
            x_twomass = cross_id_map.get(tic_id, {}).get('twomass', '')
            x_gaia = cross_id_map.get(tic_id, {}).get('gaia', '')
            other_names = f"TIC {tic_id}"
            if x_twomass:
                other_names += f", {x_twomass}"
            if x_gaia:
                other_names += f", {x_gaia}"

            # F. Generate Phase-Folded Plot for AAVSO VSX
            fig, ax = plt.subplots(figsize=(11, 6))

            plot_phases = []
            plot_fluxes = []
            for shift in [-1.0, 0.0, 1.0]:
                shift_phase = used_phase + shift
                mask = (shift_phase >= -0.2) & (shift_phase <= 1.2)
                plot_phases.append(shift_phase[mask])
                plot_fluxes.append(used_flux[mask])

            plot_phases = np.concatenate(plot_phases)
            plot_fluxes = np.concatenate(plot_fluxes)

            # Raw points
            ax.scatter(plot_phases, plot_fluxes, s=1.5, alpha=0.35, color='gray', label='TESS Observations')

            # Binned line across -0.2 to 1.2
            binned_plot_p = []
            binned_plot_f = []
            for shift in [-1.0, 0.0, 1.0]:
                shift_p = bp + shift
                mask = (shift_p >= -0.2) & (shift_p <= 1.2)
                binned_plot_p.append(shift_p[mask])
                binned_plot_f.append(bf[mask])
            binned_plot_p = np.concatenate(binned_plot_p)
            binned_plot_f = np.concatenate(binned_plot_f)
            s_idx = np.argsort(binned_plot_p)
            ax.plot(binned_plot_p[s_idx], binned_plot_f[s_idx], color='crimson', lw=2.2, label='Phase Binned (200 bins)')

            # Highlight eclipses
            ax.axvline(0.0, color='blue', linestyle='--', alpha=0.7, label='Primary Eclipse (Phase 0.0)')
            ax.axvline(1.0, color='blue', linestyle='--', alpha=0.7)
            if sec_depth > 0.001:
                ax.axvline(0.5, color='green', linestyle=':', alpha=0.7, label='Secondary Eclipse (Phase 0.5)')

            # Annotations and Titles with SNY designation
            plot_file_name = f"{sny_id.replace(' ', '_')}_vsx_phase.png"
            ax.set_title(f"{sny_id} (TIC {tic_id}) | VSX Type: {var_type} | P = {orbital_period:.6f} d | BJD_0 = {epoch_bjd:.4f}",
                         fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Orbital Phase (Phase 0 = Primary Minimum)", fontsize=11)
            ax.set_ylabel("Normalized Flux (TESS Passband)", fontsize=11)
            ax.set_xlim(-0.2, 1.2)

            subtitle = (f"Discoverer: Dr. Walter Wesley Snyder V (Dartellum) | Constellation: {constellation}\n"
                        f"RA (J2000): {ra_hms} | Dec (J2000): {dec_dms} | Cross-IDs: {other_names}\n"
                        f"Tmag: {mag_max:.2f} ({mag_max:.2f} - {mag_min_pri:.2f} T) | Pri Depth: {pri_depth*1e6:.0f} ppm "
                        f"({delta_mag_pri:.3f} mag) | Sec Depth: {sec_depth*1e6:.0f} ppm ({delta_mag_sec:.3f} mag)")
            ax.text(0.5, -0.16, subtitle, transform=ax.transAxes, ha='center', fontsize=9.2,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray'))

            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', framealpha=0.9)
            plt.tight_layout()

            plot_path = os.path.join(PLOTS_DIR, plot_file_name)
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)

            # Record row
            submission_rows.append({
                "Name": sny_id,
                "TIC_ID": f"TIC {tic_id}",
                "TwoMASS_ID": x_twomass,
                "Gaia_DR2_ID": x_gaia,
                "Other_Names": other_names,
                "Constellation": constellation,
                "RA_J2000": ra_hms,
                "Dec_J2000": dec_dms,
                "RA_deg": round(ra_deg, 6),
                "Dec_deg": round(dec_deg, 6),
                "VarType": var_type,
                "Epoch_BJD": round(epoch_bjd, 5),
                "Period_days": round(orbital_period, 6),
                "Mag_Max_T": round(mag_max, 3),
                "Mag_Min_T": round(mag_min_pri, 3),
                "Amplitude_mag": round(delta_mag_pri, 3),
                "Pri_Depth_ppm": round(pri_depth * 1e6, 1),
                "Sec_Depth_ppm": round(sec_depth * 1e6, 1),
                "Sec_Mag_T": round(mag_min_sec, 3) if not np.isnan(mag_min_sec) else "",
                "Passband": "T",
                "Discovery_Source": "TESS All-Sky Exoplanet Survey Pipeline",
                "Discoverer": "Dr. Walter Wesley Snyder V (Dartellum)",
                "Plot_File": plot_file_name
            })

            print(f"  -> Assigned: {sny_id} | Constellation: {constellation} | VarType: {var_type}")
            print(f"  -> Cross-IDs: {other_names}")
            print(f"  -> Period: {orbital_period:.5f} d | BJD_0: {epoch_bjd:.4f}")
            print(f"  -> Range: {mag_max:.2f} to {mag_min_pri:.2f} T (Pri: {pri_depth*1e6:.0f} ppm, Sec: {sec_depth*1e6:.0f} ppm)")
            print(f"  -> Saved phase plot: {plot_path}")

        except Exception as e:
            print(f"  ❌ Error processing TIC {tic_id}: {e}")

    # 4. Export CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 75)
    print(f"🎉 SUCCESS! Processed {len(submission_rows)} / {len(target_ids)} binary systems.")
    print(f"📄 Submission CSV saved to: {OUTPUT_CSV}")
    print(f"🖼️ Phased plots saved to:   {PLOTS_DIR}/")
    print("=" * 75)

if __name__ == '__main__':
    process_all_binaries()
