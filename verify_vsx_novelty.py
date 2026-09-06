import os
import sys
import warnings
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

warnings.filterwarnings('ignore')

INPUT_CSV = "binary_submissions/aavso_vsx_submission.csv"
OUTPUT_REPORT_CSV = "binary_submissions/novelty_vetting_report.csv"

def verify_all_candidates():
    print("=" * 75)
    print("🔍 VERIFYING NOVELTY AGAINST AAVSO VSX, TESS-EB & SIMBAD 🔍")
    print("=" * 75)

    if not os.path.isfile(INPUT_CSV):
        print(f"❌ Could not find {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} candidates to verify.")

    # Configure Vizier
    v_vsx = Vizier(columns=['Name', 'Type', 'Period', 'Vmag'], catalog='B/vsx/vsx')
    v_teb = Vizier(columns=['TIC', 'Period', 'DepthPri'], catalog='J/ApJS/258/16/tess-eb')

    # Configure Simbad
    s = Simbad()
    s.add_votable_fields('otype')

    report_rows = []

    print("\nQuerying catalogs for each star (15 arcsecond radius)...")
    for idx, row in df.iterrows():
        sny_id = row['Name']
        tic_id = row['TIC_ID']
        ra_deg = float(row['RA_deg'])
        dec_deg = float(row['Dec_deg'])
        period = float(row['Period_days'])

        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')

        # 1. Check AAVSO VSX
        vsx_match = "None (Novel Discovery)"
        vsx_type = "N/A"
        vsx_period = "N/A"
        try:
            res_vsx = v_vsx.query_region(coord, radius=15*u.arcsec)
            if len(res_vsx) > 0 and len(res_vsx[0]) > 0:
                vsx_match = str(res_vsx[0]['Name'][0])
                vsx_type = str(res_vsx[0]['Type'][0])
                vsx_period = str(res_vsx[0]['Period'][0]) if 'Period' in res_vsx[0].colnames else "N/A"
        except Exception as e:
            vsx_match = f"Query Error: {e}"

        # 2. Check TESS Eclipsing Binary Catalog (Prsa et al. 2022)
        teb_match = "None"
        teb_period = "N/A"
        try:
            res_teb = v_teb.query_region(coord, radius=15*u.arcsec)
            if len(res_teb) > 0 and len(res_teb[0]) > 0:
                teb_match = f"TIC {res_teb[0]['TIC'][0]}"
                teb_period = str(res_teb[0]['Period'][0]) if 'Period' in res_teb[0].colnames else "N/A"
        except Exception as e:
            teb_match = f"Query Error: {e}"

        # 3. Check SIMBAD
        simbad_id = "None"
        simbad_otype = "N/A"
        try:
            res_sim = s.query_region(coord, radius=15*u.arcsec)
            if res_sim and len(res_sim) > 0:
                simbad_id = str(res_sim['main_id'][0])
                simbad_otype = str(res_sim['otype'][0])
        except Exception as e:
            simbad_id = f"Query Error: {e}"

        status = "✅ NEW TO VSX" if "None" in vsx_match else f"⚠️ ALREADY IN VSX ({vsx_match})"

        print(f"[{idx+1}/{len(df)}] {sny_id} ({tic_id}): {status} | Simbad Type: {simbad_otype}")

        report_rows.append({
            "Name": sny_id,
            "TIC_ID": tic_id,
            "RA_J2000": row['RA_J2000'],
            "Dec_J2000": row['Dec_J2000'],
            "Our_Period_days": period,
            "VSX_Status": status,
            "VSX_Catalog_Name": vsx_match,
            "VSX_Type": vsx_type,
            "VSX_Period": vsx_period,
            "TESS_EB_Catalog": teb_match,
            "TESS_EB_Period": teb_period,
            "SIMBAD_ID": simbad_id,
            "SIMBAD_Object_Type": simbad_otype
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUTPUT_REPORT_CSV, index=False)

    print("\n" + "=" * 75)
    print("📊 NOVELTY SUMMARY:")
    new_to_vsx_count = sum(1 for r in report_rows if "✅ NEW TO VSX" in r['VSX_Status'])
    print(f"Total Candidates Verified: {len(report_rows)}")
    print(f"New to AAVSO VSX:          {new_to_vsx_count} / {len(report_rows)}")
    print(f"📄 Detailed Report Saved:   {OUTPUT_REPORT_CSV}")
    print("=" * 75)

if __name__ == '__main__':
    verify_all_candidates()
