#!/usr/bin/env python3
import duckdb
import pandas as pd
import os
import glob

# --- Config ---
DATABASE_FILE   = "tic_catalog.duckdb"
TIC_CATALOG_DIR = "TIC_CATALOG"

# --- Interactive Checker ---
def check_tic_id(tic_id):
    # 1️⃣ Query DuckDB database
    try:
        con = duckdb.connect(DATABASE_FILE, read_only=True)
        query = f"SELECT ID, Tmag, RA, Dec FROM stars WHERE ID = {tic_id}"
        db_result = con.execute(query).fetchdf()
        con.close()

        if db_result.empty:
            print(f"❌ TIC {tic_id} not found in DuckDB.")
            return
        else:
            print(f"✅ Found TIC {tic_id} in DuckDB.")
            print(db_result.to_string(index=False))
    except Exception as e:
        print(f"❌ Error querying DuckDB: {e}")
        return

    # 2️⃣ Find original CSV row for verification
    csv_files = glob.glob(os.path.join(TIC_CATALOG_DIR, "*.csv"))
    found_row = None
    for file in csv_files:
        try:
            for chunk in pd.read_csv(file, header=None, chunksize=100_000, low_memory=False, on_bad_lines='skip'):
                match = chunk[chunk.iloc[:, 0] == tic_id]
                if not match.empty:
                    found_row = match.iloc[0]
                    print(f"✅ Found TIC {tic_id} in {os.path.basename(file)}")
                    break
            if found_row is not None:
                break
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    if found_row is None:
        print("❌ Could not find TIC ID in any CSV file.")
        return

    # 3️⃣ Extract RA/Dec/Tmag from original CSV
    orig_id   = found_row.iloc[0]
    orig_tmag = found_row.iloc[14]
    orig_ra   = found_row.iloc[-4]
    orig_dec  = found_row.iloc[-3]

    print("\n--- Side-by-Side Comparison ---")
    print(f" Original CSV → ID: {orig_id}, Tmag: {orig_tmag}, RA: {orig_ra}, Dec: {orig_dec}")
    print(f" DuckDB       → ID: {db_result['ID'][0]}, Tmag: {db_result['Tmag'][0]}, RA: {db_result['RA'][0]}, Dec: {db_result['Dec'][0]}")

    if abs(orig_ra - db_result['RA'][0]) < 1e-8 and abs(orig_dec - db_result['Dec'][0]) < 1e-8:
        print("\n✅ RA and Dec match perfectly!\n")
    else:
        print("\n❌ RA/Dec mismatch — check database build.\n")

if __name__ == "__main__":
    print("--- TIC Catalog DuckDB Verification ---")
    if not os.path.isfile(DATABASE_FILE):
        print(f"Error: DuckDB database file not found at '{DATABASE_FILE}'.")
        exit()

    while True:
        try:
            user_input = input("Enter TIC ID to check (or press Enter to quit): ").strip()
            if user_input == "":
                print("Exiting.")
                break
            tic_id = int(user_input)
            check_tic_id(tic_id)
        except ValueError:
            print("Please enter a valid integer TIC ID.")
