#!/usr/bin/env python3
import duckdb
import pandas as pd
import os
import glob
from tqdm import tqdm

# --- Config ---
DATABASE_FILE   = "tic_catalog.duckdb"
TIC_CATALOG_DIR = "TIC_CATALOG"
CHUNK_SIZE      = 500_000  # rows per chunk

# --- Column Indexes ---
IDX_ID   = 0
IDX_TMAG = 14
IDX_RA   = -4
IDX_DEC  = -3

def build_database():
    # Remove old DB if exists
    if os.path.exists(DATABASE_FILE):
        print(f"🗑 Removing existing database: {DATABASE_FILE}")
        os.remove(DATABASE_FILE)

    # Connect to DuckDB
    con = duckdb.connect(DATABASE_FILE)
    con.execute("""
        CREATE TABLE stars (
            ID BIGINT,
            Tmag DOUBLE,
            RA DOUBLE,
            Dec DOUBLE
        )
    """)
    con.commit()

    # Find all TIC CSV files
    csv_files = sorted(glob.glob(os.path.join(TIC_CATALOG_DIR, "*.csv")))
    if not csv_files:
        print(f"❌ No CSV files found in {TIC_CATALOG_DIR}")
        return

    total_files = len(csv_files)
    print(f"📂 Found {total_files} TIC catalog files.")

    # Progress bar over all files
    for csv_file in tqdm(csv_files, desc="Loading TIC CSVs", unit="file"):
        try:
            # Read in chunks
            chunk_iter = pd.read_csv(
                csv_file,
                header=None,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                on_bad_lines='skip'
            )
            for chunk in chunk_iter:
                # Select only the required columns
                df = pd.DataFrame({
                    "ID":   chunk.iloc[:, IDX_ID],
                    "Tmag": chunk.iloc[:, IDX_TMAG],
                    "RA":   chunk.iloc[:, IDX_RA],
                    "Dec":  chunk.iloc[:, IDX_DEC]
                })

                # Insert into DuckDB
                con.execute("INSERT INTO stars SELECT * FROM df")

        except Exception as e:
            print(f"⚠️ Error reading {csv_file}: {e}")

    con.close()
    print(f"\n✅ Done! DuckDB database built at {DATABASE_FILE}")

if __name__ == "__main__":
    build_database()
