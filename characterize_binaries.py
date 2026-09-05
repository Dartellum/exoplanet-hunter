#!/usr/bin/env python3
import duckdb
import pandas as pd
import os

DATABASE_FILE = "tic_catalog.duckdb"
INPUT_FILE = "binary_candidates_list.txt"
OUTPUT_FILE = "binary_candidates_with_coords.csv"

def load_candidate_ids(file_path):
    """Load TIC IDs from a plain text file, one per line."""
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]

def query_duckdb(ids):
    """Query DuckDB for the given TIC IDs and return results as DataFrame."""
    con = duckdb.connect(DATABASE_FILE, read_only=True)
    id_list_str = ",".join(map(str, ids))
    query = f"""
        SELECT ID, Tmag, RA, Dec
        FROM stars
        WHERE ID IN ({id_list_str})
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df

def main():
    if not os.path.isfile(DATABASE_FILE):
        print(f"❌ DuckDB database file not found at '{DATABASE_FILE}'")
        return
    if not os.path.isfile(INPUT_FILE):
        print(f"❌ Input file not found at '{INPUT_FILE}'")
        return

    print(f"📂 Loading TIC IDs from {INPUT_FILE}...")
    candidate_ids = load_candidate_ids(INPUT_FILE)
    print(f"✅ Loaded {len(candidate_ids)} candidate IDs.")

    print("🔍 Querying DuckDB...")
    results_df = query_duckdb(candidate_ids)

    # Ensure output order matches input order
    results_df = results_df.set_index("ID").reindex(candidate_ids).reset_index()

    print(f"💾 Saving results to {OUTPUT_FILE}...")
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Results ---")
    print(results_df.to_string(index=False))
    print(f"\n✅ Done! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
