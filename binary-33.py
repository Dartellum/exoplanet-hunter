import os
import pandas as pd
from tqdm import tqdm

# === Load candidate TIC IDs ===
print("📂 Loading TIC IDs from binary_candidates_list.txt...")
candidate_ids = set()
try:
    with open("binary_candidates_list.txt", 'r') as f:
        for line in f:
            if line.strip():
                candidate_ids.add(int(line.strip()))
except FileNotFoundError:
    print("❌ Error: 'binary_candidates_list.txt' not found. Please create this file with your target TIC IDs.")
    exit()

if not candidate_ids:
    print("⚠️ Warning: No candidate IDs found in the file. The script will not find any matches.")
    exit()

print(f"✅ Loaded {len(candidate_ids)} candidate IDs.")

# === Locate TIC CSV files ===
TIC_DIR = "TIC_CATALOG"
if not os.path.isdir(TIC_DIR):
    print(f"❌ Error: TIC directory '{TIC_DIR}' not found. Please create it and place your CSVs inside.")
    exit()

tic_files = sorted([os.path.join(TIC_DIR, f) for f in os.listdir(TIC_DIR) if f.endswith(".csv")])
print(f"📂 Found {len(tic_files)} TIC CSV files in {TIC_DIR}")

results = []
found_count = 0

# === Process CSVs in Chunks ===
# This is the key change. We now process each file in small, manageable chunks.
with tqdm(total=len(tic_files), desc="🔍 Scanning TIC CSVs", unit="file") as pbar_files:
    for file in tic_files:
        try:
            print(f"\nProcessing file: {os.path.basename(file)}...")
            found_in_file = 0

            # Use 'chunksize' to read the file in manageable pieces.
            tic_chunks = pd.read_csv(
                file,
                header=None,
                dtype=str,
                low_memory=True,
                chunksize=100000,
                iterator=True,
                on_bad_lines='skip' # Skips bad lines instead of raising an error
            )

            pbar_chunks = tqdm(tic_chunks, desc="  - Reading chunks", unit="chunk")

            for chunk_df in pbar_chunks:
                # The columns we care about are 0 (ID), 14 (Tmag), 15 (RA), and 16 (Dec)
                # We select only these columns to reduce memory usage.
                chunk_df = chunk_df.iloc[:, [0, 14, 15, 16]]

                # Pad short rows to avoid index errors
                chunk_df = chunk_df.fillna(value=pd.NA)

                # Convert ID column (0) to int where possible
                chunk_df.loc[:, 0] = pd.to_numeric(chunk_df.loc[:, 0], errors='coerce').astype('Int64')

                # Filter for matches within this chunk
                matches = chunk_df[chunk_df.loc[:, 0].isin(candidate_ids)]

                for _, row in matches.iterrows():
                    try:
                        tic_id = int(row.iloc[0])
                        tmag = pd.to_numeric(row.iloc[1], errors='coerce')
                        ra = pd.to_numeric(row.iloc[2], errors='coerce')
                        dec = pd.to_numeric(row.iloc[3], errors='coerce')

                        results.append({
                            "ID": tic_id,
                            "Tmag": tmag,
                            "RA": ra,
                            "Dec": dec,
                            "SourceFile": os.path.basename(file)
                        })
                        found_in_file += 1
                        print(f"🎯 Match: TIC {tic_id} | Tmag={tmag} | RA={ra} | Dec={dec} | File={os.path.basename(file)}")

                        # Remove the found ID from the set to avoid redundant searching
                        if tic_id in candidate_ids:
                            candidate_ids.remove(tic_id)

                    except Exception as e:
                        print(f"⚠️ Error parsing match in {os.path.basename(file)}: {e}")

                # Check for an early exit condition in each chunk
                if not candidate_ids:
                    print("✅ All candidates found — stopping early.")
                    break # break from the inner chunk loop

            # Check for an early exit from the outer file loop
            if not candidate_ids:
                break # break from the outer file loop

        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(file)}: {e}")

        pbar_files.update(1)

# === Save results ===
output_file = "binary_candidates_with_coords.csv"
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"\n✅ Done! Found {len(results)} candidates. Data saved to {output_file}")
