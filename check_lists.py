import pandas as pd
import os

# --- Configuration ---
CANDIDATES_CSV = "exoplanet_candidates_results/candidates.csv"
BINARY_LIST_FILE = "binary_candidates_list.txt"

# --- Main Logic ---
print("--- Running Diagnostic Check ---")

# Check if the required files exist
if not os.path.isfile(CANDIDATES_CSV):
    print(f"Error: Main candidate file not found at '{CANDIDATES_CSV}'")
    exit()
if not os.path.isfile(BINARY_LIST_FILE):
    print(f"Error: Binary list file not found at '{BINARY_LIST_FILE}'.")
    exit()

# Read the main candidate data
all_candidates_df = pd.read_csv(CANDIDATES_CSV)
all_candidates_tics = set(all_candidates_df['tic_id'].astype(int))
print(f"\nFound {len(all_candidates_tics)} unique TIC IDs in '{CANDIDATES_CSV}'")
print(f"  - First 5 IDs: {list(all_candidates_tics)[:5]}")


# Read your list of binary TIC IDs
with open(BINARY_LIST_FILE, 'r') as f:
    binary_tics = {int(line.strip()) for line in f if line.strip()}
print(f"\nFound {len(binary_tics)} unique TIC IDs in '{BINARY_LIST_FILE}'")
print(f"  - First 5 IDs: {list(binary_tics)[:5]}")


# --- The Diagnostic Step ---
# Find the intersection (the IDs that are in BOTH files)
matching_tics = all_candidates_tics.intersection(binary_tics)

print("\n--- Diagnostic Result ---")
if not matching_tics:
    print("Result: 0 matching TIC IDs were found between the two files.")
    print("This is why the output is empty. Please verify that the IDs in")
    print(f"'{BINARY_LIST_FILE}' are present in your '{CANDIDATES_CSV}'.")
else:
    print(f"Result: Found {len(matching_tics)} matching TIC ID(s) between the files.")
    print("The following IDs should be processed correctly:")
    print(sorted(list(matching_tics)))