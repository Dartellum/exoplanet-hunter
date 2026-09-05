#!/bin/bash
#fastgrep.sh
# A simple Bash script to perform a parallel grep on a directory of large files.

# --- Configuration ---
# Set the directory containing your TIC CSV files
TIC_DIR="TIC_CATALOG"

# Set the pattern to search for (e.g., a TIC ID at the beginning of a line)
# The ^ character ensures the match is at the start of the line.
# Example: pattern="^100617302"
PATTERN="^${1}"

# Set the number of parallel jobs to run (e.g., number of CPU cores)
NUM_JOBS=16

# --- Main Logic ---

# Check if a pattern was provided as an argument
if [ -z "$PATTERN" ]; then
  echo "Usage: ./fastgrep.sh <TIC_ID>"
  exit 1
fi

echo "🔍 Starting parallel search for pattern '$PATTERN' in '$TIC_DIR'..."
echo "Using $NUM_JOBS parallel jobs."

# Find all files in the directory and its subdirectories,
# separate them with a null character (-print0) for safety.
find "$TIC_DIR" -type f -print0 | \
  parallel --progress --null -m -j "$NUM_JOBS" grep -r -a -e "$PATTERN" > "results_$1.txt"

# grep -a treats the input as text, even if it contains binary data.
# The -e flag is needed for parallel to work correctly.
# The --null flag is CRITICAL to correctly read null-delimited filenames from 'find -print0'.

echo "✅ Search complete. Results saved to results_$1.txt"

# --- End of script ---
