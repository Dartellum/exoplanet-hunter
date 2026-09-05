#!/usr/bin/env python3

import subprocess
import sys

# Define the source and destination directories
SOURCE="/media/wes/scripts"
DESTINATION="/home/wes/"

# Construct the rsync command as a list of strings
command = [
    "rsync",
    "-avh",
    "--progress",
    SOURCE,
    DESTINATION
]

print(f"🚀 Starting rsync: Moving data from {SOURCE} to {DESTINATION}")

try:
    # Execute the command
    # check=True will raise an exception if rsync returns a non-zero exit code (an error)
    subprocess.run(command, check=True)
    print("\n✅ Rsync completed successfully!")

except subprocess.CalledProcessError as e:
    print(f"❌ Error during rsync: {e}", file=sys.stderr)
    sys.exit(1) # Exit with an error code

except FileNotFoundError:
    print("❌ Error: 'rsync' command not found. Is rsync installed and in your PATH?", file=sys.stderr)
    sys.exit(1)