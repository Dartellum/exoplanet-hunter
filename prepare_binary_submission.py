import pandas as pd
import os
import glob
import re

# --- Configuration ---
BINARY_RESULTS_DIR = "binary_candidates_results"
SUBMISSION_DIR = "binary_submissions"
SUBMISSION_CSV = os.path.join(SUBMISSION_DIR, "tess_binary_submission.csv")

def prepare_submission_file():
    """
    Reads grep output files, extracts TIC IDs, and creates a submission-ready CSV.
    """
    print("--- Preparing Binary System Submission File ---")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Use a pattern to find all your grep result files
    result_files = glob.glob(os.path.join(BINARY_RESULTS_DIR, "results_*.txt"))

    if not result_files:
        print(f"No result files found in '{BINARY_RESULTS_DIR}'. Please check the directory.")
        return

    submission_data = []

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                # Assuming the grep output is one line per file
                full_line = f.readline().strip()
                if not full_line:
                    continue

                # 💡 NEW LOGIC: Split by the colon first
                data_part = full_line.split(':')[-1]

                # Now, split the data part by the comma to get the TIC ID
                first_value = data_part.split(',')[0]

                # Use a regular expression for an exact match to the number at the start of the filename
                match = re.search(r"results_(\d+)\.txt", os.path.basename(file_path))
                if match:
                    # Get the expected TIC ID from the filename
                    expected_tic_id = match.group(1)

                    # Perform an exact string match
                    if first_value == expected_tic_id:
                        tic_id = first_value
                        print(f" -> Found exact match for TIC ID: {tic_id}")
                    else:
                        print(f" -> Skipping {os.path.basename(file_path)}: TIC ID '{first_value}' does not match expected ID '{expected_tic_id}'.")
                        continue
                else:
                    print(f" !!! Could not extract expected TIC ID from filename: {file_path} !!!")
                    continue

            # --- Placeholder for your data ---
            # Now that we have the full_line and first_value, you can parse other
            # data points from the rest of the comma-separated values.
            # You would replace this with your actual data retrieval logic.
            # Example: period, depths
            period = "N/A"
            primary_depth = "N/A"
            secondary_depth = "N/A"
            status = "Ready for Submission"
            # ----------------------------------

            submission_data.append({
                'tic_id': tic_id,
                'period_days': period,
                'primary_depth_ppm': primary_depth,
                'secondary_depth_ppm': secondary_depth,
                'status': status
            })

        except Exception as e:
            print(f" !!! Error processing file {file_path}: {e} !!!")

    # Create a DataFrame from the collected data and save to CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(SUBMISSION_CSV, index=False)

    print(f"\nSubmission file created successfully at: {SUBMISSION_CSV}")
    print(f"Total entries ready for submission: {len(submission_data)}")

if __name__ == '__main__':
    prepare_submission_file()