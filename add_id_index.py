import sqlite3
import os
import time

# --- Configuration ---
DATABASE_FILE = "tic_catalog.db"

# --- Main Logic ---
if __name__ == '__main__':
    if not os.path.isfile(DATABASE_FILE):
        print(f"Error: Database file not found at '{DATABASE_FILE}'.")
        exit()

    print(f"Opening database: {DATABASE_FILE}")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    print("\n--- Creating database index for the 'ID' column ---")
    print("This is a one-time operation that may take a significant amount of time.")
    print("Please be patient and let it run to completion...")

    start_time = time.time()

    try:
        # This is the SQL command to create the index
        cursor.execute("CREATE INDEX idx_id ON stars (ID)")
        conn.commit()

        end_time = time.time()

        print("\n--- Index Creation Complete ---")
        print(f"Successfully created an index on the 'ID' column.")
        print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes.")

    except Exception as e:
        print(f"\nAn error occurred while creating the index: {e}")
        print("It's possible the index already exists, which is okay.")

    finally:
        # Close the connection
        conn.close()