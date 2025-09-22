import pandas as pd

# --- File and Sheet Definitions ---
# The original Excel file with the two sheets
INPUT_EXCEL_FILE = 'enriched_job_listings_final.xlsx'

# The names of the sheets within the Excel file
INDIVIDUAL_SHEET_NAME = 'Individual Congressional Office'
COMMITTEE_SHEET_NAME = 'Committee Offices'

# The name for the new, cleaned output Excel file
OUTPUT_EXCEL_FILE = 'enriched_job_listings_sorted.xlsx'

def sort_committee_jobs_in_excel():
    """
    Loads data from two sheets in an Excel file, moves miscategorized
    committee jobs, and saves the result to a new Excel file.
    """
    try:
        # Step 1: Load both sheets from the single Excel file
        # Note: pandas uses the 'openpyxl' engine for .xlsx files.
        # If you don't have it, run: pip install openpyxl
        df_individual = pd.read_excel(INPUT_EXCEL_FILE, sheet_name=INDIVIDUAL_SHEET_NAME)
        df_committee = pd.read_excel(INPUT_EXCEL_FILE, sheet_name=COMMITTEE_SHEET_NAME)
        
        print(f"Successfully loaded {len(df_individual)} records from the '{INDIVIDUAL_SHEET_NAME}' sheet.")
        print(f"Successfully loaded {len(df_committee)} records from the '{COMMITTEE_SHEET_NAME}' sheet.")

        # Step 2: Identify committee jobs in the individual offices sheet
        committee_jobs_in_individual = df_individual[
            df_individual['Posting_Author'].str.contains("Committee", case=False, na=False)
        ]

        if not committee_jobs_in_individual.empty:
            print(f"\nFound {len(committee_jobs_in_individual)} 'Committee' job(s) in the individual sheet to move.")

            # Step 3: Append these jobs to the committee DataFrame
            df_committee_updated = pd.concat([df_committee, committee_jobs_in_individual], ignore_index=True)

            # Step 4: Remove the moved jobs from the individual DataFrame
            df_individual_cleaned = df_individual.drop(committee_jobs_in_individual.index)

            # Step 5: Save both updated DataFrames to a new multi-sheet Excel file
            with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
                df_individual_cleaned.to_excel(writer, sheet_name=INDIVIDUAL_SHEET_NAME, index=False)
                df_committee_updated.to_excel(writer, sheet_name=COMMITTEE_SHEET_NAME, index=False)

            print(f"\nProcessing complete!")
            print(f" -> A new file named '{OUTPUT_EXCEL_FILE}' has been created with the sorted sheets.")
            print(f"    - '{INDIVIDUAL_SHEET_NAME}' sheet now has {len(df_individual_cleaned)} records.")
            print(f"    - '{COMMITTEE_SHEET_NAME}' sheet now has {len(df_committee_updated)} records.")

        else:
            print("\nNo jobs with 'Committee' in the author name were found in the individual sheet. No changes needed.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_EXCEL_FILE}' was not found.")
        print("Please make sure it's in the same directory as the script.")
    except ValueError as e:
        # This error happens if a sheet name is incorrect
        print(f"Error: {e}")
        print(f"Please ensure your Excel file contains sheets named '{INDIVIDUAL_SHEET_NAME}' and '{COMMITTEE_SHEET_NAME}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    sort_committee_jobs_in_excel()

