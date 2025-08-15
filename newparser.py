import os
import re
import json
import pandas as pd
import ollama
from pypdf import PdfReader

# --- Configuration for Score File ---
# Correctly pointing to the Excel file
SCORES_FILE_PATH = 'test.xlsx' 
NOMINATE_COL = 'First-dimension DW-NOMINATE score'
LES_COL = 'LES 1.0'
MEMBER_NAME_COL = 'Legislator name, as given in THOMAS'


def load_scores(file_path):
    """Loads the scores from the provided Excel file and creates a lookup dictionary."""
    scores_lookup = {}
    try:
        # CORRECTED: Using pd.read_excel for .xlsx files
        df_scores = pd.read_excel(file_path)
        
        required_cols = [MEMBER_NAME_COL, NOMINATE_COL, LES_COL]
        if not all(col in df_scores.columns for col in required_cols):
            print(f"Error: Scores file is missing one or more required columns: {required_cols}")
            return None

        for _, row in df_scores.iterrows():
            name_raw = row[MEMBER_NAME_COL]
            if isinstance(name_raw, str):
                name_parts = name_raw.split(',')
                last_name = name_parts[0].strip().lower()
                first_name = name_parts[1].strip().lower() if len(name_parts) > 1 else ""
                standard_name = f"{first_name} {last_name}"
                
                scores_lookup[standard_name] = {
                    "DW-NOMINATE": row[NOMINATE_COL],
                    "LES": row[LES_COL]
                }
        print(f"Successfully loaded and processed {len(scores_lookup)} records from the scores file.")
        return scores_lookup
        
    except FileNotFoundError:
        print(f"Error: Scores file not found at '{file_path}'. Score fields will be empty.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the scores file: {e}")
        return None

def standardize_author_name(name_str):
    """
    Standardizes a name from the job post (e.g., "Congressman Steven Horsford")
    to a common format ("steven horsford") for lookup.
    """
    if not isinstance(name_str, str):
        return ""
    
    name_str = re.sub(r'^(Congresswoman|Congressman|Rep\.|Representative)\s*', '', name_str, flags=re.IGNORECASE)
    name_str = name_str.split('(')[0].strip() # Handle cases like "Jayapal (WA-07)"
    
    return name_str.lower()


def parse_job_listing_with_ollama(text, congress_number=118):
    """
    Uses a local Ollama model to parse a job listing.
    """
    prompt = f"""
    You are an expert data extraction tool. Analyze the following job listing text and extract the specified information.
    Your output MUST be a single, clean JSON object. Do not include any other text, explanations, or markdown formatting.

    **JSON Fields to Extract:**
    - "Post_ID": The MEM-ID.
    - "Posting_Author": The name of the Congressperson or Committee.
    - "Congress_Number": The Congress number, which is {congress_number}.
    - "State_District": The state and district number, standardized to "ST-XX".
    - "Job_Function": The primary role (e.g., "Legislative", "Communications").
    - "Title_Parsed": The specific job title.
    - "Office_Type": "Personal" or "Committee".
    - "Committee_Affiliation": The full committee name, otherwise null.
    - "Spanish_Language": 1 if Spanish is mentioned, otherwise 0.
    - "Salary_Min": The minimum salary as an integer, otherwise null.
    - "Salary_Max": The maximum salary as an integer, otherwise null.
    - "Years_Experience": Minimum years of experience as an integer, otherwise null.

    **Job Listing Text:**
    ---
    {text}
    ---
    """
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        json_string = response['message']['content'].strip()
        if json_string.startswith("```json"):
            json_string = json_string[7:-3].strip()
        return json.loads(json_string)
    except Exception:
        return None

def process_all_pdfs_in_folder(folder_path, scores_lookup):
    """
    Processes all PDFs, parses them, and enriches the data with scores.
    """
    all_final_data = []
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder '{folder_path}' not found.")
        return pd.DataFrame()

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF file(s) to process.")

    for filename in pdf_files:
        print(f"Processing file: {filename}...")
        file_path = os.path.join(folder_path, filename)
        
        try:
            reader = PdfReader(file_path)
            full_text = "".join(page.extract_text() for page in reader.pages)
            split_text = re.split(r'(MEM-\d{3}-\d{2})', full_text)
            job_listings = [d + t for d, t in zip(split_text[1::2], split_text[2::2])]
            print(f"Found {len(job_listings)} potential listings.")

            for listing_text in job_listings:
                parsed_data = parse_job_listing_with_ollama(listing_text)
                
                if parsed_data:
                    author_name = standardize_author_name(parsed_data.get("Posting_Author", ""))
                    
                    if scores_lookup and author_name in scores_lookup:
                        parsed_data["DW-NOMINATE"] = scores_lookup[author_name]["DW-NOMINATE"]
                        parsed_data["LES"] = scores_lookup[author_name]["LES"]
                    else:
                        parsed_data["DW-NOMINATE"] = None
                        parsed_data["LES"] = None
                    
                    all_final_data.append(parsed_data)
        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")

    return pd.DataFrame(all_final_data)

# --- Main Execution ---
if __name__ == "__main__":
    scores_data = load_scores(SCORES_FILE_PATH)
    
    input_folder = "input"
    final_dataset = process_all_pdfs_in_folder(input_folder, scores_data)

    if not final_dataset.empty:
        print(f"\nSuccessfully processed all files. Created a dataset with {len(final_dataset)} listings.")
        
        column_order = [
            'Post_ID', 'Posting_Author', 'Congress_Number', 'State_District', 'DW-NOMINATE', 'LES',
            'Job_Function', 'Title_Parsed', 'Office_Type', 'Committee_Affiliation', 'Spanish_Language',
            'Salary_Min', 'Salary_Max', 'Years_Experience'
        ]
        final_dataset = final_dataset.reindex(columns=[col for col in column_order if col in final_dataset.columns])

        print("\n--- Sample of Final Combined and Enriched Dataset ---")
        print(final_dataset.head().to_markdown(index=False))
        
        output_filename = "enriched_job_listings.csv"
        final_dataset.to_csv(output_filename, index=False, float_format='%.3f')
        print(f"\nFull dataset saved to '{output_filename}'")
    else:
        print("\nNo data was successfully parsed. No output file created.")