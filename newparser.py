import os
import re
import json
import pandas as pd
import ollama
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('punkt')

# --- Configuration for Score File ---
SCORES_FILE_PATH = 'test.xlsx' 

# --- Column Names from your file ---
MEMBER_NAME_COL = 'Legislator name, as given in THOMAS'
NOMINATE_COL = 'First-dimension DW-NOMINATE score'
LES_COL = 'LES 1.0'
STATE_COL = 'Two-letter state code'
DISTRICT_COL = 'Congressional district number'
GENDER_COL = '1 = female' 
# Congress Number is now derived from the filename, so it's removed from here.

# --- NEW FUNCTION: Determines Congress Number from filename ---
def get_congress_from_filename(filename):
    """
    Extracts a date from a filename, calculates the corresponding Congress number.
    Handles multiple date formats like YYYY_MM_DD, MM.DD.YYYY, and MM-DD-YY.
    """
    # Regex for YYYY-MM-DD, YYYY_MM_DD, YYYY.MM.DD
    match = re.search(r'(\d{4})[._-](\d{1,2})[._-](\d{1,2})', filename)
    if match:
        year = int(match.group(1))
        # Congress sessions begin Jan 3 of odd years. Formula: (Year - 1789) / 2 + 1
        return int((year - 1789) / 2) + 1

    # Regex for MM-DD-YYYY, MM.DD.YYYY, MM_DD_YYYY
    match = re.search(r'(\d{1,2})[._-](\d{1,2})[._-](\d{4})', filename)
    if match:
        year = int(match.group(3))
        return int((year - 1789) / 2) + 1

    # Regex for MM-DD-YY, MM.DD.YY, MM_DD_YY
    match = re.search(r'(\d{1,2})[._-](\d{1,2})[._-](\d{2})', filename)
    if match:
        year_short = int(match.group(3))
        # Assumes years > 50 are 19xx and years < 50 are 20xx
        year = 2000 + year_short if year_short < 50 else 1900 + year_short
        return int((year - 1789) / 2) + 1

    return None # Return None if no date pattern is found

def load_scores(file_path):
    """
    Loads scores and other data from the provided Excel file.
    """
    scores_lookup = {}
    try:
        df_scores = pd.read_excel(file_path)
        
        # MODIFIED: Removed Congress number from required columns
        required_cols = [MEMBER_NAME_COL, NOMINATE_COL, LES_COL, STATE_COL, DISTRICT_COL, GENDER_COL]
        if not all(col in df_scores.columns for col in required_cols):
            print(f"Error: Your Excel file is missing one or more required columns.")
            print(f"Required: {required_cols}")
            print(f"Found in file: {list(df_scores.columns)}")
            return None

        for _, row in df_scores.iterrows():
            name_raw = row[MEMBER_NAME_COL]
            if isinstance(name_raw, str):
                name_parts = name_raw.split(',')
                last_name = name_parts[0].strip().lower()
                first_name = name_parts[1].strip().lower() if len(name_parts) > 1 else ""
                standard_name = f"{first_name} {last_name}"
                
                # MODIFIED: Removed Congress number from lookup dictionary
                scores_lookup[standard_name] = {
                    "DW-NOMINATE": row[NOMINATE_COL],
                    "LES": row[LES_COL],
                    "State": row[STATE_COL],
                    "District": row[DISTRICT_COL],
                    "Gender": row[GENDER_COL],
                }
        print(f"Successfully loaded and processed {len(scores_lookup)} records from the scores file.")
        return scores_lookup
    except FileNotFoundError:
        print(f"Error: Scores file not found at '{file_path}'.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the scores file: {e}")
        return None

def clean_posting_text(text):
    """
    Cleans extracted PDF text by fixing common encoding artifacts and re-flowing sentence breaks.
    """
    if not isinstance(text, str):
        return ""
    replacements = {'“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '™': "'", 'ﬁ': 'fi', 'ﬂ': 'fl'}
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)
    try:
        text = text.encode('latin-1').decode('utf-8', 'ignore')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = re.sub(r'(\w)-(\s*\n\s*)', r'\1', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def remove_stopwords(text):
    """Removes common English stopwords from a given text."""
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words and w.isalpha()]
    return " ".join(filtered_text)

def find_state_in_text(text):
    states = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
    text_to_search = re.sub(r'Washington,\s*D\.\s*C\.', 'WashingtonDC', text, flags=re.IGNORECASE)
    state_pattern = r'\b(' + '|'.join(states.keys()) + r')\b'
    match = re.search(state_pattern, text_to_search, re.IGNORECASE)
    if match:
        return states[match.group(1).title()]
    return None

def clean_display_name(name_str):
    if not isinstance(name_str, str): return ""
    name_str = re.sub(r'^(Congresswoman|Congressman|Rep\.|Representative)\s*', '', name_str, flags=re.IGNORECASE)
    return name_str.split('(')[0].strip()

def standardize_author_name(name_str):
    if not isinstance(name_str, str): return ""
    name_str = re.sub(r'^(Congresswoman|Congressman|Rep\.|Representative)\s*', '', name_str, flags=re.IGNORECASE)
    return name_str.split('(')[0].strip().lower()

def parse_job_listing_with_ollama(text):
    prompt = f"""
    You are an expert data extraction tool. Analyze the following job listing text and extract the specified information.
    Your output MUST be a single, clean JSON object. Do not include any other text, explanations, or markdown formatting.
    **JSON Fields to Extract:**
    - "Post_ID": The MEM-ID.
    - "Posting_Author": The name of the Congressperson or Committee.
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
    all_final_data = []
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder '{folder_path}' not found.")
        return pd.DataFrame()

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF file(s) to process.")

    for filename in pdf_files:
        print(f"Processing file: {filename}...")
        
        # --- MODIFIED: Get Congress number from the filename ---
        congress_number = get_congress_from_filename(filename)
        if congress_number is None:
            print(f"  -> Warning: Could not determine Congress number from filename '{filename}'. It will be blank.")

        file_path = os.path.join(folder_path, filename)

        try:
            reader = PdfReader(file_path)
            full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            split_text = re.split(r'(MEM-\d{3}-\d{2})', full_text)
            job_listings = [d + t for d, t in zip(split_text[1::2], split_text[2::2])]
            print(f"Found {len(job_listings)} potential listings.")

            for listing_text in job_listings:
                cleaned_text = clean_posting_text(listing_text)
                parsed_data = parse_job_listing_with_ollama(cleaned_text)

                if parsed_data:
                    # --- MODIFIED: Assign the Congress number derived from the filename ---
                    parsed_data['Congress_Number'] = congress_number
                    parsed_data['Posting_Text'] = remove_stopwords(cleaned_text)
                    raw_author = parsed_data.get("Posting_Author", "")
                    parsed_data['Posting_Author'] = clean_display_name(raw_author)
                    author_lookup_key = standardize_author_name(raw_author)
                    
                    final_state_district = parsed_data.get('State_District', 'ST-XX')
                    
                    member_info = scores_lookup.get(author_lookup_key) if scores_lookup else None
                    if member_info:
                        parsed_data.update(member_info)
                        state, district = member_info.get('State'), member_info.get('District')
                        if pd.notna(state) and pd.notna(district):
                            dist_str = f"{int(district):02d}" if int(district) > 0 else "00"
                            final_state_district = f"{state}-{dist_str}"
                    else:
                        # Fallback logic for committees (Gender is the only remaining field from lookup)
                        parsed_data['Gender'] = None
                        
                        found_state = find_state_in_text(cleaned_text)
                        if found_state:
                            final_state_district = f"{found_state}-00"
                            parsed_data['State'] = found_state
                            parsed_data['District'] = 0
                        else:
                            state_part = final_state_district.split('-')[0]
                            final_state_district = f"{state_part}-00"
                            parsed_data['State'] = state_part if state_part not in ['ST', ''] else None
                            parsed_data['District'] = 0

                    parsed_data['State_District'] = final_state_district
                    
                    is_democrat = 0
                    if member_info and member_info.get("DW-NOMINATE") is not None and member_info["DW-NOMINATE"] < 0:
                        is_democrat = 1
                    else:
                        search_text = " ".join(str(s) for s in [
                            raw_author, parsed_data.get('Committee_Affiliation', ''), cleaned_text
                        ]).lower()
                        if re.search(r'\b(democrat(ic)?|dem|d)\b', search_text):
                            is_democrat = 1
                    parsed_data["1 if Democrat"] = is_democrat
                    all_final_data.append(parsed_data)
        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")
    return pd.DataFrame(all_final_data)

# --- Main Execution ---
if __name__ == "__main__":
    scores_data = load_scores(SCORES_FILE_PATH)
    
    if scores_data is None:
        print("\nCould not load the scores file. Please check the file path and column names. Exiting.")
    else:
        input_folder = "input"
        final_dataset = process_all_pdfs_in_folder(input_folder, scores_data)

        if not final_dataset.empty:
            print(f"\nSuccessfully processed all files. Created a dataset with {len(final_dataset)} listings.")
            
            column_order = [
                'Post_ID', 'Posting_Author', 'Congress_Number', 'State_District', 'State', 'District',
                '1 if Democrat', 'Gender', 'DW-NOMINATE', 'LES', 'Job_Function', 'Title_Parsed', 'Office_Type', 
                'Committee_Affiliation', 'Spanish_Language', 'Salary_Min', 'Salary_Max', 
                'Years_Experience', 'Posting_Text'
            ]
            
            final_dataset = final_dataset.reindex(columns=[col for col in column_order if col in final_dataset.columns])
            
            print("\n--- Sample of Final Combined and Enriched Dataset ---")
            sample_cols = [col for col in column_order if col != 'Posting_Text']
            print(final_dataset[sample_cols].head().to_markdown(index=False))
            
            output_filename = "enriched_job_listings_final.csv"
            final_dataset.to_csv(output_filename, index=False, float_format='%.3f')
            print(f"\nFull dataset saved to '{output_filename}'")
        else:
            print("\nNo data was successfully parsed. No output file created.")