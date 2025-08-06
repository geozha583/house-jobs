import os
import json
import subprocess
import time
from typing import List, Dict
import re
import openai
import pandas as pd

def load_scores_from_excel(filepath: str) -> Dict:
    """
    Reads the provided Excel file and creates a lookup map for scores.
    The map's key is a formatted district string (e.g., "CA-51").
    """
    
    # --- ACTION REQUIRED: Replace these placeholder strings ---
    # --- with the actual column names from your Excel file. ---
    STATE_COL = 'Two-letter state code'  # <-- e.g., 'state' or 'st'
    DISTRICT_COL = 'Congressional district number' # <-- e.g., 'district' or 'cd'
    NOMINATE_COL = 'First-dimension DW-NOMINATE score' # This is likely correct
    LES_COL = 'LES 1.0' # This is likely correct
    CONGRESS_COL = 'Congress number' # <-- e.g., 'cong' or 'congress_num'
    # ---------------------------------------------------------

    scores_map = {}
    try:
        df = pd.read_excel(filepath)
        print(f"✅ Successfully loaded scores Excel file from '{filepath}'. Creating lookup map...")
        
        latest_congress = df[CONGRESS_COL].max()
        df_latest = df[df[CONGRESS_COL] == latest_congress]

        for index, row in df_latest.iterrows():
            state = row[STATE_COL]
            district_num = int(row[DISTRICT_COL])
            district_key = f"{state}-{district_num}"
            
            scores_map[district_key] = {
                "dw_nominate": row[NOMINATE_COL],
                "les": row[LES_COL]
            }
        print(f"🗺️ Lookup map created for {len(scores_map)} districts from congress {latest_congress}.")

    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at the path: {filepath}")
        return {}
    except KeyError as e:
        print(f"❌ ERROR: A required column was not found in the Excel file: {e}")
        print("Please check the placeholder column names in the 'load_scores_from_excel' function.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading scores: {e}")
        return {}
        
    return scores_map

def split_into_job_chunks(text: str) -> List[str]:
    """Splits the input text into chunks based on the 'MEM-' pattern."""
    chunks = re.split(r'(?=MEM-)', text)
    return [chunk.strip() for chunk in chunks[1:] if chunk.strip()]

def process_chunk(chunk: str, filename: str, scores_map: Dict) -> List[Dict]:
    """Processes a single text chunk using a LOCAL model and enriches it with scores."""
    client = openai.OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )
    
    system_prompt = """You are an expert at parsing congressional job listings into structured JSON. Extract details like the job's location, responsibilities, qualifications, and any equal opportunity statements. If a field is not present, use null. The output MUST be a valid JSON object."""
    
    user_prompt = f"""Create a JSON array of objects for each job listing with the following fields: Post_ID, Posting_Author, Congress_Number, "1 if Democrat", "1 if woman", State_District, Location, Job_Function, Title_Parsed, Office_Type, Committee_Affiliation, Responsibilities, Qualifications, Spanish_Language, Salary_Min, Salary_Max, Years_Experience, Skills_Mentioned, Equal_Opportunity_Information, Cleaned_Text.

    Here is the text to parse from the file '{filename}':
    ---
    {chunk}
    ---
    """

    try:
        response = client.chat.completions.create(
            model='llama3.1',
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            response_format={'type': 'json_object'}
        )
        parsed_jobs_str = response.choices[0].message.content
        parsed_json = json.loads(parsed_jobs_str)

        job_list = []
        if isinstance(parsed_json, dict):
            found = False
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    job_list = value
                    found = True
                    break
            if not found:
                 job_list = [parsed_json]
        elif isinstance(parsed_json, list):
            job_list = parsed_json

        final_jobs = []
        # Enrich data with scores from the map
        for job in job_list:
            # 💡 THIS IS THE FIX: Check if 'job' is a dictionary before processing it.
            if isinstance(job, dict):
                district_key = job.get("State_District")
                if district_key and district_key in scores_map:
                    job["DW-NOMINATE"] = scores_map[district_key].get("dw_nominate")
                    job["LES"] = scores_map[district_key].get("les")
                else:
                    job["DW-NOMINATE"] = None
                    job["LES"] = None
                final_jobs.append(job)
            else:
                # If the AI returned a plain string instead of a job object, we log it and skip.
                print(f"⚠️  Skipping an item because it was not a valid job object: {str(job)[:100]}...")
        
        return final_jobs

    except Exception as e:
        print(f"An error occurred in process_chunk: {e}")
        return []

def main():
    """Main function to orchestrate the parsing and enrichment of job listing files."""
    
    scores_filepath = "test.xlsx"
    scores_map = load_scores_from_excel(scores_filepath)

    if not scores_map:
        print("Halting script because the scores lookup map could not be created.")
        return

    directory_path = "output"
    output_dir = "json_pro_with_scores"
    os.makedirs(output_dir, exist_ok=True)
    processed_files = {f.replace('.json', '') for f in os.listdir(output_dir)}

    all_text_files = [f for f in os.listdir(directory_path) if 'Member' in f and f.endswith(".txt")]
    print(f"\nFound {len(all_text_files)} job files to process.")

    for filename in all_text_files:
        file_base = filename.replace('.txt', '')
        if file_base not in processed_files:
            print(f"\nProcessing {filename}...")
            
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                text_string = f.read()
            
            chunks = split_into_job_chunks(text_string)
            all_jobs = []
            
            for i, chunk in enumerate(chunks):
                print(f"  - Processing chunk {i+1}/{len(chunks)}")
                time.sleep(1) 
                
                jobs = process_chunk(chunk, filename, scores_map)
                
                if jobs:
                    all_jobs.extend(jobs)
            
            output_path = os.path.join(output_dir, f"{file_base}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_jobs, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Finished processing {filename}. Output saved to {output_path}")

if __name__ == "__main__":
    main()