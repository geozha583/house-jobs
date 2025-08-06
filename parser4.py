import os
import json
import time
from typing import List, Dict, Optional, Any
import re
import openai
import pandas as pd
from pydantic import BaseModel, ValidationError

# --- Pydantic Schema (Unchanged) ---
class Job(BaseModel):
    Post_ID: str
    Posting_Author: Optional[str] = None
    Congress_Number: Optional[int] = None
    State_District: Optional[str] = None
    Location: Optional[str] = None
    Job_Function: Optional[str] = None
    Title_Parsed: Optional[str] = None
    Office_Type: Optional[str] = None
    Responsibilities: Optional[List[str]] = None
    Qualifications: Optional[List[str]] = None
    Salary_Min: Optional[int] = None
    Salary_Max: Optional[int] = None
    Skills_Mentioned: Optional[List[str]] = None
    Equal_Opportunity_Information: Optional[str] = None
    Cleaned_Text: str
    DW_NOMINATE: Optional[float] = None
    LES: Optional[float] = None


# --- Helper Functions (Unchanged) ---
def load_scores_from_excel(filepath: str) -> Dict:
    """Reads the Excel file and creates a lookup map for scores."""
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
        for _, row in df_latest.iterrows():
            district_key = f"{row[STATE_COL]}-{int(row[DISTRICT_COL])}"
            scores_map[district_key] = {"dw_nominate": row[NOMINATE_COL], "les": row[LES_COL]}
        print(f"🗺️ Lookup map created for {len(scores_map)} districts.")
    except Exception as e:
        print(f"❌ ERROR loading scores file: {e}")
    return scores_map

def split_into_job_chunks(text: str) -> List[str]:
    """Splits the input text into chunks based on the 'MEM-' pattern."""
    chunks = re.split(r'(?=MEM-)', text)
    return [chunk.strip() for chunk in chunks[1:] if chunk.strip()]

def find_job_objects(data: Any) -> List[Dict]:
    """
    Recursively searches a nested data structure to find dictionaries that look like Job objects.
    """
    found_jobs = []
    if isinstance(data, dict):
        if 'Post_ID' in data:
            return [data]
        for key, value in data.items():
            found_jobs.extend(find_job_objects(value))
    elif isinstance(data, list):
        for item in data:
            found_jobs.extend(find_job_objects(item))
    return found_jobs


# --- process_chunk function with the FIX ---
def process_chunk(chunk: str, filename: str, scores_map: Dict) -> List[Job]:
    """
    Processes a text chunk with robust, item-level validation and retries.
    """
    client = openai.OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    system_prompt = "You are an expert parser creating structured JSON. You must include all fields from the provided schema in your JSON output. If, after double-checking, a value for an optional field truly cannot be found, you must explicitly use `null` as the value."
    user_prompt = f"""Parse the following job listing text into a JSON array of objects, where each object strictly conforms to the Job Schema. Ensure every field is present, using `null` for any missing optional values.

    Job Schema: {{ "Post_ID": "string", "Posting_Author": "string or null", "Congress_Number": "integer or null", "State_District": "string or null", "Location": "string or null", "Job_Function": "string or null", "Title_Parsed": "string or null", "Office_Type": "string or null", "Responsibilities": "array of strings or null", "Qualifications": "array of strings or null", "Salary_Min": "integer or null", "Salary_Max": "integer or null", "Skills_Mentioned": "array of strings or null", "Equal_Opportunity_Information": "string or null", "Cleaned_Text": "string" }}

    Text to parse from file '{filename}':
    ---
    {chunk}
    ---
    """
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model='llama3.2',
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
                response_format={'type': 'json_object'}
            )
            response_text = response.choices[0].message.content
            data = json.loads(response_text)
            
            job_data_list = find_job_objects(data)
            
            if not job_data_list:
                raise ValueError("Could not find any valid job objects in the LLM response.")

            # --- 💡 FIX: Using model_validate instead of the deprecated parse_obj ---
            validated_jobs = [Job.model_validate(item) for item in job_data_list]
            
            # Enrich with scores if validation is successful
            for job in validated_jobs:
                if job.State_District and job.State_District in scores_map:
                    job.DW_NOMINATE = scores_map[job.State_District].get("dw_nominate")
                    job.LES = scores_map[job.State_District].get("les")
            
            print(f"  ✅  Successfully parsed and validated chunk.")
            return validated_jobs

        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            print(f"  ⚠️  Parsing Error on attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
            if not isinstance(e, json.JSONDecodeError):
                 print(f"     Details: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  ❌ An unexpected error occurred in process_chunk: {e}")
            return []

    print("  ❌ Failed to process chunk after multiple retries.")
    return []


# --- Main Function (Unchanged) ---
def main():
    """Main function to orchestrate the parsing and enrichment of job listing files."""
    scores_filepath = "test.xlsx"
    scores_map = load_scores_from_excel(scores_filepath)

    if not scores_map:
        print("Halting script because the scores lookup map could not be created.")
        return

    directory_path = "output"
    output_dir = "json_pro_with_scores_validated"
    os.makedirs(output_dir, exist_ok=True)
    
    all_text_files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    print(f"\nFound {len(all_text_files)} job files to process.")

    for filename in all_text_files:
        print(f"\nProcessing {filename}...")
        with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
            text_string = f.read()
        
        chunks = split_into_job_chunks(text_string)
        all_jobs: List[Job] = []
        
        for i, chunk in enumerate(chunks):
            print(f"  - Processing chunk {i+1}/{len(chunks)}")
            jobs = process_chunk(chunk, filename, scores_map)
            if jobs:
                all_jobs.extend(jobs)
        
        output_path = os.path.join(output_dir, filename.replace('.txt', '.json'))
        output_data = [job.model_dump() for job in all_jobs]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Finished processing {filename}. Output saved to {output_path}")

if __name__ == "__main__":
    main()