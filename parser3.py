import os
import json
import subprocess
import time
from typing import List, Dict
import re
import openai

def split_into_job_chunks(text: str) -> List[str]:
    """Splits the input text into chunks based on the 'MEM-' pattern, keeping each job listing intact."""
    # Use a regular expression to split the text while keeping the delimiter 'MEM-'
    chunks = re.split(r'(?=MEM-)', text)
    # The first element is often empty, so it's sliced off. Each chunk is stripped of whitespace.
    return [chunk.strip() for chunk in chunks[1:] if chunk.strip()]

def process_chunk(chunk: str, filename: str) -> List[Dict]:
    """Processes a single text chunk using a LOCAL model via Ollama."""

    # Point the client to your local Ollama server
    # Ollama provides an OpenAI-compatible API
    client = openai.OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but can be any string
    )

    # The system prompt remains the same, instructing the model on its task
    system_prompt = """You are an expert at parsing congressional job listings. Your task is to extract job information into a structured JSON array.
    - Use the filename provided at the top of the text to determine the `Date_Posted`.
    - Derive `Posting_Author` from the text, often found near the position title.
    - `Post_ID` is the "MEM-XXX-XX" identifier.
    - Split the full position title into `Job_Function` (e.g., Legislative, Communications) and `Title_Parsed` (e.g., Legislative Director, Press Secretary).
    - Convert all dates to ISO 8601 format (YYYY-MM-DD).
    - Represent lists (like responsibilities or qualifications) as JSON arrays.
    - If a field is not present, use null.
    - The output MUST be a JSON object that starts with `[` and ends with `]`.
    """

    # The user prompt contains the schema and the raw text
    user_prompt = f"""Create a JSON array with objects for each job listing. Each object must have the following fields:
    - Post_ID
    - Posting_Author
    - Congress_Number
    - "1 if Democrat"
    - "1 if woman"
    - "DW-NOMINATE"
    - LES
    - Date_Posted
    - State_District
    - Job_Function
    - Title_Parsed
    - Office_Type
    - Committee_Affiliation
    - Spanish_Language
    - Salary_Min
    - Salary_Max
    - Years_Experience
    - Skills_Mentioned
    - Cleaned_Text

    Here is the text to parse, which comes from the file '{filename}':
    ---
    {chunk}
    ---
    """

    try:
        response = client.chat.completions.create(
            model='llama3.2', # The model you downloaded with Ollama
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            # Enforce JSON output
            response_format={'type': 'json_object'}
        )

        # The JSON content is in the response message
        parsed_jobs_str = response.choices[0].message.content
        # The model might return a JSON object with a key, so we need to find the array.
        # This logic might need adjustment based on how the local model formats its response.
        parsed_json = json.loads(parsed_jobs_str)

        # Find the actual list of jobs within the returned JSON
        for key, value in parsed_json.items():
            if isinstance(value, list):
                return value # Return the first list found

        # If no list is found, return the parsed object in a list (for single job chunks)
        return [parsed_json]


    except openai.APIConnectionError as e:
        print("Connection Error: Is the Ollama server running?")
        print(f"Error details: {e.__cause__}")
        return []
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {je}")
        print(f"Raw output that failed to parse: {parsed_jobs_str}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
    
def main():
    """Main function to orchestrate the parsing of job listing files."""
    directory_path = "output"
    output_dir = "json_gemini_pro"
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of files that have already been processed to avoid redundant work.
    processed_files = {f.replace('.json', '') for f in os.listdir(output_dir)}

    # Iterate over text files in the source directory.
    for filename in os.listdir(directory_path):
        if 'Member' in filename and filename.endswith(".txt"):
            file_base = filename.replace('.txt', '')
            if file_base not in processed_files:
                print(f"Processing {filename}...")
                
                with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                    text_string = f.read()
                
                # The file content is split into chunks, each representing a job listing.
                chunks = split_into_job_chunks(text_string)
                all_jobs = []
                
                # Each chunk is processed individually. A delay is added to avoid overwhelming the API.
                for i, chunk in enumerate(chunks):
                    print(f"  - Processing chunk {i+1}/{len(chunks)}")
                    time.sleep(5)  # Rate limiting
                    jobs = process_chunk(chunk, filename)
                    if jobs:
                        all_jobs.extend(jobs)
                
                # The extracted data is written to a new JSON file.
                output_path = os.path.join(output_dir, f"{file_base}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_jobs, f, indent=4, ensure_ascii=False)
                
                print(f"Finished processing {filename}. Output saved to {output_path}")

if __name__ == "__main__":
    main()