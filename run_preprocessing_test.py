import json
import os
import time
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

# Print diagnostic at start
print("[DIAG] Running run_preprocessing_test.py...", file=sys.stderr)

# --- Configuration ---
INPUT_FILE = "input files/600_son_converted.json"
OUTPUT_FILE = "preprocessing_output.json" # Define output file
BATCH_SIZE = 10
SPEAKER_TO_PROCESS = "student"

# --- Environment Setup ---
# Load environment variables from .env file
# Specify the path to the .env file explicitly if it's not in the current working directory
loaded_dotenv = load_dotenv(verbose=True) # Rely on default search 

# --- Configure Logging (after dotenv, before other imports/logic) ---
logging.basicConfig(
    level=logging.DEBUG, # Set the desired logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Example format
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get a logger for this script if needed

# --- Imports (after dotenv) ---
# It's crucial to import modules using the API key *after* loading the .env file
from analyzer_service.schemas import InputUtterance, PreprocessingOutput, PreprocessedASUnit
from analyzer_service.chains import get_preprocessing_chain

# --- Helper Functions ---
def load_and_filter_utterances(filepath: str, speaker: str) -> List[InputUtterance]:
    """Loads utterances from JSON and filters by speaker."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

    student_utterances = []
    utterance_list = data.get('utterances', [])
    if not isinstance(utterance_list, list):
        print(f"Error: 'utterances' key does not contain a list in {filepath}")
        return []
        
    for item in utterance_list: 
        if isinstance(item, dict) and item.get("speaker") == speaker:
            try:
                student_utterances.append(InputUtterance(**item))
            except Exception as e:
                print(f"Warning: Skipping item due to validation error: {item}. Error: {e}")

    print(f"Loaded {len(data)} items, filtered {len(student_utterances)} utterances for speaker '{speaker}'.")
    return student_utterances

def create_batches(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Creates batches from a list of items."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# --- Main Test Logic ---
if __name__ == "__main__":
    print("Starting preprocessing chain test...")

    # 1. Load and filter data
    utterances = load_and_filter_utterances(INPUT_FILE, SPEAKER_TO_PROCESS)

    if not utterances:
        print("No utterances to process. Exiting.")
        exit()

    # 2. Create batches
    batches = create_batches(utterances, BATCH_SIZE)
    print(f"Created {len(batches)} batches of size up to {BATCH_SIZE}.")

    if not batches:
        print("No batches created. Exiting.")
        exit()

    # 4. Get the preprocessing chain
    # Ensure API key is loaded if chain requires it implicitly
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables. Please ensure it's set in your .env file.")
        exit()

    try:
        preprocessing_chain = get_preprocessing_chain()
        print("Successfully obtained preprocessing chain.")
    except Exception as e:
        print(f"Error creating preprocessing chain: {e}")
        exit()

    # 5. Run the chain on ALL batches and collect results
    all_results: List[PreprocessingOutput] = [] 
    start_time = time.time()
    print(f"Processing {len(utterances)} utterances in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        print(f"  Processing batch {i+1}/{len(batches)} ({len(batch)} items)...", end="", flush=True)
        batch_start_time = time.time()
        try:
            # Assuming batch returns List[PreprocessingOutput], one for each input utterance
            batch_results_raw: List[PreprocessingOutput] = preprocessing_chain.batch(batch, {"max_concurrency": 5}) 

            # --- Filtering Step --- 
            filtered_batch_outputs = []
            for output_obj in batch_results_raw:
                # Filter the processed_utterances list within this PreprocessingOutput object
                original_units = getattr(output_obj, 'processed_utterances', []) # Safety check
                filtered_units_for_utterance = []
                if isinstance(original_units, list):
                    for as_unit in original_units:
                        if isinstance(as_unit, PreprocessedASUnit) and as_unit.as_unit_text and len(as_unit.as_unit_text.split()) >= 3:
                            filtered_units_for_utterance.append(as_unit)
                        else:
                            # Log the filtered out unit
                            unit_id = getattr(as_unit, 'as_unit_id', 'UNKNOWN_ID')
                            unit_text = getattr(as_unit, 'as_unit_text', 'UNKNOWN_TEXT')
                            logger.info(f"Filtering out AS unit {unit_id} due to word count < 3: '{unit_text}'")
                
                # Only keep the PreprocessingOutput object if it still has units after filtering
                if filtered_units_for_utterance:
                    # Update the object with the filtered list
                    output_obj.processed_utterances = filtered_units_for_utterance 
                    filtered_batch_outputs.append(output_obj)
                else:
                    # Log that the entire utterance output is being dropped
                    # We need the original utterance ID if available in PreprocessingOutput
                    # Assuming PreprocessingOutput might not directly hold original_utterance_id
                    # Let's log based on the first filtered unit's ID if possible, or just indicate removal
                    first_original_unit_id = getattr(original_units[0], 'original_utterance_id', 'UNKNOWN_UTTERANCE') if original_units else 'UNKNOWN_UTTERANCE'
                    logger.info(f"Dropping output for utterance (originally {first_original_unit_id}) as all AS units were filtered.")
            
            # Append the filtered PreprocessingOutput objects for this batch to the main list
            all_results.extend(filtered_batch_outputs) 
            # --- End Filtering Step --- 

            batch_end_time = time.time()
            print(f" done in {batch_end_time - batch_start_time:.2f}s")

        except Exception as e:
            print(f"\nError processing batch {i+1}: {e}")
            # continue

    end_time = time.time()
    print(f"Finished processing all batches in {end_time - start_time:.2f} seconds.")

    # 6. Write ALL collected and filtered results to the output file
    print(f"Writing {len(all_results)} filtered utterance results to {OUTPUT_FILE}...") # Corrected description
    try:
        # Convert Pydantic models to dicts for JSON serialization
        output_data = [result.model_dump() for result in all_results]
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully wrote results to file.")
    except Exception as e:
        print(f"Error writing results to {OUTPUT_FILE}: {e}")

    print("Preprocessing chain test finished.")
