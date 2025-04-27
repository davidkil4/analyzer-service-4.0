import json
import os
import time
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging

# Print diagnostic at start
print("[DIAG] Running run_preprocessing_test.py...", file=sys.stderr)

# --- Configuration ---
INPUT_FILE = "input_files/600_son_converted.json"
OUTPUT_FILE = "preprocessing_output.json" # Define output file
BATCH_SIZE = 10
TARGET_SPEAKER = "student"

# --- Environment Setup ---
# Load environment variables from .env file
# Specify the path to the .env file explicitly if it's not in the current working directory
loaded_dotenv = load_dotenv(verbose=True, override=True) # FORCE OVERRIDE

# --- Print loaded key for verification ---
# loaded_key = os.getenv("GOOGLE_API_KEY")
# print(f"[DIAG] Loaded GOOGLE_API_KEY: {'*****' + loaded_key[-4:] if loaded_key else 'Not Found!'}", file=sys.stderr)
# --- End verification ---

# --- Configure Logging (after dotenv, before other imports/logic) ---
logging.basicConfig(
    level=logging.DEBUG, # Set the desired logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Example format
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get a logger for this script if needed

# --- Imports (after dotenv) ---
# It's crucial to import modules using the API key *after* loading the .env file
from analyzer_service.schemas import InputUtterance, PreprocessingOutput, PreprocessedASUnit, ContextUtterance, AnalysisInputItem
from analyzer_service.preprocessing_chains import get_preprocessing_chain

# --- Helper Functions ---
def load_and_filter_utterances(filepath: str) -> List[Dict]:
    """Loads utterances from JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_json_data: Dict = json.load(f) # Load the whole JSON object
        
        # Get the list of utterances from the loaded data
        all_utterances_list: List[Dict] = loaded_json_data.get('utterances', [])
        if not all_utterances_list:
             logger.warning(f"No 'utterances' list found or list is empty in {filepath}")
             return []

        # Create map for context lookup from the list
        utterance_map = {utt['utterance_id']: utt for utt in all_utterances_list if 'utterance_id' in utt}

        # Filter for the target speaker from the list
        student_utterances = [
            utt for utt in all_utterances_list 
            if utt.get("speaker") == TARGET_SPEAKER and utt.get("text")
        ]
        logger.info(f"Loaded {len(all_utterances_list)} total utterances, filtered {len(student_utterances)} utterances for speaker '{TARGET_SPEAKER}'.")

        if not student_utterances:
            return []

        return student_utterances

    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

def create_batches(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Creates batches from a list of items."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def find_context(target_utterance_id: str, utterance_map: Dict[str, Dict], all_utterances: List[Dict]) -> Optional[List[ContextUtterance]]:
    """Finds the 3 utterances preceding the target utterance."""
    try:
        # Find the index of the target utterance in the original list
        target_index = -1
        for i, utt in enumerate(all_utterances):
            if utt.get('id') == target_utterance_id: # <<< Use 'id' key here
                target_index = i
                break

        if target_index == -1:
            logger.warning(f"Could not find original index for utterance_id: {target_utterance_id}")
            return []

        # Determine the start index for the context (up to 3 previous)
        start_index = max(0, target_index - 3)
        
        # Extract the preceding utterances
        context_utterances_raw = all_utterances[start_index:target_index]
        
        # Format into ContextUtterance objects
        context = [
            ContextUtterance(speaker=utt.get('speaker', 'UNKNOWN'), text=utt.get('text', ''))
            for utt in context_utterances_raw
        ]
        return context
    except Exception as e:
        logger.error(f"Error finding context for {target_utterance_id}: {e}")
        return []

# --- Main Test Logic ---
def main():
    logger.info("Starting preprocessing chain test...")

    # --- Load and Prepare Data ---
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            loaded_json_data: Dict = json.load(f) # Load the whole JSON object
        
        # Get the list of utterances from the loaded data
        all_utterances_list: List[Dict] = loaded_json_data.get('utterances', [])
        if not all_utterances_list:
             logger.warning(f"No 'utterances' list found or list is empty in {INPUT_FILE}")
             return
        logger.info(f"Loaded {len(all_utterances_list)} total utterances from the 'utterances' list.")

        # Create map for context lookup from the full list
        utterance_map = {utt['utterance_id']: utt for utt in all_utterances_list if 'utterance_id' in utt}

        # Filter for the target speaker from the full list, using the correct key 'text'
        student_utterances_dicts = [
            utt for utt in all_utterances_list 
            if utt.get("speaker") == TARGET_SPEAKER and utt.get("text") # Use 'text' key
        ]
        logger.info(f"Filtered {len(student_utterances_dicts)} utterances for speaker '{TARGET_SPEAKER}'.")

        if not student_utterances_dicts:
            logger.warning(f"No utterances found for speaker '{TARGET_SPEAKER}' with non-empty text. Exiting.")
            return

        # Convert filtered raw dicts to InputUtterance objects for the chain, using 'text'
        input_utterances_for_chain = [
            InputUtterance(
                id=utt.get("id", f"unknown_{i}"), 
                speaker=utt.get("speaker"), 
                text=utt.get("text") # Use 'text' key
            ) for i, utt in enumerate(student_utterances_dicts) # Use the filtered list
        ]

        # Create batches from the filtered chain inputs
        batches = [input_utterances_for_chain[i:i + BATCH_SIZE] for i in range(0, len(input_utterances_for_chain), BATCH_SIZE)]
        logger.info(f"Created {len(batches)} batches of size up to {BATCH_SIZE}.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_FILE}")
        return []

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
    all_results: List[List[PreprocessedASUnit]] = [] 
    start_time = time.time()
    print(f"Processing {len(input_utterances_for_chain)} utterances in {len(batches)} batches...")

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
            all_results.extend([output_obj.processed_utterances for output_obj in filtered_batch_outputs]) 

        except Exception as e:
            print(f"\nError processing batch {i+1}: {e}")
            # continue

        batch_end_time = time.time()
        print(f" done in {batch_end_time - batch_start_time:.2f}s")

    end_time = time.time()
    print(f"Finished processing all batches in {end_time - start_time:.2f} seconds.")

    # --- Process Results and Add Context ---
    final_results_with_context: List[AnalysisInputItem] = []
    processed_count = 0
    skipped_count = 0

    for result_list in all_results: 
        for preprocessed_unit in result_list:
            # Filter based on word count after preprocessing
            word_count = len(preprocessed_unit.as_unit_text.split()) # Use as_unit_text
            if word_count < 3:
                logger.info(f"Filtering out AS unit {preprocessed_unit.as_unit_id} due to word count < 3: '{preprocessed_unit.as_unit_text[:20]}...'") # Also use as_unit_text in log
                skipped_count += 1
                continue

            # Find context for the original utterance ID
            context = find_context(preprocessed_unit.original_utterance_id, utterance_map, all_utterances_list)
            if context is None:
                 # Log if context couldn't be found (shouldn't happen with the fix)
                 logger.warning(f"Could not find context for original utterance ID: {preprocessed_unit.original_utterance_id}")
                 context = [] # Assign empty list if lookup failed

            # Create the final AnalysisInputItem
            analysis_input = AnalysisInputItem(
                **preprocessed_unit.model_dump(), # Copy fields from PreprocessedASUnit
                context=context # Add the found context
            )
            final_results_with_context.append(analysis_input)
            processed_count += 1

    logger.info(f"Finished processing all batches. Processed {processed_count} AS units, skipped {skipped_count} AS units due to word count.")

    # --- Write Output --- 
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    logger.info(f"Writing {len(final_results_with_context)} final results with context to {output_path}...")
    try:
        # Use Pydantic's serialization helper for robust JSON export
        output_data = [item.model_dump(mode='json') for item in final_results_with_context]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully wrote results to file.")
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        sys.exit(1)

    print("Preprocessing chain test finished.")

if __name__ == "__main__":
    main()
