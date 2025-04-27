import json
import os
import time
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
from pydantic import ValidationError

# Print diagnostic at start
print("[DIAG] Running run_correction_test.py...", file=sys.stderr)

# --- Configuration ---
INPUT_FILE = "preprocessing_output.json" # Input from preprocessing step
OUTPUT_FILE = "correction_output.json" # Output for this step
# BATCH_SIZE = 10 # Correction chain processes items individually within the chain logic

# --- Environment Setup ---
loaded_dotenv = load_dotenv(verbose=True, override=True)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Imports (after dotenv) ---
from analyzer_service.schemas import AnalysisInputItem
from analyzer_service.analysis_chains import get_correction_chain

# --- Helper Functions ---
def load_analysis_input(filepath: str) -> List[AnalysisInputItem]:
    """Loads AnalysisInputItem objects from a JSON file."""
    logger.info(f"Attempting to load AnalysisInputItem list from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assuming the JSON file contains a list of dictionaries
        analysis_input_list = [AnalysisInputItem(**item) for item in data]
        logger.info(f"Successfully loaded {len(analysis_input_list)} AnalysisInputItem objects.")
        return analysis_input_list
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}")
        return []
    except ValidationError as e:
        logger.error(f"Error validating data from {filepath}: {e}")
        # You might want to log the specific failing item or lines
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}", exc_info=True)
        return []

# --- Main Test Logic ---
def main():
    logger.info("Starting correction chain test...")

    # 1. Load Data
    analysis_input_data = load_analysis_input(INPUT_FILE)
    if not analysis_input_data:
        logger.error("Failed to load input data. Exiting.")
        return

    # 2. Get the correction chain
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Error: GOOGLE_API_KEY not found. Please ensure it's set.")
        exit()

    try:
        # The chain itself handles batching internally by calling llm.batch
        correction_chain = get_correction_chain()
        logger.info("Successfully obtained correction chain.")
    except Exception as e:
        logger.error(f"Error creating correction chain: {e}", exc_info=True)
        exit()

    # 3. Run the chain on ALL items (sequentially invoking the chain for each item)
    all_results: List[AnalysisInputItem] = [] # Store results
    start_time = time.time()
    total_items = len(analysis_input_data)
    logger.info(f"Processing {total_items} AnalysisInputItem objects...")

    for i, input_item in enumerate(analysis_input_data):
        logger.debug(f"  Processing item {i+1}/{total_items} (ID: {input_item.as_unit_id})...",)
        item_start_time = time.time()
        try:
            # Invoke the chain for a single AnalysisInputItem
            # The chain returns the modified AnalysisInputItem
            corrected_item = correction_chain.invoke(input_item)
            all_results.append(corrected_item)
            item_end_time = time.time()
            logger.debug(f"  Finished item {i+1}/{total_items} in {item_end_time - item_start_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error processing item {i+1} (ID: {input_item.as_unit_id}): {e}", exc_info=True)
            # Optionally append the original item or None to keep track
            # all_results.append(input_item) # Or None, or skip

    end_time = time.time()
    logger.info(f"Finished processing {len(all_results)} items in {end_time - start_time:.2f} seconds.")

    # 4. Save Results to File (Guideline #5)
    logger.info(f"Attempting to save results to: {OUTPUT_FILE}")
    try:
        # Convert Pydantic objects to dictionaries for JSON serialization
        results_dict_list = [result.model_dump(mode='json') for result in all_results]
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_dict_list, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved results to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving results to {OUTPUT_FILE}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
