# run_accuracy_test.py

import json
import logging
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

from analyzer_service.schemas import AnalysisInputItem, ErrorDetail
from analyzer_service.analysis_chains import get_accuracy_analysis_chain, process_accuracy_analysis_for_unit

# --- Configuration ---
# Load environment variables (e.g., API keys)
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define input and output file paths
INPUT_FILE = Path("correction_output.json") # Use the output from the correction step
OUTPUT_FILE = Path("accuracy_output.json")

# --- Helper Functions ---
def load_analysis_input(file_path: Path) -> List[AnalysisInputItem]:
    """Loads the analysis input data (output from correction) from a JSON file."""
    logger.info(f"Loading analysis input from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate and parse using Pydantic
            items = [AnalysisInputItem(**item) for item in data]
            logger.info(f"Successfully loaded {len(items)} items.")
            return items
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

def save_analysis_output(data: List[AnalysisInputItem], file_path: Path):
    """Saves the analysis results (including errors) to a JSON file."""
    logger.info(f"Saving analysis output to: {file_path}")
    try:
        # Convert Pydantic models to dictionaries for JSON serialization
        output_data = [item.model_dump(mode='json') for item in data]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved output for {len(output_data)} items.")
    except TypeError as e:
        logger.error(f"Serialization error when saving output: {e}")
        # Attempt to log the problematic item if possible (may fail)
        try:
            problem_indices = []
            for i, item in enumerate(data):
                try:
                    item.model_dump(mode='json')
                except Exception:
                    problem_indices.append(i)
            logger.error(f"Problematic item indices (approx): {problem_indices}")
        except Exception as log_e:
            logger.error(f"Could not identify problematic item during logging: {log_e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving output to {file_path}: {e}")
        raise

# --- Main Execution ---
def main():
    logger.info("--- Starting Accuracy Analysis Test --- ")
    start_total_time = time.time()

    # 1. Load Input Data (from correction step)
    analysis_input_data = load_analysis_input(INPUT_FILE)

    # 2. Initialize the Accuracy Analysis Chain
    try:
        accuracy_chain = get_accuracy_analysis_chain()
    except Exception as e:
        logger.error(f"Failed to initialize accuracy analysis chain: {e}", exc_info=True)
        return # Cannot proceed

    # 3. Process each item
    results = []
    for i, item in enumerate(analysis_input_data):
        logger.info(f"Processing item {i+1}/{len(analysis_input_data)}: AS Unit ID {item.as_unit_id}")
        try:
            processed_item = process_accuracy_analysis_for_unit(item, accuracy_chain)
            results.append(processed_item)
        except Exception as e:
            logger.error(f"Failed to process AS Unit {item.as_unit_id} during accuracy analysis: {e}", exc_info=True)
            # Optionally append the item with an error marker or skip
            # For now, we log and continue, the item might have partial errors marked
            results.append(item) # Append original item with potential partial error marking

    # 4. Save Output
    save_analysis_output(results, OUTPUT_FILE)

    end_total_time = time.time()
    logger.info(f"--- Accuracy Analysis Test Finished --- ")
    logger.info(f"Total execution time: {end_total_time - start_total_time:.2f} seconds")
    logger.info(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
