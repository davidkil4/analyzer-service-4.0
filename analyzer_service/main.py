import json
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from analyzer_service.analysis_chains import (
    get_pattern_analysis_chain,
    process_pattern_analysis_for_unit,
    calculate_scores_for_unit # Added calculate_scores_for_unit
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# TODO: Import schemas and chains once defined
# from .schemas import InputUtterance, PreprocessedData
# from .chains import get_preprocessing_chain, get_main_analysis_chain

# Constants
BATCH_SIZE = 10

def load_utterances(filepath: Path) -> list:
    """Loads utterances from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assuming the structure is like {'utterances': [...]} or just a list [...]
        utterances = data.get('utterances', data) if isinstance(data, dict) else data
        logger.info(f"Loaded {len(utterances)} utterances from {filepath}")
        return utterances
    except FileNotFoundError:
        logger.error(f"Input file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {filepath}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}")
        return []

def process_in_batches(input_filepath: str, output_dir: str):
    """Loads data, processes it in batches, and saves results."""
    input_path = Path(input_filepath)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    utterances = load_utterances(input_path)
    if not utterances:
        return

    # TODO: Initialize chains (outside the loop for efficiency if they are stateless)
    # preprocessing_chain = get_preprocessing_chain()
    # main_analysis_chain = get_main_analysis_chain()
    pattern_chain = get_pattern_analysis_chain()

    all_results = []
    num_batches = (len(utterances) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        batch_utterances = utterances[start_index:end_index]
        batch_number = i + 1

        logger.info(f"--- Processing Batch {batch_number}/{num_batches} ({len(batch_utterances)} utterances) ---")

        batch_preprocessed_results = []
        # --- Pre-processing Stage --- TODO: Replace with actual chain invocation
        logger.info(f"Starting pre-processing for Batch {batch_number}...")
        # Example: Iterate through utterances in the batch and apply pre-processing
        # for utterance_data in batch_utterances:
        #     try:
        #         # Assuming InputUtterance is the Pydantic model for raw data
        #         # raw_utterance = InputUtterance(**utterance_data)
        #         # preprocessed_result = preprocessing_chain.invoke(raw_utterance)
        #         # batch_preprocessed_results.append(preprocessed_result)
        #         pass # Replace with actual invocation
        #     except Exception as e:
        #         logger.error(f"Error pre-processing utterance {utterance_data.get('id', 'N/A')} in Batch {batch_number}: {e}")
        logger.warning(f"Pre-processing logic for Batch {batch_number} not implemented yet.")
        # Placeholder results for now
        batch_preprocessed_results = [{"id": u.get('id'), "status": "preprocessed_placeholder"} for u in batch_utterances]

        # --- Main Analysis Stage --- TODO: Replace with actual chain invocation
        logger.info(f"Starting main analysis for Batch {batch_number}...")
        batch_final_results = []
        if batch_preprocessed_results:
            # try:
            #    # Assuming main analysis chain takes the list of preprocessed results for the batch
            #    # final_result = main_analysis_chain.invoke(batch_preprocessed_results)
            #    # batch_final_results.append(final_result)
            #    pass # Replace with actual invocation
            # except Exception as e:
            #    logger.error(f"Error in main analysis for Batch {batch_number}: {e}")
            logger.warning(f"Main analysis logic for Batch {batch_number} not implemented yet.")
            # Placeholder results
            batch_final_results = [{"id": r.get('id'), "status": "analyzed_placeholder"} for r in batch_preprocessed_results]
        else:
             logger.warning(f"Skipping main analysis for Batch {batch_number} due to empty pre-processing results.")

        # --- Pattern Analysis Step ---
        logger.info(f"Starting pattern analysis for Batch {batch_number}...")
        batch_pattern_results = []
        for item in batch_final_results:
            try:
                pattern_result = process_pattern_analysis_for_unit(item, pattern_chain)
                batch_pattern_results.append(pattern_result)
            except Exception as e:
                logger.error(f"Error during pattern analysis for item {item.get('id', 'N/A')} in Batch {batch_number}: {e}")
                # Decide how to handle: append item or skip
                batch_pattern_results.append(item)

        # --- Calculate Final Scores (Synchronous) ---
        logger.info(f"Calculating Complexity and Accuracy scores for batch {batch_number}...")
        scored_batch = []
        for item in batch_pattern_results:
            scored_item = calculate_scores_for_unit(item)
            scored_batch.append(scored_item)
            logger.debug(f"  Final scores for {item.get('id', 'N/A')}: Complexity={scored_item.get('complexity_score', 'N/A')}, Accuracy={scored_item.get('accuracy_score', 'N/A')}")

        all_results.extend(scored_batch)
        logger.info(f"--- Finished Processing Batch {batch_number}/{num_batches} ---")

    # --- Save Final Results --- TODO: Define final output format
    output_filename = input_path.stem + "_analyzed.json"
    output_filepath = output_path / output_filename
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved final results to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving final results to {output_filepath}: {e}")

if __name__ == "__main__":
    # TODO: Add argument parsing for input/output paths
    INPUT_FILE = "../input_files/600_son_converted.json" # Example
    OUTPUT_DIR = "../output_files"
    process_in_batches(INPUT_FILE, OUTPUT_DIR)
