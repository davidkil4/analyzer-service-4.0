import json
import os
import time
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
import asyncio

# --- Configuration ---
INPUT_FILE = "preprocessing_output.json"
OUTPUT_FILE = "analysis_output.json"

# --- Environment Setup ---
print("[DIAG] Running run_analysis_test.py...", file=sys.stderr)
# Specify the path to the .env file explicitly if it's not in the current working directory
loaded_dotenv = load_dotenv(verbose=True, override=True) # FORCE OVERRIDE

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to see detailed chain/LLM logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get a logger for this script

# --- Imports (after dotenv) ---
from analyzer_service.schemas import AnalysisInputItem, ContextUtterance, AlignedClause, ErrorDetail, PatternDetail
from analyzer_service.analysis_chains import (
    get_correction_chain,
    get_accuracy_analysis_chain,
    get_pattern_analysis_chain,
    process_accuracy_analysis_for_unit,
    process_pattern_analysis_for_unit,
    calculate_scores_for_unit
)

# --- Load Input Data ---
def load_input_data(filepath: str) -> List[AnalysisInputItem]:
    """Loads preprocessed data from JSON and converts to AnalysisInputItem list."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Assuming the root of the JSON is the list of PreprocessedASUnit dictionaries
        if not isinstance(data, list):
            raise ValueError("Expected JSON root to be a list of processed units.")

        analysis_items = []
        for unit_data in data:
            # Create AnalysisInputItem, Pydantic handles mapping fields from PreprocessedASUnit.
            # Context will be included if present in unit_data, otherwise defaults to None per schema.
            try:
                item = AnalysisInputItem(**unit_data)
                analysis_items.append(item)
            except Exception as e:
                logger.warning(f"Skipping item due to parsing error: {e}. Data: {unit_data.get('as_unit_id', 'N/A')}")
        
        logger.info(f"Successfully loaded {len(analysis_items)} items from {filepath}")
        return analysis_items
    except FileNotFoundError:
        logger.error(f"Input file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        sys.exit(1)

# --- Main Test Logic ---
async def main():
    logger.info("Starting analysis chain integration test...")
    start_time = time.time()

    # --- Check API Key --- 
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Error: GOOGLE_API_KEY not found in environment variables. Please ensure it's set in your .env file.")
        sys.exit(1)

    # --- Get Chains/Functions --- 
    try:
        correction_chain = get_correction_chain()
        accuracy_chain = get_accuracy_analysis_chain()
        pattern_chain = get_pattern_analysis_chain()
        # Scoring is a direct function call
        logger.info("Successfully obtained analysis chains/functions.")
    except Exception as e:
        logger.error(f"Error creating analysis chains: {e}", exc_info=True)
        sys.exit(1)

    # Load the data
    analysis_input_data = load_input_data(INPUT_FILE)
    if not analysis_input_data:
        logger.error("No data loaded, exiting.")
        return

    # Process the loaded data
    logger.info(f"Processing {len(analysis_input_data)} items from {INPUT_FILE}...")
    all_results: List[AnalysisInputItem] = []
    for i, item in enumerate(analysis_input_data):
        logger.info(f"--- Processing Item {i+1}/{len(analysis_input_data)} (ID: {item.as_unit_id}) ---")
        current_item = item.model_copy(deep=True) # Work on a copy
        item_start_time = time.time()

        try:
            # 1. Correction
            logger.debug(f"Input to Correction: {current_item.model_dump_json(indent=2)}")
            corrected_item = correction_chain.invoke(current_item) # This handles clause-level batching internally
            logger.info(f"After Correction ({time.time() - item_start_time:.2f}s): {corrected_item.model_dump_json(indent=2)}")
            # Basic Assertions
            assert isinstance(corrected_item, AnalysisInputItem)
            if corrected_item.clauses:
                 assert corrected_item.clauses[0].corrected_clause_text is not None # Check if it attempted correction

            # 2. Accuracy Analysis
            accuracy_analyzed_item = await process_accuracy_analysis_for_unit(corrected_item, accuracy_chain)
            logger.info(f"After Accuracy Analysis ({time.time() - item_start_time:.2f}s): {accuracy_analyzed_item.model_dump_json(indent=2)}")
            # Basic Assertions
            assert isinstance(accuracy_analyzed_item, AnalysisInputItem)
            if accuracy_analyzed_item.clauses and accuracy_analyzed_item.clauses[0].corrected_clause_text != accuracy_analyzed_item.clauses[0].clause_text:
                assert isinstance(accuracy_analyzed_item.clauses[0].errors_found, list)
                # assert len(accuracy_analyzed_item.clauses[0].errors_found) > 0 # Check errors were found if correction happened

            # 3. Pattern Analysis
            pattern_analyzed_item = await process_pattern_analysis_for_unit(accuracy_analyzed_item, pattern_chain)
            logger.info(f"After Pattern Analysis ({time.time() - item_start_time:.2f}s): {pattern_analyzed_item.model_dump_json(indent=2)}")
            # Basic Assertions
            assert isinstance(pattern_analyzed_item, AnalysisInputItem)
            # Pattern analysis is optional, just check structure if present
            if pattern_analyzed_item.clauses and pattern_analyzed_item.clauses[0].clause_pattern_analysis:
                assert isinstance(pattern_analyzed_item.clauses[0].clause_pattern_analysis, list)

            # 4. Scoring
            final_scored_item = calculate_scores_for_unit(pattern_analyzed_item)
            logger.info(f"After Scoring ({time.time() - item_start_time:.2f}s): {final_scored_item.model_dump_json(indent=2)}")
            # Basic Assertions
            assert isinstance(final_scored_item, AnalysisInputItem)
            assert final_scored_item.complexity_score is not None
            assert isinstance(final_scored_item.complexity_score, float)
            assert final_scored_item.accuracy_score is not None
            assert isinstance(final_scored_item.accuracy_score, float)

            all_results.append(final_scored_item)
            logger.info(f"--- Finished Item {i+1} (ID: {item.as_unit_id}) in {time.time() - item_start_time:.2f} seconds ---")

        except Exception as e:
            logger.error(f"Error processing item {item.as_unit_id}: {e}", exc_info=True)
            # Optionally add the original item or a placeholder to results
            all_results.append(item) # Add original item on error

    # --- Write Results --- 
    logger.info(f"Finished processing all items in {time.time() - start_time:.2f} seconds.")
    if all_results:
        try:
            # Convert list of Pydantic models to list of dicts for JSON serialization
            results_dict_list = [item.model_dump() for item in all_results]
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_dict_list, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully wrote {len(all_results)} results to {OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Error writing results to {OUTPUT_FILE}: {e}", exc_info=True)
    else:
        logger.warning("No results were generated.")

if __name__ == "__main__":
    asyncio.run(main())
