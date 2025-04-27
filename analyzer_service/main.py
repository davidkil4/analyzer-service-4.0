import json
import os
import logging
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# --- Import Schemas ---
from analyzer_service.schemas import (
    InputUtterance,
    PreprocessingOutput,
    PreprocessedASUnit,
    AnalysisInputItem,
    ContextUtterance
)

# --- Import Chains/Functions ---
from analyzer_service.preprocessing_chains import get_preprocessing_chain
from analyzer_service.analysis_chains import (
    get_correction_chain,
    get_accuracy_analysis_chain,
    get_pattern_analysis_chain,
    process_accuracy_analysis_for_unit,
    process_pattern_analysis_for_unit,
    calculate_scores_for_unit
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
BATCH_SIZE = 10
TARGET_SPEAKER = "student" # Define the target speaker

def load_utterances(filepath: Path) -> List[Dict[str, Any]]:
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

# --- Helper function to find context --- 
def find_context(target_utterance_id: str, utterance_map: Dict[str, Dict], all_utterances: List[Dict]) -> Optional[List[ContextUtterance]]:
    """Finds the 3 utterances preceding the target utterance.
       Handles potential missing IDs and list boundaries.
    """
    try:
        # Find the index of the target utterance in the original full list
        target_index = -1
        for i, utt in enumerate(all_utterances):
            # Use .get() for safety, assuming 'id' is the key
            if utt.get('id') == target_utterance_id:
                target_index = i
                break

        if target_index == -1:
            # Log if the original utterance ID wasn't found in the full list
            logger.warning(f"Could not find original index for utterance_id: {target_utterance_id} in the full list.")
            return [] # Return empty list, consistent with test script

        # Determine the start index for the context (up to 3 previous)
        start_index = max(0, target_index - 3)
        
        # Extract the preceding utterance dictionaries
        context_utterances_raw = all_utterances[start_index:target_index]
        
        # Format into ContextUtterance objects
        # Use .get() with defaults for robustness against missing keys
        context = [
            ContextUtterance(
                speaker=utt.get('speaker', 'UNKNOWN'), 
                text=utt.get('text', '')
            )
            for utt in context_utterances_raw
        ]
        return context
    except Exception as e:
        # Log any unexpected errors during context finding
        logger.error(f"Error finding context for utterance_id {target_utterance_id}: {e}", exc_info=True)
        return [] # Return empty list on error
# --- End Helper --- 

async def process_in_batches(input_filepath: str, output_dir: str):
    """Loads data, processes it async in batches (preprocessing) 
       and then processes analysis steps on the full dataset."""
    input_path = Path(input_filepath)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load ALL utterances first
    all_utterances_list = load_utterances(input_path)
    if not all_utterances_list:
        logger.error(f"No utterances loaded from {input_path}. Exiting.")
        return
    logger.info(f"Loaded {len(all_utterances_list)} total utterances.")

    # Create map for context lookup from the full list (assuming 'id' is the key)
    utterance_map = {utt.get('id'): utt for utt in all_utterances_list if utt.get('id')}
    if len(utterance_map) != len(all_utterances_list):
        logger.warning("Some utterances might be missing 'id' key, context lookup may fail for them.")

    # Filter for the target speaker
    student_utterances_dicts = [
        utt for utt in all_utterances_list 
        if utt.get("speaker", "").lower() == TARGET_SPEAKER.lower() and utt.get("text")
    ]
    logger.info(f"Filtered {len(student_utterances_dicts)} utterances for speaker '{TARGET_SPEAKER}'.")
    
    if not student_utterances_dicts:
        logger.warning(f"No utterances found for speaker '{TARGET_SPEAKER}' with non-empty text. Exiting.")
        return

    # Initialize Chains (Keep this logic)
    logger.info("Initializing preprocessing and analysis chains...")
    try:
        preprocessing_chain = get_preprocessing_chain()
        correction_chain = get_correction_chain()
        accuracy_chain = get_accuracy_analysis_chain()
        pattern_chain = get_pattern_analysis_chain()
        # process_accuracy_analysis_for_unit, process_pattern_analysis_for_unit, calculate_scores_for_unit are functions
        logger.info("Chains initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize chains: {e}", exc_info=True)
        return

    all_preprocessed_units: List[PreprocessedASUnit] = [] # Store all preprocessed units here
    # Calculate batches based on filtered student utterances
    num_batches = (len(student_utterances_dicts) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info("--- Starting Preprocessing Stage --- ")
    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        # Create batch from the filtered student utterances
        batch_utterances_raw = student_utterances_dicts[start_index:end_index]
        batch_number = i + 1

        logger.info(f"Starting pre-processing for Batch {batch_number}/{num_batches} ({len(batch_utterances_raw)} utterances)...")

        # --- Pre-processing Stage --- 
        # 1. Convert raw dicts to InputUtterance objects
        batch_input_objects: List[InputUtterance] = []
        for utt_data in batch_utterances_raw:
            try:
                # Ensure required fields like 'id', 'text', 'speaker' exist
                # We already filtered for speaker and non-empty text
                if not all(k in utt_data for k in ('id', 'text', 'speaker')):
                     logger.warning(f"Skipping utterance in batch {batch_number} due to missing fields (id, text, or speaker): {utt_data.get('id', 'N/A')}")
                     continue
                batch_input_objects.append(InputUtterance(**utt_data))
            except Exception as e:
                logger.error(f"Error creating InputUtterance for {utt_data.get('id', 'N/A')} in Batch {batch_number}: {e}")

        if not batch_input_objects:
            logger.warning(f"Skipping Batch {batch_number} due to no valid input utterances.")
            continue
        
        # 2. Run preprocessing chain asynchronously in batch
        try:
            results: List[Optional[PreprocessingOutput]] = await preprocessing_chain.abatch(batch_input_objects, {"max_concurrency": 5}) # Control concurrency
            
            # 3. Collect successful results
            batch_preprocessed_results = []
            for result in results:
                if result and result.processed_utterances: # Corrected attribute name
                    batch_preprocessed_results.extend(result.processed_utterances)
                elif result is None:
                    # This corresponds to utterances filtered out by the chain (e.g., too short)
                    logger.debug(f"An utterance in batch {batch_number} was filtered out during preprocessing.")
                else:
                     logger.warning(f"Preprocessing result in batch {batch_number} was empty or malformed: {result}")
            
            logger.info(f"Successfully preprocessed Batch {batch_number}, got {len(batch_preprocessed_results)} analysis units.")
            all_preprocessed_units.extend(batch_preprocessed_results)

        except Exception as e:
            logger.error(f"Error during preprocessing chain for Batch {batch_number}: {e}", exc_info=True)

        # Optional: Add a small delay between batches if needed
        # await asyncio.sleep(0.1)

    logger.info(f"--- Finished Preprocessing Stage --- Got {len(all_preprocessed_units)} total analysis units.")

    if not all_preprocessed_units:
        logger.error("No analysis units produced after preprocessing. Exiting.")
        return

    # --- Convert PreprocessedASUnit to AnalysisInputItem --- 
    logger.info("Converting preprocessed units to analysis input items...")
    analysis_input_items: List[AnalysisInputItem] = []
    for pp_unit in all_preprocessed_units:
        try:
            # Find context using the helper function
            context = find_context(pp_unit.original_utterance_id, utterance_map, all_utterances_list)
            
            # Create AnalysisInputItem including the found context
            analysis_input_items.append(
                AnalysisInputItem(
                    **pp_unit.model_dump(), # Unpack fields from PreprocessedASUnit
                    context=context        # Add the found context
                )
            )
        except Exception as e:
            logger.error(f"Error converting PreprocessedASUnit {pp_unit.as_unit_id} or finding context: {e}")
    logger.info(f"Successfully created {len(analysis_input_items)} analysis input items with context.")
    
    if not analysis_input_items:
        logger.error("No analysis items to process after conversion. Exiting.")
        return

    # --- Analysis Stages --- 
    
    # 1. Correction Stage (Async Batch)
    logger.info("--- Starting Correction Stage --- ")
    corrected_items: List[AnalysisInputItem] = []
    try:
        # Correction chain returns the modified items
        corrected_items = await correction_chain.abatch(analysis_input_items, {"max_concurrency": 5})
        logger.info(f"Successfully ran correction stage on {len(corrected_items)} items.")
    except Exception as e:
        logger.error(f"Error during correction stage: {e}", exc_info=True)
        # Decide how to proceed: maybe use original items? For now, exit.
        logger.error("Exiting due to correction stage failure.")
        return

    if not corrected_items:
        logger.error("No items left after correction stage. Exiting.")
        return

    # 2. Accuracy Analysis Stage (Async Gather)
    logger.info("--- Starting Accuracy Analysis Stage --- ")
    accuracy_processed_items_results = []
    try:
        accuracy_tasks = [process_accuracy_analysis_for_unit(item, accuracy_chain) for item in corrected_items]
        # return_exceptions=True allows processing to continue if one item fails
        accuracy_results = await asyncio.gather(*accuracy_tasks, return_exceptions=True)
        
        # Process results, separating successes from errors
        for i, result in enumerate(accuracy_results):
            if isinstance(result, Exception):
                logger.error(f"Error during accuracy analysis for item {corrected_items[i].as_unit_id}: {result}", exc_info=result)
                # Option: Append the item without accuracy analysis or skip it
                # accuracy_processed_items_results.append(corrected_items[i]) # Keep original on error
            elif result:
                 accuracy_processed_items_results.append(result) # Append successful result
            else:
                 logger.warning(f"Accuracy analysis for item {corrected_items[i].as_unit_id} returned None/empty.")

        logger.info(f"Successfully processed {len(accuracy_processed_items_results)} items through accuracy analysis stage (potential errors logged)." )
    except Exception as e:
         logger.error(f"Unexpected error during accuracy analysis gathering: {e}", exc_info=True)
         # Fallback: Use items from previous stage if gather fails catastrophically
         accuracy_processed_items_results = corrected_items

    if not accuracy_processed_items_results:
        logger.error("No items left after accuracy analysis stage. Exiting.")
        return

    # 3. Pattern Analysis Stage (Async Gather)
    logger.info("--- Starting Pattern Analysis Stage --- ")
    pattern_processed_items_results = []
    try:
        pattern_tasks = [process_pattern_analysis_for_unit(item, pattern_chain) for item in accuracy_processed_items_results]
        pattern_results = await asyncio.gather(*pattern_tasks, return_exceptions=True)

        for i, result in enumerate(pattern_results):
             if isinstance(result, Exception):
                 logger.error(f"Error during pattern analysis for item {accuracy_processed_items_results[i].as_unit_id}: {result}", exc_info=result)
                 # pattern_processed_items_results.append(accuracy_processed_items_results[i]) # Keep item on error
             elif result:
                 pattern_processed_items_results.append(result)
             else:
                 logger.warning(f"Pattern analysis for item {accuracy_processed_items_results[i].as_unit_id} returned None/empty.")

        logger.info(f"Successfully processed {len(pattern_processed_items_results)} items through pattern analysis stage (potential errors logged)." )
    except Exception as e:
         logger.error(f"Unexpected error during pattern analysis gathering: {e}", exc_info=True)
         pattern_processed_items_results = accuracy_processed_items_results
    
    if not pattern_processed_items_results:
        logger.error("No items left after pattern analysis stage. Exiting.")
        return

    # 4. Scoring Stage (Synchronous Loop)
    logger.info("--- Starting Scoring Stage --- ")
    final_analyzed_items: List[AnalysisInputItem] = []
    skipped_scoring_count = 0
    for item in pattern_processed_items_results:
        try:
            # calculate_scores_for_unit modifies the item in place and returns it
            scored_item = calculate_scores_for_unit(item)
            final_analyzed_items.append(scored_item)
        except Exception as e:
            logger.error(f"Error scoring item {item.as_unit_id}: {e}", exc_info=True)
            skipped_scoring_count += 1
            # Option: append item without scores? final_analyzed_items.append(item)
    logger.info(f"Successfully scored {len(final_analyzed_items)} items. Skipped {skipped_scoring_count} due to errors.")
    
    # --- Save Final Results --- 
    output_filename = input_path.stem + "_analyzed.json"
    output_filepath = output_path / output_filename
    try:
        # Convert Pydantic models to dictionaries for JSON serialization
        results_to_save = [item.model_dump() for item in final_analyzed_items]
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved final results to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving final results to {output_filepath}: {e}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run the Analyzer Service preprocessing and analysis pipeline.")
    parser.add_argument("-i", "--input-file", type=str, required=True,
                        help="Path to the input JSON file containing utterances.")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to the directory where output files will be saved.")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO).")

    args = parser.parse_args()

    # Configure logging level based on arguments
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_numeric) # Set root logger level
    logger.info(f"Logging level set to {args.log_level.upper()}")

    # Run the async function using parsed arguments
    asyncio.run(process_in_batches(args.input_file, args.output_dir))
