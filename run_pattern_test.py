import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from analyzer_service.analysis_chains import (
    get_pattern_analysis_chain,
    process_pattern_analysis_for_unit
)
from analyzer_service.schemas import AnalysisInputItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output to console
        # You can add logging.FileHandler('pattern_test.log') here if needed
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_JSON_PATH = Path("accuracy_output.json")
OUTPUT_JSON_PATH = Path("pattern_output.json")

async def main():
    """Main function to run the pattern analysis test."""
    logger.info("--- Starting Pattern Analysis Test ---")
    start_total_time = time.time()

    # Load environment variables (e.g., GOOGLE_API_KEY)
    load_dotenv()
    logger.info("Loaded environment variables from .env")

    # --- Load Input Data ---
    if not INPUT_JSON_PATH.exists():
        logger.error(f"Input file not found: {INPUT_JSON_PATH}")
        return

    logger.info(f"Loading analysis input from: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        try:
            # Assuming the input JSON is a list of dicts
            # If it has a top-level key like "analyzed_utterances", adjust here
            input_data = json.load(f)
            # Validate basic structure (assuming it's a list)
            if not isinstance(input_data, list):
                logger.error(f"Expected a JSON list in {INPUT_JSON_PATH}, found {type(input_data)}")
                return
            analysis_items = [AnalysisInputItem(**item) for item in input_data]
            logger.info(f"Successfully loaded and parsed {len(analysis_items)} items.")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {INPUT_JSON_PATH}")
            return
        except Exception as e:
            logger.error(f"Error parsing input data into AnalysisInputItem: {e}", exc_info=True)
            return

    # --- Initialize Chain ---
    try:
        pattern_chain = get_pattern_analysis_chain()
    except Exception as e:
        logger.error(f"Failed to initialize pattern analysis chain: {e}", exc_info=True)
        return

    # --- Process Items ---
    processed_items: List[AnalysisInputItem] = []
    tasks = []
    logger.info(f"Creating processing tasks for {len(analysis_items)} items...")
    for i, item in enumerate(analysis_items):
        logger.info(f"Processing item {i+1}/{len(analysis_items)}: AS Unit ID {item.as_unit_id}")
        tasks.append(process_pattern_analysis_for_unit(item, pattern_chain))

    # Run tasks concurrently
    logger.info(f"Running {len(tasks)} pattern analysis tasks concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Finished concurrent processing.")

    # Collect results and handle potential exceptions from gather
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing item {i+1} (AS Unit ID: {analysis_items[i].as_unit_id}): {result}", exc_info=result)
            # Decide how to handle errors: skip item, use original item, etc.
            # For now, let's add the original item back with a note or skip it
            # Skipping for simplicity, adjust if needed:
            logger.warning(f"Skipping item {i+1} due to processing error.")
            # Or append the original item: processed_items.append(analysis_items[i])
        elif isinstance(result, AnalysisInputItem):
            processed_items.append(result)
        else:
             logger.warning(f"Unexpected result type for item {i+1}: {type(result)}. Skipping.")

    logger.info(f"Successfully processed {len(processed_items)} items out of {len(analysis_items)}.")

    # --- Save Output ---    
    logger.info(f"Saving analysis output to: {OUTPUT_JSON_PATH}")
    try:
        # Convert Pydantic models back to dictionaries for JSON serialization
        output_data = [item.model_dump(mode='json') for item in processed_items]
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved output for {len(processed_items)} items.")
    except Exception as e:
        logger.error(f"Failed to save output JSON: {e}", exc_info=True)

    # --- Finish ---    
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    logger.info("--- Pattern Analysis Test Finished ---")
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    logger.info(f"Results saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
