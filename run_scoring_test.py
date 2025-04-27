import json
import logging
import argparse
from pathlib import Path
import asyncio
from typing import List, Dict, Any

from pydantic import TypeAdapter, ValidationError

from analyzer_service.schemas import AnalysisInputItem # Assuming schemas.py is accessible
from analyzer_service.analysis_chains import calculate_scores_for_unit # Assuming analysis_chains.py is accessible

# --- Constants ---
DEFAULT_INPUT_FILENAME = "pattern_output.json"
DEFAULT_OUTPUT_FILENAME = "scoring_output.json"

async def main():
    """Loads data post-pattern analysis, applies scoring, and saves."""
    parser = argparse.ArgumentParser(description="Test script for the scoring function.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILENAME,
        help=f"Input JSON file containing analyzed items (post-pattern analysis). Default: {DEFAULT_INPUT_FILENAME}"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output JSON file to save scored items. Default: {DEFAULT_OUTPUT_FILENAME}"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()

    # --- Setup Logging ---
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting scoring test script with log level {args.log_level}")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # --- Load Input Data ---
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # Assuming the input file contains a list of dicts directly
            # or a dict with a key like 'analyzed_utterances'
            input_data = json.load(f)
            if isinstance(input_data, dict):
                # Try common keys where the list might be stored
                potential_keys = ["analyzed_utterances", "results", "items", "data"]
                items_list = None
                for key in potential_keys:
                    if key in input_data and isinstance(input_data[key], list):
                        items_list = input_data[key]
                        logger.info(f"Found list of items under key '{key}' in input file.")
                        break
                if items_list is None:
                     logger.error(f"Could not find a list of items in the input dictionary structure: {input_path}")
                     return
            elif isinstance(input_data, list):
                items_list = input_data
                logger.info("Input file contains a list of items directly.")
            else:
                logger.error(f"Unexpected data structure in input file: {input_path}. Expected list or dict containing a list.")
                return

        # Use TypeAdapter for robust parsing of the list
        AnalysisInputListAdapter = TypeAdapter(List[AnalysisInputItem])
        analysis_items = AnalysisInputListAdapter.validate_python(items_list)
        logger.info(f"Successfully loaded and validated {len(analysis_items)} items.")

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from: {input_path}")
        return
    except ValidationError as e:
        logger.error(f"Data validation failed for {input_path}: {e}")
        # Optionally log more details from e.errors()
        # logger.debug(f"Validation errors: {e.errors()}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {input_path}: {e}", exc_info=True)
        return

    # --- Apply Scoring Function ---
    logger.info(f"Applying scoring function to {len(analysis_items)} items...")
    scored_items: List[AnalysisInputItem] = []
    start_time = asyncio.get_event_loop().time()
    for i, item in enumerate(analysis_items):
        try:
            scored_item = calculate_scores_for_unit(item)
            scored_items.append(scored_item)
            if (i + 1) % 10 == 0: # Log progress every 10 items
                 logger.info(f"  Processed {i + 1}/{len(analysis_items)} items...")
        except Exception as e:
            logger.error(f"Error scoring item {item.as_unit_id} (index {i}): {e}", exc_info=True)
            # Decide how to handle: skip item or add with error flag?
            # For now, we skip adding it to the output.

    end_time = asyncio.get_event_loop().time()
    logger.info(f"Scoring applied to {len(scored_items)} items (out of {len(analysis_items)}) in {end_time - start_time:.2f} seconds.")

    # --- Save Output Data ---
    logger.info(f"Saving scored items to: {output_path}")
    try:
        output_data_list = [item.model_dump(exclude_none=True) for item in scored_items]
        with open(output_path, 'w', encoding='utf-8') as f:
            # Save as a list directly
            json.dump(output_data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(scored_items)} scored items.")
    except Exception as e:
        logger.error(f"Failed to save output to {output_path}: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
