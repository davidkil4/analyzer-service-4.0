#!/usr/bin/env python3
"""
Script to run the complete learning value filtering pipeline.
This orchestrates the entire process from simplification to finalization.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path

# Import the production configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  
from analyzer_service.config.production_config import GeminiConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(prioritized_path: Path, output_dir: Path, local_mode: bool = False):
    """Runs the full Learning Value Pipeline.

    Args:
        prioritized_path: Path to the _secondary_prioritized.json file.
        output_dir: Directory where the output and intermediate files will be saved.
        local_mode: If True, runs in local mode (potentially different behavior, TBD).
    """
    # --- Ensure Output Directories Exist ---
    lv_output_dir = output_dir / "clustering_analysis" / "learning_value"
    lv_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Define File Paths ---
    base_name = prioritized_path.stem.replace('_secondary_prioritized', '')
    simplified_path = lv_output_dir / f"{base_name}_simplified_for_eval.json"
    decisions_path = lv_output_dir / f"{base_name}_evaluation_decisions.json"
    finalized_path = output_dir / "clustering_analysis" / f"{base_name}_secondary_prioritized_finalized.json"
    drill_practice_path = lv_output_dir / f"{base_name}_secondary_prioritized_finalized_drill_practice.json"
    conversational_path = lv_output_dir / f"{base_name}_secondary_prioritized_finalized_conversational.json"

    # Determine project root relative to this script's location
    # Assuming the script is run using `python -m ...` from the project root
    project_root = Path(__file__).resolve().parents[3] 
    logger.info(f"Inferred project root: {project_root}")
    
    # Determine original conversation file path (assuming convention)
    conversation_filename = f"{base_name}.json"
    # Construct path relative to project root
    conversation_path = project_root / "conversations" / conversation_filename
    if not conversation_path.is_file():
        logger.warning(f"Could not find conversation file at inferred path: {conversation_path}")
        # Handle error or proceed without context? For now, log warning.
        # sys.exit(f"Error: Conversation file not found at {conversation_path}")

    # --- Step 1: Simplify Prioritized Utterances ---
    logger.info("Step 1: Simplifying prioritized utterances")
    try:
        # Adjusted imports for scripts within the same package
        from .simplify_prioritized import simplify_prioritized_output
        simplify_prioritized_output(prioritized_path, simplified_path)
        logger.info(f"Simplified utterances saved to {simplified_path}")
    except Exception as e:
        logger.error(f"Error during simplification: {e}", exc_info=True)
        sys.exit(1)

    # --- Step 2: Evaluate Learning Value ---
    logger.info("Step 2: Evaluating learning value")
    try:
        from .evaluate_learning_value import evaluate_learning_value
        evaluate_learning_value(simplified_path, decisions_path)
        logger.info(f"Evaluation decisions saved to {decisions_path}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)

    # --- Step 3: Finalize Analysis (Apply Decisions and Split) ---
    logger.info("Step 3: Finalizing analysis")
    try:
        from .finalize_analysis import finalize_analysis
        # Pass the original prioritized file and the decisions file
        # The finalize_analysis function now internally handles splitting
        finalize_analysis(
            prioritized_path=prioritized_path,
            ai_decisions_path=decisions_path,
            output_path=finalized_path, 
            conversation_path=conversation_path # Pass the conversation file path
        )
        logger.info(f"Finalized analysis saved:")
        logger.info(f"  - Main: {finalized_path}")
        logger.info(f"  - Drill/Practice: {drill_practice_path}") # Assuming finalize_analysis creates these
        logger.info(f"  - Conversational: {conversational_path}") # Assuming finalize_analysis creates these
    except Exception as e:
        logger.error(f"Error during finalization: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Learning Value Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run the Learning Value Pipeline to filter and categorize prioritized utterances.")
    parser.add_argument("prioritized_file", type=str, help="Path to the input _secondary_prioritized.json file.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files (default: current directory).")
    parser.add_argument("--local", action="store_true", help="Run in local mode (future use).")

    args = parser.parse_args()

    prioritized_path = Path(args.prioritized_file)
    output_dir = Path(args.output_dir)

    if not prioritized_path.is_file():
        logger.error(f"Input file not found: {prioritized_path}")
        sys.exit(1)

    # Make output_dir absolute relative to the current working directory if it's relative
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
        logger.info(f"Relative output directory specified. Using absolute path: {output_dir}")

    run_pipeline(prioritized_path, output_dir, local_mode=args.local)

if __name__ == "__main__":
    # Ensure the project root is in the Python path
    # This allows imports like 'from analyzer_service.analysis_components...' if needed elsewhere
    # Determine the project root directory (adjust levels as necessary)
    project_root = Path(__file__).resolve().parents[3] 
    # Add project root to sys.path if it's not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added project root to sys.path: {project_root}")

    main()
