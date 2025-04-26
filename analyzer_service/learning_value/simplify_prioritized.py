#!/usr/bin/env python3
"""
Script to simplify prioritized utterances JSON for AI analysis.
Extracts only the essential fields needed for learning value assessment.
"""

import json
import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simplify_prioritized_output(input_path: Path, output_path: Path):
    """Reads a prioritized analysis JSON, extracts key fields, and writes a simplified version.
    
    Args:
        input_path: Path to the prioritized utterances JSON file.
        output_path: Path where the simplified utterances will be saved.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {input_path}")
        sys.exit(1)

    simplified_utterances = []
    zones = data.get("analysis_zones", [])
    
    # Process each zone
    for zone in zones:
        zone_name = zone.get("zone_name", "Unknown")
        recommendations = zone.get("recommendations", [])
        
        logger.info(f"Processing {len(recommendations)} recommendations in zone {zone_name}")
        
        # Extract essential fields from each recommendation
        for idx, utt in enumerate(recommendations):
            utt_id = f"{zone_name}_{idx}"
            
            simplified_utt = {
                "id": utt_id,
                "original": utt.get("original", ""),
                "corrected": utt.get("corrected", ""),
                "errors": [],
                "pattern_analysis": utt.get("pattern_analysis", [])
            }
            
            # Extract error information
            for error in utt.get("errors", []):
                simplified_error = {
                    "type": error.get("type", ""),
                    "severity": error.get("severity", ""),
                    "description": error.get("description", "")
                }
                simplified_utt["errors"].append(simplified_error)
            
            simplified_utterances.append(simplified_utt)
    
    # Create output data structure
    output_data = {
        "simplified_utterances": simplified_utterances
    }
    
    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Simplified output successfully written to {output_path}")
    except IOError as e:
        logger.error(f"Error writing output file to {output_path}: {e}")
        sys.exit(1)

def main():
    """Main function to run the simplification process."""
    parser = argparse.ArgumentParser(description="Simplify prioritized utterances JSON for AI analysis.")
    parser.add_argument("input_file", help="Path to the input prioritized utterances JSON file.")
    parser.add_argument("output_file", help="Path for the simplified output JSON file.")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # Basic validation
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    simplify_prioritized_output(input_path, output_path)

if __name__ == "__main__":
    main()
