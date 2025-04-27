"""Complexity and accuracy scoring component for the analyzer service.

This module calculates syntactic complexity and accuracy scores for analyzed utterances,
combining metrics from both analyses into a comprehensive score.
"""
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, Any, List
import math

from analyzer_service.analysis_components.complexity_analyzer import ComplexityAnalyzer  # pylint: disable=import-error,no-name-in-module


def configure_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )


def calculate_error_counts(all_errors: List[Dict]) -> Dict:
    """Calculate error counts by severity."""
    severity_counts = {"critical": 0, "moderate": 0, "minor": 0}
    for error in all_errors:
        severity = error.get("severity", "minor")
        severity_counts[severity] += 1
    return severity_counts


def calculate_accuracy_score(utterance: Dict) -> Dict:
    """Calculate accuracy scores using the same formula as accuracy_analyzer."""
    lambda_factor = 1.2
    severity_weights = {"critical": 0.4, "moderate": 0.2, "minor": 0.1}

    # Collect all errors from all clauses
    all_errors = []
    clause_analyses = utterance.get('clauses', [])
    for clause in clause_analyses:
        all_errors.extend(clause.get('errors_found', []))

    # Calculate error impact
    total_impact = sum(severity_weights[e["severity"]] for e in all_errors)
    accuracy_score = math.exp(-lambda_factor * total_impact)

    # Get error counts by severity
    severity_counts = calculate_error_counts(all_errors)

    # Calculate error-free ratios and errors per unit
    error_free_asunit = 1.0 if len(all_errors) == 0 else 0.0
    errors_per_asunit = len(all_errors)

    # Calculate clause-level metrics
    if clause_analyses:
        error_free_clauses = sum(1 for c in clause_analyses if c.get("is_error_free", False))
        error_free_clause_ratio = error_free_clauses / len(clause_analyses)
        errors_per_clause = len(all_errors) / len(clause_analyses)
    else:
        error_free_clause_ratio = 1.0
        errors_per_clause = 0.0

    return {
        "score": round(accuracy_score, 2),
        "breakdown": {
            "error_free_asunit_ratio": error_free_asunit,
            "errors_per_asunit": errors_per_asunit,
            "error_free_clause_ratio": error_free_clause_ratio,
            "errors_per_clause": errors_per_clause,
            "severity_counts": severity_counts
        }
    }


def process_utterance(utterance: Dict[str, Any]) -> Dict[str, Any]:
    """Process single utterance with both complexity and accuracy analysis."""
    try:
        complexity_result = ComplexityAnalyzer.analyze(utterance_data=utterance)
        accuracy_result = calculate_accuracy_score(utterance)

        return {
            **utterance,
            "complexity": complexity_result,
            "accuracy": accuracy_result
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error processing utterance %s: %s", utterance.get('n', 'unknown'), str(e))
        return {
            **utterance,
            "analysis_error": str(e)
        }


async def main():
    """Main entry point for the script."""
    configure_logging()
    parser = argparse.ArgumentParser(description='Calculate complexity and accuracy scores')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('--output-file', '-o', help='Path to output file (default: auto-generated)')
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_file = args.output_file
    
    if not output_file:
        output_file = str(input_path.parent / f"{input_path.stem}_complexity_accuracy.json")
    
    logging.info(f"Loading utterances from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    processed = [process_utterance(u) for u in input_data['preprocessed_utterances']]

    logging.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"analyzed_utterances": processed}, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully processed {len(processed)} utterances")
    print(f"\nComplexity and accuracy scoring complete!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
