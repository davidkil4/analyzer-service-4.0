# -*- coding: utf-8 -*-
from __future__ import annotations
"""Complexity and accuracy scoring component for the analyzer service.

This module calculates syntactic complexity and accuracy scores for analyzed utterances,
combining metrics from both analyses into a comprehensive score.
"""
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import math
import sys

# --- Third-Party Imports --- 
from pydantic import BaseModel, Field, parse_obj_as, ValidationError, TypeAdapter
from langchain_core.runnables import RunnableLambda

# --- First-Party Imports --- 
from analyzer_service.analysis_components.complexity_analyzer import ComplexityAnalyzer
# Import the error item schema directly from korean_analysis
from .korean_analysis import KoreanErrorItem
# Import pattern schema directly from recognizer_refactored
from .recognizer_refactored import FormulaicPattern

# Import schemas needed at RUNTIME
from .schemas import ComplexityScoredUtterance, ComplexityResult

# Import shared pipeline schemas from the new central file using TYPE_CHECKING
if TYPE_CHECKING:
    from .schemas import (
        SegmentationOutputUtterance, 
        AccuracyAnalyzedUtterance,
        KoreanVocabAnalyzedUtterance, 
        AccuracyAnalyzedASUnit, 
        AccuracyAnalyzedClause, 
        GrammarError 
    )

def configure_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def add_complexity_score(utterance: AccuracyAnalyzedUtterance) -> ComplexityScoredUtterance:
    """Calculate complexity score and add it to the utterance data."""
    try:
        utterance_dict_for_complexity = {
            "clauses": [
                clause.model_dump() 
                for unit in utterance.as_units 
                for clause in unit.clauses
            ]
        }
        
        complexity_result_dict = ComplexityAnalyzer.analyze(utterance_data=utterance_dict_for_complexity)
        complexity_result = ComplexityResult(**complexity_result_dict)
        
        return ComplexityScoredUtterance(
            **utterance.model_dump(),
            complexity=complexity_result
        )
    except Exception as e: 
        logging.error("Error calculating complexity for utterance %s: %s", utterance.n, str(e))
        return ComplexityScoredUtterance(
            **utterance.model_dump(),
            analysis_error=f"Complexity calculation failed: {str(e)}"
        )

# --- LCEL Refactoring --- 
# Wrap the complexity scoring logic in a RunnableLambda
# Input type: AccuracyAnalyzedUtterance (Imported from schemas)
# Output type: ComplexityScoredUtterance (Imported from schemas)
complexity_scoring_runnable: RunnableLambda[AccuracyAnalyzedUtterance, ComplexityScoredUtterance] = RunnableLambda(add_complexity_score)
# --- End LCEL Refactoring ---

def read_accuracy_analyzed_json(file_path: str) -> List[AccuracyAnalyzedUtterance]:
    """Read accuracy-analyzed utterances from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        utterance_dicts = data.get("analyzed_utterances", []) 
        return parse_obj_as(List[AccuracyAnalyzedUtterance], utterance_dicts)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {file_path}: {e}")
        sys.exit(1)

async def main():
    """Main entry point for the script.
       Reads accuracy-analyzed data, adds complexity scores, and saves.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description='Calculate complexity scores for accuracy-analyzed utterances')
    parser.add_argument('input_file', help='Path to input JSON file (output of accuracy analysis)')
    parser.add_argument('--output-file', '-o', help='Path to output file (default: auto-generated)')
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_file = args.output_file
    
    if not output_file:
        output_file = str(input_path.parent / f"{input_path.stem}_complexity_scored.json")
    
    logging.info(f"Loading accuracy-analyzed utterances from {input_path}")
    analyzed_utterances = read_accuracy_analyzed_json(str(input_path))

    # Use the runnable's batch method for efficiency
    scored_utterances: List[ComplexityScoredUtterance] = complexity_scoring_runnable.batch(analyzed_utterances)

    logging.info(f"Saving results to {output_file}")
    output_data = {"scored_utterances": [u.model_dump(exclude_none=True) for u in scored_utterances]}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully processed {len(scored_utterances)} utterances")
    print(f"\nComplexity scoring complete!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
