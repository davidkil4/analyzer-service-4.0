#!/usr/bin/env python3
"""
Script to evaluate the learning value of utterances using the Gemini API.
This script takes simplified utterances and returns filtering decisions.
"""

import json
import argparse
import sys
import os
import logging
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Import the production configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  
from analyzer_service.config.production_config import GeminiConfig

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
config = GeminiConfig()
MODEL_NAME = "gemini-2.0-flash"  # Using gemini-2.0-flash as requested
MAX_UTTERANCES_FOR_BATCH = 20  # Process in smaller batches to avoid token limits
MAX_RETRIES = 3 # Number of retries for API calls

# --- NEW PROMPTS ---
# Prompt template for Stage 1: Filtering
FILTERING_PROMPT = """
You are an expert English language teacher evaluating utterances for basic learning potential.

## TASK: FILTERING
For each utterance JSON object below, determine if it should be KEPT or FILTERED OUT based on these two key criteria:

1. MEANING MATCH: Does the original utterance's intended meaning probably match the corrected version? FILTER OUT if uncertain or if the corrected version adds substantial new content.
   - Example to filter: original "The beach near my house" → corrected "The beach near my house has beautiful sunset views every evening"
   - Keep if meanings probably match, even if expressed differently.

2. ERROR SIGNIFICANCE: Does the original contain actual grammatical or lexical errors worth learning from? FILTER OUT if the only changes are capitalization, fillers, contractions, or extremely minor issues.
   - Example to filter: original "I did not hear what you said" → corrected "I didn't hear what you said"
   - Keep if there are meaningful errors that would benefit the learner.

When in doubt about meaning match, FILTER OUT.

Here are the utterances to analyze:

{utterance_data_json}

For each utterance, return ONLY a JSON object with:
1. "id": The utterance ID
2. "decision": Either "KEEP" or "FILTER"
3. "reason": A brief explanation focusing on meaning match and error significance (1-2 sentences)

Return your response as a valid JSON array of these objects, with no additional text or formatting.
"""

# Prompt template for Stage 2: Categorization & Prioritization
CATEGORIZATION_PROMPT = """
You are an expert English language teacher and Second Language Acquisition (SLA) researcher.
You are given a set of utterances that have already passed an initial quality filter.

## TASK: COMPARATIVE EVALUATION AND PRIORITIZATION
Analyze all the utterances below and make a comparative judgment across the entire set. Your goal is to identify:

1. Which utterances would be MOST BENEFICIAL to teach with drill and practice
2. Which utterances would be MOST BENEFICIAL to teach conversationally

This is a relative judgment - compare utterances against each other to determine where teaching effort would have the highest impact. Consider:

**For Drill and Practice Potential:**
- Does the utterance contain high-value formulaic sequences (especially Sentence_Stem, Frame, or Pattern categories)?
- How far is the learner's original utterance from the target formulaic sequence? Greater distance suggests more need for drilling.
- Is the learner showing no familiarity with the pattern, or just using it imperfectly?
- Would practicing this pattern transfer to many other contexts?

**For Conversational Teaching Potential:**
- Is this an isolated error that's easily explained?
- Is the learner already showing partial knowledge of the formulaic sequence?
- Would a simple explanation suffice?
- Is the error specific to this context?

After analyzing all utterances, assign each one:
1. A teaching approach ("DRILL_AND_PRACTICE" or "CONVERSATIONAL_TEACHING")
2. A priority level ("HIGH", "MEDIUM", or "LOW") reflecting its relative importance compared to others

For "HIGH" priority drill and practice items, identify the specific pattern that should be drilled (patterns where the learner shows little familiarity, high transfer value, essential for communication).

Here are the pre-filtered utterances to analyze:

{utterance_data_json}

For each utterance, return ONLY a JSON object with:
1. "id": The utterance ID
2. "teaching_approach": Must be either "DRILL_AND_PRACTICE" or "CONVERSATIONAL_TEACHING"
3. "priority_level": Must be "HIGH", "MEDIUM", or "LOW"
4. "approach_reason": A brief explanation of why this teaching approach and priority level are appropriate (1-2 sentences)
5. "target_pattern": ONLY include this field if teaching_approach is "DRILL_AND_PRACTICE". The specific pattern that should be the focus of drilling

Return your response as a valid JSON array of these objects, with no additional text or formatting.
"""

# --- NEW HELPER FUNCTION ---
def call_gemini_api(prompt: str, model_name: str):
    """Helper function to call the Gemini API and parse the JSON response."""
    try:
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Error during API call to {model_name}: {e}")
        # Handle potential non-JSON errors or API issues
        response_text = "<unavailable>"
        try:
            response_text = response.text
        except:
            pass
        logger.error(f"API Response Content: {response_text}")
        return None # Indicate failure

# --- MODIFIED MAIN FUNCTION ---
def evaluate_learning_value(input_path: Path, output_path: Path):
    """Evaluates the learning value using a two-stage Gemini API process.

    Stage 1: Filters utterances (KEEP/FILTER).
    Stage 2: Categorizes and prioritizes KEPT utterances.
    
    Args:
        input_path: Path to the simplified utterances JSON file.
        output_path: Path where the final combined evaluation decisions will be saved.
    """
    try:
        # Load simplified utterances
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            
        utterances = input_data.get("simplified_utterances", [])
        if not utterances:
            logger.error(f"No simplified utterances found in {input_path}")
            sys.exit(1)
            
        logger.info(f"Processing {len(utterances)} utterances in two stages.")
        
        # --- API Configuration ---
        api_key = config.gemini_api_key
        if not api_key:
            logger.error("Google API Key not found in configuration.")
            sys.exit(1)
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Error configuring Google AI SDK: {e}")
            sys.exit(1)

        # --- Stage 1: Filtering --- 
        logger.info("--- Stage 1: Filtering --- ")
        filtering_results = {}
        # Process in batches
        for i in range(0, len(utterances), MAX_UTTERANCES_FOR_BATCH):
            batch = utterances[i:i+MAX_UTTERANCES_FOR_BATCH]
            batch_num = i // MAX_UTTERANCES_FOR_BATCH + 1
            logger.info(f"Processing filtering batch {batch_num} with {len(batch)} utterances")
            
            utterance_data_json = json.dumps(batch, indent=2, ensure_ascii=False)
            prompt = FILTERING_PROMPT.format(utterance_data_json=utterance_data_json)
            
            # Call API using helper function
            batch_decisions = call_gemini_api(prompt, MODEL_NAME) # Using same model for now, could use a faster one if needed
            
            if batch_decisions:
                logger.info(f"Received {len(batch_decisions)} filtering decisions for batch {batch_num}")
                for decision in batch_decisions:
                    # Basic validation
                    if isinstance(decision, dict) and 'id' in decision:
                        filtering_results[decision['id']] = decision
                    else:
                         logger.warning(f"Invalid decision format received in filtering batch {batch_num}: {decision}")       
            else:
                logger.error(f"Failed to get filtering decisions for batch {batch_num}. Marking batch items as FILTERED.")
                # Mark batch items as failed/filtered
                for utt in batch:
                    filtering_results[utt['id']] = {'id': utt['id'], 'decision': 'FILTER', 'reason': 'API Error during filtering.'}

        kept_utterances = [utt for utt in utterances if filtering_results.get(utt['id'], {}).get('decision') == 'KEEP']
        logger.info(f"Stage 1 complete. {len(kept_utterances)} utterances marked as KEEP.")

        # --- Stage 2: Categorization & Prioritization ---
        categorization_results = {}
        if kept_utterances:
            logger.info("--- Stage 2: Categorization & Prioritization --- ")
            # Process kept utterances in batches
            for i in range(0, len(kept_utterances), MAX_UTTERANCES_FOR_BATCH):
                batch = kept_utterances[i:i+MAX_UTTERANCES_FOR_BATCH]
                batch_num = i // MAX_UTTERANCES_FOR_BATCH + 1
                logger.info(f"Processing categorization batch {batch_num} with {len(batch)} utterances")
                
                # We only need the IDs for Stage 2 mapping, but send full data for context
                utterance_data_json = json.dumps(batch, indent=2, ensure_ascii=False)
                prompt = CATEGORIZATION_PROMPT.format(utterance_data_json=utterance_data_json)
                
                # Call API using helper function
                batch_categorizations = call_gemini_api(prompt, MODEL_NAME)
                
                if batch_categorizations:
                    logger.info(f"Received {len(batch_categorizations)} categorization decisions for batch {batch_num}")
                    for cat_decision in batch_categorizations:
                        # Basic validation
                        if isinstance(cat_decision, dict) and 'id' in cat_decision:
                            categorization_results[cat_decision['id']] = cat_decision
                        else:
                            logger.warning(f"Invalid decision format received in categorization batch {batch_num}: {cat_decision}")
                else:
                    logger.error(f"Failed to get categorization decisions for batch {batch_num}. Kept utterances from this batch won't be categorized.")
                    # Handle failed categorization - mark them with default/error values
                    for utt in batch:
                         categorization_results[utt['id']] = {'id': utt['id'], 'teaching_approach': 'UNKNOWN', 'priority_level': 'UNKNOWN', 'approach_reason': 'API Error during categorization.'}

        else:
            logger.info("No utterances kept after Stage 1. Skipping Stage 2.")

        # --- Merge Results --- 
        logger.info("--- Merging Stage 1 & Stage 2 Results ---")
        final_decisions = []
        processed_ids = set()
        for utt_id, filter_decision in filtering_results.items():
            if utt_id in processed_ids:
                continue # Should not happen with dicts, but safety check
                
            if filter_decision.get('decision') == 'KEEP':
                cat_decision = categorization_results.get(utt_id)
                if cat_decision:
                    # Merge filter reason with categorization details
                    # Ensure 'id' is not duplicated if present in both
                    merged = {**filter_decision, **{k: v for k, v in cat_decision.items() if k != 'id'}}
                    final_decisions.append(merged)
                    processed_ids.add(utt_id)
                else:
                     # This case handles if categorization API failed for a batch
                     logger.warning(f"Categorization data missing for kept utterance {utt_id}. Using filter data only with UNKNOWN categorization.")
                     # Add placeholder categorization fields
                     filter_decision['teaching_approach'] = 'UNKNOWN'
                     filter_decision['priority_level'] = 'UNKNOWN'
                     filter_decision['approach_reason'] = 'Categorization failed.'
                     final_decisions.append(filter_decision)
                     processed_ids.add(utt_id)
            else:
                # Append FILTER decisions directly
                final_decisions.append(filter_decision)
                processed_ids.add(utt_id)
        
        # Sanity check: Ensure all original utterance IDs were processed
        if len(processed_ids) != len(utterances):
             logger.warning(f"Mismatch in processed IDs ({len(processed_ids)}) vs original utterances ({len(utterances)}). Some utterances might be missing in the output.")
             # Potentially add missing ones with error status if needed

        # --- Write Output --- 
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_decisions, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Evaluation decisions successfully written to {output_path}")
        
        # Log summary statistics
        keep_count = sum(1 for d in final_decisions if d.get("decision") == "KEEP")
        filter_count = len(final_decisions) - keep_count
        logger.info(f"Summary: {keep_count} utterances to keep, {filter_count} utterances to filter out")
        
    except Exception as e:
        logger.error(f"An error occurred in the main evaluation function: {e}", exc_info=True)
        sys.exit(1)

def main():
    """Main function to run the evaluation process."""
    parser = argparse.ArgumentParser(description="Evaluate the learning value of utterances.")
    parser.add_argument("input_file", help="Path to the simplified utterances JSON file.")
    parser.add_argument("output_file", help="Path for the evaluation decisions output JSON file.")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # Basic validation
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    evaluate_learning_value(input_path, output_path)

if __name__ == "__main__":
    main()
