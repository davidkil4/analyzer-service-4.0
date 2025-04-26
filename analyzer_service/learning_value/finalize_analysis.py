#!/usr/bin/env python3
"""
Script to finalize analysis by applying AI filtering decisions to the prioritized utterances.
This is the final step in the pipeline to produce high-quality learning recommendations.
"""

import json
import argparse
import sys
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(text: Optional[str]) -> str:
    """Robust normalization: lowercase, remove fillers & punctuation, collapse whitespace."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove common fillers using word boundaries
    fillers = ['um', 'uh', 'er', 'err', 'urr']
    for filler in fillers:
        text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
    # Remove punctuation except apostrophes within words (e.g., keep "it's" but remove trailing '.')
    text = re.sub(r"[.,!?;:]", "", text) # Remove common punctuation
    text = re.sub(r"'\s|\s'", " ", text) # Handle apostrophes separated by space (like ' s -> s)
    # Remove leading/trailing single quotes if they exist after other cleaning
    text = text.strip("'")
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

def finalize_analysis(prioritized_path: Path, ai_decisions_path: Path, output_path: Path, conversation_path: Optional[Path] = None):
    """Applies AI filtering decisions and adds conversational context using normalized text matching.
    
    Args:
        prioritized_path: Path to the original prioritized utterances JSON file.
        ai_decisions_path: Path to the AI filtering decisions JSON file.
        output_path: Path where the finalized utterances will be saved.
        conversation_path: Optional path to the original conversation JSON file.
    """
    try:
        # Load prioritized data
        with open(prioritized_path, 'r', encoding='utf-8') as f:
            prioritized_data = json.load(f)
            
        # Load AI decisions
        with open(ai_decisions_path, 'r', encoding='utf-8') as f:
            ai_decisions = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON: {e}")
        sys.exit(1)
        
    # Load conversation texts if path is provided
    conversation_texts: List[str] = []
    original_utterances_data: List[Dict[str, Any]] = [] # Store original dicts for more info if needed
    normalized_conversation_texts: List[str] = [] # For robust matching

    if conversation_path and conversation_path.is_file():
        try:
            with open(conversation_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Adjust keys ('utterances', 'text') if the actual structure differs
            utterances = conversation_data.get('utterances', []) 
            if not utterances:
                 utterances = conversation_data # Handle case where the root is the list
                 if not isinstance(utterances, list):
                     logger.warning(f"Could not find 'utterances' list in {conversation_path}, or root is not a list.")
                 else:
                     original_utterances_data = utterances
            else:
                original_utterances_data = utterances
                
            # Extract just the text, handling potential missing 'text' field
            conversation_texts = [utt.get('text', '') for utt in original_utterances_data]
            # Also create a normalized version for matching
            normalized_conversation_texts = [normalize_text(text) for text in conversation_texts]
            logger.info(f"Successfully loaded and normalized {len(conversation_texts)} utterance texts from {conversation_path}.")
            
        except FileNotFoundError:
            logger.warning(f"Conversation file not found at {conversation_path}. Proceeding without context.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error processing conversation file {conversation_path}: {e}", exc_info=True)
            conversation_texts = [] # Ensure it's empty on error
            original_utterances_data = []
    elif conversation_path:
         logger.warning(f"Provided conversation path does not exist or is not a file: {conversation_path}")
    else:
         logger.info("No conversation path provided, proceeding without context.")

    # Create the final output structures
    finalized_data = {
        "metadata": prioritized_data.get("metadata", {}),
        "analysis_zones": []
    }
    
    # Add filtering metadata
    finalized_data["metadata"]["filtering"] = {
        "total_decisions": len(ai_decisions),
        "timestamp": str(Path(ai_decisions_path).stat().st_mtime)
    }
    
    # Create separate structures for drill and practice vs conversational teaching
    drill_practice_data = {
        "metadata": {
            "approach": "DRILL_AND_PRACTICE",
            "timestamp": str(Path(ai_decisions_path).stat().st_mtime)
        },
        "utterances": []
    }
    
    conversational_data = {
        "metadata": {
            "approach": "CONVERSATIONAL_TEACHING",
            "timestamp": str(Path(ai_decisions_path).stat().st_mtime)
        },
        "utterances": []
    }
        
    # Extract decisions into a lookup dictionary
    keep_decisions = {}
    for decision in ai_decisions:
        utt_id = decision.get("id", "")
        keep = decision.get("decision", "") == "KEEP"
        reason = decision.get("reason", "")
        teaching_approach = decision.get("teaching_approach", "")
        priority_level = decision.get("priority_level", "")
        approach_reason = decision.get("approach_reason", "")
        target_pattern = decision.get("target_pattern", "")
        
        if utt_id:
            keep_decisions[utt_id] = {
                "keep": keep,
                "reason": reason,
                "teaching_approach": teaching_approach,
                "priority_level": priority_level,
                "approach_reason": approach_reason,
                "target_pattern": target_pattern
            }
            
    logger.info(f"Loaded {len(keep_decisions)} AI filtering decisions")
    
    # Apply filtering to each zone
    finalized_zones = []
    zones = prioritized_data.get("analysis_zones", [])
    
    for zone in zones:
        zone_name = zone.get("zone_name", "Unknown")
        recommendations = zone.get("recommendations", [])
        
        logger.info(f"Processing {len(recommendations)} recommendations in zone {zone_name}")
        
        # Filter recommendations based on AI decisions
        filtered_recommendations = []
        for idx, utt in enumerate(recommendations):
            utt_id = f"{zone_name}_{idx}"
            
            # Check if we should keep this utterance
            decision = keep_decisions.get(utt_id, {"keep": True, "reason": "No decision found (keeping by default)"})
            
            if decision["keep"]:
                # We are keeping this utterance
                
                # Create the restructured utterance, excluding 'metrics' and top-level 'errors'
                input_clauses = utt.get('clauses', [])
                combined_pattern_analysis = []
                processed_clauses = []

                # Iterate through input clauses to collect pattern_analysis and create cleaned clauses
                for clause in input_clauses:
                    if isinstance(clause, dict):
                        # Collect pattern_analysis
                        clause_patterns = clause.get('pattern_analysis')
                        if isinstance(clause_patterns, list):
                            combined_pattern_analysis.extend(clause_patterns)
                        
                        # Create a copy of the clause without pattern_analysis
                        cleaned_clause = clause.copy()
                        cleaned_clause.pop('pattern_analysis', None)
                        processed_clauses.append(cleaned_clause)
                    else:
                        # Keep non-dict items as is (though clauses should be dicts)
                        processed_clauses.append(clause)

                # De-duplicate the combined pattern analysis list
                unique_patterns = []
                seen_patterns = set()
                if combined_pattern_analysis:
                    for pattern_dict in combined_pattern_analysis:
                        # Convert dict to a tuple of sorted items for hashing
                        # Sorting ensures order doesn't affect uniqueness check
                        try:
                            pattern_tuple = tuple(sorted(pattern_dict.items())) 
                            if pattern_tuple not in seen_patterns:
                                unique_patterns.append(pattern_dict)
                                seen_patterns.add(pattern_tuple)
                        except TypeError as e:
                            logging.warning(f"Skipping non-hashable pattern analysis item: {pattern_dict}. Error: {e}")

                restructured_utt = {
                    "original": utt.get('original'),
                    "corrected": utt.get('corrected'),
                    "priority_score": utt.get('priority_score'),
                    "clauses": processed_clauses, # Use the cleaned clauses
                }

                # Add the UNIQUE pattern_analysis at the top level if it's not empty
                if unique_patterns:
                    restructured_utt['pattern_analysis'] = unique_patterns

                # Remove None values for cleaner output (do this after modifications)
                restructured_utt = {k: v for k, v in restructured_utt.items() if v is not None}

                # --- Add Conversation Context (up to 3 preceding utterances with speaker) ---
                original_text = utt.get('original')
                preceding_context_list = [] # Initialize as an empty list
                
                normalized_original = normalize_text(original_text)
                match_index = -1 # Initialize as not found

                if normalized_original and normalized_conversation_texts and original_utterances_data:
                    logger.debug(f"Attempting to find normalized context for: '{normalized_original}' (Original: '{original_text}')")
                    
                    # --- Implement Containment Matching Logic --- 
                    match_index = -1 # Initialize to -1 (not found)
                    for i, normalized_conv_text in enumerate(normalized_conversation_texts):
                        # Check if the (potentially shorter) normalized original utterance 
                        # is contained within the normalized conversation utterance
                        if normalized_original in normalized_conv_text:
                            match_index = i
                            logger.debug(f"Found containment match for '{normalized_original}' within transcript line {i}: '{normalized_conv_text}'")
                            break # Use the first match found
                    # --- End Containment Matching Logic ---    
                    
                    if match_index != -1:
                        # Get up to 3 preceding utterances *with speaker info* using the found index
                        start_index = max(0, match_index - 3)
                        preceding_context_list = [
                            {"speaker": original_utterances_data[i].get('speaker', 'Unknown'), "text": original_utterances_data[i].get('text', '')} 
                            for i in range(start_index, match_index)
                        ]
                        logger.debug(f"Collected preceding context: {preceding_context_list}")

                elif not normalized_conversation_texts:
                    logger.debug("Skipping context search: conversation_texts list is empty or not loaded.")
                elif not normalized_original:
                     logger.debug("Skipping context search: utterance has no 'original' text or normalizes to empty.")

                # Add the context list (even if empty) to the utterance
                restructured_utt['preceding_context'] = preceding_context_list
                # --- End Context Addition ---
                
                # Add the AI's reason and teaching approach as metadata
                restructured_utt["filtering_metadata"] = {
                    "decision": "KEEP",
                    "reason": decision["reason"],
                    "teaching_approach": decision["teaching_approach"],
                    "priority_level": decision["priority_level"],
                    "approach_reason": decision["approach_reason"]
                }
                
                # Add target pattern if it's a drill and practice item
                if decision["teaching_approach"] == "DRILL_AND_PRACTICE" and decision["target_pattern"]:
                    restructured_utt["filtering_metadata"]["target_pattern"] = decision["target_pattern"]
                filtered_recommendations.append(restructured_utt)
                
                # Add to the appropriate teaching approach file
                if decision["teaching_approach"] == "DRILL_AND_PRACTICE":
                    drill_practice_data["utterances"].append(restructured_utt)
                elif decision["teaching_approach"] == "CONVERSATIONAL_TEACHING":
                    conversational_data["utterances"].append(restructured_utt)
            else:
                logger.info(f"Filtering out utterance {utt_id}: {decision['reason']}")
                
        # Create filtered zone
        filtered_zone = {
            "zone_name": zone_name,
            "recommendations": filtered_recommendations
        }
        finalized_zones.append(filtered_zone)
        
        logger.info(f"Zone {zone_name}: Kept {len(filtered_recommendations)} out of {len(recommendations)} recommendations")
    
    # Set the finalized zones in the output data
    finalized_data["analysis_zones"] = finalized_zones
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the final outputs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(finalized_data, f, indent=2, ensure_ascii=False)
        
        # Create learning_value subdirectory
        learning_value_dir = output_path.parent / "learning_value"
        learning_value_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate paths for the teaching approach files
        base_name = output_path.stem
        drill_practice_path = learning_value_dir / f"{base_name}_drill_practice.json"
        conversational_path = learning_value_dir / f"{base_name}_conversational.json"
        
        # Write the teaching approach files
        with open(drill_practice_path, 'w', encoding='utf-8') as f:
            json.dump(drill_practice_data, f, indent=2, ensure_ascii=False)
            
        with open(conversational_path, 'w', encoding='utf-8') as f:
            json.dump(conversational_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Finalized analysis successfully written to {output_path}")
        logger.info(f"Drill and practice utterances written to {drill_practice_path}")
        logger.info(f"Conversational teaching utterances written to {conversational_path}")
    except IOError as e:
        logger.error(f"Error writing output file to {output_path}: {e}")
        sys.exit(1)

def restructure_utterance(utterance):
    """Restructures an utterance to have a single pattern_analysis field.
    
    Args:
        utterance: The original utterance dictionary.
        
    Returns:
        A restructured utterance dictionary with a single pattern_analysis field.
    """
    # Create a copy of the utterance to avoid modifying the original
    restructured = utterance.copy()
    
    # Extract all pattern_analysis from clauses
    all_patterns = []
    clauses = restructured.get('clauses', [])
    
    # Create new clauses list without pattern_analysis
    new_clauses = []
    
    for clause in clauses:
        # Extract patterns from this clause
        patterns = clause.get('pattern_analysis', [])
        all_patterns.extend(patterns)
        
        # Create a new clause without pattern_analysis
        new_clause = {k: v for k, v in clause.items() if k != 'pattern_analysis'}
        new_clauses.append(new_clause)
    
    # Remove duplicates from patterns while preserving order
    unique_patterns = []
    pattern_keys = set()
    
    for pattern in all_patterns:
        # Create a key from the pattern's component and category
        key = (pattern.get('component', ''), pattern.get('category', ''))
        if key not in pattern_keys:
            pattern_keys.add(key)
            unique_patterns.append(pattern)
    
    # Update the utterance with restructured data
    restructured['clauses'] = new_clauses
    restructured['pattern_analysis'] = unique_patterns
    
    return restructured

def main():
    """Main function to run the finalization process."""
    parser = argparse.ArgumentParser(description="Finalize analysis by applying AI filtering decisions.")
    parser.add_argument("prioritized_file", help="Path to the original prioritized utterances JSON file.")
    parser.add_argument("ai_decisions_file", help="Path to the AI filtering decisions JSON file.")
    parser.add_argument("output_file", help="Path for the finalized output JSON file.")
    parser.add_argument("--conversation_file", help="Optional path to the original conversation JSON file.")

    args = parser.parse_args()

    prioritized_path = Path(args.prioritized_file)
    ai_decisions_path = Path(args.ai_decisions_file)
    output_path = Path(args.output_file)
    conversation_path = Path(args.conversation_file) if args.conversation_file else None

    # Basic validation
    if not prioritized_path.is_file():
        logger.error(f"Prioritized file not found: {prioritized_path}")
        sys.exit(1)
    if not ai_decisions_path.is_file():
        logger.error(f"AI decisions file not found: {ai_decisions_path}")
        sys.exit(1)

    finalize_analysis(prioritized_path, ai_decisions_path, output_path, conversation_path)

if __name__ == "__main__":
    main()
