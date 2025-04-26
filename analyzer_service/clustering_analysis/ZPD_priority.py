"""
This script calculates priority scores for utterances in each ZPD zone.

IMPORTANT: This script processes JSON analysis files containing utterance data and proficiency tiers.
The script expects a JSON structure with regions containing:
- proficiency_tier field showing error types and ratios
- utterances with clauses containing errors and pattern analysis
"""

import json
import logging
import os
import statistics
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(
    filename='analysis_errors.log',
    format='[%(levelname)s] %(message)s',
    level=logging.DEBUG
)

# Configuration
CONFIG = {
    'per_zone_max': 10,  # Changed from 6 to 10
    'severity_weights': {
        'critical': 0.4,
        'moderate': 0.2,
        'minor': 0.2
    },
    'zone_order': ['ZPD/2', 'ZPD-1', 'ZPD', 'ZPD+1'], 
    'processing_order': ['ZPD/2', 'ZPD-1', 'ZPD', 'ZPD+1']
}

# Constants for metric keys
METRICS = {
    'complexity': 'C',  # Structural complexity score
    'accuracy': 'A',    # Grammatical accuracy score
    'fluency': 'F'      # Language fluency score
}

# Insert helper functions for configuration validation after CONFIG definition

def _validate_config_keys() -> None:
    required_keys = {'per_zone_max', 'severity_weights', 'zone_order', 'processing_order'}
    missing = required_keys - CONFIG.keys()
    if missing:
        raise RuntimeError(f"Missing config keys: {missing}")


def _validate_severity_weights() -> None:
    required_severity = {'critical', 'moderate', 'minor'}
    if not isinstance(CONFIG['severity_weights'], dict):
        raise RuntimeError("severity_weights must be a dictionary")
    if set(CONFIG['severity_weights'].keys()) != required_severity:
        raise RuntimeError(f"severity_weights must contain exactly: {required_severity}")


def _validate_zone_logic() -> None:
    if not CONFIG['zone_order']:
        raise RuntimeError("zone_order cannot be empty")
    try:
        if CONFIG['zone_order'].index('ZPD') < CONFIG['zone_order'].index('ZPD-1'):
            logging.warning("Zone order puts ZPD before ZPD-1 - intentional?")
    except ValueError:
        raise RuntimeError("zone_order must contain both 'ZPD' and 'ZPD-1'")
    if not CONFIG['processing_order']:
        raise RuntimeError("processing_order cannot be empty")
    if set(CONFIG['processing_order']) != set(CONFIG['zone_order']):
        raise RuntimeError("processing_order must match zone_order")

def validate_config() -> None:
    """Validate configuration settings.

    Raises:
        RuntimeError: If configuration is invalid
    """
    _validate_config_keys()
    _validate_severity_weights()
    _validate_zone_logic()
    if not isinstance(CONFIG['per_zone_max'], int) or CONFIG['per_zone_max'] <= 0:
        raise RuntimeError("per_zone_max must be a positive integer")

    # Check zone order logic
    try:
        zpd_idx = CONFIG['zone_order'].index('ZPD')
        zpd1_idx = CONFIG['zone_order'].index('ZPD-1')
        if zpd_idx < zpd1_idx:
            logging.warning("Zone order puts ZPD before ZPD-1 - intentional?")
    except ValueError:
        raise RuntimeError("zone_order must contain both 'ZPD' and 'ZPD-1'")

    # Validate processing_order
    if not CONFIG['processing_order']:
        raise RuntimeError("processing_order cannot be empty")

    if set(CONFIG['processing_order']) != set(CONFIG['zone_order']):
        raise RuntimeError("processing_order must match zone_order")

# Constants for pattern analysis
PATTERN_DEFAULTS = {
    'frequency_level': 1.0,
    'intention': 'unknown',
    'category': 'uncategorized',
    'component': 'unknown'
}

def extract_patterns(utterance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract patterns from utterance with safe defaults.

    Args:
        utterance: Utterance data containing patterns

    Returns:
        List of processed patterns with normalized frequencies
    """
    patterns = []
    # Iterate through clauses and the pattern_analysis within each clause
    for clause in utterance.get('clauses', []):
        for pattern in clause.get('pattern_analysis', []): 
            if not isinstance(pattern, dict):
                 logging.warning(f"Skipping non-dictionary item in clause patterns: {pattern}")
                 continue
            pattern_data = {
                'frequency': pattern.get('frequency_level', PATTERN_DEFAULTS['frequency_level']),
                'intention': pattern.get('intention', PATTERN_DEFAULTS['intention']),
                'category': pattern.get('category', PATTERN_DEFAULTS['category']),
                'component': pattern.get('component', PATTERN_DEFAULTS['component']),
                'context': pattern.get('context', ''),
                'note': pattern.get('note', '')
            }
            # Normalize frequency to valid range
            pattern_data['frequency'] = max(1.0, min(5.0, pattern_data['frequency']))
            patterns.append(pattern_data)
    return patterns # Fixed indentation

def process_json_utterance(
    utterance: Dict[str, Any],
    error_ratios: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    """Process a single utterance with error handling.

    Example input JSON structure:
    {
        "original": "Required text",
        "corrected": "Corrected version",
        "metrics": {
            "C": 0.7,  # complexity score
            "A": 0.9,  # accuracy score
            "F": 0.6   # fluency score
        },
        "clauses": [{
            "text": "Original text",
            "corrected_segment": "Corrected version",
            "errors": [{
                "category": "Grammatical",
                "severity": "critical",
                "error": "Verb tense mismatch",
                "correction": "Suggested fix"
            }],
            "pattern_analysis": [{
                "intention": "Purpose of pattern",
                "category": "Verb Patterns",
                "component": "Past tense narration",
                "frequency": 4.9,
                "context": "Narrative sections",
                "note": "Optional analysis note"
            }]
        }]
    }

    Args:
        utterance: Raw utterance data to process
        error_ratios: Pre-parsed error ratios from proficiency tier

    Returns:
        Processed utterance dictionary or None if invalid
    """
    if not utterance or not isinstance(utterance, dict):
        logging.warning(f"Skipping invalid utterance data: {utterance}")
        return None

    # Check mandatory fields more robustly
    original_text = utterance.get('original', '')
    corrected_text = utterance.get('corrected', '') # Corrected text might be None or empty
    metrics_dict = utterance.get('metrics', {})
    errors_list = utterance.get('errors', [])
    clauses_list = utterance.get('clauses', [])
    pattern_analysis_list = utterance.get('pattern_analysis', []) # Get pattern_analysis, default to []

    if not original_text:
        logging.warning(f"Skipping utterance with missing 'original' text: {utterance.get('id', 'N/A')}")
        return None
    if not isinstance(metrics_dict, dict):
        logging.warning(f"Invalid 'metrics' format for utterance: {utterance.get('id', 'N/A')}. Using defaults.")
        metrics_dict = {}
    if not isinstance(errors_list, list):
        logging.warning(f"Invalid 'errors' format for utterance: {utterance.get('id', 'N/A')}. Skipping errors.")
        errors_list = []
    if not isinstance(clauses_list, list):
        logging.warning(f"Invalid 'clauses' format for utterance: {utterance.get('id', 'N/A')}. Skipping clauses.")
        clauses_list = []
    if not isinstance(pattern_analysis_list, list):
        logging.warning(f"Invalid 'pattern_analysis' format for utterance: {utterance.get('id', 'N/A')}. Skipping patterns.")
        pattern_analysis_list = []

    # Process metrics
    processed_metrics = {
        'C': metrics_dict.get('C', 0.0),
        'A': metrics_dict.get('A', 0.0),
        'F': metrics_dict.get('F', 0.0)
    }

    # Process errors - ensure structure is valid
    processed_errors = []
    for error in errors_list:
        if isinstance(error, dict) and 'type' in error and 'severity' in error:
            processed_errors.append({
                'type': error.get('type', 'Unknown'),
                'severity': error.get('severity', 'minor'),
                'description': error.get('description', ''),
                'correction': error.get('correction', '')
            })
        else:
            logging.warning(f"Skipping malformed error entry: {error} in utterance {utterance.get('id', 'N/A')}")

    # Extract patterns from pattern_analysis or clauses
    pattern_analysis_list = utterance.get('pattern_analysis', [])
    
    # If no pattern_analysis at utterance level, try to extract from clauses
    if not pattern_analysis_list and clauses_list:
        for clause in clauses_list:
            if 'pattern_analysis' in clause and clause['pattern_analysis']:
                pattern_analysis_list.extend(clause['pattern_analysis'])

    # Ensure patterns key exists and defaults correctly
    patterns = pattern_analysis_list if pattern_analysis_list else []
    
    # Log detailed pattern extraction for debugging
    with open(os.path.join(os.path.dirname(__file__), 'pattern_extraction_debug.txt'), 'a') as f:
        f.write(f"\n\n==== PATTERN EXTRACTION ====\n")
        f.write(f"Utterance: {original_text}\n")
        f.write(f"Pattern analysis list: {pattern_analysis_list}\n")
        f.write(f"Clauses count: {len(clauses_list)}\n")
        for i, clause in enumerate(clauses_list):
            f.write(f"Clause {i} pattern_analysis: {clause.get('pattern_analysis', [])}\n")
        f.write(f"Final patterns extracted: {patterns}\n")

    return {
        'original': original_text,
        'corrected': corrected_text if corrected_text is not None else original_text, # Provide fallback for corrected
        'priority_score': 0.0,  # Initialize with zero, will be updated later
        'metrics': processed_metrics,
        'errors': processed_errors,
        'clauses': clauses_list # Pass clauses through
        # Removed redundant 'patterns' field since pattern information is already in clauses
    }

def parse_json_analysis(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process JSON analysis data with validation.

    Args:
        json_data: Raw JSON data

    Returns:
        Processed analysis by zone
    """
    # Validate JSON structure
    if 'regions' not in json_data:
        raise ValueError("Missing 'regions' key in analysis JSON")

    required_zone_keys = {'utterances', 'proficiency_tier'}
    analysis = {}

    for zone_name, zone_data in json_data['regions'].items():
        # Skip invalid zones
        if zone_name not in CONFIG['zone_order']:
            logging.info(f"Skipping zone '{zone_name}' (not in processing order)")
            continue

        # Validate zone structure
        if not required_zone_keys.issubset(zone_data.keys()):
            logging.error(f"Zone {zone_name} missing required keys")
            continue

        # Parse proficiency tier once per zone
        error_ratios = extract_proficiency_errors(zone_data['proficiency_tier'])

        # Process valid utterances
        valid_utterances = [
            u for u in (
                process_json_utterance(u, error_ratios)
                for u in zone_data['utterances']
            ) if u is not None
        ]

        if valid_utterances:
            # The sorting based on priority score happens later in the main loop
            # after scores are actually calculated.
            analysis[zone_name] = {
                'utterances': valid_utterances[:CONFIG['per_zone_max']],
                'stats': zone_data.get('stats', {}),
                'error_ratios': error_ratios
            }
        else:
            logging.warning(f"No valid utterances found in zone: {zone_name}")

    return analysis

def extract_proficiency_errors(proficiency_str: str) -> Dict[str, float]:
    """Parse proficiency tier string into error ratios.

    Example input: "Advanced (Noun Agreement (0.75), Verb Tense (0.25))"

    Args:
        proficiency_str: String containing proficiency tier and error ratios

    Returns:
        Dictionary mapping error types to their ratios
    """
    error_ratios = {}
    # Corrected regex to parse "Noun Agreement (0.75)"
    pattern = r"([\w\s\/-]+?)\((\d+\.\d+)\)"
    matches = re.findall(pattern, proficiency_str)

    for error_type, ratio in matches:
        error_type = error_type.strip().lower()
        try:
            ratio = float(ratio)
            if 0.0 <= ratio <= 1.0:
                error_ratios[error_type] = ratio
            else:
                logging.warning(f"Invalid ratio {ratio} for {error_type}, skipping")
        except ValueError:
            logging.warning(f"Could not parse ratio for {error_type}, skipping")

    return error_ratios

def calculate_formulaic_score(patterns):
    """Calculate normalized formulaic score based on highest frequency pattern.

    Args:
        patterns: List of pattern dictionaries containing frequency information
        
    Returns:
        Normalized score between 0.0-1.0
    """
    # Log the raw patterns input for debugging
    with open(os.path.join(os.path.dirname(__file__), 'patterns_debug.txt'), 'a') as f:
        f.write(f"\n\n==== PATTERNS INPUT ====\n")
        f.write(f"Patterns count: {len(patterns)}\n")
        for i, p in enumerate(patterns):
            f.write(f"Pattern {i}: {p}\n")
    
    if not patterns:
        logging.debug(f"Formulaic score calculation: No patterns found, returning default score 0.0")
        return 0.0
    
    try:
        # Extract frequency values from patterns
        frequencies = [float(pattern.get('frequency', 0.0)) for pattern in patterns]
        # Calculate score based on highest frequency pattern
        max_frequency = max(frequencies) if frequencies else 0.0
        # Normalize to 0.0-1.0 range (assuming max frequency is 5.0)
        normalized_score = max_frequency / 5.0 if max_frequency > 0 else 0.0
        
        with open(os.path.join(os.path.dirname(__file__), 'formulaic_score_debug.txt'), 'a') as f:
            f.write(f"\n\nFrequencies: {frequencies}\n")
            f.write(f"Max frequency: {max_frequency}\n")
            f.write(f"Normalized score: {normalized_score}\n")
        
        logging.debug(f"Formulaic score calculation: frequencies={frequencies}, max={max_frequency}, normalized={normalized_score}")
        return normalized_score
    except Exception as e:
        logging.error(f"Error calculating formulaic score: {e}")
        with open(os.path.join(os.path.dirname(__file__), 'formulaic_score_debug.txt'), 'a') as f:
            f.write(f"\n\nERROR: {e}\n")
            f.write(f"Patterns: {patterns}\n")
        return 0.0

def calculate_accuracy_factor(accuracy: float) -> float:
    """Calculate accuracy factor based on distance from 0.6 using linear interpolation."""
    factor = 1.0 - abs(accuracy - 0.6) * 2.5
    return max(0.1, min(factor, 1.0))

def calculate_priority(
    errors: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    metrics: Dict[str, float],
    zone_name: str,
    error_ratios: Dict[str, float]
) -> float:
    # Log detailed input for debugging
    with open(os.path.join(os.path.dirname(__file__), 'priority_calculation_debug.txt'), 'a') as f:
        f.write(f"\n\n==== CALCULATE PRIORITY DETAILED INPUT ====\n")
        f.write(f"Zone: {zone_name}\n")
        f.write(f"Errors count: {len(errors)}\n")
        f.write(f"Patterns count: {len(patterns)}\n")
        f.write(f"Patterns: {patterns}\n")
        f.write(f"Metrics: {metrics}\n")
        f.write(f"Error ratios: {error_ratios}\n")
    # Add detailed debug information
    with open(os.path.join(os.path.dirname(__file__), 'priority_debug.txt'), 'a') as f:
        f.write(f"\n\n==== CALCULATE PRIORITY INPUTS ====\n")
        f.write(f"Zone: {zone_name}\n")
        f.write(f"Errors count: {len(errors)}\n")
        f.write(f"Patterns count: {len(patterns)}\n")
        f.write(f"Metrics: {metrics}\n")
        f.write(f"Error ratios: {error_ratios}\n")
    """Calculate priority score with bounds checking.

    Args:
        errors: List of error dictionaries with type and severity
        patterns: List of pattern dictionaries with frequency and category
        metrics: Dictionary with complexity, accuracy, and fluency scores
        zone_name: Current zone name (for logging)
        error_ratios: Pre-parsed error ratios from proficiency tier

    Returns:
        Priority score (can exceed 1.0 for high priority utterances)
    """
    logging.debug("\nPriority calculation:")
    logging.debug(f"Zone: {zone_name}")
    logging.debug(f"Number of errors: {len(errors)}")
    logging.debug(f"Number of patterns: {len(patterns)}")
    logging.debug(f"Metrics: {metrics}")
    logging.debug(f"Error ratios: {error_ratios}")

    # Assign weights to errors
    errors_with_weights = []
    for error in errors:
        err_type = error['type'].lower()
        severity = error['severity'].lower()

        # Determine weight based on ratio tiers or severity
        if err_type in error_ratios:
            ratio = error_ratios[err_type]
            if ratio > 0.4:
                weight = 0.6
                logging.debug(f"Error type {err_type} with ratio {ratio} gets weight {weight}")
            elif ratio > 0.3:
                weight = 0.5
                logging.debug(f"Error type {err_type} with ratio {ratio} gets weight {weight}")
            elif ratio > 0.25:
                weight = 0.4
                logging.debug(f"Error type {err_type} with ratio {ratio} gets weight {weight}")
            else:
                # Fall back to severity weights for low ratios
                weight = CONFIG['severity_weights'][severity]
                logging.debug(f"Error type {err_type} with low ratio {ratio}, falling back to severity weight {weight}")
        else:
            weight = CONFIG['severity_weights'][severity]
            logging.debug(f"Error type {err_type} not in proficiency tier, using severity {severity} weight {weight}")

        errors_with_weights.append({
            'type': err_type,
            'severity': severity,
            'weight': weight
        })

    # Sort errors by weight in descending order
    sorted_errors = sorted(errors_with_weights, key=lambda x: -x['weight'])

    # Calculate boost with new diminishing returns formula
    total_boost = 0
    for i, error in enumerate(sorted_errors):
        weight = error['weight']
        if i == 0:  # First error: full weight
            delta = weight
        elif i == 1:  # Second error: one-third weight
            delta = weight / 3
        else:  # Third and subsequent errors: subtract full weight
            delta = -weight

        total_boost += delta
    
    # Calculate error weight and boost
    error_weight = sum(error['weight'] for error in errors_with_weights) if errors_with_weights else 0.0
    error_weight = min(error_weight, 1.0)  # Normalize to max 1.0
    error_boost = total_boost if total_boost > 0 else 0.0
    
    # Calculate formulaic score
    formulaic_score = calculate_formulaic_score(patterns)
    
    # Calculate accuracy factor
    accuracy = metrics.get('A', 0.5)  # Default to 0.5 if not present
    accuracy_factor = calculate_accuracy_factor(accuracy)
    
    # Calculate final priority score - ensure it's never zero by adding a small base value
    base_value = 0.1  # Base value to ensure non-zero scores
    priority_score = round(base_value + (
        (error_boost * 0.6) +
        (formulaic_score * 0.3) +
        (accuracy_factor * 0.1)
    ), 3)  # Round to 3 decimal places
    
    # Add detailed debug information
    with open(os.path.join(os.path.dirname(__file__), 'priority_debug.txt'), 'a') as f:
        f.write(f"\n\n==== PRIORITY CALCULATION ====\n")
        f.write(f"Zone: {zone_name}\n")
        f.write(f"Error weight: {error_weight}\n")
        f.write(f"Error boost: {error_boost}\n")
        f.write(f"Formulaic score: {formulaic_score}\n")
        f.write(f"Accuracy factor: {accuracy_factor}\n")
        f.write(f"Base value: {base_value}\n")
        f.write(f"Priority score: {priority_score}\n")
    
    return priority_score

def process_analysis(analysis: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Process analysis data and return prioritized utterances per zone."""
    results = {}

    # Process each zone in order
    for zone in CONFIG['processing_order']:
        if zone not in analysis:
            logging.info(f"Skipping zone '{zone}' (not found)")
            continue

        logging.info(f"Processing zone: {zone}")
        utterances = []

        # Process each utterance
        for u in analysis[zone]['utterances']:
            try:
                # Extract patterns from clauses
                all_patterns = []
                # Extract errors from clauses and convert to expected format
                all_errors = []
                
                for clause in u['clauses']:
                    all_patterns.extend(clause.get('pattern_analysis', []))
                    # Extract errors from each clause and convert to expected format
                    if 'errors' in clause:
                        for error in clause.get('errors', []):
                            # Convert from 'category' to 'type' and 'error' to 'description'
                            all_errors.append({
                                'type': error.get('category', ''),
                                'severity': error.get('severity', ''),
                                'description': error.get('error', ''),
                                'correction': error.get('correction', '')
                            })

                # Add errors to utterance for later use in format_output
                u['errors'] = all_errors

                # Calculate priority score using current zone
                u['priority'] = calculate_priority(
                    errors=all_errors,
                    patterns=all_patterns,
                    metrics=u.get('metrics', {}),
                    zone_name=zone,
                    error_ratios=analysis[zone]['error_ratios']
                )

                utterances.append(u)

            except KeyError as e:
                logging.error(f"Error processing utterance: {e}")
                continue

        # Sort by priority score descending
        utterances.sort(key=lambda x: x['priority'], reverse=True)
        results[zone] = utterances

    return results

def format_output(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Format results into the final JSON structure.

    Args:
        results: Processed results by zone

    Returns:
        Formatted output structure
    """
    return {
        'analysis_zones': [
            {
                'zone_name': zone,
                'recommendations': [
                    {
                        'original': utterance['original'],
                        'corrected': utterance['corrected'],
                        'priority_score': utterance['priority'],
                        'metrics': utterance['metrics'],
                        'errors': [
                            {
                                'type': error['type'],
                                'severity': error['severity'],
                                'description': error['description'],
                                'correction': error['correction']
                            }
                            for error in utterance['errors']
                        ],
                        # Reverted: Directly copy clauses from input utterance
                        'clauses': utterance['clauses']
                    }
                    for utterance in results[zone]
                ]
            }
            for zone in sorted(results.keys())
        ]
    }

def main(input_file: str) -> str:
    """Main function to process JSON analysis file and output prioritized utterances.

    Args:
        input_file: Path to input JSON file

    Returns:
        Path to output JSON file

    Raises:
        RuntimeError: If configuration is invalid
        ValueError: If JSON structure is invalid
        FileNotFoundError: If input file doesn't exist
    """
    try:
        # Validate configuration first
        validate_config()

        # Load and process JSON data
        with open(input_file) as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in input file: {e}")

        # Process JSON data
        analysis = parse_json_analysis(json_data)

        # Process analysis
        results = process_analysis(analysis)

        # Removed the redundant processing loop that recalculated scores and potentially dropped data.
        # The 'results' dictionary from process_analysis already contains the correctly scored and sorted utterances.
        prioritized_output: Dict[str, List[Dict[str, Any]]] = {
            'metadata': json_data.get('metadata', {}),
            'analysis_zones': []
        }

        # Use the 'results' from process_analysis directly to format the output
        for zone_name in CONFIG['processing_order']:
            if zone_name in results:
                # Get the top N utterances for the zone based on the already calculated priority
                top_utterances = results[zone_name][:CONFIG['per_zone_max']]
                
                # Format the recommendations for this zone
                zone_recommendations = [
                    {
                        'original': utterance['original'],
                        'corrected': utterance['corrected'],
                        'priority_score': utterance.get('priority', 0.0), # Use pre-calculated priority
                        'metrics': utterance['metrics'],
                        'errors': [
                            {
                                'type': error['type'],
                                'severity': error['severity'],
                                'description': error['description'],
                                'correction': error['correction']
                            }
                            for error in utterance.get('errors', []) # Use errors stored during process_analysis
                        ],
                        'clauses': utterance.get('clauses', []) # Use clauses stored during process_analysis
                    }
                    for utterance in top_utterances
                ]
                
                prioritized_output['analysis_zones'].append({
                    'zone_name': zone_name,
                    'recommendations': zone_recommendations
                })
            else:
                logging.warning(f"Zone '{zone_name}' processed but not found in final results. Adding empty zone.")
                prioritized_output['analysis_zones'].append({'zone_name': zone_name, 'recommendations': []})

        # Save the correctly formatted output to JSON file
        output_file = input_file.replace('.json', '_prioritized.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prioritized_output, f, indent=2, ensure_ascii=False) # Use prioritized_output
        logging.info(f"Results written to: {output_file}")

        # Also print human-readable format to console
        # (Optional: Adjust this print section if needed, but the core logic is fixed)
        for zone in prioritized_output['analysis_zones']:
            print(f"\n{zone['zone_name']} Top Candidates ({len(zone['recommendations'])}):")
            for rec in zone['recommendations']:
                print(f"  Original: {rec['original']}")
                print(f"  Correct: {rec['corrected']}")
                print(f"  Priority score: {rec['priority_score']:.3f}")
                print(f"  Metrics: C={rec['metrics'].get('C', 0.0):.2f}, A={rec['metrics'].get('A', 0.0):.2f}, F={rec['metrics'].get('F', 0.0):.2f}")
                
                if rec['errors']:
                    print("  Errors:")
                    for error in rec['errors']:
                        print(f"  - {error['type']} ({error['severity']}): {error['description']}")
                        # print(f"    Correction: {error['correction']}") # Optional: uncomment if needed
                else:
                    print("  Errors: None")
                    
                # Optionally print clauses/patterns if needed for console output
                # if rec['clauses']:
                #    print("  Clauses: ... ") # Add detailed clause/pattern printing if required
                print()

        return output_file

    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logging.error(f"Error processing file: {e}")
        raise

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ZPD_priority.py <input_json_file>")
        sys.exit(1)
    main(sys.argv[1])
