import json
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import re
from collections import defaultdict

# Configure basic logging for verification
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --- Copied Configuration & Constants from ZPD_priority.py ---
CONFIG = {
    'per_zone_max': 10,
    'severity_weights': {
        'critical': 0.4,
        'moderate': 0.2,
        'minor': 0.2
    },
    # Zone order isn't strictly needed for verification logic itself
}

METRICS = {
    'complexity': 'C',
    'accuracy': 'A',
    'fluency': 'F'
}

PATTERN_DEFAULTS = {
    'frequency_level': 1.0,
    'intention': 'unknown',
    'category': 'uncategorized',
    'component': 'unknown'
}

# --- Copied Helper Functions from ZPD_priority.py ---

def extract_proficiency_errors(proficiency_str: str) -> Dict[str, float]:
    """Parse proficiency tier string into error ratios."""
    error_ratios = {}
    match = re.search(r'\((.*?)\)', proficiency_str)
    if match:
        errors_part = match.group(1)
        # Regex to find error types and their ratios
        pattern = re.compile(r'([\w\s/-]+?)\s*\((\d\.\d+)\)')
        matches = pattern.findall(errors_part)
        for error_type, ratio in matches:
            # Handle potential variations like 'korean_vocabulary'
            processed_type = error_type.strip().lower()
            try:
                error_ratios[processed_type] = float(ratio)
            except ValueError:
                logging.warning(f"Could not parse ratio for {error_type}: {ratio}")
    return error_ratios

def extract_patterns(utterance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract patterns from utterance with safe defaults."""
    patterns = []
    for clause in utterance.get('clauses', []):
        for pattern in clause.get('pattern_analysis', []):
            pattern_data = {
                'frequency': pattern.get('frequency', PATTERN_DEFAULTS['frequency_level']), # Correct key is 'frequency'
                'intention': pattern.get('intention', PATTERN_DEFAULTS['intention']),
                'category': pattern.get('category', PATTERN_DEFAULTS['category']),
                'component': pattern.get('component', PATTERN_DEFAULTS['component']),
                'context': pattern.get('context', ''),
                'note': pattern.get('note', '')
            }
            # Normalize frequency
            try:
                freq = float(pattern_data['frequency'])
                pattern_data['frequency'] = max(1.0, min(5.0, freq))
            except (ValueError, TypeError):
                 pattern_data['frequency'] = 1.0 # Default if conversion fails

            patterns.append(pattern_data)
    return patterns


def calculate_formulaic_score(patterns: List[Dict[str, Any]]) -> float:
    """Calculate normalized formulaic score based on highest frequency pattern."""
    if not patterns:
        return 0.0

    max_freq = 0.0
    for pattern in patterns:
         try:
             freq = float(pattern.get('frequency', 1.0))
             max_freq = max(max_freq, freq)
         except (ValueError, TypeError):
             continue # Ignore patterns with invalid frequency

    # Normalize score: 1.0 -> 0.0, 5.0 -> 1.0
    normalized_score = (max(1.0, max_freq) - 1.0) / 4.0
    return round(normalized_score, 3)


def calculate_accuracy_factor(accuracy: float) -> float:
    """Calculate accuracy factor based on distance from 0.6."""
    # Clamp accuracy between 0 and 1
    accuracy = max(0.0, min(1.0, accuracy))
    # Linear decay away from 0.6
    factor = 1.0 - abs(accuracy - 0.6) / 0.4
    return round(max(0.0, factor), 3) # Ensure non-negative


def calculate_priority(
    errors: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    metrics: Dict[str, float],
    zone_name: str,
    error_ratios: Dict[str, float]
) -> float:
    """Calculate priority score."""
    # Assign weights to errors
    errors_with_weights = []
    for error in errors:
        # Added lower() for robustness
        err_type = error.get('type', 'unknown').lower()
        severity = error.get('severity', 'minor').lower()

        # Check if severity is valid before accessing weight
        if severity not in CONFIG['severity_weights']:
            logging.warning(f"Unknown severity '{severity}' for error type '{err_type}' in zone {zone_name}. Using minor weight.")
            severity = 'minor'


        # Determine weight based on ratio tiers or severity
        weight = 0.0 # Default weight
        if err_type in error_ratios:
            ratio = error_ratios[err_type]
            if ratio > 0.4:
                weight = 0.6
            elif ratio > 0.3:
                weight = 0.5
            elif ratio > 0.25:
                weight = 0.4
            else:
                weight = CONFIG['severity_weights'][severity]
        else:
             weight = CONFIG['severity_weights'][severity]

        errors_with_weights.append({'weight': weight})

    # Sort errors by weight in descending order
    sorted_errors = sorted(errors_with_weights, key=lambda x: -x['weight'])

    # Calculate boost with diminishing returns
    total_boost = 0
    for i, error in enumerate(sorted_errors):
        weight = error['weight']
        if i == 0: delta = weight
        elif i == 1: delta = weight / 3
        else: delta = -weight
        total_boost += delta

    formulaic_score = calculate_formulaic_score(patterns)
    accuracy = metrics.get(METRICS['accuracy'], 0.6)
    accuracy_factor = calculate_accuracy_factor(accuracy)

    final_score = round(formulaic_score * (1 + total_boost) * accuracy_factor, 3)
    return max(0.0, final_score)

def process_json_utterance_for_verification(
    utterance: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Simplified processing just to get data needed for calculate_priority."""
    try:
        if not utterance.get('original') or not utterance.get('clauses'):
            return None

        errors = []
        for clause in utterance.get('clauses', []):
            for error in clause.get('errors', []):
                if error.get('category') and error.get('severity'):
                    errors.append({
                        'type': error['category'],
                        'severity': error['severity'],
                    })
        if not errors: # Skip utterances without identified errors
             return None

        metrics = utterance.get('metrics', {})
        # Ensure metrics needed are present with defaults
        processed_metrics = {
            METRICS['accuracy']: metrics.get(METRICS['accuracy'], 0.6),
            METRICS['complexity']: metrics.get(METRICS['complexity'], 0.5),
            METRICS['fluency']: metrics.get(METRICS['fluency'], 0.5)
        }


        patterns = extract_patterns(utterance)

        return {
            'original': utterance['original'],
            'errors': errors,
            'patterns': patterns,
            'metrics': processed_metrics
        }
    except Exception as e:
        logging.error(f"Error processing utterance {utterance.get('original', 'Unknown')}: {e}")
        return None

# --- Main Verification Logic ---

def verify_selection(secondary_path: Path, prioritized_path: Path):
    """Verifies the top N selection from secondary to prioritized JSON."""
    try:
        with open(secondary_path, 'r') as f:
            secondary_data = json.load(f)
        with open(prioritized_path, 'r') as f:
            prioritized_data_wrapper = json.load(f)
            # Adjust based on the actual structure of prioritized JSON
            prioritized_data = {zone['zone_name']: zone['recommendations']
                                for zone in prioritized_data_wrapper.get('analysis_zones', [])}

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return

    logging.info(f"Starting verification for {secondary_path.name} against {prioritized_path.name}")

    # Iterate through zones defined within the 'regions' key
    if 'regions' not in secondary_data:
        logging.error("Key 'regions' not found in secondary data. Cannot verify ZPD zones.")
        return
    zones_to_check = secondary_data['regions'].keys()

    all_match = True
    for zone_name in zones_to_check:
        logging.info(f"--- Verifying Zone: {zone_name} ---")

        # Access the specific zone data within 'regions'
        zone_data = secondary_data['regions'][zone_name]
        proficiency_tier = zone_data.get('proficiency_tier', "")
        error_ratios = extract_proficiency_errors(proficiency_tier)
        utterances_in_zone = zone_data.get('utterances', [])

        calculated_priorities = []
        for utt in utterances_in_zone:
            processed_utt = process_json_utterance_for_verification(utt)
            if processed_utt:
                score = calculate_priority(
                    errors=processed_utt['errors'],
                    patterns=processed_utt['patterns'],
                    metrics=processed_utt['metrics'],
                    zone_name=zone_name,
                    error_ratios=error_ratios
                )
                calculated_priorities.append({'original': processed_utt['original'], 'score': score})

        # Sort by score (descending), then by original text (ascending) for stable sort
        calculated_priorities.sort(key=lambda x: (-x['score'], x['original']))

        # Get top N based on config
        top_n = CONFIG['per_zone_max']
        calculated_top_n_originals = {item['original'] for item in calculated_priorities[:top_n]}

        # Get originals from the prioritized file for this zone
        prioritized_originals = set()
        if zone_name in prioritized_data:
            prioritized_originals = {item['original'] for item in prioritized_data[zone_name]}
        else:
             logging.warning(f"Zone '{zone_name}' not found in prioritized file.")


        # Compare
        if calculated_top_n_originals == prioritized_originals:
            logging.info(f"Zone '{zone_name}': OK - Top {len(calculated_top_n_originals)} utterances match.")
        else:
            all_match = False
            logging.error(f"Zone '{zone_name}': MISMATCH!")
            logging.error(f"  Calculated Top {top_n} Originals ({len(calculated_top_n_originals)}): {sorted(list(calculated_top_n_originals))}")
            logging.error(f"  Prioritized File Originals ({len(prioritized_originals)}): {sorted(list(prioritized_originals))}")
            missing_in_prioritized = calculated_top_n_originals - prioritized_originals
            extra_in_prioritized = prioritized_originals - calculated_top_n_originals
            if missing_in_prioritized:
                 logging.error(f"    Missing in prioritized: {sorted(list(missing_in_prioritized))}")
            if extra_in_prioritized:
                 logging.error(f"    Extra in prioritized: {sorted(list(extra_in_prioritized))}")

    logging.info("--- Verification Complete ---")
    if all_match:
        logging.info("All zones match successfully!")
    else:
        logging.error("Some zones have discrepancies.")


if __name__ == "__main__":
    # Define the paths relative to the script location or use absolute paths
    # Assuming the script is in analyzer_service/clustering_analysis
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent # Go up two levels
    secondary_file = project_root / "analysis_output/660_golf/clustering_analysis/660_golf_secondary.json"
    prioritized_file = project_root / "analysis_output/660_golf/clustering_analysis/660_golf_secondary_prioritized.json"

    if not secondary_file.exists():
         print(f"Error: Secondary file not found at {secondary_file}")
    elif not prioritized_file.exists():
         print(f"Error: Prioritized file not found at {prioritized_file}")
    else:
        verify_selection(secondary_file, prioritized_file)
