import re
import json
import sys
import os
import pathlib

# Add the parent directory to sys.path to allow importing from clustering_analysis
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from clustering_analysis.ZPD_analyzer import CAFClusterAnalyzer

def parse_primary_output(json_path):
    """Load utterance and zone data from the primary JSON"""
    print("\nParsing primary output file...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract all utterances from clusters
    utterances = []
    for cluster in data['clusters']:
        utterances.extend(cluster['utterances'])

    print(f"Total utterances parsed: {len(utterances)}")
    print("\nSample complexities found:")
    for i, u in enumerate(utterances[:5]):  # Show first 5 utterances
        print(f"  Utterance {i+1}: C={u['metrics']['C']:.3f}")

    return data

def write_secondary_analysis(output_file, data, region_utterances, region_bounds):
    print("Debug - Starting write_secondary_analysis")
    with open(output_file, 'w') as f:
        # Write global statistics
        f.write("Global Statistics:\n")
        f.write("================\n")
        global_stats = data['global_stats']
        f.write(f"Total Utterances: {global_stats['total_utterances']}\n\n")

        f.write("Feature Averages:\n")
        for feature, stats in global_stats['feature_averages'].items():
            if feature != 'error_distribution':
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
                f.write(f"  {feature.capitalize()}: mean={mean_val:.3f}, std={std_val:.3f}\n")

        f.write("\nError Distribution:\n")
        # Retrieve error_distribution from within feature_averages
        ed_global = global_stats["feature_averages"].get("error_distribution", {})
        print("Debug - Global error_distribution:", ed_global)
        for error_type, stats in ed_global.items():
            print(f"Debug - Processing error_type {error_type} with stats: {stats}")
            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 0.0)
            f.write(f"  {error_type.capitalize()}: mean={mean_val:.3f}, std={std_val:.3f}\n")

        # Log warnings for clusters with incomplete error_distribution
        if 'clusters' in data:
            for cluster in data['clusters']:
                ed = cluster.get('feature_averages', {}).get('error_distribution', {})
                for etype in ["critical", "moderate", "minor"]:
                    if etype not in ed or "mean" not in ed.get(etype, {}) or "std" not in ed.get(etype, {}):
                        print(f"WARNING: Cluster {cluster.get('zone_id', 'unknown')} has incomplete error_distribution for {etype}: {ed.get(etype)}")

        # Write tendency zone information
        tendency = data['tendency_zone']
        f.write("\nTendency Zone:\n")
        f.write("=============\n")
        f.write(f"Size: {tendency['size']} utterances\n")
        f.write(f"Distance from global mean: {tendency['distance_from_global_mean']:.3f}\n")

        # Write tendency zone feature averages and error distribution
        f.write("\nTendency Zone Feature Averages:\n")
        for feature, stats in tendency['feature_averages'].items():
            if feature != 'error_distribution':
                f.write(f"  {feature.capitalize()}: mean={stats['mean']:.3f}, std={stats['std']:.3f}\n")

        f.write("\nTendency Zone Error Distribution:\n")
        ed_tendency = tendency['feature_averages'].get('error_distribution', {})
        print("Debug - Tendency zone error_distribution:", ed_tendency)
        for error_type, stats in ed_tendency.items():
            print(f"Debug - Processing tendency error_type {error_type} with stats: {stats}")
            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 0.0)
            f.write(f"  {error_type.capitalize()}: mean={mean_val:.3f}, std={std_val:.3f}\n")

        # Write region statistics
        for region_name, utterances in region_utterances.items():
            f.write(f"\n{region_name} Region Statistics:\n")
            f.write("=" * (len(region_name) + 20) + "\n")
            f.write(f"  Number of utterances: {len(utterances)}\n")

            if utterances:
                avg_complexity = sum(u['metrics']['C'] for u in utterances) / len(utterances)
                avg_accuracy = sum(u['metrics']['A'] for u in utterances) / len(utterances)

                f.write(f"  Average Complexity: {avg_complexity:.3f}\n")
                f.write(f"  Average Accuracy: {avg_accuracy:.3f}\n")

                # Calculate error averages from metrics
                avg_critical = sum(u['metrics'].get('Critical', 0) for u in utterances) / len(utterances)
                avg_moderate = sum(u['metrics'].get('Moderate', 0) for u in utterances) / len(utterances)
                avg_minor = sum(u['metrics'].get('Minor', 0) for u in utterances) / len(utterances)

                f.write(f"  Average Error Rates:\n")
                f.write(f"    Critical: {avg_critical:.3f}\n")
                f.write(f"    Moderate: {avg_moderate:.3f}\n")
                f.write(f"    Minor: {avg_minor:.3f}\n")
            f.write("\n")

        # Write detailed utterance information for each region
        for region_name, utterances in region_utterances.items():
            f.write(f"\n{region_name} Region Utterances:\n")
            f.write("=" * (len(region_name) + 18) + "\n")

            # Add zone width information if available
            if region_name in region_bounds and utterances:
                lower, upper = region_bounds[region_name]
                avg_complexity = sum(u['metrics']['C'] for u in utterances) / len(utterances)
                f.write(f"Zone Width: {lower:.3f} <= complexity({avg_complexity:.3f}) <= {upper:.3f}\n")

            f.write("\n")

            # Sort utterances by complexity (low to high)
            utterances.sort(key=lambda x: x['metrics']['C'])

            for utterance in utterances:
                # Show actual original/corrected fields (which are empty)
                f.write(f"  Original: {utterance['original']}\n")
                f.write(f"  Corrected: {utterance['corrected']}\n")

                # Write normalized features
                metrics = utterance['metrics']
                f.write(f"  Metrics:\n")
                f.write(f"    C: {metrics['C']:.3f}, A: {metrics['A']:.3f}\n")
                if 'Critical' in metrics:
                    f.write(f"    Error Rates: Critical={metrics['Critical']:.3f}, ")
                    f.write(f"Moderate={metrics['Moderate']:.3f}, ")
                    f.write(f"Minor={metrics['Minor']:.3f}\n")

                # Show clauses content separately
                if 'clauses' in utterance and utterance['clauses']:
                    f.write("  Clauses:\n")
                    for i, clause in enumerate(utterance['clauses']):
                        f.write(f"    Clause {i+1}:\n")
                        f.write(f"      Text: {clause.get('clause_text', '')}\n")
                        if not clause.get('is_error_free', False):
                            f.write(f"      Corrected: {clause.get('corrected_segment', clause.get('clause_text', ''))}\n")

                        # Write pattern analysis from clauses
                        patterns_source = clause.get('pattern_analysis') # Get potential patterns
                        if not patterns_source and 'pattern_analysis' in utterance:
                            # If no pattern analysis in clause, use the utterance-level pattern analysis
                            patterns_source = utterance.get('pattern_analysis')

                        if patterns_source:
                            f.write("      Pattern Analysis:\n")
                            # Check if patterns_source is a dict containing 'patterns' or a list itself
                            actual_patterns = []
                            if isinstance(patterns_source, dict):
                                actual_patterns = patterns_source.get('patterns', [])
                            elif isinstance(patterns_source, list):
                                actual_patterns = patterns_source # Assume it's already a list of patterns

                            for pattern in actual_patterns:
                                if not isinstance(pattern, dict):
                                    print(f"WARNING in write_secondary_analysis: Found non-dict item in 'patterns' list: {repr(pattern)}")
                                    continue # Skip this item
                                f.write(f"        - Intention: {pattern.get('intention', 'N/A')}\n")
                                f.write(f"          Category: {pattern.get('category', 'N/A')}\n")
                                f.write(f"          Component: {pattern.get('component', 'N/A')}\n")
                                f.write(f"          Frequency: {pattern.get('frequency_level', 'N/A')}\n")
                                f.write(f"          Context: {pattern.get('usage_context', 'N/A')}\n")
                                if 'relative_note' in pattern:
                                    f.write(f"          Note: {pattern['relative_note']}\n") # Keep direct access if key presence is checked

                        # Write errors from clauses
                        errors = clause.get('errors_found', [])
                        if errors:
                            f.write("      Errors:\n")
                            for error in errors:
                                f.write(f"        - {error['category']} ({error['severity']}): {error['error']}\n")
                                if error['category'] != 'korean_vocabulary' and 'correction' in error:
                                    f.write(f"          Correction: {error['correction']}\n")

                f.write("\n")

    print("Debug - Finishing write_secondary_analysis_to_text")

def write_secondary_json(output_file, data, region_utterances, region_bounds):
    print("Debug - Starting write_secondary_json")
    # Create dummy analyzer instance for error analysis
    analyzer = CAFClusterAnalyzer.__new__(CAFClusterAnalyzer)
    analyzer.error_weights = {
        'critical': 1.0,  # All weights set to 1.0 to count raw frequency
        'moderate': 1.0,
        'minor': 1.0
    }

    if not isinstance(data, dict):
        print(f"CRITICAL ERROR in write_secondary_json: Input 'data' is not a dict! Type: {type(data)}")
        return # Cannot proceed

    required_top_keys = ['global_stats', 'tendency_zone']
    for key in required_top_keys:
        if key not in data:
            print(f"CRITICAL ERROR in write_secondary_json: Input 'data' is missing key '{key}'!")
            return # Cannot proceed
        if not isinstance(data[key], dict):
            print(f"CRITICAL ERROR in write_secondary_json: 'data[\'{key}\']' is not a dict! Type: {type(data[key])}")
            return # Cannot proceed

    # Specific checks for nested structures needed for initialization
    if 'feature_averages' not in data['global_stats'] or not isinstance(data['global_stats']['feature_averages'], dict):
        print(f"CRITICAL ERROR: 'data[\'global_stats\'][\'feature_averages\']' is missing or not a dict!")
        return # Cannot proceed
    if 'total_utterances' not in data['global_stats']:
        print(f"CRITICAL ERROR: 'data[\'global_stats\'][\'total_utterances\']' is missing!")
        return # Cannot proceed
    if 'size' not in data['tendency_zone'] or 'distance_from_global_mean' not in data['tendency_zone']:
        print(f"CRITICAL ERROR: 'data[\'tendency_zone\']' is missing keys ('size', 'distance_from_global_mean')!")
        return # Cannot proceed

    analysis = {
        "global_statistics": {
            "total_utterances": data['global_stats']['total_utterances'],
            "feature_averages": {k: dict(v) for k, v in data['global_stats']['feature_averages'].items()},
            "error_distribution": {k: dict(v) for k, v in data['global_stats']['feature_averages'].get('error_distribution', {}).items()}
        },
        "tendency_zone": {
            "size": data['tendency_zone']['size'],
            "distance_from_global_mean": data['tendency_zone']['distance_from_global_mean']
        },
        "regions": {}
    }

    # Process each region
    for region_name, utterances in region_utterances.items():
        region_data = {
            "statistics": {
                "utterance_count": len(utterances),
                "complexity_bounds": region_bounds.get(region_name, (None, None)),
                "averages": {}
            }
        }

        if utterances:
            # Calculate averages - Potential error source
            try:
                region_data['statistics']['averages'] = {
                    "complexity": sum(u['metrics']['C'] for u in utterances) / len(utterances),
                    "accuracy": sum(u['metrics']['A'] for u in utterances) / len(utterances),
                    "error_rates": {
                        "critical": sum(u['metrics'].get('Critical', 0) for u in utterances) / len(utterances),
                        "moderate": sum(u['metrics'].get('Moderate', 0) for u in utterances) / len(utterances),
                        "minor": sum(u['metrics'].get('Minor', 0) for u in utterances) / len(utterances)
                    }
                }
            except (TypeError, KeyError) as e:
                print(f"CRITICAL ERROR calculating averages for region {region_name}: {e}")
                print(f"Problematic utterance list sample: {repr(utterances[:2])}")
                # Set defaults or skip if critical
                region_data['statistics']['averages'] = {"complexity": 0, "accuracy": 0, "error_rates": {"critical":0, "moderate":0, "minor":0}}
                # Optionally continue to next region: continue

            # Add proficiency tier - Potential error source
            if region_name != 'Filtered':
                try:
                    avg_caf = [
                        region_data['statistics']['averages']['complexity'],
                        region_data['statistics']['averages']['accuracy'],
                    ]
                    # Generate proficiency tier with error patterns
                    region_data['proficiency_tier'] = analyzer._generate_zone_name(avg_caf, utterances)
                except (TypeError, KeyError, AttributeError) as e: # Catch AttributeError too
                    print(f"CRITICAL ERROR generating proficiency tier for region {region_name}: {type(e).__name__}: {e}") # Log specific error
                    print(f"Problematic utterance list sample: {repr(utterances[:2])}")
                    region_data['proficiency_tier'] = "Error Generating Tier"
                    # Optionally continue to next region: continue

        # Add utterances list after statistics and proficiency_tier
        region_data['utterances'] = []

        # Process individual utterances
        for i, u in enumerate(utterances):
            # Ensure 'u' is a dictionary before attempting key access
            if not isinstance(u, dict):
                print(f"CRITICAL ERROR: Utterance item {i} in region {region_name} is not a dict: {type(u)}. Skipping.")
                continue # Skip this non-dictionary utterance

            # --- Initialize basic utterance data --- 
            utterance_data = {
                "original": u.get('original', ''),
                "corrected": u.get('corrected', ''),
                "metrics": {}, # Initialize empty, populate safely
                "clauses": [], # Initialize empty, populate in try block
                # Include utterance-level pattern analysis if available
                "pattern_analysis": [] 
            }
            # Safely populate metrics
            if isinstance(u.get('metrics'), dict):
                 utterance_data['metrics'] = {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in u['metrics'].items()}
            else:
                 print(f"WARNING: Metrics missing or not a dict for utterance {i} in region {region_name}. Using empty metrics.")

            # --- Try processing clauses and utterance-level patterns --- 
            try:
                # Process clauses
                for clause in u.get('clauses', []):
                    # ... (Existing clause processing logic) ...
                    clause_data = {
                        "text": clause.get('clause_text', ''),
                        **({} if clause.get('is_error_free', False) else {"corrected_segment": clause.get('corrected_segment', clause.get('clause_text', ''))}),
                        "errors": [],
                        "pattern_analysis": []
                    }
                    # Process errors safely
                    for e in clause.get('errors_found', []):
                        if isinstance(e, dict):
                            clause_data['errors'].append({
                                "category": e.get('category', 'Unknown'),
                                "severity": e.get('severity', 'Unknown'),
                                "error": e.get('error', 'Unknown'),
                                **({'correction': e['correction']} if e.get('category') != 'korean_vocabulary' and 'correction' in e else {})
                            })
                        else:
                            print(f"WARNING: Non-dict item found in errors_found for clause in utterance {i}, region {region_name}: {repr(e)}")
                    
                    # Process clause-level pattern analysis safely
                    clause_patterns_source = clause.get('pattern_analysis')
                    if clause_patterns_source:
                        actual_patterns = []
                        if isinstance(clause_patterns_source, dict):
                            actual_patterns = clause_patterns_source.get('patterns', [])
                        elif isinstance(clause_patterns_source, list):
                             actual_patterns = clause_patterns_source # Assume list contains pattern dicts
                        else:
                             print(f"WARNING: Unexpected type for clause pattern_analysis in utterance {i}, region {region_name}: {type(clause_patterns_source)}")
                        
                        for p in actual_patterns:
                             if isinstance(p, dict):
                                 clause_data['pattern_analysis'].append({
                                     "intention": p.get('intention', 'N/A'),
                                     "category": p.get('category', 'N/A'),
                                     "component": p.get('component', 'N/A'),
                                     "frequency": p.get('frequency_level', 'N/A'),
                                     "context": p.get('usage_context', 'N/A'),
                                     "note": p.get('relative_note', None)
                                 })
                             else:
                                 print(f"WARNING in write_secondary_json: Found non-dict item in clause 'patterns' list: {repr(p)}")

                    utterance_data['clauses'].append(clause_data)

                # Process utterance-level pattern analysis safely (if clauses didn't have any)
                if not any(c.get('pattern_analysis') for c in utterance_data['clauses']): # Check if any clause processed patterns
                    utterance_patterns_source = u.get('pattern_analysis')
                    if utterance_patterns_source:
                        actual_patterns = []
                        if isinstance(utterance_patterns_source, dict):
                             actual_patterns = utterance_patterns_source.get('patterns', [])
                        elif isinstance(utterance_patterns_source, list):
                             actual_patterns = utterance_patterns_source
                        else:
                             print(f"WARNING: Unexpected type for utterance pattern_analysis in utterance {i}, region {region_name}: {type(utterance_patterns_source)}")
                        
                        for p in actual_patterns:
                            if isinstance(p, dict):
                                utterance_data['pattern_analysis'].append({
                                    "intention": p.get('intention', 'N/A'),
                                    "category": p.get('category', 'N/A'),
                                    "component": p.get('component', 'N/A'),
                                    "frequency": p.get('frequency_level', 'N/A'),
                                    "context": p.get('usage_context', 'N/A'),
                                    "note": p.get('relative_note', None)
                                })
                            else:
                                print(f"WARNING in write_secondary_json: Found non-dict item in utterance 'patterns' list: {repr(p)}")

            except Exception as e:
                # Catch errors during clause/pattern processing specifically
                print(f"ERROR processing clauses/patterns for utterance {i} in region {region_name}: {type(e).__name__}: {e}")
                print(f"Problematic utterance data (first 100 chars): {repr(u)[:100]}")
                # Don't continue; proceed to append partially processed data

            # --- Append utterance data (always runs) --- 
            region_data['utterances'].append(utterance_data)

        analysis['regions'][region_name] = region_data

    # Write final JSON output
    try:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=4)
    except Exception as e:
        print(f"CRITICAL ERROR writing final JSON file {output_file}: {type(e).__name__}: {e}") # Log JSON write error

    print("Debug - Finishing write_secondary_json")

def define_learning_regions(data):
    print("Debug - Starting define_learning_regions")
    print("Debug - Tendency zone:", data['tendency_zone'])
    tendency = data['tendency_zone']
    mean_complexity = tendency['feature_averages']['complexity']['mean']
    std_complexity = tendency['feature_averages']['complexity']['std']

    # Define region bounds using mean and standard deviation
    regions = {
        'ZPD/2': (mean_complexity - 2*std_complexity, mean_complexity - std_complexity),
        'ZPD-1': (mean_complexity - std_complexity, mean_complexity),
        'ZPD': (mean_complexity, mean_complexity + std_complexity),
        'ZPD+1': (mean_complexity + std_complexity, mean_complexity + 3*std_complexity)
    }

    return regions

def generate_secondary_analysis(data, regions, output_base):
    """Generate the secondary analysis output file."""
    print("\nGenerating secondary analysis...")

    # Extract all utterances from clusters
    utterances = []
    for cluster in data['clusters']:
        utterances.extend(cluster['utterances'])

    # First, separate out utterances with perfect accuracy
    filtered_utterances = []
    imperfect_utterances = []
    for u in utterances:
        if u['metrics']['A'] == 1.0:
            filtered_utterances.append(u)
        else:
            imperfect_utterances.append(u)

    # Print region bounds
    print("\nRegion bounds:")
    for region_name, bounds in regions.items():
        print(f"  {region_name}: {bounds}")

    # Classify remaining utterances into regions
    region_utterances = {
        'ZPD/2': [],
        'ZPD-1': [],
        'ZPD': [],
        'ZPD+1': [],
        'Filtered': filtered_utterances
    }

    # Only classify imperfect utterances into ZPD regions
    for utterance in imperfect_utterances:
        if not isinstance(utterance, dict):
            print(f"CRITICAL ERROR classifying utterance: Item is not a dict! Type: {type(utterance)}, Value: {repr(utterance)}")
            continue
        if 'metrics' not in utterance or 'C' not in utterance['metrics']:
            print(f"CRITICAL ERROR classifying utterance: Missing 'metrics' or 'C' key! Value: {repr(utterance)}")
            continue

        complexity = utterance['metrics']['C']

        # Check each region in order
        for region_name, (lower, upper) in regions.items():
            if lower <= complexity <= upper:
                region_utterances[region_name].append(utterance)
                break

    # Print classification summary
    print("\nClassification summary:")
    for region_name, utterances in region_utterances.items():
        print(f"  {region_name}: {len(utterances)} utterances")

    # Write both formats
    write_secondary_analysis(f"{output_base}_secondary.txt", data, region_utterances, regions)
    print("Debug - About to call write_secondary_json")
    write_secondary_json(f"{output_base}_secondary.json", data, region_utterances, regions)

def main():
    if len(sys.argv) != 3:
        print("Usage: python ZPD_regions.py <input_primary_json> <output_base>")
        print("\nExample:")
        print("  python ZPD_regions.py analysis_output/660_golf/660_golf_primary.json analysis_output/660_golf/660_golf")
        sys.exit(1)

    input_primary = sys.argv[1]
    output_base = sys.argv[2]
    secondary_txt = f"{output_base}_secondary.txt"

    print("\nStage 2: Secondary Analysis")
    print(f"Input Primary JSON: {input_primary}")
    print(f"Output: {secondary_txt}\n")

    try:
        # Parse the primary output to get all data
        data = parse_primary_output(input_primary)

        # Define learning regions based on tendency zone
        regions = define_learning_regions(data)

        # Generate secondary analysis
        generate_secondary_analysis(data, regions, output_base)

        print("\nStage 2 Complete! Output files:")
        print(f"- Secondary Analysis: {secondary_txt}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
