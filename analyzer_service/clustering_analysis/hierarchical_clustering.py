"""Hierarchical clustering script for language analysis data."""
import sys
import os
import pathlib

# Add the parent directory to sys.path to allow importing from clustering_analysis
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from clustering_analysis.ZPD_analyzer import CAFClusterAnalyzer

def main():
    """Main entry point for hierarchical clustering analysis."""
    if len(sys.argv) != 3:
        print("Usage: python hierarchical_clustering.py <input_json> <output_base>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_base = sys.argv[2]

    print("\nStage 1: Primary Clustering Analysis")
    print(f"Input: {input_json}")
    print(f"Output Base: {output_base}\n")

    try:
        analyzer = CAFClusterAnalyzer(input_json)

        print("Extracting features...")
        analyzer.extract_features()

        print("\nPerforming primary clustering...")
        analyzer.perform_clustering()

        # Save state for potential future use
        state_path = f"{output_base}_stage1_state.json"
        print(f"\nSaving analysis state to: {state_path}")
        analyzer.save_state(state_path)

        # Generate analysis reports
        print("\nGenerating primary analysis reports...")
        analyzer.generate_cluster_analysis(f"{output_base}_primary.json")
        analyzer.generate_dendrogram(f"{output_base}_primary_dendrogram.png")

        # Determine tendency zone (result used in logs)
        analyzer.determine_tendency_zone()

        print("\nStage 1 Complete! Output files:")
        print(f"- Primary Analysis: {output_base}_primary.json")
        print(f"- Primary Dendrogram: {output_base}_primary_dendrogram.png")
        print(f"- State File: {state_path}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
