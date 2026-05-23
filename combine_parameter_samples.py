#!/usr/bin/env python3
"""
Script to combine FATES parameter samples from multiple cases and add case identifiers.

This script reads parameter samples from files specified in the configuration,
adds case identifiers, and creates a combined parameter file that can be used
with the multi-case processing system.
"""

import pandas as pd
import os
import glob
from config_utils import get_cleaning_metadata

def combine_parameter_samples():
    """
    Combine parameter samples from multiple cases and add case IDs.

    Reads parameter files based on configuration and naming patterns.
    Creates combined parameter files with case identifiers.
    """

    print("Starting parameter sample combination...")

    # Load configuration to get case IDs
    config = get_cleaning_metadata()
    print(f"Found {len(config)} cases in configuration:")
    for idx, row in config.iterrows():
        print(f"  Case {row['CASE_ID']}: {row['CASE']} (Drive: {row['DRIVE']})")

    # Find all parameter sample files in the directory
    param_files = glob.glob('FATES_Param_samples_*.csv')
    print(f"\nFound {len(param_files)} parameter files:")
    for i, file in enumerate(param_files, 1):
        print(f"  {i}. {file}")

    if len(param_files) == 0:
        raise FileNotFoundError("No parameter sample files found matching pattern 'FATES_Param_samples_*.csv'")

    # Load parameter samples and match with cases
    print("\nLoading and processing parameter samples...")
    case_params = {}

    for idx, row in config.iterrows():
        case_id = row['CASE_ID']
        case_name = row['CASE']

        # Try to find the most relevant parameter file for this case
        # Look for files that contain the case name or similar patterns
        matching_files = [f for f in param_files if case_name.replace('_2pft.IELMFATES', '').lower() in f.lower()]

        if matching_files:
            # Use the first matching file
            param_file = matching_files[0]
            print(f"  Loading parameters for Case {case_id} ({case_name}): {param_file}")
            params = pd.read_csv(param_file)

            # Add case metadata
            params['case_name'] = case_name
            params['case_id'] = case_id
            case_params[case_id] = params
        else:
            print(f"  Warning: No specific parameter file found for Case {case_id} ({case_name})")
            print(f"  Available files: {param_files}")

    if not case_params:
        raise ValueError("Could not match any parameter files to configured cases")

    # Combine all case parameters
    print("\nCombining datasets...")
    combined_params = pd.concat(case_params.values(), ignore_index=True)

    # Add model IDs (sequential across all samples)
    combined_params['Model'] = range(1, len(combined_params) + 1)

    print(f"Combined dataset: {combined_params.shape[0]} samples, {combined_params.shape[1]} columns")

    # Show case distribution
    case_distribution = combined_params['case_id'].value_counts().sort_index()
    print(f"Case distribution:")
    for case_id, count in case_distribution.items():
        case_name = combined_params[combined_params['case_id'] == case_id]['case_name'].iloc[0]
        print(f"  Case {case_id} ({case_name}): {count} samples")

    # Show sample of the combined data
    print("\nSample of combined parameters:")
    # Get first 3 parameter columns (excluding metadata)
    param_cols = [col for col in combined_params.columns if col not in ['Model', 'case_id', 'case_name']][:3]
    sample_cols = ['Model', 'case_name', 'case_id'] + param_cols
    print(combined_params[sample_cols].head())

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save combined parameters
    output_file = 'data/FATES_Param_samples_combined.csv'
    combined_params.to_csv(output_file, index=False)
    print(f"\nSaved combined parameters to: {output_file}")

    # Also save a version without the case metadata for compatibility
    params_no_metadata = combined_params.drop(['case_name'], axis=1)
    output_file_clean = 'FATES_Param_samples_combined_clean.csv'
    params_no_metadata.to_csv(output_file_clean, index=False)
    print(f"Saved clean combined parameters (no metadata) to: {output_file_clean}")

    # Summary statistics
    print(f"\nSummary:")
    print(f"- Total samples: {len(combined_params)}")
    for case_id, count in case_distribution.items():
        case_name = combined_params[combined_params['case_id'] == case_id]['case_name'].iloc[0]
        print(f"- {case_name} (case_id={case_id}): {count} samples")
    print(f"- Parameters per sample: {len(param_cols)}")

    return combined_params

if __name__ == "__main__":
    combined_params = combine_parameter_samples()
    print("\nParameter combination completed successfully!")