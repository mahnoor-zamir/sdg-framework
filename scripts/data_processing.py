#!/usr/bin/env python3
"""
Convert OSDG dataset to multi-label format with multi-hot vectors.

This script converts the OSDG community dataset from single-label format
to multi-label format where each text is mapped to a multi-hot vector
representing which SDGs it belongs to based on agreement threshold.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
import argparse
from pathlib import Path


def load_sdg_info(sdg_file_path):
    """Load SDG information from JSON file."""
    with open(sdg_file_path, 'r') as f:
        sdg_data = json.load(f)
    
    sdg_names = {}
    for sdg_num, sdg_info in sdg_data.items():
        sdg_names[int(sdg_num)] = sdg_info['name']
    
    return sdg_names


def create_multilabel_dataset(csv_file_path, agreement_threshold=0.5, output_format='csv'):
    """
    Convert OSDG dataset to multi-label format.
    
    Args:
        csv_file_path (str): Path to the OSDG CSV file
        agreement_threshold (float): Minimum agreement score to include a label
        output_format (str): Output format ('csv', 'json', or 'both')
    
    Returns:
        pandas.DataFrame: Multi-label dataset
    """
    print(f"Loading dataset from {csv_file_path}")
    
    # Load the dataset
    df = pd.read_csv(csv_file_path, sep='\t')
    print(f"Loaded {len(df)} rows")
    
    # Print dataset info
    print(f"Columns: {list(df.columns)}")
    print(f"Agreement scores range: {df['agreement'].min():.3f} to {df['agreement'].max():.3f}")
    
    # Filter by agreement threshold and positive > negative labels
    df_filtered = df[
        (df['agreement'] >= agreement_threshold) & 
        (df['labels_positive'] > df['labels_negative'])
    ].copy()
    print(f"After filtering by agreement >= {agreement_threshold} and positive > negative: {len(df_filtered)} rows")
    
    # Get unique SDGs in the dataset
    unique_sdgs = sorted([int(x) for x in df_filtered['sdg'].unique() if str(x).isdigit()])
    print(f"Unique SDGs in dataset: {unique_sdgs}")
    
    # Group by text_id to create multi-label entries
    print("Creating multi-label dataset...")
    multilabel_data = []
    
    # Group by text_id and text content
    grouped = df_filtered.groupby(['text_id', 'text'])
    
    for (text_id, text), group in grouped:
        # Create multi-hot vector for this text
        sdg_labels = sorted(group['sdg'].unique())
        
        # Create binary vector for all possible SDGs (1-17)
        multi_hot_vector = [0] * 17  # SDGs 1-17
        sdg_list = []
        
        for sdg in sdg_labels:
            if str(sdg).isdigit():
                sdg_int = int(sdg)
                if 1 <= sdg_int <= 17:
                    multi_hot_vector[sdg_int - 1] = 1  # Index 0 = SDG 1
                    sdg_list.append(sdg_int)
        
        # Calculate average agreement for this text across all its SDG labels
        avg_agreement = group['agreement'].mean()
        
        multilabel_data.append({
            'text_id': text_id,
            'text': text,
            'sdg_labels': sdg_list,
            'multi_hot_vector': multi_hot_vector,
            'avg_agreement': avg_agreement,
            'num_labels': len(sdg_list)
        })
    
    # Convert to DataFrame
    multilabel_df = pd.DataFrame(multilabel_data)
    
    print(f"Created multi-label dataset with {len(multilabel_df)} unique texts")
    print(f"Label distribution:")
    print(f"  Single label: {sum(multilabel_df['num_labels'] == 1)} texts")
    print(f"  Multi label: {sum(multilabel_df['num_labels'] > 1)} texts")
    print(f"  Max labels per text: {multilabel_df['num_labels'].max()}")
    print(f"  Average labels per text: {multilabel_df['num_labels'].mean():.2f}")
    
    return multilabel_df, unique_sdgs


def save_dataset(multilabel_df, unique_sdgs, output_dir, agreement_threshold, output_format):
    """Save the multi-label dataset in specified format(s)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    base_filename = f"osdg_multilabel_threshold_{agreement_threshold}"
    
    if output_format in ['csv', 'both']:
        # Save as CSV with expanded multi-hot columns
        csv_data = multilabel_df.copy()
        
        # Expand multi-hot vector into separate columns
        multi_hot_array = np.array(csv_data['multi_hot_vector'].tolist())
        sdg_columns = [f'sdg_{i+1}' for i in range(17)]
        
        # Add SDG columns
        for i, col in enumerate(sdg_columns):
            csv_data[col] = multi_hot_array[:, i]
        
        # Remove the list column for CSV
        csv_data = csv_data.drop(['multi_hot_vector'], axis=1)
        
        csv_path = output_path / f"{base_filename}.csv"
        csv_data.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")
    
    if output_format in ['json', 'both']:
        # Save as JSON with metadata
        json_data = {
            'metadata': {
                'agreement_threshold': agreement_threshold,
                'total_texts': len(multilabel_df),
                'unique_sdgs': unique_sdgs,
                'sdg_names': {i+1: f"SDG {i+1}" for i in range(17)},
                'description': 'Multi-label OSDG dataset with multi-hot vectors'
            },
            'data': multilabel_df.to_dict('records')
        }
        
        json_path = output_path / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON to: {json_path}")
    
    # Save statistics
    stats = {
        'dataset_stats': {
            'total_texts': len(multilabel_df),
            'agreement_threshold': agreement_threshold,
            'label_distribution': {
                'single_label': int(sum(multilabel_df['num_labels'] == 1)),
                'multi_label': int(sum(multilabel_df['num_labels'] > 1)),
                'max_labels_per_text': int(multilabel_df['num_labels'].max()),
                'avg_labels_per_text': float(multilabel_df['num_labels'].mean())
            },
            'sdg_frequency': {}
        }
    }
    
    # Calculate SDG frequency
    all_vectors = np.array(multilabel_df['multi_hot_vector'].tolist())
    for i in range(17):
        sdg_num = i + 1
        frequency = int(all_vectors[:, i].sum())
        stats['dataset_stats']['sdg_frequency'][f'sdg_{sdg_num}'] = frequency
    
    stats_path = output_path / f"{base_filename}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert OSDG dataset to multi-label format')
    parser.add_argument('--input', '-i', 
                        default='data/raw/osdg-community-dataset-v21-09-30.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', 
                        default='data/processed',
                        help='Output directory')
    parser.add_argument('--threshold', '-t', 
                        type=float, 
                        default=0.5,
                        help='Agreement threshold (default: 0.5)')
    parser.add_argument('--format', '-f', 
                        choices=['csv', 'json', 'both'],
                        default='both',
                        help='Output format (default: both)')
    
    args = parser.parse_args()
    
    print("=== OSDG Multi-Label Dataset Converter ===")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Agreement threshold: {args.threshold}")
    print(f"Output format: {args.format}")
    print()
    
    # Create multi-label dataset
    multilabel_df, unique_sdgs = create_multilabel_dataset(
        args.input, 
        agreement_threshold=args.threshold,
        output_format=args.format
    )
    
    # Save dataset
    save_dataset(multilabel_df, unique_sdgs, args.output, args.threshold, args.format)
    
    print("\n=== Conversion completed successfully! ===")


if __name__ == "__main__":
    main()
