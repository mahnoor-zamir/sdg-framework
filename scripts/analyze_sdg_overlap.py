#!/usr/bin/env python3
"""
Analyze SDG label overlap between original OSDG dataset and similarity-based multi-label dataset.

This script compares how many texts with a specific SDG in the original dataset
also have that same SDG assigned in the similarity-based multi-label dataset.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_datasets(osdg_file: str, similarity_file: str):
    """Load both datasets and merge on text_id."""
    
    print("Loading datasets...")
    
    # Load original OSDG dataset
    osdg_df = pd.read_csv(osdg_file)
    print(f"Loaded OSDG dataset: {len(osdg_df)} rows")
    print(f"OSDG columns: {list(osdg_df.columns)}")
    
    # Load similarity-based dataset
    sim_df = pd.read_csv(similarity_file)
    print(f"Loaded similarity dataset: {len(sim_df)} rows")
    print(f"Similarity columns: {list(sim_df.columns)}")
    
    # Merge datasets on text_id
    merged_df = pd.merge(osdg_df, sim_df, on='text_id', suffixes=('_original', '_similarity'))
    print(f"Merged dataset: {len(merged_df)} rows")
    print(f"Merged columns: {list(merged_df.columns)}")
    
    return merged_df


def parse_sdg_list(sdg_string):
    """Parse SDG list from string format [1, 2, 3] to actual list."""
    if pd.isna(sdg_string) or sdg_string == '[]':
        return []
    
    # Remove brackets and split by comma
    sdg_string = str(sdg_string).strip('[]')
    if not sdg_string:
        return []
    
    try:
        sdgs = [int(x.strip()) for x in sdg_string.split(',')]
        return sdgs
    except:
        return []


def analyze_sdg_overlap(merged_df: pd.DataFrame):
    """Analyze overlap for each SDG."""
    
    print("Analyzing SDG overlap...")
    
    # Parse both datasets' SDG lists
    merged_df['original_sdgs_parsed'] = merged_df['sdg_labels'].apply(parse_sdg_list)
    merged_df['similarity_sdgs_parsed'] = merged_df['similarity_assigned_sdgs'].apply(parse_sdg_list)
    
    results = {}
    overall_stats = {
        'total_texts': len(merged_df),
        'texts_with_original_sdg': 0,
        'texts_with_similarity_sdgs': 0,
        'total_matches': 0
    }
    
    for sdg_num in range(1, 18):
        print(f"Analyzing SDG {sdg_num}...")
        
        # Find texts with this SDG in original dataset (using binary columns for accuracy)
        original_col = f'sdg_{sdg_num}_original'
        texts_with_original = merged_df[original_col].sum()
        
        # Find how many of these also have this SDG in similarity dataset
        similarity_col = f'sdg_{sdg_num}_similarity'
        overlap_mask = (merged_df[original_col] == 1) & (merged_df[similarity_col] == 1)
        similarity_matches = overlap_mask.sum()
        
        # Calculate preservation rate
        preservation_rate = (similarity_matches / texts_with_original * 100) if texts_with_original > 0 else 0
        
        # Find texts with this SDG in similarity dataset (regardless of original)
        texts_with_similarity = merged_df[similarity_col].sum()
        
        results[f'sdg_{sdg_num}'] = {
            'original_count': int(texts_with_original),
            'similarity_count': int(texts_with_similarity),
            'preserved_count': int(similarity_matches),
            'preservation_rate': float(preservation_rate),
            'similarity_precision': float((similarity_matches / texts_with_similarity * 100) if texts_with_similarity > 0 else 0)
        }
        
        overall_stats['total_matches'] += int(similarity_matches)
    
    # Calculate overall statistics
    overall_stats['texts_with_original_sdg'] = int((merged_df['original_sdgs_parsed'].apply(len) > 0).sum())
    overall_stats['texts_with_similarity_sdgs'] = int((merged_df['similarity_sdgs_parsed'].apply(len) > 0).sum())
    overall_stats['overall_preservation_rate'] = float((overall_stats['total_matches'] / overall_stats['texts_with_original_sdg'] * 100) if overall_stats['texts_with_original_sdg'] > 0 else 0)
    
    return results, overall_stats


def create_summary_table(results: dict):
    """Create a summary table of the analysis."""
    
    data = []
    for sdg_key, stats in results.items():
        sdg_num = int(sdg_key.split('_')[1])
        data.append({
            'SDG': sdg_num,
            'Original Count': stats['original_count'],
            'Similarity Count': stats['similarity_count'],
            'Preserved Count': stats['preserved_count'],
            'Preservation Rate (%)': round(stats['preservation_rate'], 2),
            'Similarity Precision (%)': round(stats['similarity_precision'], 2)
        })
    
    return pd.DataFrame(data).sort_values('SDG')


def save_analysis(results: dict, overall_stats: dict, summary_df: pd.DataFrame, output_dir: str):
    """Save analysis results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results as JSON
    analysis_data = {
        'metadata': {
            'description': 'SDG label overlap analysis between original OSDG dataset and similarity-based multi-label dataset',
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'overall_statistics': overall_stats,
        'per_sdg_results': results
    }
    
    json_path = output_path / 'sdg_overlap_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Saved detailed analysis to: {json_path}")
    
    # Save summary table as CSV
    csv_path = output_path / 'sdg_overlap_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary table to: {csv_path}")
    
    return json_path, csv_path


def create_visualizations(summary_df: pd.DataFrame, output_dir: str):
    """Create visualizations of the analysis."""
    
    output_path = Path(output_dir)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SDG Label Overlap Analysis', fontsize=16, fontweight='bold')
    
    # 1. Preservation Rate by SDG
    ax1 = axes[0, 0]
    bars1 = ax1.bar(summary_df['SDG'], summary_df['Preservation Rate (%)'], 
                    color='steelblue', alpha=0.7)
    ax1.set_title('Preservation Rate by SDG')
    ax1.set_xlabel('SDG Number')
    ax1.set_ylabel('Preservation Rate (%)')
    ax1.set_xticks(summary_df['SDG'])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Original vs Similarity Counts
    ax2 = axes[0, 1]
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, summary_df['Original Count'], width, 
                     label='Original Dataset', alpha=0.7, color='orange')
    bars2b = ax2.bar(x + width/2, summary_df['Similarity Count'], width, 
                     label='Similarity Dataset', alpha=0.7, color='green')
    
    ax2.set_title('Text Counts by SDG')
    ax2.set_xlabel('SDG Number')
    ax2.set_ylabel('Number of Texts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['SDG'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Preservation Rate vs Original Count (Scatter)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(summary_df['Original Count'], summary_df['Preservation Rate (%)'],
                         s=summary_df['Preserved Count']*3, alpha=0.6, c=summary_df['SDG'], 
                         cmap='viridis')
    ax3.set_title('Preservation Rate vs Original Count\n(Bubble size = Preserved Count)')
    ax3.set_xlabel('Original Count')
    ax3.set_ylabel('Preservation Rate (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add SDG labels to points
    for i, row in summary_df.iterrows():
        ax3.annotate(f'SDG{row["SDG"]}', 
                    (row['Original Count'], row['Preservation Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Precision vs Recall-like metric
    ax4 = axes[1, 1]
    # Calculate recall-like metric (how many of original were found)
    recall_like = summary_df['Preserved Count'] / summary_df['Original Count'] * 100
    precision = summary_df['Similarity Precision (%)']
    
    bars4 = ax4.bar(summary_df['SDG'], precision, alpha=0.7, color='purple')
    ax4.set_title('Similarity Precision by SDG')
    ax4.set_xlabel('SDG Number')
    ax4.set_ylabel('Precision (%)')
    ax4.set_xticks(summary_df['SDG'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'sdg_overlap_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {plot_path}")
    plt.show()


def print_detailed_results(results: dict, overall_stats: dict, summary_df: pd.DataFrame):
    """Print detailed analysis results."""
    
    print("\n" + "="*80)
    print("SDG LABEL OVERLAP ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total texts analyzed: {overall_stats['total_texts']:,}")
    print(f"Texts with original SDG labels: {overall_stats['texts_with_original_sdg']:,}")
    print(f"Texts with similarity SDG labels: {overall_stats['texts_with_similarity_sdgs']:,}")
    print(f"Total preservation matches: {overall_stats['total_matches']:,}")
    print(f"Overall preservation rate: {overall_stats['overall_preservation_rate']:.2f}%")
    
    print(f"\nDETAILED RESULTS BY SDG:")
    print("-" * 100)
    print(f"{'SDG':<4} {'Original':<9} {'Similarity':<10} {'Preserved':<9} {'Preservation':<12} {'Precision':<10}")
    print(f"{'Num':<4} {'Count':<9} {'Count':<10} {'Count':<9} {'Rate (%)':<12} {'Rate (%)':<10}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['SDG']:<4} {row['Original Count']:<9} {row['Similarity Count']:<10} "
              f"{row['Preserved Count']:<9} {row['Preservation Rate (%)']:<12.1f} "
              f"{row['Similarity Precision (%)']:<10.1f}")
    
    print("-" * 100)
    
    # Identify best and worst performing SDGs
    best_preservation = summary_df.loc[summary_df['Preservation Rate (%)'].idxmax()]
    worst_preservation = summary_df.loc[summary_df['Preservation Rate (%)'].idxmin()]
    
    print(f"\nKEY INSIGHTS:")
    print(f"• Best preservation: SDG {best_preservation['SDG']} ({best_preservation['Preservation Rate (%)']:.1f}%)")
    print(f"• Worst preservation: SDG {worst_preservation['SDG']} ({worst_preservation['Preservation Rate (%)']:.1f}%)")
    
    # Most/least represented SDGs
    most_original = summary_df.loc[summary_df['Original Count'].idxmax()]
    least_original = summary_df.loc[summary_df['Original Count'].idxmin()]
    
    print(f"• Most represented in original: SDG {most_original['SDG']} ({most_original['Original Count']} texts)")
    print(f"• Least represented in original: SDG {least_original['SDG']} ({least_original['Original Count']} texts)")


def main():
    parser = argparse.ArgumentParser(description='Analyze SDG label overlap between datasets')
    parser.add_argument('--osdg-file', '-o',
                        default='data/processed/osdg_multilabel_threshold_0.6.csv',
                        help='Original OSDG dataset file')
    parser.add_argument('--similarity-file', '-s',
                        default='data/processed/similarity_multilabel_embeddings_p0.4_s0.3.csv',
                        help='Similarity-based multilabel dataset file')
    parser.add_argument('--output', '-d',
                        default='data/analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--visualize', '-v',
                        action='store_true',
                        help='Create visualizations')
    
    args = parser.parse_args()
    
    print("=== SDG Label Overlap Analysis ===")
    print(f"Original OSDG file: {args.osdg_file}")
    print(f"Similarity file: {args.similarity_file}")
    print(f"Output directory: {args.output}")
    print()
    
    # Load and merge datasets
    merged_df = load_datasets(args.osdg_file, args.similarity_file)
    
    # Perform analysis
    results, overall_stats = analyze_sdg_overlap(merged_df)
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Save results
    json_path, csv_path = save_analysis(results, overall_stats, summary_df, args.output)
    
    # Print results
    print_detailed_results(results, overall_stats, summary_df)
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(summary_df, args.output)
    
    print(f"\n=== Analysis completed! ===")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
