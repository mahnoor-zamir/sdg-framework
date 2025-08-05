#!/usr/bin/env python3
"""
Create a multi-label SDG dataset using text similarity.

This script combines the SDG paragraph dataset with the OSDG multilabel dataset
to create a new multi-label dataset where SDG labels are assigned based on 
text similarity between excerpts and SDG descriptions.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import argparse
import re
from typing import List, Dict, Tuple


def clean_text(text: str) -> str:
    """Clean and preprocess text for similarity calculation."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
    
    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_text_similarity(texts1: List[str], texts2: List[str], method='embeddings') -> np.ndarray:
    """Calculate similarity matrix between two sets of texts."""
    
    if method == 'embeddings':
        print("Loading sentence transformer model...")
        # Use a multilingual model that works well for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Generating embeddings for texts...")
        # Generate embeddings for both text sets
        embeddings1 = model.encode(texts1, show_progress_bar=True, batch_size=32)
        embeddings2 = model.encode(texts2, show_progress_bar=True, batch_size=32)
        
        print("Calculating cosine similarity...")
        # Calculate cosine similarity between embeddings
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
    elif method == 'tfidf':
        # Keep TF-IDF as fallback option
        all_texts = texts1 + texts2
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        vectorizer.fit(all_texts)
        vectors1 = vectorizer.transform(texts1)
        vectors2 = vectorizer.transform(texts2)
        similarity_matrix = cosine_similarity(vectors1, vectors2)
        
    else:
        raise ValueError(f"Unsupported similarity method: {method}")
    
    return similarity_matrix


def assign_multilabel_sdgs(similarity_scores: np.ndarray, 
                          sdg_numbers: List[int],
                          primary_threshold: float = 0.3,
                          secondary_threshold: float = 0.2,
                          max_labels: int = 5) -> Dict:
    """
    Assign multiple SDG labels based on similarity scores.
    
    Args:
        similarity_scores: Array of similarity scores for each SDG
        sdg_numbers: List of SDG numbers corresponding to scores
        primary_threshold: Minimum similarity for primary labels
        secondary_threshold: Minimum similarity for secondary labels
        max_labels: Maximum number of labels to assign
    
    Returns:
        Dictionary with label assignment information
    """
    
    # Get indices sorted by similarity (highest first)
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_scores = similarity_scores[sorted_indices]
    sorted_sdgs = [sdg_numbers[i] for i in sorted_indices]
    
    # Assign labels based on thresholds
    primary_labels = []
    secondary_labels = []
    
    for i, (sdg, score) in enumerate(zip(sorted_sdgs, sorted_scores)):
        if i >= max_labels:
            break
            
        if score >= primary_threshold:
            primary_labels.append(sdg)
        elif score >= secondary_threshold:
            secondary_labels.append(sdg)
    
    # Create multi-hot vector
    multi_hot_vector = [0] * 17
    all_assigned_sdgs = primary_labels + secondary_labels
    
    for sdg in all_assigned_sdgs:
        if 1 <= sdg <= 17:
            multi_hot_vector[sdg - 1] = 1
    
    return {
        'primary_sdgs': primary_labels,
        'secondary_sdgs': secondary_labels,
        'all_sdgs': all_assigned_sdgs,
        'multi_hot_vector': multi_hot_vector,
        'max_similarity': float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0,
        'similarity_scores': {int(sdg): float(score) for sdg, score in zip(sorted_sdgs, sorted_scores)},
        'num_labels': len(all_assigned_sdgs)
    }


def create_similarity_based_multilabel_dataset(osdg_file: str, 
                                             sdg_paragraphs_file: str,
                                             primary_threshold: float = 0.3,
                                             secondary_threshold: float = 0.2,
                                             max_labels: int = 5,
                                             similarity_method: str = 'embeddings') -> pd.DataFrame:
    """Create multi-label dataset using text similarity."""
    
    print("Loading datasets...")
    
    # Load OSDG dataset
    osdg_df = pd.read_csv(osdg_file)
    print(f"Loaded OSDG dataset: {len(osdg_df)} rows")
    
    # Load SDG paragraphs
    sdg_df = pd.read_csv(sdg_paragraphs_file)
    print(f"Loaded SDG paragraphs: {len(sdg_df)} SDGs")
    
    # Prepare texts for similarity calculation
    print("Preprocessing texts...")
    
    # Clean OSDG texts
    osdg_texts = [clean_text(text) for text in osdg_df['text']]
    
    # Clean SDG paragraph texts
    sdg_texts = [clean_text(text) for text in sdg_df['text']]
    sdg_numbers = sdg_df['sdg'].tolist()
    
    print(f"Calculating text similarities using {similarity_method}...")
    
    # Calculate similarity matrix
    # Shape: (num_osdg_texts, num_sdgs)
    similarity_matrix = calculate_text_similarity(osdg_texts, sdg_texts, method=similarity_method)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Create new multi-label dataset
    print("Assigning multi-labels based on similarity...")
    
    multilabel_data = []
    
    for i, (_, row) in enumerate(osdg_df.iterrows()):
        if i % 1000 == 0:
            print(f"Processing text {i+1}/{len(osdg_df)}")
        
        # Get similarity scores for this text
        text_similarities = similarity_matrix[i]
        
        # Assign multi-labels
        label_assignment = assign_multilabel_sdgs(
            text_similarities, 
            sdg_numbers,
            primary_threshold=primary_threshold,
            secondary_threshold=secondary_threshold,
            max_labels=max_labels
        )
        
        # Create record
        record = {
            'text_id': row['text_id'],
            'text': row['text'],
            'original_sdg': row['sdg'] if 'sdg' in row.index else None,
            'original_avg_agreement': row['avg_agreement'] if 'avg_agreement' in row.index else None,
            'similarity_assigned_sdgs': label_assignment['all_sdgs'],
            'primary_sdgs': label_assignment['primary_sdgs'],
            'secondary_sdgs': label_assignment['secondary_sdgs'],
            'multi_hot_vector': label_assignment['multi_hot_vector'],
            'max_similarity_score': label_assignment['max_similarity'],
            'num_assigned_labels': label_assignment['num_labels'],
            'similarity_scores': label_assignment['similarity_scores']
        }
        
        # Add individual SDG columns for easy analysis
        for sdg_num in range(1, 18):
            record[f'sdg_{sdg_num}'] = label_assignment['multi_hot_vector'][sdg_num - 1]
        
        multilabel_data.append(record)
    
    return pd.DataFrame(multilabel_data)


def analyze_label_assignment(df: pd.DataFrame) -> Dict:
    """Analyze the label assignment results."""
    
    stats = {
        'total_texts': len(df),
        'label_distribution': {},
        'similarity_stats': {},
        'sdg_frequency': {},
        'comparison_with_original': {}
    }
    
    # Label distribution
    num_labels = df['num_assigned_labels']
    stats['label_distribution'] = {
        'mean_labels_per_text': float(num_labels.mean()),
        'median_labels_per_text': float(num_labels.median()),
        'min_labels': int(num_labels.min()),
        'max_labels': int(num_labels.max()),
        'zero_labels': int(sum(num_labels == 0)),
        'single_label': int(sum(num_labels == 1)),
        'multi_label': int(sum(num_labels > 1))
    }
    
    # Similarity statistics
    max_sim = df['max_similarity_score']
    stats['similarity_stats'] = {
        'mean_max_similarity': float(max_sim.mean()),
        'median_max_similarity': float(max_sim.median()),
        'min_similarity': float(max_sim.min()),
        'max_similarity': float(max_sim.max())
    }
    
    # SDG frequency
    for sdg_num in range(1, 18):
        col_name = f'sdg_{sdg_num}'
        if col_name in df.columns:
            stats['sdg_frequency'][f'sdg_{sdg_num}'] = int(df[col_name].sum())
    
    # Comparison with original labels (if available)
    if 'original_sdg' in df.columns:
        original_match = 0
        for _, row in df.iterrows():
            if pd.notna(row['original_sdg']) and int(row['original_sdg']) in row['similarity_assigned_sdgs']:
                original_match += 1
        
        stats['comparison_with_original'] = {
            'texts_with_original_label': int(df['original_sdg'].notna().sum()),
            'original_label_preserved': original_match,
            'preservation_rate': float(original_match / df['original_sdg'].notna().sum()) if df['original_sdg'].notna().sum() > 0 else 0.0
        }
    
    return stats


def save_multilabel_dataset(df: pd.DataFrame, stats: Dict, output_dir: str, 
                           primary_threshold: float, secondary_threshold: float, 
                           similarity_method: str = 'embeddings'):
    """Save the multi-label dataset and analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create filename with parameters
    base_name = f"similarity_multilabel_{similarity_method}_p{primary_threshold}_s{secondary_threshold}"
    
    # Save main dataset
    csv_path = output_path / f"{base_name}.csv"
    
    # Prepare CSV version (without complex columns)
    csv_df = df.drop(['similarity_scores'], axis=1, errors='ignore')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved dataset to: {csv_path}")
    
    # Save full dataset with metadata as JSON
    json_data = {
        'metadata': {
            'primary_threshold': primary_threshold,
            'secondary_threshold': secondary_threshold,
            'similarity_method': similarity_method,
            'total_texts': len(df),
            'creation_method': 'text_similarity',
            'description': f'Multi-label SDG dataset created using {similarity_method} similarity between OSDG texts and SDG paragraph descriptions'
        },
        'statistics': stats,
        'data': df.to_dict('records')
    }
    
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved full dataset (JSON) to: {json_path}")
    
    # Save statistics
    stats_path = output_path / f"{base_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Create similarity-based multi-label SDG dataset')
    parser.add_argument('--osdg-file', '-o',
                        default='data/processed/osdg_multilabel_threshold_0.6.csv',
                        help='OSDG multilabel dataset file')
    parser.add_argument('--sdg-paragraphs', '-s',
                        default='data/processed/sdg_paragraph_dataset.csv',
                        help='SDG paragraph dataset file')
    parser.add_argument('--primary-threshold', '-p',
                        type=float,
                        default=0.3,
                        help='Primary similarity threshold (default: 0.3)')
    parser.add_argument('--secondary-threshold', '-t',
                        type=float,
                        default=0.2,
                        help='Secondary similarity threshold (default: 0.2)')
    parser.add_argument('--max-labels', '-m',
                        type=int,
                        default=5,
                        help='Maximum labels per text (default: 5)')
    parser.add_argument('--method', '-M',
                        choices=['embeddings', 'tfidf'],
                        default='embeddings',
                        help='Similarity calculation method (default: embeddings)')
    parser.add_argument('--output', '-d',
                        default='data/processed',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("=== Similarity-Based Multi-Label SDG Dataset Creator ===")
    print(f"OSDG file: {args.osdg_file}")
    print(f"SDG paragraphs file: {args.sdg_paragraphs}")
    print(f"Similarity method: {args.method}")
    print(f"Primary threshold: {args.primary_threshold}")
    print(f"Secondary threshold: {args.secondary_threshold}")
    print(f"Max labels: {args.max_labels}")
    print(f"Output directory: {args.output}")
    print()
    
    # Create multi-label dataset
    df = create_similarity_based_multilabel_dataset(
        args.osdg_file,
        args.sdg_paragraphs,
        primary_threshold=args.primary_threshold,
        secondary_threshold=args.secondary_threshold,
        max_labels=args.max_labels,
        similarity_method=args.method
    )
    
    print(f"\nCreated multi-label dataset with {len(df)} texts")
    
    # Analyze results
    print("Analyzing label assignments...")
    stats = analyze_label_assignment(df)
    
    # Save dataset
    save_multilabel_dataset(df, stats, args.output, args.primary_threshold, 
                           args.secondary_threshold, args.method)
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total texts: {stats['total_texts']}")
    print(f"Average labels per text: {stats['label_distribution']['mean_labels_per_text']:.2f}")
    print(f"Texts with multiple labels: {stats['label_distribution']['multi_label']}")
    print(f"Texts with single label: {stats['label_distribution']['single_label']}")
    print(f"Texts with no labels: {stats['label_distribution']['zero_labels']}")
    print(f"Average max similarity: {stats['similarity_stats']['mean_max_similarity']:.3f}")
    
    if 'comparison_with_original' in stats and stats['comparison_with_original']:
        print(f"Original label preservation rate: {stats['comparison_with_original']['preservation_rate']:.2%}")
    
    print("\n=== Multi-label dataset creation completed! ===")


if __name__ == "__main__":
    main()
