#!/usr/bin/env python3
"""
Experiment 1: Cosine Similarity + MiniLM + Global Thresholds
============================================================

Clean implementation of cosine similarity classification with:
- all-MiniLM-L6-v2 embeddings
- Fixed global thresholds (primary/secondary)
- Comprehensive evaluation metrics

Author: Research Team
Date: August 21, 2025
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Configuration
CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'distance_metric': 'cosine',
    'primary_threshold': 0.4,
    'secondary_threshold': 0.3,
    'max_labels_per_text': 5,
    'experiment_name': 'cosine_minilm_global_threshold'
}

def load_data():
    """Load OSDG dataset and SDG descriptions."""
    print("Loading datasets...")
    
    # Load OSDG data
    osdg_path = Path("../../data/processed/osdg_multilabel_threshold_0.6.csv")
    osdg_df = pd.read_csv(osdg_path)
    print(f"Loaded {len(osdg_df)} OSDG texts")
    
    # Load SDG descriptions
    sdg_path = Path("../../data/processed/sdg_paragraph_dataset.csv")
    sdg_df = pd.read_csv(sdg_path)
    print(f"Loaded {len(sdg_df)} SDG descriptions")
    
    return osdg_df, sdg_df

def generate_embeddings(texts, model, batch_size=32):
    """Generate embeddings for texts using sentence transformer."""
    print(f"Generating embeddings for {len(texts)} texts...")
    start_time = time.time()
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    elapsed = time.time() - start_time
    print(f"Embedding generation completed in {elapsed:.2f} seconds")
    
    return embeddings

def calculate_cosine_similarities(text_embeddings, sdg_embeddings):
    """Calculate cosine similarities between texts and SDG descriptions."""
    print("Calculating cosine similarities...")
    start_time = time.time()
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(text_embeddings, sdg_embeddings)
    
    elapsed = time.time() - start_time
    print(f"Similarity calculation completed in {elapsed:.2f} seconds")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: {similarity_matrix.min():.4f} to {similarity_matrix.max():.4f}")
    
    return similarity_matrix

def apply_global_thresholds(similarity_matrix, primary_thresh, secondary_thresh, max_labels):
    """Apply global thresholds to assign labels."""
    print(f"Applying global thresholds: Primary={primary_thresh}, Secondary={secondary_thresh}")
    
    assignments = []
    stats = {
        'primary_assignments': 0,
        'secondary_assignments': 0,
        'total_assignments': 0,
        'texts_with_labels': 0,
        'zero_label_texts': 0
    }
    
    for i, similarities in enumerate(similarity_matrix):
        text_labels = []
        
        # Primary threshold assignments
        primary_indices = np.where(similarities >= primary_thresh)[0]
        text_labels.extend(primary_indices.tolist())
        stats['primary_assignments'] += len(primary_indices)
        
        # Secondary threshold assignments (if not already assigned)
        if len(text_labels) < max_labels:
            secondary_indices = np.where(
                (similarities >= secondary_thresh) & 
                (similarities < primary_thresh)
            )[0]
            
            # Add secondary up to max_labels limit
            remaining_slots = max_labels - len(text_labels)
            secondary_to_add = secondary_indices[:remaining_slots]
            text_labels.extend(secondary_to_add.tolist())
            stats['secondary_assignments'] += len(secondary_to_add)
        
        # Convert to SDG numbers (1-based)
        sdg_assignments = [idx + 1 for idx in text_labels[:max_labels]]
        assignments.append(sdg_assignments)
        
        # Update stats
        if len(sdg_assignments) > 0:
            stats['texts_with_labels'] += 1
        else:
            stats['zero_label_texts'] += 1
        
        stats['total_assignments'] += len(sdg_assignments)
    
    # Calculate summary statistics
    avg_labels = stats['total_assignments'] / len(similarity_matrix)
    coverage = (stats['texts_with_labels'] / len(similarity_matrix)) * 100
    
    print(f"Assignment completed:")
    print(f"  Average labels per text: {avg_labels:.2f}")
    print(f"  Coverage rate: {coverage:.1f}%")
    print(f"  Primary assignments: {stats['primary_assignments']}")
    print(f"  Secondary assignments: {stats['secondary_assignments']}")
    print(f"  Zero-label texts: {stats['zero_label_texts']}")
    
    return assignments, stats

def calculate_evaluation_metrics(osdg_df, assignments, sdg_columns):
    """Calculate comprehensive evaluation metrics."""
    print("Calculating evaluation metrics...")
    
    # Create prediction matrix
    n_texts = len(osdg_df)
    n_sdgs = len(sdg_columns)
    y_pred = np.zeros((n_texts, n_sdgs))
    
    for i, text_assignments in enumerate(assignments):
        for sdg_num in text_assignments:
            if 1 <= sdg_num <= n_sdgs:
                y_pred[i, sdg_num - 1] = 1
    
    # Create ground truth matrix
    y_true = osdg_df[sdg_columns].values
    
    # Sample-based metrics
    sample_precision = []
    sample_recall = []
    sample_f1 = []
    sample_jaccard = []
    
    for i in range(n_texts):
        if y_pred[i].sum() > 0:  # Only if predictions exist
            prec = precision_score(y_true[i], y_pred[i], average='binary', zero_division=0)
            rec = recall_score(y_true[i], y_pred[i], average='binary', zero_division=0)
            f1 = f1_score(y_true[i], y_pred[i], average='binary', zero_division=0)
            jacc = jaccard_score(y_true[i], y_pred[i], average='binary', zero_division=0)
            
            sample_precision.append(prec)
            sample_recall.append(rec)
            sample_f1.append(f1)
            sample_jaccard.append(jacc)
    
    # Label-based metrics
    micro_precision = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_recall = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Conservation rate (preservation of original labels)
    original_labels = y_true.sum()
    preserved_labels = (y_true * y_pred).sum()
    conservation_rate = preserved_labels / original_labels if original_labels > 0 else 0
    
    metrics = {
        'sample_based': {
            'precision': np.mean(sample_precision) if sample_precision else 0,
            'recall': np.mean(sample_recall) if sample_recall else 0,
            'f1_score': np.mean(sample_f1) if sample_f1 else 0,
            'jaccard': np.mean(sample_jaccard) if sample_jaccard else 0
        },
        'label_based': {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        },
        'conservation_rate': conservation_rate,
        'coverage_rate': len([a for a in assignments if len(a) > 0]) / len(assignments)
    }
    
    return metrics

def save_results(osdg_df, assignments, similarity_matrix, stats, metrics, config):
    """Save experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/global_threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results DataFrame
    results_df = osdg_df.copy()
    
    # Add assigned SDG columns
    for i, text_assignments in enumerate(assignments):
        sdg_list = ','.join(map(str, text_assignments)) if text_assignments else ''
        results_df.loc[i, 'assigned_sdgs'] = sdg_list
        results_df.loc[i, 'num_assigned_sdgs'] = len(text_assignments)
    
    # Add similarity scores
    for sdg_idx in range(similarity_matrix.shape[1]):
        results_df[f'sim_sdg_{sdg_idx + 1}'] = similarity_matrix[:, sdg_idx]
    
    # Save CSV
    csv_path = output_dir / f"experiment_1_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save comprehensive stats
    full_stats = {
        'experiment_config': config,
        'assignment_stats': stats,
        'evaluation_metrics': metrics,
        'timestamp': timestamp,
        'total_texts': len(osdg_df),
        'avg_labels_per_text': stats['total_assignments'] / len(osdg_df)
    }
    
    stats_path = output_dir / f"experiment_1_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(full_stats, f, indent=2, default=str)
    print(f"Stats saved to: {stats_path}")
    
    return csv_path, stats_path

def print_experiment_report(stats, metrics, config):
    """Print comprehensive experiment report."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: COSINE SIMILARITY + GLOBAL THRESHOLDS")
    print("="*60)
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"Distance Metric: {config['distance_metric']}")
    print(f"Primary Threshold: {config['primary_threshold']}")
    print(f"Secondary Threshold: {config['secondary_threshold']}")
    print(f"Max Labels per Text: {config['max_labels_per_text']}")
    print()
    
    print("ASSIGNMENT STATISTICS:")
    print(f"  Average labels per text: {stats['total_assignments'] / (stats['texts_with_labels'] + stats['zero_label_texts']):.2f}")
    print(f"  Coverage rate: {(stats['texts_with_labels'] / (stats['texts_with_labels'] + stats['zero_label_texts'])) * 100:.1f}%")
    print(f"  Total assignments: {stats['total_assignments']:,}")
    print(f"  Primary assignments: {stats['primary_assignments']:,}")
    print(f"  Secondary assignments: {stats['secondary_assignments']:,}")
    print()
    
    print("EVALUATION METRICS:")
    print(f"Sample-based metrics:")
    print(f"  Precision: {metrics['sample_based']['precision']:.4f}")
    print(f"  Recall: {metrics['sample_based']['recall']:.4f}")
    print(f"  F1-Score: {metrics['sample_based']['f1_score']:.4f}")
    print(f"  Jaccard: {metrics['sample_based']['jaccard']:.4f}")
    print()
    print(f"Label-based metrics:")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['label_based']['macro_f1']:.4f}")
    print()
    print(f"Conservation rate: {metrics['conservation_rate']:.4f}")
    print(f"Coverage rate: {metrics['coverage_rate']:.4f}")
    print("="*60)

def main():
    """Main experiment execution."""
    print("Starting Experiment 1: Cosine Similarity + Global Thresholds")
    print(f"Configuration: {CONFIG}")
    
    # Load data
    osdg_df, sdg_df = load_data()
    
    # Initialize embedding model
    print(f"Loading embedding model: {CONFIG['embedding_model']}")
    model = SentenceTransformer(CONFIG['embedding_model'])
    
    # Generate embeddings
    text_embeddings = generate_embeddings(osdg_df['text'].tolist(), model)
    sdg_embeddings = generate_embeddings(sdg_df['text'].tolist(), model)
    
    # Calculate similarities
    similarity_matrix = calculate_cosine_similarities(text_embeddings, sdg_embeddings)
    
    # Apply global thresholds
    assignments, stats = apply_global_thresholds(
        similarity_matrix,
        CONFIG['primary_threshold'],
        CONFIG['secondary_threshold'],
        CONFIG['max_labels_per_text']
    )
    
    # Calculate evaluation metrics
    sdg_columns = [f'sdg_{i}' for i in range(1, 18)]
    metrics = calculate_evaluation_metrics(osdg_df, assignments, sdg_columns)
    
    # Save results
    csv_path, stats_path = save_results(osdg_df, assignments, similarity_matrix, stats, metrics, CONFIG)
    
    # Print report
    print_experiment_report(stats, metrics, CONFIG)
    
    print(f"\nExperiment 1 completed successfully!")
    print(f"Results: {csv_path}")
    print(f"Stats: {stats_path}")

if __name__ == "__main__":
    main()
