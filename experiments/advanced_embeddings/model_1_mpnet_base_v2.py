#!/usr/bin/env python3
"""
Advanced Embeddings Model 1: MPNet Base v2 with Threshold Optimization
====================================================================

Implementation of cosine similarity classification with:
- all-mpnet-base-v2 embeddings (high-performance sentence transformer)
- Grid search for optimal primary/secondary thresholds
- Comprehensive evaluation metrics

Author: Research Team
Date: August 24, 2025
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
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'model_name': 'all-mpnet-base-v2',
    'max_labels_per_text': 5,
    'experiment_name': 'mpnet_base_v2',
    'threshold_search': {
        'primary_range': np.arange(0.1, 0.7, 0.05),
        'secondary_range': np.arange(0.05, 0.5, 0.05)
    },
    'random_state': 42
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

def apply_thresholds(similarity_matrix, primary_thresh, secondary_thresh, max_labels):
    """Apply thresholds to assign labels."""
    assignments = []
    
    for i, similarities in enumerate(similarity_matrix):
        text_labels = []
        
        # Primary threshold assignments
        primary_indices = np.where(similarities >= primary_thresh)[0]
        text_labels.extend(primary_indices.tolist())
        
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
        
        # Convert to SDG numbers (1-based)
        sdg_assignments = [idx + 1 for idx in text_labels[:max_labels]]
        assignments.append(sdg_assignments)
    
    return assignments

def calculate_f1_score(osdg_df, assignments, sdg_columns):
    """Calculate F1 score for threshold optimization."""
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
    
    # Calculate micro F1 score
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    return micro_f1

def optimize_thresholds(similarity_matrix, osdg_df, sdg_columns, config):
    """Find optimal primary and secondary thresholds using grid search."""
    print("Starting threshold optimization...")
    print(f"Primary threshold range: {config['threshold_search']['primary_range']}")
    print(f"Secondary threshold range: {config['threshold_search']['secondary_range']}")
    
    best_f1 = 0
    best_params = None
    best_assignments = None
    results = []
    
    # Grid search
    param_combinations = list(product(
        config['threshold_search']['primary_range'],
        config['threshold_search']['secondary_range']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, (primary_thresh, secondary_thresh) in enumerate(param_combinations):
        if secondary_thresh >= primary_thresh:
            continue  # Skip invalid combinations
            
        # Apply thresholds
        assignments = apply_thresholds(
            similarity_matrix, primary_thresh, secondary_thresh, config['max_labels_per_text']
        )
        
        # Calculate F1 score
        f1 = calculate_f1_score(osdg_df, assignments, sdg_columns)
        
        results.append({
            'primary_threshold': primary_thresh,
            'secondary_threshold': secondary_thresh,
            'f1_score': f1,
            'avg_labels': np.mean([len(a) for a in assignments]),
            'coverage': len([a for a in assignments if len(a) > 0]) / len(assignments)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = (primary_thresh, secondary_thresh)
            best_assignments = assignments
            
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(param_combinations)} combinations. Best F1: {best_f1:.4f}")
    
    print(f"Optimization completed. Best F1: {best_f1:.4f}")
    print(f"Best parameters: Primary={best_params[0]}, Secondary={best_params[1]}")
    
    return best_params, best_assignments, results

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
    
    # Weighted metrics
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Conservation rate
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
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        },
        'conservation_rate': conservation_rate,
        'coverage_rate': len([a for a in assignments if len(a) > 0]) / len(assignments)
    }
    
    return metrics

def save_results(osdg_df, assignments, similarity_matrix, optimization_results, metrics, best_params, config):
    """Save experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
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
    
    # Save main results CSV
    csv_path = output_dir / f"mpnet_base_v2_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save optimization results
    optimization_df = pd.DataFrame(optimization_results)
    opt_path = output_dir / f"mpnet_base_v2_optimization_{timestamp}.csv"
    optimization_df.to_csv(opt_path, index=False)
    print(f"Optimization results saved to: {opt_path}")
    
    # Save comprehensive stats
    stats = {
        'experiment_config': config,
        'optimal_thresholds': {
            'primary': best_params[0],
            'secondary': best_params[1]
        },
        'evaluation_metrics': metrics,
        'optimization_summary': {
            'total_combinations_tested': len(optimization_results),
            'best_f1_score': max([r['f1_score'] for r in optimization_results]),
            'optimization_results': optimization_results
        },
        'timestamp': timestamp,
        'total_texts': len(osdg_df),
        'avg_labels_per_text': np.mean([len(a) for a in assignments])
    }
    
    stats_path = output_dir / f"mpnet_base_v2_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats saved to: {stats_path}")
    
    return csv_path, opt_path, stats_path

def print_experiment_report(optimization_results, metrics, best_params, config):
    """Print comprehensive experiment report."""
    print("\n" + "="*70)
    print("ADVANCED EMBEDDINGS MODEL 1: ALL-MPNET-BASE-V2")
    print("="*70)
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"Distance Metric: {config['distance_metric']}")
    print(f"Max Labels per Text: {config['max_labels_per_text']}")
    print()
    
    print("THRESHOLD OPTIMIZATION RESULTS:")
    print(f"  Combinations tested: {len(optimization_results)}")
    print(f"  Best F1 score: {max([r['f1_score'] for r in optimization_results]):.4f}")
    print(f"  Optimal primary threshold: {best_params[0]}")
    print(f"  Optimal secondary threshold: {best_params[1]}")
    print()
    
    print("FINAL EVALUATION METRICS:")
    print("Sample-based metrics:")
    print(f"  Precision: {metrics['sample_based']['precision']:.4f}")
    print(f"  Recall: {metrics['sample_based']['recall']:.4f}")
    print(f"  F1-Score: {metrics['sample_based']['f1_score']:.4f}")
    print(f"  Jaccard: {metrics['sample_based']['jaccard']:.4f}")
    print()
    print("Label-based metrics:")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['label_based']['macro_f1']:.4f}")
    print()
    print(f"Conservation rate: {metrics['conservation_rate']:.4f}")
    print(f"Coverage rate: {metrics['coverage_rate']:.4f}")
    print("="*70)

def main():
    """Main experiment execution."""
    print("Starting Advanced Embeddings Model 1: all-mpnet-base-v2")
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
    
    # Optimize thresholds
    sdg_columns = [f'sdg_{i}' for i in range(1, 18)]
    best_params, best_assignments, optimization_results = optimize_thresholds(
        similarity_matrix, osdg_df, sdg_columns, CONFIG
    )
    
    # Calculate final evaluation metrics
    metrics = calculate_evaluation_metrics(osdg_df, best_assignments, sdg_columns)
    
    # Save results
    csv_path, opt_path, stats_path = save_results(
        osdg_df, best_assignments, similarity_matrix, optimization_results, 
        metrics, best_params, CONFIG
    )
    
    # Print report
    print_experiment_report(optimization_results, metrics, best_params, CONFIG)
    
    print(f"\nModel 1 (MPNet Base v2) experiment completed successfully!")
    print(f"Results: {csv_path}")
    print(f"Optimization: {opt_path}")
    print(f"Stats: {stats_path}")

if __name__ == "__main__":
    main()
