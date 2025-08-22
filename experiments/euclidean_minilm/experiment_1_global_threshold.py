#!/usr/bin/env python3
"""
Euclidean Distance + MiniLM Experiment 1: Global Fixed Thresholds
================================================================

This experiment tests SDG classification using Euclidean distance with fixed global thresholds.
Based on previous research, we'll optimize thresholds to avoid over-assignment issues.

Key Differences from Cosine:
- Lower distance = higher similarity (inverted logic)
- Different optimal threshold ranges
- Tendency to over-assign labels (needs control)

Author: Research Team
Date: August 21, 2025
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import time
from datetime import datetime
import os

def calculate_euclidean_distances(embeddings1, embeddings2):
    """Calculate Euclidean distances between two sets of embeddings."""
    # Expand dimensions for broadcasting
    emb1_expanded = embeddings1[:, np.newaxis, :]  # (n_texts, 1, embedding_dim)
    emb2_expanded = embeddings2[np.newaxis, :, :]  # (1, n_sdgs, embedding_dim)
    
    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((emb1_expanded - emb2_expanded) ** 2, axis=2))
    return distances

def assign_labels_euclidean(distances, primary_threshold, secondary_threshold, max_labels=5):
    """
    Assign SDG labels based on Euclidean distances.
    Lower distance = higher similarity (opposite of cosine)
    """
    assignments = []
    
    for i, text_distances in enumerate(distances):
        # Get SDG indices sorted by distance (ascending - closest first)
        sorted_indices = np.argsort(text_distances)
        sorted_distances = text_distances[sorted_indices]
        
        # Assign labels based on distance thresholds
        assigned_sdgs = []
        
        # Primary threshold assignments (closest matches)
        for j, (sdg_idx, dist) in enumerate(zip(sorted_indices, sorted_distances)):
            if dist <= primary_threshold and len(assigned_sdgs) < max_labels:
                assigned_sdgs.append(sdg_idx + 1)  # SDG numbers are 1-indexed
        
        # Secondary threshold assignments if we have room
        if len(assigned_sdgs) < max_labels:
            for j, (sdg_idx, dist) in enumerate(zip(sorted_indices, sorted_distances)):
                if (primary_threshold < dist <= secondary_threshold and 
                    (sdg_idx + 1) not in assigned_sdgs and 
                    len(assigned_sdgs) < max_labels):
                    assigned_sdgs.append(sdg_idx + 1)
        
        assignments.append(sorted(assigned_sdgs))
    
    return assignments

def calculate_comprehensive_metrics(y_true_binary, y_pred_binary, y_true_lists, y_pred_lists):
    """Calculate comprehensive evaluation metrics."""
    
    # Sample-based metrics
    sample_precision = []
    sample_recall = []
    sample_f1 = []
    sample_jaccard = []
    
    for i in range(len(y_true_lists)):
        true_set = set(y_true_lists[i])
        pred_set = set(y_pred_lists[i])
        
        if len(pred_set) == 0:
            prec = 1.0 if len(true_set) == 0 else 0.0
            rec = 1.0 if len(true_set) == 0 else 0.0
        else:
            intersection = len(true_set & pred_set)
            prec = intersection / len(pred_set)
            rec = intersection / len(true_set) if len(true_set) > 0 else 1.0
        
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # Jaccard similarity
        union = len(true_set | pred_set)
        jaccard = len(true_set & pred_set) / union if union > 0 else 1.0
        
        sample_precision.append(prec)
        sample_recall.append(rec)
        sample_f1.append(f1)
        sample_jaccard.append(jaccard)
    
    # Label-based metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )
    
    return {
        'sample_based': {
            'precision': np.mean(sample_precision),
            'recall': np.mean(sample_recall),
            'f1_score': np.mean(sample_f1),
            'jaccard': np.mean(sample_jaccard)
        },
        'label_based': {
            'micro_precision': precision_micro,
            'micro_recall': recall_micro,
            'micro_f1': f1_micro,
            'macro_precision': precision_macro,
            'macro_recall': recall_macro,
            'macro_f1': f1_macro
        }
    }

def main():
    """Main experiment execution."""
    
    print("=" * 70)
    print("EUCLIDEAN DISTANCE + MiniLM EXPERIMENT 1: GLOBAL THRESHOLDS")
    print("=" * 70)
    
    # Configuration - Optimized for Euclidean distance
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'distance_metric': 'euclidean',
        'primary_threshold': 1.03,    # Based on distance analysis (25th percentile of min distances)
        'secondary_threshold': 1.10,  # Based on distance analysis (50th percentile of min distances)
        'max_labels_per_text': 5,
        'experiment_name': 'euclidean_minilm_global_threshold'
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    print("Loading dataset...")
    data_path = '/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/osdg_multilabel_threshold_0.6.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} texts from OSDG dataset")
    
    # Load SDG descriptions
    sdg_path = '/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/sdg_paragraph_dataset.csv'
    sdg_df = pd.read_csv(sdg_path)
    sdg_descriptions = sdg_df['text'].tolist()
    print(f"Loaded {len(sdg_descriptions)} SDG descriptions")
    
    # Initialize model
    print("Loading embedding model...")
    model = SentenceTransformer(config['embedding_model'])
    
    # Generate embeddings
    print("Generating embeddings...")
    start_time = time.time()
    
    text_embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    sdg_embeddings = model.encode(sdg_descriptions, show_progress_bar=True)
    
    embedding_time = time.time() - start_time
    print(f"Embedding generation completed in {embedding_time:.2f} seconds")
    
    # Calculate Euclidean distances
    print("Calculating Euclidean distances...")
    distances = calculate_euclidean_distances(text_embeddings, sdg_embeddings)
    
    # Assign labels
    print("Assigning labels based on distance thresholds...")
    predicted_labels = assign_labels_euclidean(
        distances, 
        config['primary_threshold'], 
        config['secondary_threshold'],
        config['max_labels_per_text']
    )
    
    # Prepare ground truth labels
    sdg_columns = [col for col in df.columns if col.startswith('sdg_') and col != 'sdg_text']
    y_true_lists = []
    
    for _, row in df.iterrows():
        true_labels = []
        for col in sdg_columns:
            if row[col] == 1:
                sdg_num = int(col.split('_')[1])
                true_labels.append(sdg_num)
        y_true_lists.append(true_labels)
    
    # Convert to binary format for sklearn metrics
    all_labels = list(range(1, 18))
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(predicted_labels)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_comprehensive_metrics(y_true_binary, y_pred_binary, y_true_lists, predicted_labels)
    
    # Assignment statistics
    total_assignments = sum(len(labels) for labels in predicted_labels)
    texts_with_labels = sum(1 for labels in predicted_labels if len(labels) > 0)
    zero_label_texts = len(predicted_labels) - texts_with_labels
    avg_labels_per_text = total_assignments / len(predicted_labels)
    
    # Calculate distance statistics
    avg_min_distances = np.mean(np.min(distances, axis=1))
    
    # Primary/Secondary assignment breakdown
    primary_assignments = 0
    secondary_assignments = 0
    
    for text_distances in distances:
        min_distance = np.min(text_distances)
        distances_in_primary = np.sum(text_distances <= config['primary_threshold'])
        distances_in_secondary = np.sum((text_distances > config['primary_threshold']) & 
                                      (text_distances <= config['secondary_threshold']))
        primary_assignments += distances_in_primary
        secondary_assignments += distances_in_secondary
    
    # Conservation rate
    total_true_labels = sum(len(labels) for labels in y_true_lists)
    total_correct_predictions = sum(len(set(true) & set(pred)) for true, pred in zip(y_true_lists, predicted_labels))
    conservation_rate = total_correct_predictions / total_true_labels if total_true_labels > 0 else 0
    coverage_rate = texts_with_labels / len(df)
    
    # Print results
    print("\n" + "=" * 50)
    print("EXPERIMENT 1 RESULTS - GLOBAL THRESHOLDS")
    print("=" * 50)
    
    print(f"\nASSIGNMENT STATISTICS:")
    print(f"  Average labels per text: {avg_labels_per_text:.2f}")
    print(f"  Coverage rate: {coverage_rate:.1%}")
    print(f"  Total assignments: {total_assignments:,}")
    print(f"  Primary assignments: {primary_assignments:,}")
    print(f"  Secondary assignments: {secondary_assignments:,}")
    print(f"  Zero-label texts: {zero_label_texts:,}")
    print(f"  Average minimum distance: {avg_min_distances:.4f}")
    
    print(f"\nEVALUATION METRICS:")
    print(f"Sample-based metrics:")
    print(f"  Precision: {metrics['sample_based']['precision']:.4f}")
    print(f"  Recall: {metrics['sample_based']['recall']:.4f}")
    print(f"  F1-Score: {metrics['sample_based']['f1_score']:.4f}")
    print(f"  Jaccard: {metrics['sample_based']['jaccard']:.4f}")
    
    print(f"\nLabel-based metrics:")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['label_based']['macro_f1']:.4f}")
    
    print(f"\nConservation rate: {conservation_rate:.4f}")
    print(f"Coverage rate: {coverage_rate:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '/Users/mahnoorzamir/Desktop/mitacs/project/experiments/euclidean_minilm/results/global_threshold'
    
    # Save CSV results
    results_df = df.copy()
    results_df['predicted_sdgs'] = [','.join(map(str, labels)) if labels else '' for labels in predicted_labels]
    results_df['num_predicted_labels'] = [len(labels) for labels in predicted_labels]
    results_df['min_distance'] = np.min(distances, axis=1)
    
    # Add individual distance columns
    for i in range(17):
        results_df[f'distance_sdg_{i+1}'] = distances[:, i]
    
    csv_path = os.path.join(results_dir, f'experiment_1_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save statistics JSON
    stats = {
        'experiment_config': config,
        'assignment_stats': {
            'primary_assignments': int(primary_assignments),
            'secondary_assignments': int(secondary_assignments),
            'total_assignments': int(total_assignments),
            'texts_with_labels': int(texts_with_labels),
            'zero_label_texts': int(zero_label_texts)
        },
        'evaluation_metrics': metrics,
        'conservation_rate': conservation_rate,
        'coverage_rate': coverage_rate,
        'distance_stats': {
            'avg_min_distance': float(avg_min_distances),
            'primary_threshold': config['primary_threshold'],
            'secondary_threshold': config['secondary_threshold']
        },
        'timestamp': timestamp,
        'total_texts': len(df),
        'avg_labels_per_text': avg_labels_per_text
    }
    
    json_path = os.path.join(results_dir, f'experiment_1_stats_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {json_path}")
    
    print("\n" + "=" * 50)
    print("EXPERIMENT 1 COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == "__main__":
    main()
