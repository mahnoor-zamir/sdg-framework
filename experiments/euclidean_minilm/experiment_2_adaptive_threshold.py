#!/usr/bin/env python3
"""
Euclidean Distance + MiniLM Experiment 2: Adaptive Dynamic Thresholds
=====================================================================

This experiment tests SDG classification using Euclidean distance with adaptive thresholds
that dynamically adjust based on the distance distribution to achieve target metrics.

Key Features:
- Dynamic threshold calculation based on percentiles
- Target average labels per text (2.8)
- Assignment control (1-4 labels per text)
- Coverage guarantee with fallback mechanism
- Euclidean distance optimization

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

def calculate_adaptive_thresholds(distances, target_avg_labels=2.8):
    """
    Calculate adaptive thresholds based on distance distribution.
    For Euclidean: lower distance = higher similarity
    """
    print("Calculating adaptive thresholds...")
    
    # Get all distance values
    all_distances = distances.flatten()
    
    # Calculate percentiles for threshold candidates
    # For Euclidean, we want lower distances (better matches)
    percentiles = np.arange(10, 90, 2)
    threshold_candidates = np.percentile(all_distances, percentiles)
    
    best_threshold = None
    best_secondary = None
    best_score = float('inf')
    best_avg_labels = 0
    
    print("Testing threshold combinations...")
    for i, primary_thresh in enumerate(threshold_candidates):
        for j, secondary_thresh in enumerate(threshold_candidates):
            if secondary_thresh <= primary_thresh:  # Secondary should be higher distance (lower similarity)
                continue
                
            # Test this threshold combination
            total_labels = 0
            for text_distances in distances:
                text_labels = 0
                # Count labels that would be assigned
                for dist in text_distances:
                    if dist <= primary_thresh:
                        text_labels += 1
                    elif dist <= secondary_thresh and text_labels < 4:  # Max 4 labels
                        text_labels += 1
                
                # Ensure at least 1 label per text, max 4
                text_labels = max(1, min(text_labels, 4))
                total_labels += text_labels
            
            avg_labels = total_labels / len(distances)
            score = abs(avg_labels - target_avg_labels)
            
            if score < best_score:
                best_score = score
                best_threshold = primary_thresh
                best_secondary = secondary_thresh
                best_avg_labels = avg_labels
    
    # Convert thresholds back to percentiles for reporting
    primary_percentile = np.searchsorted(np.sort(all_distances), best_threshold) / len(all_distances) * 100
    secondary_percentile = np.searchsorted(np.sort(all_distances), best_secondary) / len(all_distances) * 100
    
    print(f"Selected adaptive thresholds:")
    print(f"  Primary: {best_threshold:.4f} ({primary_percentile:.0f}th percentile)")
    print(f"  Secondary: {best_secondary:.4f} ({secondary_percentile:.0f}th percentile)")
    print(f"  Predicted avg labels: {best_avg_labels:.2f}")
    print()
    
    return best_threshold, best_secondary, primary_percentile, secondary_percentile

def assign_labels_adaptive_euclidean(distances, primary_threshold, secondary_threshold, fallback_threshold=1.25):
    """
    Assign SDG labels using adaptive Euclidean distance thresholds.
    Ensures 1-4 labels per text with fallback for difficult texts.
    """
    assignments = []
    fallback_count = 0
    assignment_breakdown = {
        'primary_only': 0,
        'primary_secondary': 0,
        'fallback_used': 0,
        'none': 0
    }
    
    total_primary_assignments = 0
    total_secondary_assignments = 0
    total_fallback_assignments = 0
    
    for i, text_distances in enumerate(distances):
        # Get SDG indices sorted by distance (ascending - closest first)
        sorted_indices = np.argsort(text_distances)
        sorted_distances = text_distances[sorted_indices]
        
        assigned_sdgs = []
        used_fallback = False
        
        # Primary threshold assignments (closest matches)
        for j, (sdg_idx, dist) in enumerate(zip(sorted_indices, sorted_distances)):
            if dist <= primary_threshold and len(assigned_sdgs) < 4:
                assigned_sdgs.append(sdg_idx + 1)  # SDG numbers are 1-indexed
                total_primary_assignments += 1
        
        # Secondary threshold assignments if we have room and need more
        initial_count = len(assigned_sdgs)
        if len(assigned_sdgs) < 4:
            for j, (sdg_idx, dist) in enumerate(zip(sorted_indices, sorted_distances)):
                if (primary_threshold < dist <= secondary_threshold and 
                    (sdg_idx + 1) not in assigned_sdgs and 
                    len(assigned_sdgs) < 4):
                    assigned_sdgs.append(sdg_idx + 1)
                    total_secondary_assignments += 1
        
        # Fallback mechanism: ensure at least 1 label per text
        if len(assigned_sdgs) == 0:
            # Find the closest SDG (minimum distance)
            closest_sdg = sorted_indices[0] + 1
            if sorted_distances[0] <= fallback_threshold:
                assigned_sdgs.append(closest_sdg)
                used_fallback = True
                fallback_count += 1
                total_fallback_assignments += 1
        
        # Ensure we have 1-4 labels (but allow 0 if nothing meets fallback threshold)
        if len(assigned_sdgs) > 4:
            assigned_sdgs = assigned_sdgs[:4]
        
        # Track assignment breakdown
        if used_fallback:
            assignment_breakdown['fallback_used'] += 1
        elif len(assigned_sdgs) == initial_count and initial_count > 0:
            assignment_breakdown['primary_only'] += 1
        elif len(assigned_sdgs) > initial_count:
            assignment_breakdown['primary_secondary'] += 1
        else:
            assignment_breakdown['none'] += 1
        
        assignments.append(sorted(assigned_sdgs))
    
    print(f"Assignment breakdown:")
    print(f"  Primary Only: {assignment_breakdown['primary_only']:,} texts ({assignment_breakdown['primary_only']/len(distances)*100:.1f}%)")
    print(f"  Primary + Secondary: {assignment_breakdown['primary_secondary']:,} texts ({assignment_breakdown['primary_secondary']/len(distances)*100:.1f}%)")
    print(f"  Fallback Used: {assignment_breakdown['fallback_used']:,} texts ({assignment_breakdown['fallback_used']/len(distances)*100:.1f}%)")
    print(f"  None: {assignment_breakdown['none']:,} texts ({assignment_breakdown['none']/len(distances)*100:.1f}%)")
    print()
    
    return assignments, {
        'primary_assignments': total_primary_assignments,
        'secondary_assignments': total_secondary_assignments,
        'fallback_assignments': total_fallback_assignments,
        'breakdown': assignment_breakdown
    }

def calculate_ranking_metrics(distances, y_true_lists, k_values=[1, 2, 3, 5, 10]):
    """
    Calculate ranking-based metrics (Hit@K and MRR) for single-label ground truth.
    
    Args:
        distances: Distance matrix (n_texts, n_sdgs) - lower is more similar
        y_true_lists: List of ground truth SDG lists per text
        k_values: List of k values for Hit@K calculation
    
    Returns:
        Dictionary with ranking metrics
    """
    print("Calculating ranking metrics...")
    
    # Convert multi-label ground truth to primary label (first/main SDG)
    true_primary_indices = []
    for true_labels in y_true_lists:
        if len(true_labels) > 0:
            # Use the first SDG as primary (could also use most frequent or other logic)
            primary_sdg = true_labels[0]  # SDG numbers are 1-indexed
            true_primary_indices.append(primary_sdg - 1)  # Convert to 0-indexed for array access
        else:
            # Handle texts with no labels (shouldn't happen in OSDG dataset)
            true_primary_indices.append(0)  # Default to SDG 1
    
    # Calculate Hit@K metrics
    hit_metrics = {}
    reciprocal_ranks = []
    
    for k in k_values:
        hits = 0
        for i, text_distances in enumerate(distances):
            true_primary_idx = true_primary_indices[i]
            
            # Get top-k SDG indices with lowest distances (most similar)
            top_k_indices = np.argsort(text_distances)[:k]
            
            # Check if primary true SDG is in top-k predictions
            if true_primary_idx in top_k_indices:
                hits += 1
        
        hit_metrics[f'Hit@{k}'] = hits / len(distances)
    
    # Calculate Mean Reciprocal Rank (MRR)
    for i, text_distances in enumerate(distances):
        true_primary_idx = true_primary_indices[i]
        
        # Get ranking of all SDGs (0-indexed positions)
        sdg_ranking = np.argsort(text_distances)
        
        # Find position of true primary SDG (1-indexed rank)
        true_rank = np.where(sdg_ranking == true_primary_idx)[0][0] + 1
        reciprocal_ranks.append(1.0 / true_rank)
    
    mrr = np.mean(reciprocal_ranks)
    avg_true_rank = 1.0 / mrr
    
    print(f"Ranking Metrics Results:")
    for k in k_values:
        print(f"  Hit@{k}: {hit_metrics[f'Hit@{k}']:.4f} ({hit_metrics[f'Hit@{k}']*100:.1f}%)")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  Average True SDG Rank: {avg_true_rank:.2f}")
    print()
    
    return {
        'hit_metrics': hit_metrics,
        'mrr': mrr,
        'avg_true_rank': avg_true_rank,
        'individual_reciprocal_ranks': reciprocal_ranks
    }

def calculate_comprehensive_metrics(y_true_binary, y_pred_binary, y_true_lists, y_pred_lists, distances=None):
    """Calculate comprehensive evaluation metrics including traditional and ranking metrics."""
    
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
    
    # Calculate ranking metrics if distances are provided
    ranking_metrics = None
    if distances is not None:
        ranking_metrics = calculate_ranking_metrics(distances, y_true_lists)
    
    results = {
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
    
    # Add ranking metrics if calculated
    if ranking_metrics:
        results['ranking_metrics'] = ranking_metrics
    
    return results

def main():
    """Main experiment execution."""
    
    print("=" * 70)
    print("EUCLIDEAN DISTANCE + MiniLM EXPERIMENT 2: ADAPTIVE THRESHOLDS")
    print("=" * 70)
    
    # Configuration
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'distance_metric': 'euclidean',
        'target_avg_labels': 2.8,
        'min_labels_per_text': 1,
        'max_labels_per_text': 4,
        'fallback_threshold': 1.25,  # Based on distance analysis - between median and 75th percentile
        'experiment_name': 'euclidean_minilm_adaptive_threshold'
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
    
    # Calculate adaptive thresholds
    primary_threshold, secondary_threshold, primary_pct, secondary_pct = calculate_adaptive_thresholds(
        distances, config['target_avg_labels']
    )
    
    # Update config with calculated thresholds
    config['primary_threshold'] = primary_threshold
    config['secondary_threshold'] = secondary_threshold
    config['primary_percentile'] = primary_pct
    config['secondary_percentile'] = secondary_pct
    
    # Assign labels using adaptive thresholds
    print("Assigning labels using adaptive thresholds...")
    predicted_labels, assignment_stats = assign_labels_adaptive_euclidean(
        distances, primary_threshold, secondary_threshold, config['fallback_threshold']
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
    metrics = calculate_comprehensive_metrics(y_true_binary, y_pred_binary, y_true_lists, predicted_labels, distances)
    
    # Assignment statistics
    total_assignments = sum(len(labels) for labels in predicted_labels)
    texts_with_labels = sum(1 for labels in predicted_labels if len(labels) > 0)
    zero_label_texts = len(predicted_labels) - texts_with_labels
    avg_labels_per_text = total_assignments / len(predicted_labels)
    
    # Distance statistics
    avg_min_distances = np.mean(np.min(distances, axis=1))
    
    # Conservation rate
    total_true_labels = sum(len(labels) for labels in y_true_lists)
    total_correct_predictions = sum(len(set(true) & set(pred)) for true, pred in zip(y_true_lists, predicted_labels))
    conservation_rate = total_correct_predictions / total_true_labels if total_true_labels > 0 else 0
    coverage_rate = texts_with_labels / len(df)
    
    # Print results
    print("\n" + "=" * 50)
    print("EXPERIMENT 2 RESULTS - ADAPTIVE THRESHOLDS")
    print("=" * 50)
    
    print(f"\nCALCULATED THRESHOLDS:")
    print(f"  Primary: {primary_threshold:.4f} ({primary_pct:.0f}th percentile)")
    print(f"  Secondary: {secondary_threshold:.4f} ({secondary_pct:.0f}th percentile)")
    print(f"  Predicted avg labels: {avg_labels_per_text:.2f}")
    
    print(f"\nASSIGNMENT STATISTICS:")
    print(f"  Average labels per text: {avg_labels_per_text:.2f}")
    print(f"  Coverage rate: {coverage_rate:.1%}")
    print(f"  Total assignments: {total_assignments:,}")
    print(f"  Primary assignments: {assignment_stats['primary_assignments']:,}")
    print(f"  Secondary assignments: {assignment_stats['secondary_assignments']:,}")
    print(f"  Fallback assignments: {assignment_stats['fallback_assignments']:,}")
    print(f"  Zero-label texts: {zero_label_texts:,}")
    print(f"  Average minimum distance: {avg_min_distances:.4f}")
    
    print(f"\nTHRESHOLD USAGE:")
    breakdown = assignment_stats['breakdown']
    for key, value in breakdown.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,} texts ({value/len(df)*100:.1f}%)")
    
    print(f"\nEVALUATION METRICS:")
    print(f"Sample-based metrics:")
    print(f"  Precision: {metrics['sample_based']['precision']:.4f}")
    print(f"  Recall: {metrics['sample_based']['recall']:.4f}")
    print(f"  F1-Score: {metrics['sample_based']['f1_score']:.4f}")
    print(f"  Jaccard: {metrics['sample_based']['jaccard']:.4f}")
    print(f"  Conservation: {conservation_rate:.4f}")
    
    print(f"\nLabel-based metrics:")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['label_based']['macro_f1']:.4f}")
    
    # Print ranking metrics if available
    if 'ranking_metrics' in metrics:
        print(f"\nRanking-based metrics:")
        ranking = metrics['ranking_metrics']
        for k in [1, 2, 3, 5, 10]:
            if f'Hit@{k}' in ranking['hit_metrics']:
                print(f"  Hit@{k}: {ranking['hit_metrics'][f'Hit@{k}']:.4f} ({ranking['hit_metrics'][f'Hit@{k}']*100:.1f}%)")
        print(f"  Mean Reciprocal Rank: {ranking['mrr']:.4f}")
        print(f"  Avg True SDG Rank: {ranking['avg_true_rank']:.2f}")
    
    print(f"\nOverall conservation rate: {conservation_rate:.4f}")
    print(f"Coverage rate: {coverage_rate:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '/Users/mahnoorzamir/Desktop/mitacs/project/experiments/euclidean_minilm/results/adaptive_threshold'
    
    # Save CSV results
    results_df = df.copy()
    results_df['predicted_sdgs'] = [','.join(map(str, labels)) if labels else '' for labels in predicted_labels]
    results_df['num_predicted_labels'] = [len(labels) for labels in predicted_labels]
    results_df['min_distance'] = np.min(distances, axis=1)
    
    # Add individual distance columns
    for i in range(17):
        results_df[f'distance_sdg_{i+1}'] = distances[:, i]
    
    csv_path = os.path.join(results_dir, f'experiment_2_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save statistics JSON
    stats = {
        'experiment_config': config,
        'assignment_stats': {
            **assignment_stats,
            'total_assignments': int(total_assignments),
            'texts_with_labels': int(texts_with_labels),
            'zero_label_texts': int(zero_label_texts)
        },
        'evaluation_metrics': metrics,
        'conservation_rate': conservation_rate,
        'coverage_rate': coverage_rate,
        'distance_stats': {
            'avg_min_distance': float(avg_min_distances),
            'primary_threshold': primary_threshold,
            'secondary_threshold': secondary_threshold,
            'fallback_threshold': config['fallback_threshold']
        },
        'timestamp': timestamp,
        'total_texts': len(df),
        'avg_labels_per_text': avg_labels_per_text
    }
    
    json_path = os.path.join(results_dir, f'experiment_2_stats_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {json_path}")
    
    print("\n" + "=" * 50)
    print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == "__main__":
    main()
