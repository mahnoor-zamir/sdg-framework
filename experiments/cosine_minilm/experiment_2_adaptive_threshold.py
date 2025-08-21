#!/usr/bin/env python3
"""
Experiment 2: Cosine Similarity + MiniLM + Adaptive Thresholds
==============================================================

Advanced implementation of cosine similarity classification with:
- all-MiniLM-L6-v2 embeddings
- Adaptive threshold strategy based on target metrics
- Assignment control to prevent over-labeling
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
    'target_avg_labels': 2.8,  # Target average labels per text
    'min_labels': 1,
    'max_labels': 4,
    'adaptive_strategy': 'dynamic_percentile',
    'coverage_guarantee': True,  # Ensure all texts get at least one label
    'fallback_threshold': 0.25,  # Minimum threshold for coverage
    'experiment_name': 'cosine_minilm_adaptive_threshold'
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

def calculate_adaptive_thresholds(similarity_matrix, target_avg_labels, min_labels, max_labels):
    """Calculate adaptive thresholds based on target metrics."""
    print("Calculating adaptive thresholds...")
    
    # Get all similarity scores for analysis
    all_scores = similarity_matrix.flatten()
    max_similarities = np.max(similarity_matrix, axis=1)
    
    # Calculate initial percentile-based thresholds
    total_texts = similarity_matrix.shape[0]
    target_total_assignments = int(target_avg_labels * total_texts)
    
    # Try different percentile combinations to hit target
    best_thresholds = None
    best_score = float('inf')
    
    for primary_percentile in np.arange(60, 85, 2):
        for secondary_percentile in np.arange(45, primary_percentile, 2):
            primary_thresh = np.percentile(all_scores, primary_percentile)
            secondary_thresh = np.percentile(all_scores, secondary_percentile)
            
            # Simulate assignments with these thresholds
            simulated_assignments = 0
            for i in range(total_texts):
                similarities = similarity_matrix[i]
                
                # Primary assignments
                primary_count = np.sum(similarities >= primary_thresh)
                
                # Secondary assignments (up to max_labels)
                if primary_count < max_labels:
                    secondary_candidates = np.sum(
                        (similarities >= secondary_thresh) & 
                        (similarities < primary_thresh)
                    )
                    secondary_count = min(secondary_candidates, max_labels - primary_count)
                else:
                    secondary_count = 0
                
                total_text_assignments = min(primary_count + secondary_count, max_labels)
                total_text_assignments = max(total_text_assignments, min_labels)  # Ensure minimum
                simulated_assignments += total_text_assignments
            
            # Calculate how close we are to target
            avg_simulated = simulated_assignments / total_texts
            score = abs(avg_simulated - target_avg_labels)
            
            if score < best_score:
                best_score = score
                best_thresholds = {
                    'primary': primary_thresh,
                    'secondary': secondary_thresh,
                    'primary_percentile': primary_percentile,
                    'secondary_percentile': secondary_percentile,
                    'predicted_avg': avg_simulated
                }
    
    print(f"Optimal adaptive thresholds found:")
    print(f"  Primary: {best_thresholds['primary']:.4f} ({best_thresholds['primary_percentile']}th percentile)")
    print(f"  Secondary: {best_thresholds['secondary']:.4f} ({best_thresholds['secondary_percentile']}th percentile)")
    print(f"  Predicted avg labels: {best_thresholds['predicted_avg']:.2f}")
    
    return best_thresholds

def apply_adaptive_thresholds(similarity_matrix, thresholds, config):
    """Apply adaptive thresholds with assignment control."""
    print("Applying adaptive thresholds with assignment control...")
    
    primary_thresh = thresholds['primary']
    secondary_thresh = thresholds['secondary']
    fallback_thresh = config['fallback_threshold']
    min_labels = config['min_labels']
    max_labels = config['max_labels']
    
    assignments = []
    stats = {
        'primary_assignments': 0,
        'secondary_assignments': 0,
        'fallback_assignments': 0,
        'total_assignments': 0,
        'texts_with_labels': 0,
        'zero_label_texts': 0,
        'threshold_distribution': {
            'primary_only': 0,
            'primary_secondary': 0,
            'fallback_used': 0,
            'none': 0
        }
    }
    
    for i, similarities in enumerate(similarity_matrix):
        text_labels = []
        assignment_type = 'none'
        
        # Sort indices by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[sorted_indices]
        
        # Primary threshold assignments
        primary_indices = sorted_indices[sorted_similarities >= primary_thresh]
        text_labels.extend(primary_indices.tolist())
        stats['primary_assignments'] += len(primary_indices)
        
        if len(primary_indices) > 0:
            assignment_type = 'primary_only'
        
        # Secondary threshold assignments (if space available)
        if len(text_labels) < max_labels:
            remaining_indices = sorted_indices[
                (sorted_similarities >= secondary_thresh) & 
                (sorted_similarities < primary_thresh)
            ]
            
            slots_available = max_labels - len(text_labels)
            secondary_to_add = remaining_indices[:slots_available]
            text_labels.extend(secondary_to_add.tolist())
            stats['secondary_assignments'] += len(secondary_to_add)
            
            if len(secondary_to_add) > 0:
                assignment_type = 'primary_secondary'
        
        # Fallback mechanism for coverage guarantee
        if len(text_labels) == 0 and config['coverage_guarantee']:
            # Use highest similarity if above fallback threshold
            best_idx = sorted_indices[0]
            best_sim = sorted_similarities[0]
            
            if best_sim >= fallback_thresh:
                text_labels.append(best_idx)
                stats['fallback_assignments'] += 1
                assignment_type = 'fallback_used'
        
        # Ensure minimum labels (if any assignments made)
        if len(text_labels) > 0 and len(text_labels) < min_labels:
            # Add more labels up to min_labels
            remaining_needed = min_labels - len(text_labels)
            available_indices = [idx for idx in sorted_indices 
                               if idx not in text_labels and 
                               similarities[idx] >= fallback_thresh]
            
            additional_labels = available_indices[:remaining_needed]
            text_labels.extend(additional_labels)
            stats['fallback_assignments'] += len(additional_labels)
        
        # Convert to SDG numbers (1-based) and limit to max_labels
        sdg_assignments = [idx + 1 for idx in text_labels[:max_labels]]
        assignments.append(sdg_assignments)
        
        # Update stats
        if len(sdg_assignments) > 0:
            stats['texts_with_labels'] += 1
        else:
            stats['zero_label_texts'] += 1
        
        stats['total_assignments'] += len(sdg_assignments)
        stats['threshold_distribution'][assignment_type] += 1
    
    # Calculate summary statistics
    avg_labels = stats['total_assignments'] / len(similarity_matrix)
    coverage = (stats['texts_with_labels'] / len(similarity_matrix)) * 100
    
    print(f"Adaptive assignment completed:")
    print(f"  Average labels per text: {avg_labels:.2f}")
    print(f"  Coverage rate: {coverage:.1f}%")
    print(f"  Primary assignments: {stats['primary_assignments']}")
    print(f"  Secondary assignments: {stats['secondary_assignments']}")
    print(f"  Fallback assignments: {stats['fallback_assignments']}")
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
    sample_conservation = []
    
    for i in range(n_texts):
        if y_pred[i].sum() > 0 or y_true[i].sum() > 0:
            # Precision and Recall
            if y_pred[i].sum() > 0:
                prec = np.sum(y_true[i] * y_pred[i]) / np.sum(y_pred[i])
            else:
                prec = 0.0
            
            if y_true[i].sum() > 0:
                rec = np.sum(y_true[i] * y_pred[i]) / np.sum(y_true[i])
            else:
                rec = 1.0 if y_pred[i].sum() == 0 else 0.0
            
            # F1 and Jaccard
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            
            union = np.sum((y_true[i] + y_pred[i]) > 0)
            if union > 0:
                jacc = np.sum(y_true[i] * y_pred[i]) / union
            else:
                jacc = 1.0
            
            # Conservation rate (for this text)
            if y_true[i].sum() > 0:
                conservation = np.sum(y_true[i] * y_pred[i]) / np.sum(y_true[i])
            else:
                conservation = 1.0 if y_pred[i].sum() == 0 else 0.0
            
            sample_precision.append(prec)
            sample_recall.append(rec)
            sample_f1.append(f1)
            sample_jaccard.append(jacc)
            sample_conservation.append(conservation)
    
    # Label-based metrics
    micro_precision = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_recall = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Overall conservation rate
    total_original = y_true.sum()
    total_preserved = (y_true * y_pred).sum()
    overall_conservation = total_preserved / total_original if total_original > 0 else 0
    
    metrics = {
        'sample_based': {
            'precision': np.mean(sample_precision) if sample_precision else 0,
            'recall': np.mean(sample_recall) if sample_recall else 0,
            'f1_score': np.mean(sample_f1) if sample_f1 else 0,
            'jaccard': np.mean(sample_jaccard) if sample_jaccard else 0,
            'conservation': np.mean(sample_conservation) if sample_conservation else 0
        },
        'label_based': {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        },
        'overall_conservation_rate': overall_conservation,
        'coverage_rate': len([a for a in assignments if len(a) > 0]) / len(assignments)
    }
    
    return metrics

def save_results(osdg_df, assignments, similarity_matrix, thresholds, stats, metrics, config):
    """Save experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/adaptive_threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results DataFrame
    results_df = osdg_df.copy()
    
    # Add assigned SDG columns
    for i, text_assignments in enumerate(assignments):
        sdg_list = ','.join(map(str, text_assignments)) if text_assignments else ''
        results_df.loc[i, 'assigned_sdgs'] = sdg_list
        results_df.loc[i, 'num_assigned_sdgs'] = len(text_assignments)
        results_df.loc[i, 'max_similarity'] = np.max(similarity_matrix[i])
    
    # Add similarity scores
    for sdg_idx in range(similarity_matrix.shape[1]):
        results_df[f'sim_sdg_{sdg_idx + 1}'] = similarity_matrix[:, sdg_idx]
    
    # Save CSV
    csv_path = output_dir / f"experiment_2_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save comprehensive stats
    full_stats = {
        'experiment_config': config,
        'adaptive_thresholds': thresholds,
        'assignment_stats': stats,
        'evaluation_metrics': metrics,
        'timestamp': timestamp,
        'total_texts': len(osdg_df),
        'avg_labels_per_text': stats['total_assignments'] / len(osdg_df)
    }
    
    stats_path = output_dir / f"experiment_2_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(full_stats, f, indent=2, default=str)
    print(f"Stats saved to: {stats_path}")
    
    return csv_path, stats_path

def print_experiment_report(thresholds, stats, metrics, config):
    """Print comprehensive experiment report."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: COSINE SIMILARITY + ADAPTIVE THRESHOLDS")
    print("="*70)
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"Distance Metric: {config['distance_metric']}")
    print(f"Target Avg Labels: {config['target_avg_labels']}")
    print(f"Label Range: {config['min_labels']}-{config['max_labels']}")
    print(f"Coverage Guarantee: {config['coverage_guarantee']}")
    print()
    
    print("ADAPTIVE THRESHOLDS:")
    print(f"  Primary: {thresholds['primary']:.4f} ({thresholds['primary_percentile']}th percentile)")
    print(f"  Secondary: {thresholds['secondary']:.4f} ({thresholds['secondary_percentile']}th percentile)")
    print(f"  Fallback: {config['fallback_threshold']:.4f}")
    print()
    
    print("ASSIGNMENT STATISTICS:")
    total_texts = stats['texts_with_labels'] + stats['zero_label_texts']
    print(f"  Average labels per text: {stats['total_assignments'] / total_texts:.2f}")
    print(f"  Coverage rate: {(stats['texts_with_labels'] / total_texts) * 100:.1f}%")
    print(f"  Total assignments: {stats['total_assignments']:,}")
    print(f"  Primary assignments: {stats['primary_assignments']:,}")
    print(f"  Secondary assignments: {stats['secondary_assignments']:,}")
    print(f"  Fallback assignments: {stats['fallback_assignments']:,}")
    print()
    
    print("THRESHOLD USAGE:")
    for assignment_type, count in stats['threshold_distribution'].items():
        percentage = (count / total_texts) * 100
        print(f"  {assignment_type.replace('_', ' ').title()}: {count:,} texts ({percentage:.1f}%)")
    print()
    
    print("EVALUATION METRICS:")
    print("Sample-based metrics:")
    print(f"  Precision: {metrics['sample_based']['precision']:.4f}")
    print(f"  Recall: {metrics['sample_based']['recall']:.4f}")
    print(f"  F1-Score: {metrics['sample_based']['f1_score']:.4f}")
    print(f"  Jaccard: {metrics['sample_based']['jaccard']:.4f}")
    print(f"  Conservation: {metrics['sample_based']['conservation']:.4f}")
    print()
    print("Label-based metrics:")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['label_based']['macro_f1']:.4f}")
    print()
    print(f"Overall conservation rate: {metrics['overall_conservation_rate']:.4f}")
    print(f"Coverage rate: {metrics['coverage_rate']:.4f}")
    print("="*70)

def main():
    """Main experiment execution."""
    print("Starting Experiment 2: Cosine Similarity + Adaptive Thresholds")
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
    
    # Calculate adaptive thresholds
    thresholds = calculate_adaptive_thresholds(
        similarity_matrix, 
        CONFIG['target_avg_labels'],
        CONFIG['min_labels'],
        CONFIG['max_labels']
    )
    
    # Apply adaptive thresholds
    assignments, stats = apply_adaptive_thresholds(similarity_matrix, thresholds, CONFIG)
    
    # Calculate evaluation metrics
    sdg_columns = [f'sdg_{i}' for i in range(1, 18)]
    metrics = calculate_evaluation_metrics(osdg_df, assignments, sdg_columns)
    
    # Save results
    csv_path, stats_path = save_results(
        osdg_df, assignments, similarity_matrix, thresholds, stats, metrics, CONFIG
    )
    
    # Print report
    print_experiment_report(thresholds, stats, metrics, CONFIG)
    
    print(f"\nExperiment 2 completed successfully!")
    print(f"Results: {csv_path}")
    print(f"Stats: {stats_path}")

if __name__ == "__main__":
    main()
