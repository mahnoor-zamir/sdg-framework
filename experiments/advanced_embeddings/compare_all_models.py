#!/usr/bin/env python3
"""
Advanced Embeddings Model Compari    'threshold_search': {
        'primary_range': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        'secondary_range': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    },===================================

This script runs all three advanced embedding models, finds their optimal thresholds,
and compares their best performances. It provides a comprehensive comparison report.

Models compared:
1. all-mpnet-base-v2
2. all-distilroberta-v1  
3. multi-qa-mpnet-base-dot-v1

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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Models to compare
MODELS = [
    {
        'name': 'MiniLM L6 v2',
        'model_id': 'all-MiniLM-L6-v2',
        'filename_prefix': 'minilm_l6_v2'
    },
    {
        'name': 'MPNet Base v2',
        'model_id': 'all-mpnet-base-v2',
        'filename_prefix': 'mpnet_base_v2'
    },
    {
        'name': 'DistilRoBERTa v1',
        'model_id': 'all-distilroberta-v1', 
        'filename_prefix': 'distilroberta_v1'
    },
    {
        'name': 'Multi-QA MPNet',
        'model_id': 'multi-qa-mpnet-base-dot-v1',
        'filename_prefix': 'multi_qa_mpnet'
    }
]

# Configuration
CONFIG = {
    'distance_metric': 'cosine',
    'max_labels_per_text': 5,
    'threshold_search': {
        'primary_range': [0.2, 0.3, 0.4, 0.5, 0.6],
        'secondary_range': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    }
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
    
    similarity_matrix = cosine_similarity(text_embeddings, sdg_embeddings)
    
    elapsed = time.time() - start_time
    print(f"Similarity calculation completed in {elapsed:.2f} seconds")
    
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
            
            remaining_slots = max_labels - len(text_labels)
            secondary_to_add = secondary_indices[:remaining_slots]
            text_labels.extend(secondary_to_add.tolist())
        
        # Convert to SDG numbers (1-based)
        sdg_assignments = [idx + 1 for idx in text_labels[:max_labels]]
        assignments.append(sdg_assignments)
    
    return assignments

def calculate_f1_score(osdg_df, assignments, sdg_columns):
    """Calculate F1 score for threshold optimization."""
    n_texts = len(osdg_df)
    n_sdgs = len(sdg_columns)
    y_pred = np.zeros((n_texts, n_sdgs))
    
    for i, text_assignments in enumerate(assignments):
        for sdg_num in text_assignments:
            if 1 <= sdg_num <= n_sdgs:
                y_pred[i, sdg_num - 1] = 1
    
    y_true = osdg_df[sdg_columns].values
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    return micro_f1

def optimize_thresholds(similarity_matrix, osdg_df, sdg_columns):
    """Find optimal primary and secondary thresholds using grid search."""
    print("Starting threshold optimization...")
    
    best_f1 = 0
    best_params = None
    best_assignments = None
    results = []
    
    param_combinations = list(product(
        CONFIG['threshold_search']['primary_range'],
        CONFIG['threshold_search']['secondary_range']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, (primary_thresh, secondary_thresh) in enumerate(param_combinations):
        if secondary_thresh >= primary_thresh:
            continue
            
        assignments = apply_thresholds(
            similarity_matrix, primary_thresh, secondary_thresh, CONFIG['max_labels_per_text']
        )
        
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
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(param_combinations)} combinations. Best F1: {best_f1:.4f}")
    
    print(f"Optimization completed. Best F1: {best_f1:.4f}")
    print(f"Best parameters: Primary={best_params[0]}, Secondary={best_params[1]}")
    
    return best_params, best_assignments, results

def calculate_comprehensive_metrics(osdg_df, assignments, sdg_columns):
    """Calculate comprehensive evaluation metrics with detailed precision and recall."""
    print("Calculating comprehensive metrics...")
    
    n_texts = len(osdg_df)
    n_sdgs = len(sdg_columns)
    y_pred = np.zeros((n_texts, n_sdgs))
    
    for i, text_assignments in enumerate(assignments):
        for sdg_num in text_assignments:
            if 1 <= sdg_num <= n_sdgs:
                y_pred[i, sdg_num - 1] = 1
    
    y_true = osdg_df[sdg_columns].values
    
    # Sample-based metrics
    sample_precision = []
    sample_recall = []
    sample_f1 = []
    sample_jaccard = []
    
    for i in range(n_texts):
        if y_pred[i].sum() > 0:
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
    
    # Weighted metrics (accounts for label imbalance)
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-SDG metrics
    per_sdg_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_sdg_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_sdg_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Conservation rate
    original_labels = y_true.sum()
    preserved_labels = (y_true * y_pred).sum()
    conservation_rate = preserved_labels / original_labels if original_labels > 0 else 0
    
    # Additional detailed metrics
    true_positives = (y_true * y_pred).sum()
    false_positives = ((1 - y_true) * y_pred).sum()
    false_negatives = (y_true * (1 - y_pred)).sum()
    true_negatives = ((1 - y_true) * (1 - y_pred)).sum()
    
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
        'per_sdg_metrics': {
            'precision': per_sdg_precision.tolist(),
            'recall': per_sdg_recall.tolist(),
            'f1_score': per_sdg_f1.tolist(),
            'sdg_labels': [f'SDG_{i+1}' for i in range(len(per_sdg_precision))]
        },
        'confusion_matrix_components': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives)
        },
        'conservation_rate': conservation_rate,
        'coverage_rate': len([a for a in assignments if len(a) > 0]) / len(assignments),
        'avg_labels_per_text': np.mean([len(a) for a in assignments]),
        'total_predictions': int(y_pred.sum()),
        'total_ground_truth': int(y_true.sum())
    }
    
    return metrics

def run_single_model_experiment(model_info, osdg_df, sdg_df, sdg_columns):
    """Run experiment for a single model."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {model_info['name']} ({model_info['model_id']})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Load model
    print(f"Loading model: {model_info['model_id']}")
    model = SentenceTransformer(model_info['model_id'])
    
    # Generate embeddings
    text_embeddings = generate_embeddings(osdg_df['text'].tolist(), model)
    sdg_embeddings = generate_embeddings(sdg_df['text'].tolist(), model)
    
    # Calculate similarities
    similarity_matrix = calculate_cosine_similarities(text_embeddings, sdg_embeddings)
    
    # Optimize thresholds
    best_params, best_assignments, optimization_results = optimize_thresholds(
        similarity_matrix, osdg_df, sdg_columns
    )
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(osdg_df, best_assignments, sdg_columns)
    
    total_time = time.time() - start_time
    
    # Package results
    result = {
        'model_info': model_info,
        'best_params': best_params,
        'metrics': metrics,
        'optimization_results': optimization_results,
        'similarity_matrix': similarity_matrix,
        'best_assignments': best_assignments,
        'total_time': total_time
    }
    
    print(f"Model {model_info['name']} completed in {total_time/60:.1f} minutes")
    print(f"Best F1 Score: {metrics['label_based']['micro_f1']:.4f}")
    
    return result

def create_comparison_visualizations(results):
    """Create visualizations comparing model performances."""
    print("Creating comparison visualizations...")
    
    # Extract metrics for comparison
    model_names = [r['model_info']['name'] for r in results]
    
    # Key metrics to compare
    metrics_data = {
        'Micro F1': [r['metrics']['label_based']['micro_f1'] for r in results],
        'Macro F1': [r['metrics']['label_based']['macro_f1'] for r in results],
        'Weighted F1': [r['metrics']['label_based']['weighted_f1'] for r in results],
        'Sample F1': [r['metrics']['sample_based']['f1_score'] for r in results],
        'Micro Precision': [r['metrics']['label_based']['micro_precision'] for r in results],
        'Macro Precision': [r['metrics']['label_based']['macro_precision'] for r in results],
        'Micro Recall': [r['metrics']['label_based']['micro_recall'] for r in results],
        'Macro Recall': [r['metrics']['label_based']['macro_recall'] for r in results],
        'Coverage Rate': [r['metrics']['coverage_rate'] for r in results],
        'Conservation Rate': [r['metrics']['conservation_rate'] for r in results]
    }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(metrics_data, index=model_names)
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Advanced Embeddings Models Performance Comparison (Including MiniLM)', fontsize=16, fontweight='bold')
    
    # Plot 1: F1 Scores Comparison
    ax1 = axes[0, 0]
    f1_metrics = comparison_df[['Micro F1', 'Macro F1', 'Weighted F1', 'Sample F1']]
    f1_metrics.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('F1 Score Comparison')
    ax1.set_ylabel('F1 Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Precision Comparison
    ax2 = axes[0, 1]
    precision_metrics = comparison_df[['Micro Precision', 'Macro Precision']]
    precision_metrics.plot(kind='bar', ax=ax2, rot=45, color=['skyblue', 'lightcoral'])
    ax2.set_title('Precision Comparison')
    ax2.set_ylabel('Precision Score')
    ax2.legend()
    
    # Plot 3: Recall Comparison
    ax3 = axes[0, 2]
    recall_metrics = comparison_df[['Micro Recall', 'Macro Recall']]
    recall_metrics.plot(kind='bar', ax=ax3, rot=45, color=['lightgreen', 'orange'])
    ax3.set_title('Recall Comparison')
    ax3.set_ylabel('Recall Score')
    ax3.legend()
    
    # Plot 4: Coverage vs Conservation
    ax4 = axes[1, 0]
    colors = ['blue', 'red', 'green', 'purple']
    ax4.scatter(comparison_df['Coverage Rate'], comparison_df['Conservation Rate'], 
               s=100, alpha=0.7, c=colors[:len(model_names)])
    for i, model in enumerate(model_names):
        ax4.annotate(model, (comparison_df.loc[model, 'Coverage Rate'], 
                            comparison_df.loc[model, 'Conservation Rate']),
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Coverage Rate')
    ax4.set_ylabel('Conservation Rate')
    ax4.set_title('Coverage vs Conservation')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Optimal Thresholds
    ax5 = axes[1, 1]
    primary_thresholds = [r['best_params'][0] for r in results]
    secondary_thresholds = [r['best_params'][1] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax5.bar(x - width/2, primary_thresholds, width, label='Primary Threshold', alpha=0.8)
    ax5.bar(x + width/2, secondary_thresholds, width, label='Secondary Threshold', alpha=0.8)
    ax5.set_xlabel('Models')
    ax5.set_ylabel('Threshold Value')
    ax5.set_title('Optimal Thresholds')
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Processing Time
    ax6 = axes[1, 2]
    processing_times = [r['total_time']/60 for r in results]  # Convert to minutes
    bars = ax6.bar(model_names, processing_times, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax6.set_ylabel('Processing Time (minutes)')
    ax6.set_title('Processing Time Comparison')
    plt.setp(ax6.get_xticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m', ha='center', va='bottom')
    
    # Plot 7: Precision vs Recall
    ax7 = axes[2, 0]
    ax7.scatter(comparison_df['Micro Recall'], comparison_df['Micro Precision'], 
               s=100, alpha=0.7, c=colors[:len(model_names)], label='Micro')
    ax7.scatter(comparison_df['Macro Recall'], comparison_df['Macro Precision'], 
               s=100, alpha=0.7, c=colors[:len(model_names)], marker='s', label='Macro')
    for i, model in enumerate(model_names):
        ax7.annotate(model, (comparison_df.loc[model, 'Micro Recall'], 
                            comparison_df.loc[model, 'Micro Precision']),
                    xytext=(5, 5), textcoords='offset points')
    ax7.set_xlabel('Recall')
    ax7.set_ylabel('Precision')
    ax7.set_title('Precision vs Recall')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Average Labels per Text
    ax8 = axes[2, 1]
    avg_labels = [r['metrics']['avg_labels_per_text'] for r in results]
    bars = ax8.bar(model_names, avg_labels, color=['gold', 'orange', 'tomato', 'lightblue'])
    ax8.set_ylabel('Average Labels per Text')
    ax8.set_title('Average Labels Assignment')
    plt.setp(ax8.get_xticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 9: Overall Performance Ranking
    ax9 = axes[2, 2]
    # Create a composite score based on F1, precision, and recall
    composite_scores = []
    for r in results:
        score = (r['metrics']['label_based']['micro_f1'] + 
                r['metrics']['label_based']['micro_precision'] + 
                r['metrics']['label_based']['micro_recall']) / 3
        composite_scores.append(score)
    
    # Sort models by composite score
    sorted_indices = np.argsort(composite_scores)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_scores = [composite_scores[i] for i in sorted_indices]
    
    bars = ax9.bar(range(len(sorted_names)), sorted_scores, 
                   color=['gold', 'silver', '#CD7F32', 'lightgray'][:len(sorted_names)])
    ax9.set_xlabel('Model Ranking')
    ax9.set_ylabel('Composite Score (F1+P+R)/3')
    ax9.set_title('Overall Performance Ranking')
    ax9.set_xticks(range(len(sorted_names)))
    ax9.set_xticklabels(sorted_names, rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = output_dir / f"models_comparison_with_minilm_{timestamp}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved: {viz_path}")
    
    return comparison_df, viz_path

def save_comprehensive_results(results, comparison_df):
    """Save all results and comparison data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual model results
    for result in results:
        model_prefix = result['model_info']['filename_prefix']
        
        # Save optimization results
        opt_df = pd.DataFrame(result['optimization_results'])
        opt_path = output_dir / f"{model_prefix}_optimization_{timestamp}.csv"
        opt_df.to_csv(opt_path, index=False)
        
        # Save model stats
        stats = {
            'model_info': result['model_info'],
            'optimal_thresholds': {
                'primary': result['best_params'][0],
                'secondary': result['best_params'][1]
            },
            'evaluation_metrics': result['metrics'],
            'processing_time_minutes': result['total_time'] / 60,
            'timestamp': timestamp
        }
        
        stats_path = output_dir / f"{model_prefix}_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    # Save comparison summary
    comparison_path = output_dir / f"models_comparison_summary_{timestamp}.csv"
    comparison_df.to_csv(comparison_path)
    print(f"Comparison summary saved: {comparison_path}")
    
    # Save comprehensive comparison report
    comprehensive_stats = {
        'comparison_summary': {
            'timestamp': timestamp,
            'models_compared': len(results),
            'best_performing_model': {
                'name': max(results, key=lambda x: x['metrics']['label_based']['micro_f1'])['model_info']['name'],
                'micro_f1': max(r['metrics']['label_based']['micro_f1'] for r in results)
            }
        },
        'individual_results': []
    }
    
    for result in results:
        comprehensive_stats['individual_results'].append({
            'model_name': result['model_info']['name'],
            'model_id': result['model_info']['model_id'],
            'optimal_thresholds': {
                'primary': result['best_params'][0],
                'secondary': result['best_params'][1]
            },
            'performance_metrics': result['metrics'],
            'processing_time_minutes': result['total_time'] / 60
        })
    
    comprehensive_path = output_dir / f"comprehensive_comparison_{timestamp}.json"
    with open(comprehensive_path, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2, default=str)
    print(f"Comprehensive comparison saved: {comprehensive_path}")
    
    return comparison_path, comprehensive_path

def print_final_comparison_report(results, comparison_df):
    """Print final comparison report."""
    print("\n" + "="*120)
    print("ADVANCED EMBEDDINGS MODELS - FINAL COMPARISON REPORT (INCLUDING MINILM)")
    print("="*120)
    
    # Find best performing model
    best_model = max(results, key=lambda x: x['metrics']['label_based']['micro_f1'])
    
    print(f"🏆 BEST PERFORMING MODEL: {best_model['model_info']['name']}")
    print(f"   Model ID: {best_model['model_info']['model_id']}")
    print(f"   Best Micro F1: {best_model['metrics']['label_based']['micro_f1']:.4f}")
    print(f"   Micro Precision: {best_model['metrics']['label_based']['micro_precision']:.4f}")
    print(f"   Micro Recall: {best_model['metrics']['label_based']['micro_recall']:.4f}")
    print(f"   Optimal Thresholds: Primary={best_model['best_params'][0]}, Secondary={best_model['best_params'][1]}")
    print()
    
    print("DETAILED COMPARISON:")
    print("-" * 120)
    print(f"{'Model':<15} {'Micro F1':<8} {'Micro P':<8} {'Micro R':<8} {'Macro F1':<8} {'Macro P':<8} {'Macro R':<8} {'Coverage':<9} {'Conservation':<12} {'Time (min)':<10}")
    print("-" * 120)
    
    for result in sorted(results, key=lambda x: x['metrics']['label_based']['micro_f1'], reverse=True):
        model_name = result['model_info']['name']
        metrics = result['metrics']
        time_min = result['total_time'] / 60
        
        print(f"{model_name:<15} "
              f"{metrics['label_based']['micro_f1']:<8.4f} "
              f"{metrics['label_based']['micro_precision']:<8.4f} "
              f"{metrics['label_based']['micro_recall']:<8.4f} "
              f"{metrics['label_based']['macro_f1']:<8.4f} "
              f"{metrics['label_based']['macro_precision']:<8.4f} "
              f"{metrics['label_based']['macro_recall']:<8.4f} "
              f"{metrics['coverage_rate']:<9.4f} "
              f"{metrics['conservation_rate']:<12.4f} "
              f"{time_min:<10.1f}")
    
    print("-" * 120)
    print()
    
    # Detailed analysis
    print("DETAILED PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    # Best in each category
    best_micro_f1 = max(results, key=lambda x: x['metrics']['label_based']['micro_f1'])
    best_macro_f1 = max(results, key=lambda x: x['metrics']['label_based']['macro_f1'])
    best_precision = max(results, key=lambda x: x['metrics']['label_based']['micro_precision'])
    best_recall = max(results, key=lambda x: x['metrics']['label_based']['micro_recall'])
    fastest_model = min(results, key=lambda x: x['total_time'])
    best_coverage = max(results, key=lambda x: x['metrics']['coverage_rate'])
    best_conservation = max(results, key=lambda x: x['metrics']['conservation_rate'])
    
    print(f"🎯 Best Micro F1:     {best_micro_f1['model_info']['name']} ({best_micro_f1['metrics']['label_based']['micro_f1']:.4f})")
    print(f"📊 Best Macro F1:     {best_macro_f1['model_info']['name']} ({best_macro_f1['metrics']['label_based']['macro_f1']:.4f})")
    print(f"🎯 Best Precision:    {best_precision['model_info']['name']} ({best_precision['metrics']['label_based']['micro_precision']:.4f})")
    print(f"🎯 Best Recall:       {best_recall['model_info']['name']} ({best_recall['metrics']['label_based']['micro_recall']:.4f})")
    print(f"⚡ Fastest Model:     {fastest_model['model_info']['name']} ({fastest_model['total_time']/60:.1f} min)")
    print(f"📈 Best Coverage:     {best_coverage['model_info']['name']} ({best_coverage['metrics']['coverage_rate']:.4f})")
    print(f"💾 Best Conservation: {best_conservation['model_info']['name']} ({best_conservation['metrics']['conservation_rate']:.4f})")
    print()
    
    # Performance insights
    print("PERFORMANCE INSIGHTS:")
    print("📊 Metrics Explanation:")
    print("   • Micro F1/P/R: Overall performance across all labels (treats each prediction equally)")
    print("   • Macro F1/P/R: Average performance across all SDGs (treats each SDG equally)")
    print("   • Coverage: Percentage of texts that received at least one label")
    print("   • Conservation: Percentage of original labels preserved")
    print()
    
    # Model-specific insights
    print("MODEL-SPECIFIC INSIGHTS:")
    for result in results:
        model_name = result['model_info']['name']
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  • Strengths: ", end="")
        strengths = []
        if metrics['label_based']['micro_precision'] > 0.3:
            strengths.append("High Precision")
        if metrics['label_based']['micro_recall'] > 0.3:
            strengths.append("High Recall")
        if metrics['coverage_rate'] > 0.8:
            strengths.append("Good Coverage")
        if metrics['conservation_rate'] > 0.5:
            strengths.append("Good Conservation")
        if result['total_time'] < 600:  # Less than 10 minutes
            strengths.append("Fast Processing")
        print(", ".join(strengths) if strengths else "Baseline Performance")
        
        print(f"  • Optimal Thresholds: Primary={result['best_params'][0]}, Secondary={result['best_params'][1]}")
        print(f"  • Processing Time: {result['total_time']/60:.1f} minutes")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("🎯 For highest accuracy:", f"Use {best_micro_f1['model_info']['name']} (Micro F1: {best_micro_f1['metrics']['label_based']['micro_f1']:.4f})")
    print("⚡ For fastest processing:", f"Use {fastest_model['model_info']['name']} ({fastest_model['total_time']/60:.1f} min)")
    
    if best_micro_f1 != fastest_model:
        accuracy_diff = best_micro_f1['metrics']['label_based']['micro_f1'] - fastest_model['metrics']['label_based']['micro_f1']
        time_diff = best_micro_f1['total_time'] - fastest_model['total_time']
        print(f"💡 Trade-off: {accuracy_diff:.4f} F1 improvement costs {time_diff/60:.1f} extra minutes")
    
    # Balance recommendation
    balanced_scores = []
    for r in results:
        # Composite score considering F1, processing time, and balance
        f1_score = r['metrics']['label_based']['micro_f1']
        time_penalty = min(1.0, r['total_time'] / 3600)  # Penalty for >1 hour
        balance_score = (r['metrics']['label_based']['micro_precision'] + r['metrics']['label_based']['micro_recall']) / 2
        composite = f1_score * 0.5 + balance_score * 0.3 + (1 - time_penalty) * 0.2
        balanced_scores.append(composite)
    
    best_balanced_idx = np.argmax(balanced_scores)
    best_balanced = results[best_balanced_idx]
    print(f"⚖️ Best balanced choice: {best_balanced['model_info']['name']} (composite score: {balanced_scores[best_balanced_idx]:.4f})")
    
    print("\n" + "="*120)

def main():
    """Main comparison execution."""
    print("🚀 Starting Advanced Embeddings Models Comparison (Including MiniLM)")
    print(f"Models to compare: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model['name']} ({model['model_id']})")
    print()
    
    overall_start_time = time.time()
    
    # Load data once for all models
    osdg_df, sdg_df = load_data()
    sdg_columns = [f'sdg_{i}' for i in range(1, 18)]
    
    # Run experiments for each model
    results = []
    for model_info in MODELS:
        try:
            result = run_single_model_experiment(model_info, osdg_df, sdg_df, sdg_columns)
            results.append(result)
        except Exception as e:
            print(f"❌ Error running {model_info['name']}: {str(e)}")
            continue
    
    if not results:
        print("❌ No models completed successfully!")
        return
    
    # Create visualizations and comparisons
    comparison_df, viz_path = create_comparison_visualizations(results)
    
    # Save all results
    comparison_path, comprehensive_path = save_comprehensive_results(results, comparison_df)
    
    # Print final report
    print_final_comparison_report(results, comparison_df)
    
    total_time = time.time() - overall_start_time
    print(f"\n🎉 Complete comparison finished in {total_time/60:.1f} minutes!")
    print(f"📁 Results saved in: results/")
    print(f"📊 Visualization: {viz_path}")
    print(f"📋 Summary: {comparison_path}")
    print(f"📖 Comprehensive: {comprehensive_path}")
    
    # Print quick summary
    print(f"\n📈 QUICK SUMMARY:")
    best_f1 = max(results, key=lambda x: x['metrics']['label_based']['micro_f1'])
    best_precision = max(results, key=lambda x: x['metrics']['label_based']['micro_precision'])
    best_recall = max(results, key=lambda x: x['metrics']['label_based']['micro_recall'])
    print(f"   Best F1: {best_f1['model_info']['name']} ({best_f1['metrics']['label_based']['micro_f1']:.4f})")
    print(f"   Best Precision: {best_precision['model_info']['name']} ({best_precision['metrics']['label_based']['micro_precision']:.4f})")
    print(f"   Best Recall: {best_recall['model_info']['name']} ({best_recall['metrics']['label_based']['micro_recall']:.4f})")

if __name__ == "__main__":
    main()
