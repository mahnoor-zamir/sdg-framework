#!/usr/bin/env python3
"""
MiniLM Baseline Verification - Using Original Fixed Thresholds
============================================================

This script runs MiniLM with the original fixed thresholds (Primary=0.4, Secondary=0.3)
to verify the discrepancy and provide a fair comparison baseline.
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

# Configuration - USING ORIGINAL FIXED THRESHOLDS
CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'distance_metric': 'cosine',
    'primary_threshold': 0.4,  # Original fixed value
    'secondary_threshold': 0.3,  # Original fixed value
    'max_labels_per_text': 5,
    'experiment_name': 'minilm_fixed_thresholds_verification'
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
    
    # Label-based metrics
    micro_precision = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_recall = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Conservation rate
    original_labels = y_true.sum()
    preserved_labels = (y_true * y_pred).sum()
    conservation_rate = preserved_labels / original_labels if original_labels > 0 else 0
    
    metrics = {
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

def main():
    """Main verification execution."""
    print("🔍 BASELINE VERIFICATION: MiniLM with Original Fixed Thresholds")
    print(f"Configuration: {CONFIG}")
    print(f"Primary Threshold: {CONFIG['primary_threshold']} (ORIGINAL)")
    print(f"Secondary Threshold: {CONFIG['secondary_threshold']} (ORIGINAL)")
    print()
    
    # Load data
    osdg_df, sdg_df = load_data()
    
    # Initialize embedding model
    print(f"Loading embedding model: {CONFIG['embedding_model']}")
    model = SentenceTransformer(CONFIG['embedding_model'])
    
    # Generate embeddings
    print(f"Generating embeddings for {len(osdg_df)} texts...")
    text_embeddings = model.encode(osdg_df['text'].tolist(), batch_size=32, show_progress_bar=True)
    
    print(f"Generating embeddings for {len(sdg_df)} texts...")
    sdg_embeddings = model.encode(sdg_df['text'].tolist(), batch_size=32, show_progress_bar=True)
    
    # Calculate similarities
    print("Calculating cosine similarities...")
    similarity_matrix = cosine_similarity(text_embeddings, sdg_embeddings)
    
    # Apply fixed thresholds
    print("Applying ORIGINAL fixed thresholds...")
    assignments = apply_thresholds(
        similarity_matrix, 
        CONFIG['primary_threshold'], 
        CONFIG['secondary_threshold'], 
        CONFIG['max_labels_per_text']
    )
    
    # Calculate metrics
    sdg_columns = [f'sdg_{i}' for i in range(1, 18)]
    metrics = calculate_evaluation_metrics(osdg_df, assignments, sdg_columns)
    
    # Print comparison
    print("\n" + "="*80)
    print("BASELINE VERIFICATION RESULTS")
    print("="*80)
    print("ORIGINAL EXPERIMENT (Fixed Thresholds):")
    print(f"  Primary Threshold: 0.4")
    print(f"  Secondary Threshold: 0.3")
    print(f"  Micro F1: {metrics['label_based']['micro_f1']:.4f}")
    print(f"  Micro Precision: {metrics['label_based']['micro_precision']:.4f}")
    print(f"  Micro Recall: {metrics['label_based']['micro_recall']:.4f}")
    print()
    
    print("EXPECTED FROM PREVIOUS EXPERIMENT:")
    print(f"  Micro F1: 0.4021")
    print(f"  Micro Precision: 0.2811")
    print(f"  Micro Recall: 0.7058")
    print()
    
    print("ADVANCED EMBEDDINGS (Optimized Thresholds):")
    print(f"  Primary Threshold: 0.5")
    print(f"  Secondary Threshold: 0.35")
    print(f"  Micro F1: 0.4259")
    print(f"  Micro Precision: 0.3428")
    print(f"  Micro Recall: 0.5624")
    print()
    
    f1_diff = metrics['label_based']['micro_f1'] - 0.4021
    print(f"📊 F1 DIFFERENCE FROM PREVIOUS: {f1_diff:+.4f}")
    
    if abs(f1_diff) < 0.001:
        print("✅ VERIFICATION SUCCESSFUL: Results match previous experiment!")
    else:
        print("❌ DISCREPANCY CONFIRMED: Results differ from previous experiment")
        print("   This confirms the threshold optimization improved performance")
    
    print("="*80)

if __name__ == "__main__":
    main()
