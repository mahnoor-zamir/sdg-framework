#!/usr/bin/env python3
"""
Summarized SDG Classification Experiment
=======================================

This script tests classification performance using summarized SDG descriptions
vs original descriptions with the same cosine similarity approach.

Comparison:
- Original SDG descriptions + Cosine Global (baseline from previous experiments)  
- Summarized SDG descriptions + Cosine Global (new approach)

Author: Experimental Framework
Date: August 22, 2025
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

class SummarizedSDGClassifier:
    def __init__(self, summarized_sdg_path=None):
        """Initialize classifier with summarized SDG descriptions."""
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load test data
        self.load_test_data()
        
        # Load SDG descriptions (original and summarized)
        self.load_sdg_descriptions(summarized_sdg_path)
        
    def load_test_data(self):
        """Load the test dataset."""
        print("Loading test dataset...")
        
        # Load the OSDG multilabel data (use CSV format like winning experiment)
        data_path = '/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/osdg_multilabel_threshold_0.6.csv'
        import pandas as pd
        osdg_df = pd.read_csv(data_path)
        
        # Convert to list format for compatibility with existing code
        self.data = []
        for _, row in osdg_df.iterrows():
            # Extract SDG labels from binary columns (same as Experiment 1)
            sdg_labels = []
            for i in range(1, 18):
                if row[f'sdg_{i}'] == 1:
                    sdg_labels.append(i)
            
            self.data.append({
                'text': row['text'],
                'sdg_labels': sdg_labels
            })
            
        # Use complete dataset  
        print(f"Using complete dataset: {len(self.data)} texts for classification")
    
    def load_sdg_descriptions(self, summarized_path=None):
        """Load both original and summarized SDG descriptions."""
        print("Loading SDG descriptions...")
        
        # Load original descriptions
        original_df = pd.read_csv('/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/sdg_paragraph_dataset.csv')
        self.original_sdgs = {}
        for _, row in original_df.iterrows():
            self.original_sdgs[row['sdg']] = row['text']
        
        # Load summarized descriptions if provided
        if summarized_path and os.path.exists(summarized_path):
            with open(summarized_path, 'r') as f:
                summary_data = json.load(f)
                self.summarized_sdgs = summary_data['summarized_sdgs']
                # Convert string keys to integers
                self.summarized_sdgs = {int(k): v for k, v in self.summarized_sdgs.items()}
        else:
            print("No summarized descriptions provided - will generate them")
            self.summarized_sdgs = None
        
        print(f"Loaded original SDG descriptions: {len(self.original_sdgs)}")
        if self.summarized_sdgs:
            print(f"Loaded summarized SDG descriptions: {len(self.summarized_sdgs)}")
    
    def create_embeddings(self):
        """Create embeddings for both original and summarized SDG descriptions."""
        print("Creating SDG embeddings...")
        
        # Original SDG embeddings
        original_texts = [self.original_sdgs[i] for i in range(1, 18)]
        self.original_sdg_embeddings = self.model.encode(original_texts)
        print(f"Original SDG embeddings shape: {self.original_sdg_embeddings.shape}")
        
        # Summarized SDG embeddings (if available)
        if self.summarized_sdgs:
            summarized_texts = [self.summarized_sdgs[i] for i in range(1, 18)]
            self.summarized_sdg_embeddings = self.model.encode(summarized_texts)
            print(f"Summarized SDG embeddings shape: {self.summarized_sdg_embeddings.shape}")
        else:
            self.summarized_sdg_embeddings = None
    
    def classify_with_cosine_global(self, sdg_embeddings, approach_name):
        """Classify texts using cosine similarity with global thresholds."""
        print(f"\nRunning classification with {approach_name}...")
        
        # Global thresholds (same as Experiment 1)
        PRIMARY_THRESHOLD = 0.4
        SECONDARY_THRESHOLD = 0.3
        MAX_LABELS_PER_TEXT = 5  # Same as Experiment 1
        
        predictions = []
        true_labels = []
        
        # Create text embeddings in batches for efficiency
        texts = [item['text'] for item in self.data]
        print("Creating text embeddings...")
        text_embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        
        print("Running classification...")
        
        # Calculate similarity matrix using sklearn (same as Experiment 1)
        similarity_matrix = cosine_similarity(text_embeddings, sdg_embeddings)
        
        for i, item in enumerate(self.data):
            true_label = item['sdg_labels']  # Use 'sdg_labels' field
            similarities = similarity_matrix[i]
            
            # Apply global thresholds (exact same logic as Experiment 1)
            text_labels = []
            
            # Primary threshold assignments
            primary_indices = np.where(similarities >= PRIMARY_THRESHOLD)[0]
            text_labels.extend(primary_indices.tolist())
            
            # Secondary threshold assignments (if not already assigned AND under max_labels)
            if len(text_labels) < MAX_LABELS_PER_TEXT:
                secondary_indices = np.where(
                    (similarities >= SECONDARY_THRESHOLD) & 
                    (similarities < PRIMARY_THRESHOLD)
                )[0]
                
                # Add secondary up to max_labels limit
                remaining_slots = MAX_LABELS_PER_TEXT - len(text_labels)
                secondary_to_add = secondary_indices[:remaining_slots]
                text_labels.extend(secondary_to_add.tolist())
            
            # Convert to SDG numbers (1-indexed, same as Experiment 1)
            predicted_labels = [idx + 1 for idx in text_labels[:MAX_LABELS_PER_TEXT]]
            
            predictions.append(predicted_labels)
            true_labels.append(true_label)
        
        # Calculate metrics
        results = self.calculate_metrics(predictions, true_labels, approach_name)
        return results
    
    def calculate_metrics(self, predictions, true_labels, approach_name):
        """Calculate classification metrics using sample-based evaluation (like Experiment 1)."""
        print(f"Calculating metrics for {approach_name}...")
        
        # Create binary matrices for SDGs 1-17
        n_texts = len(true_labels)
        n_sdgs = 17
        
        y_true_binary = np.zeros((n_texts, n_sdgs))
        y_pred_binary = np.zeros((n_texts, n_sdgs))
        
        # Fill true labels matrix
        for i, labels in enumerate(true_labels):
            for label in labels:
                if 1 <= label <= 17:
                    y_true_binary[i][label - 1] = 1
        
        # Fill predictions matrix
        for i, labels in enumerate(predictions):
            for label in labels:
                if 1 <= label <= 17:
                    y_pred_binary[i][label - 1] = 1
        
        # Sample-based metrics (same as Experiment 1)
        sample_precision = []
        sample_recall = []
        sample_f1 = []
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        for i in range(n_texts):
            if y_pred_binary[i].sum() > 0:  # Only if predictions exist (same as Experiment 1)
                prec = precision_score(y_true_binary[i], y_pred_binary[i], average='binary', zero_division=0)
                rec = recall_score(y_true_binary[i], y_pred_binary[i], average='binary', zero_division=0)
                f1 = f1_score(y_true_binary[i], y_pred_binary[i], average='binary', zero_division=0)
                
                sample_precision.append(prec)
                sample_recall.append(rec)
                sample_f1.append(f1)
        
        # Average sample-based metrics (same as Experiment 1)
        precision = np.mean(sample_precision) if sample_precision else 0
        recall = np.mean(sample_recall) if sample_recall else 0
        f1 = np.mean(sample_f1) if sample_f1 else 0
        
        # Calculate additional metrics
        total_predictions = sum(len(pred) for pred in predictions)
        total_texts = len(predictions)
        avg_labels_per_text = total_predictions / total_texts if total_texts > 0 else 0
        
        # Coverage (texts with at least one prediction)
        texts_with_predictions = sum(1 for pred in predictions if len(pred) > 0)
        coverage = texts_with_predictions / total_texts if total_texts > 0 else 0
        
        results = {
            'approach': approach_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'coverage': coverage,
            'avg_labels_per_text': avg_labels_per_text,
            'total_texts': total_texts,
            'total_predictions': total_predictions
        }
        
        return results
    
    def run_comparison_experiment(self):
        """Run comparison between original and summarized SDG descriptions."""
        print("\n" + "="*60)
        print("SUMMARIZED SDG CLASSIFICATION EXPERIMENT")
        print("="*60)
        
        # Create embeddings
        self.create_embeddings()
        
        results = {}
        
        # Test with original SDG descriptions
        print("\nTesting with ORIGINAL SDG descriptions...")
        results['original'] = self.classify_with_cosine_global(
            self.original_sdg_embeddings, "Original SDG Descriptions"
        )
        
        # Test with summarized SDG descriptions (if available)
        if self.summarized_sdg_embeddings is not None:
            print("\nTesting with SUMMARIZED SDG descriptions...")
            results['summarized'] = self.classify_with_cosine_global(
                self.summarized_sdg_embeddings, "Summarized SDG Descriptions"
            )
        else:
            print("\nSkipping summarized test - no summarized descriptions available")
            results['summarized'] = None
        
        return results
    
    def display_results(self, results):
        """Display comparison results."""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS COMPARISON")
        print("="*60)
        
        if results['summarized'] is None:
            print("Only original results available")
            self.print_result_table([results['original']])
            return
        
        # Compare results
        orig = results['original']
        summ = results['summarized']
        
        print(f"\n{'Metric':<25} {'Original':<12} {'Summarized':<12} {'Improvement':<12}")
        print("-" * 65)
        print(f"{'F1-Score':<25} {orig['f1_score']:<12.4f} {summ['f1_score']:<12.4f} {((summ['f1_score'] - orig['f1_score']) / orig['f1_score'] * 100):>+10.1f}%")
        print(f"{'Precision':<25} {orig['precision']:<12.4f} {summ['precision']:<12.4f} {((summ['precision'] - orig['precision']) / orig['precision'] * 100):>+10.1f}%")
        print(f"{'Recall':<25} {orig['recall']:<12.4f} {summ['recall']:<12.4f} {((summ['recall'] - orig['recall']) / orig['recall'] * 100):>+10.1f}%")
        print(f"{'Coverage':<25} {orig['coverage']:<12.4f} {summ['coverage']:<12.4f} {((summ['coverage'] - orig['coverage']) / orig['coverage'] * 100):>+10.1f}%")
        print(f"{'Avg Labels/Text':<25} {orig['avg_labels_per_text']:<12.2f} {summ['avg_labels_per_text']:<12.2f} {((summ['avg_labels_per_text'] - orig['avg_labels_per_text']) / orig['avg_labels_per_text'] * 100):>+10.1f}%")
        
        # Determine winner
        if summ['f1_score'] > orig['f1_score']:
            winner = "SUMMARIZED"
            improvement = (summ['f1_score'] - orig['f1_score']) / orig['f1_score'] * 100
        else:
            winner = "ORIGINAL"
            improvement = (orig['f1_score'] - summ['f1_score']) / summ['f1_score'] * 100
        
        print(f"\nWINNER: {winner} (F1 improvement: +{improvement:.1f}%)")
        
        return results
    
    def save_results(self, results):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_path = f'/Users/mahnoorzamir/Desktop/mitacs/project/experiments/summarized_sdg_minilm/results/summarized_experiment_results_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        return results_path

def main():
    """Main execution function."""
    print("="*60)
    print("SUMMARIZED SDG CLASSIFICATION EXPERIMENT")
    print("="*60)
    
    # Check if we have summarized descriptions already in the results folder
    results_dir = '/Users/mahnoorzamir/Desktop/mitacs/project/experiments/summarized_sdg_minilm/results'
    summary_files = []
    
    if os.path.exists(results_dir):
        summary_files = [f for f in os.listdir(results_dir) 
                        if f.startswith('sdg_summaries_') and f.endswith('.json')]
    
    summarized_path = None
    if summary_files:
        # Use the most recent summary file
        latest_summary = sorted(summary_files)[-1]
        summarized_path = f'{results_dir}/{latest_summary}'
        print(f"Found existing summaries: {latest_summary}")
    else:
        print("No existing summaries found. Run sdg_summarizer.py first.")
    
    # Initialize classifier
    classifier = SummarizedSDGClassifier(summarized_path)
    
    # Run comparison experiment
    results = classifier.run_comparison_experiment()
    
    # Display and save results
    classifier.display_results(results)
    results_path = classifier.save_results(results)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
