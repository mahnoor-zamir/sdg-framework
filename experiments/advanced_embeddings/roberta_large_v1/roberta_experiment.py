#!/usr/bin/env python3
"""
Advanced Embedding Models Experiment: all-roberta-large-v1
=========================================================

Tests the performance of sentence-transformers/all-roberta-large-v1 (1024D)
vs baseline all-MiniLM-L6-v2 (384D) for SDG text classification.

Model Details:
- all-roberta-large-v1: 1024 dimensions, RoBERTa-based sentence transformer
- Larger model with potentially better semantic understanding
- Expected improvement: 8-18% F1-score increase
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
import os
import sys

class RoBERTaSDGClassifier:
    def __init__(self):
        print("="*60)
        print("ADVANCED EMBEDDING EXPERIMENT: ALL-ROBERTA-LARGE-V1")
        print("="*60)
        
        self.model_name = "sentence-transformers/all-roberta-large-v1"
        self.baseline_model = "all-MiniLM-L6-v2"
        
        # Load models
        self.load_models()
        
        # Load test data
        self.load_test_data()
        
        # Load SDG descriptions
        self.load_sdg_descriptions()
        
        # Configuration for global thresholds (winning approach)
        self.primary_threshold = 0.4
        self.secondary_threshold = 0.3
        
    def load_models(self):
        """Load both RoBERTa and baseline models for comparison."""
        print("Loading embedding models...")
        print(f"Target model: {self.model_name} (1024D)")
        print(f"Baseline model: {self.baseline_model} (384D)")
        
        # Load RoBERTa model
        print("Loading RoBERTa model (this may take a while for first download)...")
        self.roberta_model = SentenceTransformer(self.model_name)
        print(f"RoBERTa model loaded: {self.roberta_model.get_sentence_embedding_dimension()}D embeddings")
        
        # Load baseline model for comparison
        print("Loading baseline model...")
        self.baseline_model_obj = SentenceTransformer(self.baseline_model)
        print(f"Baseline model loaded: {self.baseline_model_obj.get_sentence_embedding_dimension()}D embeddings")
        
    def load_test_data(self):
        """Load the OSDG test dataset."""
        print("Loading test dataset...")
        
        data_path = '/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/osdg_multilabel_threshold_0.6.json'
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            self.data = full_data['data']
            
        print(f"Loaded {len(self.data)} texts for classification")
        
    def load_sdg_descriptions(self):
        """Load SDG descriptions from the paragraph dataset."""
        print("Loading SDG descriptions...")
        
        # Load from the consistent SDG paragraph dataset
        sdg_path = '/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/sdg_paragraph_dataset.csv'
        sdg_df = pd.read_csv(sdg_path)
        
        # Create SDG descriptions dictionary
        self.sdg_descriptions = {}
        for _, row in sdg_df.iterrows():
            sdg_num = int(row['sdg'])
            self.sdg_descriptions[sdg_num] = row['text']
            
        print(f"Loaded {len(self.sdg_descriptions)} SDG descriptions")
        
    def create_embeddings(self, model, texts, batch_size=16):
        """Create embeddings for texts using specified model. Smaller batch for large model."""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            if i // batch_size % 20 == 0:
                print(f"  Processed batch {i//batch_size + 1}/{total_batches}")
                
        return np.array(embeddings)
    
    def calculate_similarities(self, text_embeddings, sdg_embeddings):
        """Calculate cosine similarities between text and SDG embeddings."""
        # Normalize embeddings for cosine similarity
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        sdg_norm = sdg_embeddings / np.linalg.norm(sdg_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(text_norm, sdg_norm.T)
        return similarities
    
    def apply_global_thresholds(self, similarities):
        """Apply global thresholds to similarity matrix."""
        predictions = []
        assignment_stats = {'primary_only': 0, 'primary_secondary': 0, 'zero_labels': 0}
        
        for sim_row in similarities:
            # Apply thresholds
            primary_matches = sim_row >= self.primary_threshold
            secondary_matches = (sim_row >= self.secondary_threshold) & (sim_row < self.primary_threshold)
            
            assigned_sdgs = []
            
            # Add primary matches
            primary_sdgs = np.where(primary_matches)[0] + 1  # SDGs are 1-indexed
            assigned_sdgs.extend(primary_sdgs)
            
            # Add secondary matches
            secondary_sdgs = np.where(secondary_matches)[0] + 1
            assigned_sdgs.extend(secondary_sdgs)
            
            # Track assignment statistics
            if len(primary_sdgs) > 0 and len(secondary_sdgs) == 0:
                assignment_stats['primary_only'] += 1
            elif len(primary_sdgs) > 0 or len(secondary_sdgs) > 0:
                assignment_stats['primary_secondary'] += 1
            else:
                assignment_stats['zero_labels'] += 1
                
            predictions.append(assigned_sdgs)
            
        return predictions, assignment_stats
    
    def evaluate_performance(self, predictions, true_labels, model_name):
        """Evaluate classification performance."""
        print(f"\nEvaluating {model_name} performance...")
        
        # Convert to multilabel format
        y_true = np.zeros((len(true_labels), 17))
        y_pred = np.zeros((len(predictions), 17))
        
        for i, labels in enumerate(true_labels):
            for sdg in labels:
                if 1 <= sdg <= 17:
                    y_true[i, sdg-1] = 1
                    
        for i, labels in enumerate(predictions):
            for sdg in labels:
                if 1 <= sdg <= 17:
                    y_pred[i, sdg-1] = 1
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        micro_f1 = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)[2]
        macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]
        
        # Coverage and efficiency metrics
        coverage = np.mean(np.sum(y_pred, axis=1) > 0)
        avg_labels = np.mean(np.sum(y_pred, axis=1))
        
        results = {
            'model': model_name,
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1),
            'coverage': float(coverage),
            'avg_labels_per_text': float(avg_labels),
            'total_texts': len(predictions)
        }
        
        return results
    
    def run_experiment(self):
        """Run the complete RoBERTa vs baseline comparison experiment."""
        
        # Extract texts and true labels
        texts = [item['text'] for item in self.data]
        true_labels = [item['sdg_labels'] for item in self.data]
        
        print(f"\nStarting experiment with {len(texts)} texts...")
        
        results = {}
        
        # Test RoBERTa model
        print(f"\n{'='*60}")
        print(f"TESTING: ALL-ROBERTA-LARGE-V1")
        print(f"{'='*60}")
        
        # Create embeddings (smaller batches for large model)
        print("Creating text embeddings...")
        roberta_text_embeddings = self.create_embeddings(self.roberta_model, texts, batch_size=16)
        
        print("Creating SDG embeddings...")
        sdg_texts = [self.sdg_descriptions[i] for i in range(1, 18)]
        roberta_sdg_embeddings = self.create_embeddings(self.roberta_model, sdg_texts, batch_size=17)
        
        # Calculate similarities
        print("Calculating similarities...")
        roberta_similarities = self.calculate_similarities(roberta_text_embeddings, roberta_sdg_embeddings)
        
        # Apply thresholds
        print("Applying global thresholds...")
        roberta_predictions, roberta_stats = self.apply_global_thresholds(roberta_similarities)
        
        # Evaluate
        roberta_results = self.evaluate_performance(roberta_predictions, true_labels, "all-roberta-large-v1")
        roberta_results['assignment_stats'] = roberta_stats
        results['roberta'] = roberta_results
        
        # Test baseline model
        print(f"\n{'='*60}")
        print(f"TESTING: {self.baseline_model.upper()} (BASELINE)")
        print(f"{'='*60}")
        
        # Create embeddings
        print("Creating text embeddings...")
        baseline_text_embeddings = self.create_embeddings(self.baseline_model_obj, texts, batch_size=32)
        
        print("Creating SDG embeddings...")
        baseline_sdg_embeddings = self.create_embeddings(self.baseline_model_obj, sdg_texts, batch_size=17)
        
        # Calculate similarities
        print("Calculating similarities...")
        baseline_similarities = self.calculate_similarities(baseline_text_embeddings, baseline_sdg_embeddings)
        
        # Apply thresholds
        print("Applying global thresholds...")
        baseline_predictions, baseline_stats = self.apply_global_thresholds(baseline_similarities)
        
        # Evaluate
        baseline_results = self.evaluate_performance(baseline_predictions, true_labels, self.baseline_model)
        baseline_results['assignment_stats'] = baseline_stats
        results['baseline'] = baseline_results
        
        # Compare results
        self.compare_results(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def compare_results(self, results):
        """Compare RoBERTa vs baseline results."""
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        
        roberta = results['roberta']
        baseline = results['baseline']
        
        print(f"\nMetric                    RoBERTa-Large-v1     MiniLM-L6-v2         Improvement")
        print("-" * 80)
        
        metrics = ['f1_score', 'precision', 'recall', 'coverage', 'avg_labels_per_text']
        for metric in metrics:
            roberta_val = roberta[metric]
            baseline_val = baseline[metric]
            improvement = ((roberta_val - baseline_val) / baseline_val) * 100
            
            print(f"{metric:<25} {roberta_val:<20.4f} {baseline_val:<20.4f} {improvement:+6.1f}%")
        
        # Overall winner
        f1_improvement = ((roberta['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
        winner = "RoBERTa" if roberta['f1_score'] > baseline['f1_score'] else "Baseline"
        
        print(f"\nWINNER: {winner} (F1 improvement: {f1_improvement:+.1f}%)")
        
    def save_results(self, results):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results/roberta_experiment_results_{timestamp}.json'
        
        # Add experiment metadata
        experiment_data = {
            'experiment': 'advanced_embeddings_roberta',
            'timestamp': timestamp,
            'configuration': {
                'target_model': self.model_name,
                'baseline_model': self.baseline_model,
                'primary_threshold': self.primary_threshold,
                'secondary_threshold': self.secondary_threshold,
                'dataset_size': len(self.data)
            },
            'results': results
        }
        
        with open(results_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        return results_path

def main():
    """Run the RoBERTa embedding experiment."""
    classifier = RoBERTaSDGClassifier()
    results = classifier.run_experiment()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
