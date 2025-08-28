#!/usr/bin/env python3
"""
Euclidean Distance with Min-Max Scaling Threshold Validation
===========================================================

This script implements threshold validation using Euclidean distance with min-max scaling.

Key Concepts:
1. Euclidean Distance: Measures geometric distance in embedding space (lower = more similar)
2. Min-Max Scaling: Normalizes distances to [0,1] range for stable threshold selection  
3. Threshold Sweeping: Systematically tests multiple threshold combinations

CORRECTED EVALUATION APPROACH:
==============================
The dataset is SINGLE-LABEL classification (each document has exactly 1 SDG), 
but our model makes MULTILABEL predictions (can predict multiple SDGs per document).

This is a "threshold-based multilabel prediction on single-label ground truth" problem.

Key Metrics:
- ACCURACY: % of documents where the true SDG is included in predictions (most important!)
- Sample F1: Average F1 score per document 
- Macro F1: Traditional multilabel F1 (for comparison)

The most meaningful metric is ACCURACY since it answers:
"What percentage of documents got their correct SDG predicted?"

Author: Research Team
Date: August 27, 2025 (Corrected)
Purpose: Validate SDG classification with proper single-label vs multilabel prediction evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class EuclideanMinMaxThresholdValidator:
    """
    Validates SDG classification using Euclidean distance with min-max scaling
    """
    
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2'):
        """Initialize validator"""
        self.data_path = data_path
        self.model_name = model_name
        self.model = None
        self.texts = None
        self.true_labels = None
        self.text_embeddings = None
        self.sdg_embeddings = None
        self.sdg_descriptions = None
        self.raw_distances = None
        self.scaled_distances = None
        self.scaler = MinMaxScaler()
        
        print(f"Initializing Euclidean Distance + Min-Max Scaling Validator")
        print(f"Model: {model_name}")
        print(f"Data path: {data_path}")
    
    def load_data_and_model(self):
        """
        Step 1: Load dataset, SDG descriptions, and embedding model
        
        Why: We need the ground truth data to validate our approach
        """
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA AND MODEL")
        print("="*70)
        
        # Load OSDG dataset
        osdg_path = os.path.join(self.data_path, 'data', 'processed', 'osdg_multilabel_threshold_0.6.csv')
        print(f"Loading OSDG data from: {osdg_path}")
        
        if not os.path.exists(osdg_path):
            raise FileNotFoundError(f"OSDG dataset not found at {osdg_path}")
            
        df = pd.read_csv(osdg_path)
        self.texts = df['text'].tolist()
        
        # Parse true labels (multilabel format)
        label_columns = [col for col in df.columns if col.startswith('sdg_') and col != 'sdg_labels']
        if not label_columns:
            raise ValueError("No SDG label columns found in dataset")
        
        # Convert to numeric (they should already be 0/1 but ensure proper type)
        self.true_labels = df[label_columns].values.astype(int)
        print(f"Loaded {len(self.texts)} texts with {len(label_columns)} SDG labels")
        
        # Load SDG descriptions (using first 17 descriptions)
        sdg_data_path = os.path.join(self.data_path, 'data', 'processed', 'sdg_paragraph_dataset.csv')
        if os.path.exists(sdg_data_path):
            sdg_df = pd.read_csv(sdg_data_path)
            self.sdg_descriptions = sdg_df['text'].tolist()[:17]  # Take first 17
            print(f"Loaded {len(self.sdg_descriptions)} SDG descriptions")
        else:
            # Fallback: create basic SDG descriptions
            self.sdg_descriptions = [
                f"Sustainable Development Goal {i}: End poverty, ensure food security, health, education, "
                f"gender equality, clean water, energy, decent work, innovation, reduce inequalities, "
                f"sustainable cities, responsible consumption, climate action, marine life, terrestrial life, "
                f"peace and justice, partnerships for development."[i*50:(i+1)*50]
                for i in range(17)
            ]
            print("Using fallback SDG descriptions")
        
        # Load embedding model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Data and model loading completed!")
    
    def generate_embeddings(self):
        """
        Step 2: Generate embeddings for texts and SDG descriptions
        
        Why: Convert text to numerical vectors that capture semantic meaning
        """
        print("\n" + "="*70)
        print("STEP 2: GENERATING EMBEDDINGS")
        print("="*70)
        
        print("Generating text embeddings...")
        print(f"Processing all {len(self.texts)} texts (this may take several minutes)...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, batch_size=32)
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
        
        print("Generating SDG embeddings...")
        self.sdg_embeddings = self.model.encode(self.sdg_descriptions, show_progress_bar=True)
        print(f"SDG embeddings shape: {self.sdg_embeddings.shape}")
        
        # Note: Using full dataset - no sampling applied
        
        print("Embedding generation completed!")
    
    def calculate_euclidean_distances(self):
        """
        Step 3: Calculate Euclidean distances between text and SDG embeddings
        
        Why: Euclidean distance measures geometric proximity in embedding space
        Lower distance = higher semantic similarity
        """
        print("\n" + "="*70)
        print("STEP 3: CALCULATING EUCLIDEAN DISTANCES")
        print("="*70)
        
        print("Computing Euclidean distances...")
        self.raw_distances = euclidean_distances(self.text_embeddings, self.sdg_embeddings)
        print(f"Raw distances shape: {self.raw_distances.shape}")
        
        # Show distance statistics
        print(f"Distance statistics:")
        print(f"   Min distance: {self.raw_distances.min():.4f}")
        print(f"   Max distance: {self.raw_distances.max():.4f}")
        print(f"   Mean distance: {self.raw_distances.mean():.4f}")
        print(f"   Std distance: {self.raw_distances.std():.4f}")
        
        print("Euclidean distance calculation completed!")
    
    def apply_minmax_scaling(self):
        """
        Step 4: Apply Min-Max scaling to normalize distances to [0,1] range
        
        Why Min-Max Scaling:
        1. Normalizes distances to interpretable [0,1] range
        2. Makes threshold selection stable across different embedding dimensions
        3. 0 = most similar (minimum distance), 1 = least similar (maximum distance)
        """
        print("\n" + "="*70)
        print("STEP 4: APPLYING MIN-MAX SCALING")
        print("="*70)
        
        print("Applying Min-Max scaling to distance matrix...")
        
        # Reshape for sklearn scaler (needs 2D input)
        distances_reshaped = self.raw_distances.reshape(-1, 1)
        
        # Fit and transform
        distances_scaled_reshaped = self.scaler.fit_transform(distances_reshaped)
        
        # Reshape back to original shape
        self.scaled_distances = distances_scaled_reshaped.reshape(self.raw_distances.shape)
        
        print(f"Scaled distances shape: {self.scaled_distances.shape}")
        
        # Show scaling results
        print(f"After Min-Max scaling:")
        print(f"   Min scaled distance: {self.scaled_distances.min():.4f}")
        print(f"   Max scaled distance: {self.scaled_distances.max():.4f}")
        print(f"   Mean scaled distance: {self.scaled_distances.mean():.4f}")
        print(f"   Std scaled distance: {self.scaled_distances.std():.4f}")
        
        # Show the transformation
        print(f"\nScaling transformation:")
        print(f"   Original range: [{self.raw_distances.min():.4f}, {self.raw_distances.max():.4f}]")
        print(f"   Scaled range: [{self.scaled_distances.min():.4f}, {self.scaled_distances.max():.4f}]")
        
        print("Min-Max scaling completed!")
    
    def evaluate_single_threshold(self, primary_thresh, secondary_thresh=None):
        """
        Step 5: Evaluate performance for a single threshold combination
        
        Args:
            primary_thresh: Primary threshold (lower distance = higher confidence assignments)
            secondary_thresh: Secondary threshold for additional assignments (optional)
            
        Why Two Thresholds:
        - Primary: High-confidence assignments (lower distance threshold)
        - Secondary: Additional potential assignments (higher distance threshold)
        """
        predictions = []
        
        for i in range(len(self.scaled_distances)):
            text_distances = self.scaled_distances[i]
            assigned_labels = []
            
            # Primary assignments (most confident - lowest distances)
            primary_matches = np.where(text_distances <= primary_thresh)[0]
            assigned_labels.extend(primary_matches)
            
            # Secondary assignments (if specified and not already assigned)
            if secondary_thresh is not None and secondary_thresh > primary_thresh:
                secondary_matches = np.where(
                    (text_distances <= secondary_thresh) & 
                    (text_distances > primary_thresh)
                )[0]
                # Limit secondary assignments to avoid over-assignment
                assigned_labels.extend(secondary_matches[:2])
            
            # Convert to binary vector
            binary_pred = np.zeros(len(self.sdg_descriptions))
            for label_idx in assigned_labels:
                binary_pred[label_idx] = 1
                
            predictions.append(binary_pred)
        
        predictions = np.array(predictions)
        
        # CORRECTED EVALUATION FOR SINGLE-LABEL GROUND TRUTH vs MULTILABEL PREDICTIONS
        # Each document has exactly 1 true SDG, but we predict multiple SDGs
        # This is threshold-based multilabel prediction on single-label classification
        
        # 1. Calculate metrics treating this as multilabel problem (macro/micro)
        macro_f1 = f1_score(self.true_labels, predictions, average='macro', zero_division=0)
        macro_precision = precision_score(self.true_labels, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(self.true_labels, predictions, average='macro', zero_division=0)
        
        micro_f1 = f1_score(self.true_labels, predictions, average='micro', zero_division=0)
        micro_precision = precision_score(self.true_labels, predictions, average='micro', zero_division=0)
        micro_recall = recall_score(self.true_labels, predictions, average='micro', zero_division=0)
        
        # 2. Calculate single-label focused metrics
        # For each sample: does our prediction set contain the true label?
        correct_predictions = []
        precision_per_sample = []
        recall_per_sample = []
        f1_per_sample = []
        
        for i in range(len(predictions)):
            # True label (single SDG index)
            true_sdg_idx = np.where(self.true_labels[i] == 1)[0][0]
            
            # Predicted labels (multiple SDG indices)  
            pred_sdg_indices = np.where(predictions[i] == 1)[0]
            
            # Is the true SDG in our predictions?
            is_correct = true_sdg_idx in pred_sdg_indices
            correct_predictions.append(is_correct)
            
            # Sample-wise precision, recall, F1
            if len(pred_sdg_indices) > 0:
                sample_precision = 1.0 if is_correct else 0.0
                precision_per_sample.append(sample_precision)
            else:
                precision_per_sample.append(0.0)
            
            # Recall is always 1.0 if correct, 0.0 if incorrect (single label)
            sample_recall = 1.0 if is_correct else 0.0
            recall_per_sample.append(sample_recall)
            
            # F1 score
            if len(pred_sdg_indices) > 0:
                sample_f1 = sample_precision  # Same as precision for single label
            else:
                sample_f1 = 0.0
            f1_per_sample.append(sample_f1)
        
        # 3. Single-label specific metrics
        accuracy = np.mean(correct_predictions)  # What % of documents got their true SDG predicted?
        avg_sample_precision = np.mean(precision_per_sample)
        avg_sample_recall = np.mean(recall_per_sample)
        avg_sample_f1 = np.mean(f1_per_sample)
        
        # Calculate assignment statistics
        avg_assignments = np.mean([pred.sum() for pred in predictions])
        
        return {
            'primary_threshold': primary_thresh,
            'secondary_threshold': secondary_thresh,
            # Traditional multilabel metrics (treating as multilabel problem)
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            # Single-label focused metrics (more appropriate for this problem)
            'accuracy': accuracy,  # % documents where true SDG was predicted
            'sample_f1': avg_sample_f1,
            'sample_precision': avg_sample_precision,
            'sample_recall': avg_sample_recall,
            # Statistics
            'avg_assignments': avg_assignments,
            'n_predictions': len(predictions),
            'n_samples_with_predictions': len([p for p in predictions if p.sum() > 0]),
            'n_correct_predictions': sum(correct_predictions)
        }
    
    def sweep_thresholds(self, primary_range=None, secondary_range=None, step=0.05):
        """
        Step 6: Systematically sweep through threshold combinations
        
        Args:
            primary_range: Tuple (start, end) for primary thresholds
            secondary_range: Tuple (start, end) for secondary thresholds  
            step: Step size for threshold increments
            
        Why Threshold Sweeping:
        - Finds optimal threshold combinations systematically
        - Avoids manual guessing and selection bias
        - Provides comprehensive performance landscape
        """
        print("\n" + "="*70)
        print("STEP 6: THRESHOLD SWEEPING")
        print("="*70)
        
        # Default ranges based on scaled distances
        if primary_range is None:
            primary_range = (0.1, 0.7)  # Conservative range for primary (low distances)
        if secondary_range is None:
            secondary_range = (0.3, 0.8)  # Extended range for secondary
            
        print(f"Primary threshold range: {primary_range}")
        print(f"Secondary threshold range: {secondary_range}")
        print(f"Step size: {step}")
        
        # Generate threshold combinations
        primary_thresholds = np.arange(primary_range[0], primary_range[1] + step, step)
        secondary_thresholds = np.arange(secondary_range[0], secondary_range[1] + step, step)
        
        print(f"Testing {len(primary_thresholds)} x {len(secondary_thresholds)} = {len(primary_thresholds) * len(secondary_thresholds)} combinations")
        
        results = []
        
        # Test all combinations
        for i, primary_thresh in enumerate(primary_thresholds):
            print(f"Processing primary threshold {i+1}/{len(primary_thresholds)}: {primary_thresh:.3f}")
            
            for secondary_thresh in secondary_thresholds:
                # Only test if secondary > primary (makes sense)
                if secondary_thresh >= primary_thresh:
                    result = self.evaluate_single_threshold(primary_thresh, secondary_thresh)
                    results.append(result)
        
        self.results = pd.DataFrame(results)
        print(f"Completed {len(self.results)} threshold evaluations")
        
        return self.results
    
    def analyze_results(self):
        """
        Step 7: Analyze and visualize threshold sweep results
        
        Why Analysis:
        - Identifies optimal threshold combinations
        - Reveals performance trade-offs
        - Guides final threshold selection
        """
        print("\n" + "="*70)
        print("STEP 7: ANALYZING RESULTS")
        print("="*70)
        
        if not hasattr(self, 'results'):
            print("No results to analyze. Run sweep_thresholds() first.")
            return
            
        # Find best performing combinations for different metrics
        best_accuracy_idx = self.results['accuracy'].idxmax()
        best_accuracy_result = self.results.loc[best_accuracy_idx]
        
        best_macro_f1_idx = self.results['macro_f1'].idxmax()
        best_macro_f1_result = self.results.loc[best_macro_f1_idx]
        
        best_sample_f1_idx = self.results['sample_f1'].idxmax()
        best_sample_f1_result = self.results.loc[best_sample_f1_idx]
        
        print("BEST PERFORMANCE RESULTS (SINGLE-LABEL GROUND TRUTH):")
        print("-" * 70)
        print("🎯 ACCURACY is the most important metric here!")
        print("   (What % of documents got their true SDG predicted?)")
        print()
        print(f"Best Accuracy: {best_accuracy_result['accuracy']:.4f} ({best_accuracy_result['accuracy']*100:.1f}%)")
        print(f"   Primary threshold: {best_accuracy_result['primary_threshold']:.3f}")
        print(f"   Secondary threshold: {best_accuracy_result['secondary_threshold']:.3f}")
        print(f"   Correct predictions: {best_accuracy_result['n_correct_predictions']:.0f}/{best_accuracy_result['n_predictions']:.0f}")
        print(f"   Avg assignments per doc: {best_accuracy_result['avg_assignments']:.2f}")
        print(f"   Sample F1: {best_accuracy_result['sample_f1']:.4f}")
        print()
        
        print("📊 Alternative metrics (for comparison):")
        print(f"Best Macro F1: {best_macro_f1_result['macro_f1']:.4f}")
        print(f"   Accuracy: {best_macro_f1_result['accuracy']:.4f} ({best_macro_f1_result['accuracy']*100:.1f}%)")
        print(f"   Primary/Secondary thresholds: {best_macro_f1_result['primary_threshold']:.3f}/{best_macro_f1_result['secondary_threshold']:.3f}")
        print()
        
        print(f"Best Sample F1: {best_sample_f1_result['sample_f1']:.4f}")
        print(f"   Accuracy: {best_sample_f1_result['accuracy']:.4f} ({best_sample_f1_result['accuracy']*100:.1f}%)")
        print(f"   Primary/Secondary thresholds: {best_sample_f1_result['primary_threshold']:.3f}/{best_sample_f1_result['secondary_threshold']:.3f}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS (SINGLE-LABEL PROBLEM):")
        print("-" * 70)
        print(f"🎯 Accuracy - Mean: {self.results['accuracy'].mean():.4f}, Std: {self.results['accuracy'].std():.4f}")
        print(f"   Best: {self.results['accuracy'].max():.4f}, Worst: {self.results['accuracy'].min():.4f}")
        print()
        print("📊 Other metrics:")
        print(f"Sample F1 - Mean: {self.results['sample_f1'].mean():.4f}, Std: {self.results['sample_f1'].std():.4f}")
        print(f"Macro F1 - Mean: {self.results['macro_f1'].mean():.4f}, Std: {self.results['macro_f1'].std():.4f}")
        print(f"Avg Assignments - Mean: {self.results['avg_assignments'].mean():.2f}, Std: {self.results['avg_assignments'].std():.2f}")
        
        return {
            'best_accuracy': best_accuracy_result,
            'best_macro_f1': best_macro_f1_result,
            'best_sample_f1': best_sample_f1_result,
            'summary_stats': self.results[['accuracy', 'sample_f1', 'macro_f1', 'avg_assignments']].describe()
        }
    
    def create_visualizations(self):
        """
        Step 8: Create visualizations to understand threshold performance
        
        Why Visualizations:
        - Shows performance landscape across threshold combinations  
        - Reveals optimal regions and trade-offs
        - Aids in threshold selection decision
        """
        print("\n" + "="*70)
        print("STEP 8: CREATING VISUALIZATIONS")
        print("="*70)
        
        if not hasattr(self, 'results'):
            print("No results to visualize. Run sweep_thresholds() first.")
            return
            
        # Create pivot tables for heatmaps - focus on accuracy as main metric
        accuracy_pivot = self.results.pivot(
            index='primary_threshold', 
            columns='secondary_threshold', 
            values='accuracy'
        )
        
        sample_f1_pivot = self.results.pivot(
            index='primary_threshold', 
            columns='secondary_threshold', 
            values='sample_f1'
        )
        
        macro_f1_pivot = self.results.pivot(
            index='primary_threshold', 
            columns='secondary_threshold', 
            values='macro_f1'
        )
        
        assignments_pivot = self.results.pivot(
            index='primary_threshold', 
            columns='secondary_threshold', 
            values='avg_assignments'
        )
        
        # Create comprehensive visualization with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Euclidean Distance + Min-Max Scaling: Single-Label Performance Analysis', fontsize=16)
        
        # Accuracy Heatmap (MOST IMPORTANT)
        sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='Greens', ax=axes[0,0])
        axes[0,0].set_title('🎯 ACCURACY by Threshold Combination (MAIN METRIC)')
        axes[0,0].set_xlabel('Secondary Threshold')
        axes[0,0].set_ylabel('Primary Threshold')
        
        # Sample F1 Score Heatmap
        sns.heatmap(sample_f1_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Sample F1 Score by Threshold Combination')
        axes[0,1].set_xlabel('Secondary Threshold')
        axes[0,1].set_ylabel('Primary Threshold')
        
        # Macro F1 Score Heatmap (for comparison)
        sns.heatmap(macro_f1_pivot, annot=True, fmt='.3f', cmap='Oranges', ax=axes[1,0])
        axes[1,0].set_title('Macro F1 Score by Threshold Combination')
        axes[1,0].set_xlabel('Secondary Threshold')
        axes[1,0].set_ylabel('Primary Threshold')
        
        # Average Assignments Heatmap
        sns.heatmap(assignments_pivot, annot=True, fmt='.1f', cmap='Purples', ax=axes[1,1])
        axes[1,1].set_title('Average SDG Assignments per Document')
        axes[1,1].set_xlabel('Secondary Threshold')
        axes[1,1].set_ylabel('Primary Threshold')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.data_path, 'experiments', 'euclidean_minmax_scaling', 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'euclidean_minmax_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """
        Step 9: Save results to files for further analysis
        
        Why Save Results:
        - Preserves experimental results
        - Enables comparison with other methods
        - Supports reproducible research
        """
        print("\n" + "="*70)
        print("STEP 9: SAVING RESULTS")
        print("="*70)
        
        if not hasattr(self, 'results'):
            print("No results to save. Run sweep_thresholds() first.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = os.path.join(self.data_path, 'experiments', 'euclidean_minmax_scaling', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        csv_path = os.path.join(results_dir, f'euclidean_minmax_results_{timestamp}.csv')
        self.results.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Save summary analysis
        analysis = self.analyze_results()
        
        summary = {
            'experiment_info': {
                'method': 'Euclidean Distance + Min-Max Scaling (Single-Label Ground Truth)',
                'model': self.model_name,
                'timestamp': timestamp,
                'n_texts': len(self.texts),
                'n_sdgs': len(self.sdg_descriptions),
                'n_threshold_combinations': len(self.results),
                'evaluation_approach': 'Single-label ground truth vs multilabel predictions',
                'problem_type': 'Threshold-based multilabel prediction on single-label classification'
            },
            'best_performance': {
                'best_accuracy': {
                    'accuracy': float(analysis['best_accuracy']['accuracy']),
                    'accuracy_percentage': float(analysis['best_accuracy']['accuracy'] * 100),
                    'sample_f1': float(analysis['best_accuracy']['sample_f1']),
                    'macro_f1': float(analysis['best_accuracy']['macro_f1']),
                    'primary_threshold': float(analysis['best_accuracy']['primary_threshold']),
                    'secondary_threshold': float(analysis['best_accuracy']['secondary_threshold']),
                    'avg_assignments': float(analysis['best_accuracy']['avg_assignments']),
                    'correct_predictions': int(analysis['best_accuracy']['n_correct_predictions'])
                },
                'best_sample_f1': {
                    'sample_f1': float(analysis['best_sample_f1']['sample_f1']),
                    'accuracy': float(analysis['best_sample_f1']['accuracy']),
                    'macro_f1': float(analysis['best_sample_f1']['macro_f1']),
                    'primary_threshold': float(analysis['best_sample_f1']['primary_threshold']),
                    'secondary_threshold': float(analysis['best_sample_f1']['secondary_threshold']),
                    'avg_assignments': float(analysis['best_sample_f1']['avg_assignments'])
                }
            },
            'scaling_info': {
                'original_distance_range': [float(self.raw_distances.min()), float(self.raw_distances.max())],
                'scaled_distance_range': [float(self.scaled_distances.min()), float(self.scaled_distances.max())],
                'scaler_params': {
                    'feature_range': self.scaler.feature_range,
                    'data_min': float(self.scaler.data_min_[0]),
                    'data_max': float(self.scaler.data_max_[0]),
                    'scale': float(self.scaler.scale_[0])
                }
            }
        }
        
        json_path = os.path.join(results_dir, f'euclidean_minmax_summary_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {json_path}")
        
        return csv_path, json_path
    
    def run_complete_analysis(self):
        """
        Step 10: Run the complete analysis pipeline
        
        Why Complete Pipeline:
        - Executes all steps in correct order
        - Provides comprehensive analysis
        - Ensures reproducible results
        """
        print("\n" + "="*70)
        print("EUCLIDEAN DISTANCE + MIN-MAX SCALING ANALYSIS")
        print("="*70)
        
        try:
            # Execute all steps
            self.load_data_and_model()
            self.generate_embeddings() 
            self.calculate_euclidean_distances()
            self.apply_minmax_scaling()
            
            # Run threshold sweeping
            results = self.sweep_thresholds()
            
            # Analyze results
            analysis = self.analyze_results()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            csv_path, json_path = self.save_results()
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            return {
                'results': results,
                'analysis': analysis,
                'files': {'csv': csv_path, 'json': json_path}
            }
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise

def main():
    """Main execution function"""
    # Initialize validator
    data_path = "/Users/mahnoorzamir/Desktop/mitacs/project"
    validator = EuclideanMinMaxThresholdValidator(data_path)
    
    # Run complete analysis
    results = validator.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()
