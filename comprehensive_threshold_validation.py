#!/usr/bin/env python3
"""
Comprehensive Threshold Robustness Validation
============================================

This script implements systematic validation to address threshold selection bias by testing performance across multiple threshold configurations.

Author: Research Team  
Date: August 21, 2025
Purpose: Validate SDG classification results with statistical rigor
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
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ComprehensiveThresholdValidator:
    """
    Validates model performance across multiple threshold configurations
    to address methodological concerns about threshold selection bias.
    """
    
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2'):
        """Initialize validator with dataset and model"""
        self.data_path = data_path
        self.model_name = model_name
        self.model = None
        self.texts = None
        self.true_labels = None
        self.text_embeddings = None
        self.sdg_embeddings = None
        self.sdg_descriptions = None
        
        print(f"Initializing Comprehensive Threshold Validator")
        print(f"Model: {model_name}")
        print(f"Data path: {data_path}")
    
    def load_data_and_model(self):
        """Load dataset, SDG descriptions, and embedding model"""
        print("\n" + "="*60)
        print("LOADING DATA AND MODEL")
        print("="*60)
        
        # Load OSDG dataset
        osdg_path = os.path.join(self.data_path, 'data', 'processed', 'osdg_multilabel_threshold_0.6.csv')
        print(f"Loading OSDG data from: {osdg_path}")
        
        if not os.path.exists(osdg_path):
            raise FileNotFoundError(f"OSDG dataset not found at {osdg_path}")
            
        df = pd.read_csv(osdg_path)
        self.texts = df['text'].tolist()
        
        # Parse true labels (assuming multilabel format)
        label_columns = [col for col in df.columns if col.startswith('sdg_')]
        if not label_columns:
            # Alternative: look for labels column
            if 'labels' in df.columns:
                self.true_labels = df['labels'].apply(eval).tolist()
            else:
                raise ValueError("No label columns found in dataset")
        else:
            self.true_labels = df[label_columns].values.tolist()
        
        print(f"Loaded {len(self.texts)} texts with {len(label_columns) if label_columns else 'multilabel'} labels")
        
        # Load SDG descriptions
        sdg_data_path = os.path.join(self.data_path, 'data', 'processed', 'sdg_paragraph_dataset.csv')
        if os.path.exists(sdg_data_path):
            sdg_df = pd.read_csv(sdg_data_path)
            self.sdg_descriptions = sdg_df['text'].tolist()
            print(f"Loaded {len(self.sdg_descriptions)} SDG descriptions")
        else:
            # Fallback: create basic SDG descriptions
            self.sdg_descriptions = [
                f"Sustainable Development Goal {i}: SDG-{i} related content"
                for i in range(1, 18)
            ]
            print("Using fallback SDG descriptions")
        
        # Load embedding model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        print("Data and model loading completed!")
    
    def generate_embeddings(self):
        """Generate embeddings for texts and SDG descriptions"""
        print("\n" + "="*60)
        print("GENERATING EMBEDDINGS")
        print("="*60)
        
        print("Generating text embeddings...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, batch_size=32)
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
        
        print("Generating SDG embeddings...")
        self.sdg_embeddings = self.model.encode(self.sdg_descriptions, show_progress_bar=True)
        print(f"SDG embeddings shape: {self.sdg_embeddings.shape}")
        
        print("Embedding generation completed!")
    
    def calculate_similarities(self, distance_metric='cosine'):
        """Calculate similarity/distance matrix between texts and SDGs"""
        if distance_metric == 'cosine':
            similarities = cosine_similarity(self.text_embeddings, self.sdg_embeddings)
            return similarities
        elif distance_metric == 'euclidean':
            distances = euclidean_distances(self.text_embeddings, self.sdg_embeddings)
            return distances
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    def evaluate_threshold_combination(self, primary_thresh, secondary_thresh, 
                                     similarities, distance_metric='cosine'):
        """Evaluate performance for specific threshold combination"""
        predictions = []
        
        for i in range(len(similarities)):
            text_similarities = similarities[i]
            assigned_labels = []
            
            if distance_metric == 'cosine':
                # Higher similarity = better match
                primary_matches = np.where(text_similarities >= primary_thresh)[0]
                secondary_matches = np.where(text_similarities >= secondary_thresh)[0]
                
                # Remove primary matches from secondary
                secondary_only = [idx for idx in secondary_matches if idx not in primary_matches]
                
                # Assign labels (primary weight = 1.0, secondary weight = 0.5)
                for idx in primary_matches:
                    assigned_labels.append(idx)
                for idx in secondary_only[:2]:  # Limit secondary assignments
                    assigned_labels.append(idx)
                    
            else:  # euclidean
                # Lower distance = better match
                primary_matches = np.where(text_similarities <= primary_thresh)[0]
                secondary_matches = np.where(text_similarities <= secondary_thresh)[0]
                
                # Remove primary matches from secondary
                secondary_only = [idx for idx in secondary_matches if idx not in primary_matches]
                
                # Assign labels
                for idx in primary_matches:
                    assigned_labels.append(idx)
                for idx in secondary_only[:2]:  # Limit secondary assignments
                    assigned_labels.append(idx)
            
            # Convert to binary vector
            binary_pred = np.zeros(len(self.sdg_descriptions))
            for label_idx in assigned_labels:
                binary_pred[label_idx] = 1
                
            predictions.append(binary_pred)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        # Convert true labels to binary matrix if needed
        if isinstance(self.true_labels[0], list):
            true_binary = np.zeros((len(self.true_labels), len(self.sdg_descriptions)))
            for i, labels in enumerate(self.true_labels):
                for label in labels:
                    if isinstance(label, int) and 0 <= label < len(self.sdg_descriptions):
                        true_binary[i, label] = 1
        else:
            true_binary = np.array(self.true_labels)
        
        # Calculate sample-based metrics
        f1_samples = []
        precision_samples = []
        recall_samples = []
        
        for i in range(len(predictions)):
            if predictions[i].sum() > 0:  # Avoid division by zero
                p = precision_score(true_binary[i], predictions[i], average='binary', zero_division=0)
                r = recall_score(true_binary[i], predictions[i], average='binary', zero_division=0)
                f1 = f1_score(true_binary[i], predictions[i], average='binary', zero_division=0)
                
                precision_samples.append(p)
                recall_samples.append(r)
                f1_samples.append(f1)
            else:
                precision_samples.append(0.0)
                recall_samples.append(0.0)  
                f1_samples.append(0.0)
        
        # Calculate overall metrics
        f1_mean = np.mean(f1_samples)
        precision_mean = np.mean(precision_samples)
        recall_mean = np.mean(recall_samples)
        
        # Calculate coverage
        coverage = np.mean(predictions.sum(axis=1) > 0)
        
        # Calculate average labels per text
        avg_labels = np.mean(predictions.sum(axis=1))
        
        return {
            'f1_score': f1_mean,
            'precision': precision_mean,
            'recall': recall_mean,
            'coverage': coverage,
            'avg_labels': avg_labels
        }
    
    def grid_search_validation(self, distance_metric='cosine', n_combinations=50):
        """Test multiple threshold combinations systematically"""
        print(f"\n{'='*60}")
        print(f"GRID SEARCH VALIDATION - {distance_metric.upper()}")
        print(f"{'='*60}")
        
        # Calculate similarities/distances
        similarities = self.calculate_similarities(distance_metric)
        
        # Define threshold ranges based on distance metric
        if distance_metric == 'cosine':
            # Test cosine similarity thresholds
            primary_range = np.linspace(0.25, 0.55, 15)
            secondary_range = np.linspace(0.15, 0.45, 15)
        else:  # euclidean
            # Test euclidean distance thresholds  
            primary_range = np.linspace(0.85, 1.25, 15)
            secondary_range = np.linspace(0.95, 1.35, 15)
        
        results = []
        combinations_tested = 0
        
        print(f"Testing threshold combinations...")
        
        for primary_thresh in primary_range:
            for secondary_thresh in secondary_range:
                # Skip invalid combinations
                if distance_metric == 'cosine' and primary_thresh <= secondary_thresh:
                    continue
                if distance_metric == 'euclidean' and primary_thresh >= secondary_thresh:
                    continue
                
                if combinations_tested >= n_combinations:
                    break
                
                # Evaluate this threshold combination
                metrics = self.evaluate_threshold_combination(
                    primary_thresh, secondary_thresh, similarities, distance_metric
                )
                
                results.append({
                    'primary_threshold': primary_thresh,
                    'secondary_threshold': secondary_thresh,
                    'distance_metric': distance_metric,
                    **metrics
                })
                
                combinations_tested += 1
                
                if combinations_tested % 10 == 0:
                    print(f"Tested {combinations_tested} combinations...")
            
            if combinations_tested >= n_combinations:
                break
        
        results_df = pd.DataFrame(results)
        
        print(f"\nCompleted grid search with {len(results_df)} threshold combinations")
        print(f"F1-Score range: {results_df['f1_score'].min():.4f} - {results_df['f1_score'].max():.4f}")
        print(f"F1-Score mean ± std: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
        
        return results_df
    
    def train_test_validation(self, distance_metric='cosine', test_size=0.3):
        """Proper train-test split validation (fixed for correct indexing)"""
        print(f"\n{'='*60}")
        print(f"TRAIN-TEST SPLIT VALIDATION - {distance_metric.upper()}")
        print(f"{'='*60}")
        
        # Split data
        indices = np.arange(len(self.texts))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        print(f"Training set: {len(train_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
        
        # Calculate similarities for all data ONCE, then slice
        all_similarities = self.calculate_similarities(distance_metric)
        train_similarities = all_similarities[train_idx]
        test_similarities = all_similarities[test_idx]
        
        # Create train/test subsets of true labels
        if isinstance(self.true_labels[0], list):
            train_labels = [self.true_labels[i] for i in train_idx]
            test_labels = [self.true_labels[i] for i in test_idx]
        else:
            train_labels = np.array(self.true_labels)[train_idx].tolist()
            test_labels = np.array(self.true_labels)[test_idx].tolist()
        
        # Store original data
        original_labels = self.true_labels
        original_text_embeddings = self.text_embeddings
        
        # Optimize thresholds on training set (using direct evaluation instead of grid search)
        print("Finding optimal thresholds on training set...")
        
        # Define threshold ranges for training optimization
        if distance_metric == 'cosine':
            primary_range = np.linspace(0.35, 0.45, 5)
            secondary_range = np.linspace(0.25, 0.35, 5)
        else:
            primary_range = np.linspace(0.95, 1.15, 5)
            secondary_range = np.linspace(1.05, 1.25, 5)
        
        # Set training data temporarily
        self.true_labels = train_labels
        self.text_embeddings = self.text_embeddings[train_idx]
        
        best_f1 = 0
        best_config = None
        
        for primary_thresh in primary_range:
            for secondary_thresh in secondary_range:
                # Skip invalid combinations
                if distance_metric == 'cosine' and primary_thresh <= secondary_thresh:
                    continue
                if distance_metric == 'euclidean' and primary_thresh >= secondary_thresh:
                    continue
                
                # Evaluate on training data
                train_metrics = self.evaluate_threshold_combination(
                    primary_thresh, secondary_thresh, train_similarities, distance_metric
                )
                
                if train_metrics['f1_score'] > best_f1:
                    best_f1 = train_metrics['f1_score']
                    best_config = {
                        'primary_threshold': primary_thresh,
                        'secondary_threshold': secondary_thresh,
                        **train_metrics
                    }
        
        if best_config is None:
            raise ValueError("No valid threshold configuration found")
        
        print(f"Best training thresholds: {best_config['primary_threshold']:.3f}/{best_config['secondary_threshold']:.3f}")
        print(f"Training F1: {best_config['f1_score']:.4f}")
        
        # Evaluate on test set
        self.true_labels = test_labels
        self.text_embeddings = original_text_embeddings[test_idx]
        
        print("Evaluating on test set...")
        test_metrics = self.evaluate_threshold_combination(
            best_config['primary_threshold'],
            best_config['secondary_threshold'],
            test_similarities,
            distance_metric
        )
        
        # Restore original data
        self.true_labels = original_labels
        self.text_embeddings = original_text_embeddings
        
        print(f"Test F1: {test_metrics['f1_score']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test Coverage: {test_metrics['coverage']:.3f}")
        
        # Calculate performance degradation
        degradation = (best_config['f1_score'] - test_metrics['f1_score']) / best_config['f1_score']
        print(f"Performance degradation: {degradation:.2%}")
        
        return {
            'train_f1': best_config['f1_score'],
            'test_f1': test_metrics['f1_score'],
            'degradation': degradation,
            'best_thresholds': (best_config['primary_threshold'], best_config['secondary_threshold']),
            'test_metrics': test_metrics
        }
    
    def statistical_comparison(self, cosine_results, euclidean_results):
        """Statistical significance testing between methods"""
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*60}")
        
        cosine_f1 = cosine_results['f1_score'].values
        euclidean_f1 = euclidean_results['f1_score'].values
        
        # Descriptive statistics
        print("Cosine Similarity Results:")
        print(f"  Mean F1: {cosine_f1.mean():.4f} ± {cosine_f1.std():.4f}")
        print(f"  Range: [{cosine_f1.min():.4f}, {cosine_f1.max():.4f}]")
        print(f"  Median: {np.median(cosine_f1):.4f}")
        
        print("\nEuclidean Distance Results:")
        print(f"  Mean F1: {euclidean_f1.mean():.4f} ± {euclidean_f1.std():.4f}")
        print(f"  Range: [{euclidean_f1.min():.4f}, {euclidean_f1.max():.4f}]")
        print(f"  Median: {np.median(euclidean_f1):.4f}")
        
        # Statistical tests
        if len(cosine_f1) == len(euclidean_f1):
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(cosine_f1, euclidean_f1)
            test_type = "Paired t-test"
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(cosine_f1, euclidean_f1)
            test_type = "Independent t-test"
        
        print(f"\n{test_type} Results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((cosine_f1.var() + euclidean_f1.var()) / 2)
        cohens_d = (cosine_f1.mean() - euclidean_f1.mean()) / pooled_std
        
        print(f"\nEffect Size:")
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        if abs(cohens_d) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d) < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        print(f"  Interpretation: {effect_size} effect")
        
        # Win rate analysis
        cosine_wins = np.sum(cosine_f1.max() > euclidean_f1)
        euclidean_wins = np.sum(euclidean_f1.max() > cosine_f1)
        
        print(f"\nRobustness Analysis:")
        print(f"  Cosine superior in: {(cosine_wins/len(cosine_f1))*100:.1f}% of configurations")
        print(f"  Euclidean superior in: {(euclidean_wins/len(euclidean_f1))*100:.1f}% of configurations")
        
        return {
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'cosine_mean': cosine_f1.mean(),
            'euclidean_mean': euclidean_f1.mean(),
            'cosine_std': cosine_f1.std(),
            'euclidean_std': euclidean_f1.std()
        }
    
    def generate_comprehensive_report(self):
        """Generate complete validation report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE THRESHOLD ROBUSTNESS VALIDATION")
        print(f"{'='*80}")
        
        # Load data and generate embeddings
        self.load_data_and_model()
        self.generate_embeddings()
        
        # Run grid search validations
        cosine_results = self.grid_search_validation('cosine', n_combinations=40)
        euclidean_results = self.grid_search_validation('euclidean', n_combinations=40)
        
        # Run train-test validations
        cosine_traintest = self.train_test_validation('cosine')
        euclidean_traintest = self.train_test_validation('euclidean')
        
        # Statistical comparison
        stats_results = self.statistical_comparison(cosine_results, euclidean_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        cosine_results.to_csv(f'validation_cosine_grid_search_{timestamp}.csv', index=False)
        euclidean_results.to_csv(f'validation_euclidean_grid_search_{timestamp}.csv', index=False)
        
        # Generate summary report
        report = {
            'timestamp': timestamp,
            'dataset_size': len(self.texts),
            'model': self.model_name,
            'cosine_grid_search': {
                'n_combinations': len(cosine_results),
                'best_f1': cosine_results['f1_score'].max(),
                'mean_f1': cosine_results['f1_score'].mean(),
                'std_f1': cosine_results['f1_score'].std(),
                'best_config': cosine_results.loc[cosine_results['f1_score'].idxmax()].to_dict()
            },
            'euclidean_grid_search': {
                'n_combinations': len(euclidean_results),
                'best_f1': euclidean_results['f1_score'].max(),
                'mean_f1': euclidean_results['f1_score'].mean(),
                'std_f1': euclidean_results['f1_score'].std(),
                'best_config': euclidean_results.loc[euclidean_results['f1_score'].idxmax()].to_dict()
            },
            'train_test_validation': {
                'cosine': cosine_traintest,
                'euclidean': euclidean_traintest
            },
            'statistical_analysis': stats_results
        }
        
        # Save summary report
        with open(f'threshold_validation_summary_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY FOR THESIS DEFENSE")
        print(f"{'='*60}")
        
        print(f"\n📊 GRID SEARCH RESULTS:")
        print(f"   Cosine: {len(cosine_results)} combinations, F1={cosine_results['f1_score'].mean():.4f}±{cosine_results['f1_score'].std():.4f}")
        print(f"   Euclidean: {len(euclidean_results)} combinations, F1={euclidean_results['f1_score'].mean():.4f}±{euclidean_results['f1_score'].std():.4f}")
        
        print(f"\n🔄 TRAIN-TEST VALIDATION:")
        print(f"   Cosine degradation: {cosine_traintest['degradation']:.2%}")
        print(f"   Euclidean degradation: {euclidean_traintest['degradation']:.2%}")
        
        print(f"\n📈 STATISTICAL SIGNIFICANCE:")
        print(f"   p-value: {stats_results['p_value']:.6f}")
        print(f"   Effect size: {stats_results['effect_size']} (d={stats_results['cohens_d']:.4f})")
        print(f"   Statistically significant: {stats_results['significant']}")
        
        print(f"\n✅ THESIS DEFENSE READINESS:")
        if stats_results['significant']:
            print("   🎯 Results are statistically significant - strong defense position")
            print("   📊 Can claim robust performance differences with confidence")
        else:
            print("   ⚠️  Results may not be statistically significant")
            print("   📊 Emphasize methodological rigor and acknowledge limitations")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   - validation_cosine_grid_search_{timestamp}.csv")
        print(f"   - validation_euclidean_grid_search_{timestamp}.csv")
        print(f"   - threshold_validation_summary_{timestamp}.json")
        
        return report

# Usage example
if __name__ == "__main__":
    # Initialize validator
    project_path = "/Users/mahnoorzamir/Desktop/mitacs/project"
    validator = ComprehensiveThresholdValidator(project_path)
    
    # Generate comprehensive validation report
    print("Starting comprehensive threshold robustness validation...")
    print("This addresses methodological concerns for thesis defense.")
    
    try:
        report = validator.generate_comprehensive_report()
        print("\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print("Results ready for thesis defense!")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {str(e)}")
        print("Check data paths and requirements.")
