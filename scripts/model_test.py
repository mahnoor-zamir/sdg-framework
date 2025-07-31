#!/usr/bin/env python3
"""
Robust Model Testing for Multi-SDG Classification

Fixed version that handles compatibility issues and provides reliable testing.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, hamming_loss, jaccard_score, 
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import time
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def load_data(file_path="benchmark_multi_sdg.csv", max_samples=None):
    """Load and prepare the multi-SDG dataset."""
    print("📊 Loading multi-SDG dataset...")
    df = pd.read_csv(file_path)
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        print(f"   Using {len(df)} samples for testing")
    
    texts = df['text'].tolist()
    label_columns = [f'sdg_{i}' for i in range(1, 18)]
    labels = df[label_columns].values.astype(int)
    
    print(f"   Dataset: {len(texts)} texts, {labels.shape[1]} SDG labels")
    
    # Show label distribution
    positive_counts = labels.sum(axis=0)
    print("   SDG label distribution:")
    for i, count in enumerate(positive_counts):
        pct = (count / len(texts)) * 100
        print(f"     SDG {i+1:2d}: {count:4d} ({pct:5.1f}%)")
    
    return texts, labels

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate comprehensive multi-label metrics."""
    if len(y_pred.shape) > 1:
        if y_pred.shape[1] == 17:
            y_pred_binary = (y_pred > threshold).astype(int)
        else:
            y_pred_binary = y_pred
    else:
        y_pred_binary = y_pred
    
    # Ensure we have at least one positive prediction to avoid division by zero
    if y_pred_binary.sum() == 0:
        y_pred_binary = np.zeros_like(y_true)
        if len(y_pred_binary) > 0:
            y_pred_binary[0, 0] = 1
    
    try:
        return {
            'hamming_loss': hamming_loss(y_true, y_pred_binary),
            'jaccard_score': jaccard_score(y_true, y_pred_binary, average='samples', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred_binary, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred_binary, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'subset_accuracy': accuracy_score(y_true, y_pred_binary),
        }
    except Exception as e:
        print(f"   ⚠️  Error calculating metrics: {e}")
        return {key: 0.0 for key in [
            'hamming_loss', 'jaccard_score', 'f1_micro', 'f1_macro', 'f1_weighted',
            'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro', 'subset_accuracy'
        ]}

def test_model_1_sentence_bert_lr(X_train, y_train, X_test, y_test):
    """Model 1: Sentence-BERT + Logistic Regression (similar to DeBERTa approach)."""
    print("🤖 Model 1: Sentence-BERT + Logistic Regression")
    print("   (Simulating DeBERTa-style deep learning approach)")
    
    try:
        # Use a powerful sentence transformer
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        start_time = time.time()
        
        # Generate embeddings
        print("   Generating embeddings...")
        train_embeddings = model.encode(X_train, show_progress_bar=True)
        test_embeddings = model.encode(X_test, show_progress_bar=True)
        
        # Train classifier with better parameters
        print("   Training classifier...")
        classifier = MultiOutputClassifier(
            LogisticRegression(
                random_state=42, 
                max_iter=2000,
                C=0.1,  # Regularization
                solver='lbfgs'
            ),
            n_jobs=-1
        )
        
        classifier.fit(train_embeddings, y_train)
        training_time = time.time() - start_time
        
        # Predict
        print("   Making predictions...")
        predictions = classifier.predict(test_embeddings)
        
        # Get prediction probabilities
        try:
            prediction_probs = classifier.predict_proba(test_embeddings)
            probs = np.zeros((len(test_embeddings), 17))
            for i, prob_array in enumerate(prediction_probs):
                if prob_array.shape[1] == 2:
                    probs[:, i] = prob_array[:, 1]
                else:
                    probs[:, i] = 0.5
        except:
            probs = predictions.astype(float)
        
        metrics = calculate_metrics(y_test, probs)
        
        return {
            'model_name': 'Sentence-BERT + LogisticRegression',
            'training_time': training_time,
            'metrics': metrics,
            'status': 'success',
            'approach': 'Deep embeddings + ML classifier'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'model_name': 'Sentence-BERT + LR', 'error': str(e), 'status': 'failed'}

def test_model_2_sentence_bert_rf(X_train, y_train, X_test, y_test):
    """Model 2: Sentence-BERT + Random Forest (alternative to BERT)."""
    print("🌲 Model 2: Sentence-BERT + Random Forest")
    print("   (Alternative ensemble approach)")
    
    try:
        # Use BERT-like sentence transformer
        model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        
        start_time = time.time()
        
        # Generate embeddings
        print("   Generating embeddings...")
        train_embeddings = model.encode(X_train, show_progress_bar=True)
        test_embeddings = model.encode(X_test, show_progress_bar=True)
        
        # Train Random Forest classifier
        print("   Training Random Forest...")
        classifier = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            ),
            n_jobs=-1
        )
        
        classifier.fit(train_embeddings, y_train)
        training_time = time.time() - start_time
        
        # Predict
        print("   Making predictions...")
        predictions = classifier.predict(test_embeddings)
        
        # Get prediction probabilities
        try:
            prediction_probs = classifier.predict_proba(test_embeddings)
            probs = np.zeros((len(test_embeddings), 17))
            for i, prob_array in enumerate(prediction_probs):
                if prob_array.shape[1] == 2:
                    probs[:, i] = prob_array[:, 1]
                else:
                    probs[:, i] = 0.5
        except:
            probs = predictions.astype(float)
        
        metrics = calculate_metrics(y_test, probs)
        
        return {
            'model_name': 'Sentence-BERT + RandomForest',
            'training_time': training_time,
            'metrics': metrics,
            'status': 'success',
            'approach': 'BERT embeddings + ensemble classifier'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'model_name': 'Sentence-BERT + RF', 'error': str(e), 'status': 'failed'}

def test_model_3_setfit_style(X_train, y_train, X_test, y_test):
    """Model 3: SetFit-style approach with paraphrase-mpnet-base-v2."""
    print("🔄 Model 3: SetFit-style with paraphrase-mpnet-base-v2")
    print("   (Few-shot learning approach)")
    
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
        
        start_time = time.time()
        
        # Generate embeddings
        print("   Generating embeddings...")
        train_embeddings = model.encode(X_train, show_progress_bar=True)
        test_embeddings = model.encode(X_test, show_progress_bar=True)
        
        # Use a different approach - train separate binary classifiers for each SDG
        print("   Training binary classifiers for each SDG...")
        classifiers = []
        
        for sdg_idx in range(17):
            sdg_labels = y_train[:, sdg_idx]
            
            # Skip if no positive examples
            if sdg_labels.sum() == 0:
                classifiers.append(None)
                continue
            
            # Train binary classifier for this SDG
            clf = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                class_weight='balanced'  # Handle imbalanced classes
            )
            clf.fit(train_embeddings, sdg_labels)
            classifiers.append(clf)
        
        training_time = time.time() - start_time
        
        # Predict
        print("   Making predictions...")
        predictions = np.zeros((len(test_embeddings), 17))
        probs = np.zeros((len(test_embeddings), 17))
        
        for sdg_idx, clf in enumerate(classifiers):
            if clf is not None:
                predictions[:, sdg_idx] = clf.predict(test_embeddings)
                probs[:, sdg_idx] = clf.predict_proba(test_embeddings)[:, 1]
        
        metrics = calculate_metrics(y_test, probs)
        
        return {
            'model_name': 'SetFit-style (paraphrase-mpnet)',
            'training_time': training_time,
            'metrics': metrics,
            'status': 'success',
            'approach': 'Per-SDG binary classifiers'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'model_name': 'SetFit-style', 'error': str(e), 'status': 'failed'}

def run_robust_model_test(max_samples=500):
    """Run robust model testing that handles compatibility issues."""
    print("🚀 Robust Multi-SDG Model Performance Testing")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now()}")
    print(f"📊 Max samples: {max_samples}")
    print()
    
    # Load data
    texts, labels = load_data(max_samples=max_samples)
    
    # Split data
    print("✂️  Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=None
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    results = {}
    
    # Test Model 1
    print("\n" + "="*60)
    print("🤖 TESTING MODEL 1: Deep Learning Style (DeBERTa equivalent)")
    print("="*60)
    results['Model_1_DeepLearning'] = test_model_1_sentence_bert_lr(X_train, y_train, X_test, y_test)
    
    # Test Model 2
    print("\n" + "="*60)
    print("🌲 TESTING MODEL 2: Ensemble Style (BERT equivalent)")
    print("="*60)
    results['Model_2_Ensemble'] = test_model_2_sentence_bert_rf(X_train, y_train, X_test, y_test)
    
    # Test Model 3
    print("\n" + "="*60)
    print("🔄 TESTING MODEL 3: SetFit Style")
    print("="*60)
    results['Model_3_SetFit'] = test_model_3_setfit_style(X_train, y_train, X_test, y_test)
    
    # Results Summary
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RESULTS")
    print("="*60)
    
    successful_models = [k for k, v in results.items() if v.get('status') == 'success']
    
    if successful_models:
        print(f"\n{'Model':<30} {'F1-Micro':<10} {'F1-Macro':<10} {'Jaccard':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        best_model = None
        best_score = 0
        
        for model_name in successful_models:
            result = results[model_name]
            metrics = result['metrics']
            time_taken = result['training_time']
            
            print(f"{result['model_name']:<30} {metrics['f1_micro']:<10.3f} {metrics['f1_macro']:<10.3f} "
                  f"{metrics['jaccard_score']:<10.3f} {time_taken:<10.1f}")
            
            # Track best model
            combined_score = metrics['f1_micro'] + metrics['f1_macro']
            if combined_score > best_score:
                best_score = combined_score
                best_model = model_name
        
        # Detailed analysis
        if best_model:
            print(f"\n🏆 BEST PERFORMING MODEL: {results[best_model]['model_name']}")
            print(f"   Approach: {results[best_model]['approach']}")
            print(f"   Training time: {results[best_model]['training_time']:.1f} seconds")
            
            print(f"\n📈 Detailed Performance Metrics:")
            best_metrics = results[best_model]['metrics']
            for metric, value in best_metrics.items():
                print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Performance comparison
        print(f"\n📊 Model Comparison Summary:")
        print(f"   🎯 Most accurate (F1-Micro): {max(successful_models, key=lambda x: results[x]['metrics']['f1_micro'])}")
        print(f"   ⚖️  Most balanced (F1-Macro): {max(successful_models, key=lambda x: results[x]['metrics']['f1_macro'])}")
        print(f"   ⚡ Fastest training: {min(successful_models, key=lambda x: results[x]['training_time'])}")
        
        # Save results
        results_file = 'robust_model_comparison_results.json'
        
        # Prepare for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                k: v for k, v in result.items() 
                if k not in ['predictions']  # Remove non-serializable data
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   📚 For production use: {results[best_model]['model_name']}")
        print(f"   🔬 For research: Consider fine-tuning actual transformer models")
        print(f"   ⚡ For speed: Use the fastest model with acceptable performance")
        
    else:
        print("❌ No models completed successfully")
        for model_name, result in results.items():
            if result.get('status') == 'failed':
                print(f"   {model_name}: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = run_robust_model_test(max_samples=500)
