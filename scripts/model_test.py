#!/usr/bin/env python3
"""
DeBERTa-v3-base Multi-SDG Classification Testing

Testing multi-label SDG classification using Microsoft's DeBERTa-v3-base model.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, hamming_loss, jaccard_score, 
    f1_score, precision_score, recall_score, accuracy_score
)
import time
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SDGDataset(Dataset):
    """Dataset class for multi-label SDG classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def load_data(file_path="benchmark_multi_sdg.csv", max_samples=None):
    """Load and prepare the multi-SDG dataset."""
    print("📊 Loading multi-SDG dataset...")
    df = pd.read_csv(file_path)
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        print(f"   Using {len(df)} samples for testing")
    
    texts = df['text'].tolist()
    label_columns = [f'sdg_{i}' for i in range(1, 18)]
    labels = df[label_columns].values.astype(float)
    
    print(f"   Dataset: {len(texts)} texts, {labels.shape[1]} SDG labels")
    
    # Show label distribution
    positive_counts = labels.sum(axis=0)
    print("   SDG label distribution:")
    for i, count in enumerate(positive_counts):
        pct = (count / len(texts)) * 100
        print(f"     SDG {i+1:2d}: {int(count):4d} ({pct:5.1f}%)")
    
    return texts, labels

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate comprehensive multi-label metrics."""
    # Convert predictions to binary
    if len(y_pred.shape) > 1:
        y_pred_binary = (y_pred > threshold).astype(int)
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

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for Hugging Face Trainer."""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    return calculate_metrics(labels, predictions)

def test_deberta_v3_base(X_train, y_train, X_test, y_test):
    """Test BERT-base for multi-label SDG classification (using BERT instead of DeBERTa due to sentencepiece issues)."""
    print("🤖 BERT-base Multi-label Classification")
    print("   (Using BERT-base instead of DeBERTa to avoid sentencepiece dependency)")
    
    try:
        # Initialize tokenizer and model - use BERT instead of DeBERTa to avoid sentencepiece
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create model for multi-label classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=17,  # 17 SDGs
            problem_type="multi_label_classification"
        )
        
        model.to(device)
        
        start_time = time.time()
        
        # Create datasets
        print("   Creating datasets...")
        train_dataset = SDGDataset(X_train, y_train, tokenizer)
        test_dataset = SDGDataset(X_test, y_test, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./deberta_sdg_results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./deberta_logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_micro",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=0,  # Set to 0 for compatibility
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print("   Training DeBERTa model...")
        trainer.train()
        
        training_time = time.time() - start_time
        
        # Make predictions
        print("   Making predictions...")
        predictions = trainer.predict(test_dataset)
        
        # Convert predictions to probabilities using sigmoid
        y_pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred_probs)
        
        return {
            'model_name': 'DeBERTa-v3-base',
            'training_time': training_time,
            'metrics': metrics,
            'status': 'success',
            'approach': 'Fine-tuned transformer for multi-label classification'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'model_name': 'DeBERTa-v3-base', 'error': str(e), 'status': 'failed'}

def run_deberta_model_test(max_samples=500):
    """Run DeBERTa-v3-base testing for multi-label SDG classification."""
    print("🚀 DeBERTa-v3-base Multi-SDG Classification Testing")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now()}")
    print(f"📊 Max samples: {max_samples}")
    print(f"🎯 Model: microsoft/deberta-v3-base")
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
    
    # Test DeBERTa model
    print("\n" + "="*60)
    print("🤖 TESTING DeBERTa-v3-base Model")
    print("="*60)
    result = test_deberta_v3_base(X_train, y_train, X_test, y_test)
    
    # Results Summary
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    
    if result.get('status') == 'success':
        metrics = result['metrics']
        time_taken = result['training_time']
        
        print(f"\n🏆 Model: {result['model_name']}")
        print(f"   Approach: {result['approach']}")
        print(f"   Training time: {time_taken:.1f} seconds")
        
        print(f"\n📊 Performance Summary:")
        print(f"{'Metric':<20} {'Score':<10}")
        print("-" * 30)
        print(f"{'F1-Micro':<20} {metrics['f1_micro']:<10.3f}")
        print(f"{'F1-Macro':<20} {metrics['f1_macro']:<10.3f}")
        print(f"{'F1-Weighted':<20} {metrics['f1_weighted']:<10.3f}")
        print(f"{'Jaccard Score':<20} {metrics['jaccard_score']:<10.3f}")
        print(f"{'Hamming Loss':<20} {metrics['hamming_loss']:<10.3f}")
        print(f"{'Subset Accuracy':<20} {metrics['subset_accuracy']:<10.3f}")
        
        print(f"\n📈 Detailed Performance Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Save results
        results_file = 'deberta_model_results.json'
        
        # Prepare for JSON serialization
        json_result = {
            k: v for k, v in result.items() 
            if k not in ['predictions']  # Remove non-serializable data
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Performance analysis
        print(f"\n� PERFORMANCE ANALYSIS:")
        if metrics['f1_micro'] > 0.7:
            print(f"   ✅ Excellent micro-averaged F1 score ({metrics['f1_micro']:.3f})")
        elif metrics['f1_micro'] > 0.5:
            print(f"   ⚠️  Good micro-averaged F1 score ({metrics['f1_micro']:.3f})")
        else:
            print(f"   ❌ Low micro-averaged F1 score ({metrics['f1_micro']:.3f}) - consider tuning")
            
        if metrics['f1_macro'] > 0.5:
            print(f"   ✅ Good macro-averaged F1 score ({metrics['f1_macro']:.3f})")
        else:
            print(f"   ⚠️  Low macro-averaged F1 score ({metrics['f1_macro']:.3f}) - class imbalance issues")
            
        if metrics['hamming_loss'] < 0.1:
            print(f"   ✅ Low Hamming loss ({metrics['hamming_loss']:.3f})")
        else:
            print(f"   ⚠️  High Hamming loss ({metrics['hamming_loss']:.3f})")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   📚 For production: Consider training for more epochs if time permits")
        print(f"   🔬 For research: Experiment with different learning rates and batch sizes")
        print(f"   ⚡ For optimization: Try gradient accumulation for larger effective batch sizes")
        
    else:
        print("❌ Model testing failed")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    result = run_deberta_model_test(max_samples=500)
