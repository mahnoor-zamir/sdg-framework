#!/usr/bin/env python3
"""
Summarization Approach 2: Both texts are summarized, then similarity is calculated.

This script extends the first approach by summarizing BOTH the SDG descriptions 
and the OSDG texts before calculating embedding-based similarity. This approach
should reduce noise and focus on the core concepts in both sets of texts.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import argparse
import re
from typing import List, Dict, Tuple
import time

# Summarization models
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class DualSDGSummarizer:
    """Handle summarization of both SDG descriptions and OSDG texts."""
    
    def __init__(self, method='facebook/bart-large-cnn'):
        self.method = method
        self.summarizer = None
        self._initialize_summarizer()
    
    def _initialize_summarizer(self):
        """Initialize the summarization model."""
        print(f"Loading summarization model: {self.method}")
        
        if self.method == 'facebook/bart-large-cnn':
            # Best for general summarization, good balance of quality and speed
            self.summarizer = pipeline(
                "summarization", 
                model=self.method,
                tokenizer=self.method,
                max_length=150,  # Shorter summaries
                min_length=30,
                do_sample=False
            )
            
        elif self.method == 'google/pegasus-xsum':
            # Excellent for abstractive summarization, more concise
            self.summarizer = pipeline(
                "summarization",
                model=self.method,
                max_length=100,
                min_length=20,
                do_sample=False
            )
            
        elif self.method == 'microsoft/DialoGPT-medium':
            # For keyword/concept extraction approach
            self.summarizer = pipeline(
                "text-generation",
                model=self.method,
                max_length=50
            )
            
        elif self.method == 'extractive':
            # Simple extractive summarization (no model needed)
            self.summarizer = None
            
        elif self.method == 'keywords':
            # Keyword extraction approach
            try:
                from keybert import KeyBERT
                self.summarizer = KeyBERT()
            except ImportError:
                print("KeyBERT not installed. Install with: pip install keybert")
                self.summarizer = None
    
    def summarize_text(self, text: str, text_type: str = 'osdg', max_length: int = 100) -> str:
        """Summarize a single text using the specified method."""
        
        if not text or len(text.strip()) < 30:
            return text
        
        try:
            if self.method == 'extractive':
                return self._extractive_summarize(text, text_type, max_length)
            
            elif self.method == 'keywords':
                return self._keyword_summarize(text, text_type)
            
            elif self.method in ['facebook/bart-large-cnn', 'google/pegasus-xsum']:
                # Abstractive summarization
                text_words = text.split()
                
                # More conservative approach for short texts
                if len(text_words) < 40:  # Reduced threshold
                    return text
                    
                # Adjust summarization parameters based on text type
                if text_type == 'sdg':
                    # SDG descriptions are already quite concise - be more conservative
                    max_len = min(max_length, max(50, len(text_words) // 2))
                    min_len = min(30, len(text_words) // 3)
                else:
                    # OSDG texts - preserve more content for better matching
                    max_len = min(max_length, max(80, len(text_words) // 2))
                    min_len = min(40, len(text_words) // 3)
                
                # Handle length warnings by adjusting max_length
                if len(text_words) < max_len:
                    max_len = max(min_len + 10, len(text_words) // 2)
                
                # Truncate if too long for model
                max_input_length = 1024
                if len(text_words) > max_input_length:
                    text = ' '.join(text_words[:max_input_length])
                
                summary = self.summarizer(text, max_length=max_len, min_length=min_len)
                return summary[0]['summary_text']
            
            else:
                return text
                
        except Exception as e:
            print(f"Summarization failed for {text_type} text: {str(e)[:100]}...")
            return text  # Return original text if summarization fails
    
    def _extractive_summarize(self, text: str, text_type: str, max_length: int = 100) -> str:
        """Simple extractive summarization - select key sentences."""
        sentences = text.split('. ')
        
        # Different key terms based on text type
        if text_type == 'sdg':
            key_terms = ['goal', 'target', 'sustainable', 'development', 'ensure', 'promote', 'achieve', 'access', 'reduce', 'improve']
        else:
            # For OSDG texts, focus on research and policy terms
            key_terms = ['research', 'study', 'analysis', 'policy', 'impact', 'development', 'social', 'economic', 'environmental', 'sustainable']
        
        scored_sentences = []
        for sent in sentences:
            score = sum(1 for term in key_terms if term.lower() in sent.lower())
            scored_sentences.append((score, sent))
        
        # Sort by score and select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        summary = ""
        for score, sent in scored_sentences:
            if len(summary) + len(sent) < max_length:
                summary += sent + ". "
            else:
                break
        
        return summary.strip() if summary else text[:max_length]
    
    def _keyword_summarize(self, text: str, text_type: str) -> str:
        """Extract keywords and create a keyword-based summary."""
        if self.summarizer is None:
            # Fallback to simple keyword extraction
            return self._simple_keyword_extract(text, text_type)
        
        try:
            # Adjust keyword extraction parameters based on text type
            top_k = 6 if text_type == 'sdg' else 8
            
            keywords = self.summarizer.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english',
                top_k=top_k,
                use_maxsum=True
            )
            
            # Convert keywords to summary text
            keyword_phrases = [kw[0] for kw in keywords]
            return ", ".join(keyword_phrases)
            
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return self._simple_keyword_extract(text, text_type)
    
    def _simple_keyword_extract(self, text: str, text_type: str) -> str:
        """Simple keyword extraction as fallback."""
        # Common SDG-related important terms
        if text_type == 'sdg':
            important_terms = [
                'poverty', 'hunger', 'health', 'education', 'gender', 'water', 'energy', 
                'work', 'industry', 'inequality', 'cities', 'consumption', 'climate', 
                'ocean', 'land', 'peace', 'partnership', 'sustainable', 'development',
                'access', 'quality', 'clean', 'affordable', 'decent', 'innovation',
                'reduced', 'responsible', 'action', 'life', 'justice', 'global'
            ]
            max_terms = 6
        else:
            # For OSDG texts, include research and policy terms
            important_terms = [
                'research', 'study', 'analysis', 'policy', 'implementation', 'impact',
                'social', 'economic', 'environmental', 'sustainable', 'development',
                'poverty', 'health', 'education', 'gender', 'water', 'energy', 'climate',
                'innovation', 'governance', 'inequality', 'urban', 'rural', 'agriculture'
            ]
            max_terms = 8
        
        found_terms = []
        text_lower = text.lower()
        
        for term in important_terms:
            if term in text_lower and term not in found_terms:
                found_terms.append(term)
        
        return ", ".join(found_terms[:max_terms])


def load_sdg_paragraphs(file_path: str) -> pd.DataFrame:
    """Load the SDG paragraph dataset."""
    return pd.read_csv(file_path)


def create_dual_summarized_descriptions(sdg_df: pd.DataFrame, 
                                       osdg_texts: List[str], 
                                       summarization_method: str) -> Tuple[Dict[int, str], List[str]]:
    """Create summarized versions of both SDG descriptions and OSDG texts."""
    
    print(f"Creating dual summaries using method: {summarization_method}")
    
    summarizer = DualSDGSummarizer(method=summarization_method)
    
    # Summarize SDG descriptions
    print("Summarizing SDG descriptions...")
    summarized_sdgs = {}
    
    for _, row in sdg_df.iterrows():
        sdg_num = row['sdg']
        original_text = row['text']
        
        print(f"Summarizing SDG {sdg_num}...")
        
        # Different summary lengths based on method
        if summarization_method == 'keywords':
            max_length = 50
        elif summarization_method == 'google/pegasus-xsum':
            max_length = 70
        else:
            max_length = 100
        
        summarized_text = summarizer.summarize_text(original_text, text_type='sdg', max_length=max_length)
        summarized_sdgs[sdg_num] = summarized_text
        
        print(f"  Original ({len(original_text)} chars): {original_text[:100]}...")
        print(f"  Summary ({len(summarized_text)} chars): {summarized_text}")
        print()
    
    # Summarize OSDG texts
    print(f"Summarizing {len(osdg_texts)} OSDG texts...")
    summarized_osdg_texts = []
    
    # Different summary lengths for OSDG texts
    if summarization_method == 'keywords':
        osdg_max_length = 60
    elif summarization_method == 'google/pegasus-xsum':
        osdg_max_length = 80
    else:
        osdg_max_length = 120
    
    # Process in smaller batches for better progress tracking
    batch_size = 50  # Reduced batch size for better progress
    total_batches = (len(osdg_texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(osdg_texts))
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (texts {start_idx+1} to {end_idx})...")
        
        batch_texts = osdg_texts[start_idx:end_idx]
        batch_summaries = []
        
        for j, text in enumerate(batch_texts):
            try:
                summarized_text = summarizer.summarize_text(str(text), text_type='osdg', max_length=osdg_max_length)
                batch_summaries.append(summarized_text)
                
                # Progress within batch
                if (j + 1) % 10 == 0:
                    print(f"  Completed {j + 1}/{len(batch_texts)} texts in current batch")
                    
            except Exception as e:
                print(f"  Error summarizing text {start_idx + j + 1}: {str(e)[:50]}...")
                batch_summaries.append(str(text))  # Use original text as fallback
        
        summarized_osdg_texts.extend(batch_summaries)
        
        # Progress update
        completed = len(summarized_osdg_texts)
        print(f"  Batch {batch_idx + 1} completed. Total progress: {completed}/{len(osdg_texts)} texts ({completed/len(osdg_texts)*100:.1f}%)")
        
        # Memory cleanup
        if batch_idx % 10 == 0:
            import gc
            gc.collect()
    
    print(f"Completed summarization of all texts.")
    
    return summarized_sdgs, summarized_osdg_texts


def calculate_dual_summary_similarity(summarized_osdg_texts: List[str], 
                                     summarized_sdgs: Dict[int, str],
                                     embedding_model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Calculate similarity using both summarized OSDG texts and summarized SDG descriptions."""
    
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    # Prepare SDG summaries in order
    sdg_summaries = [summarized_sdgs[i] for i in range(1, 18)]
    
    print("Generating embeddings for summarized OSDG texts...")
    osdg_embeddings = model.encode(summarized_osdg_texts, show_progress_bar=True, batch_size=32)
    
    print("Generating embeddings for summarized SDG descriptions...")
    sdg_embeddings = model.encode(sdg_summaries, show_progress_bar=True, batch_size=32)
    
    print("Calculating cosine similarity between summaries...")
    similarity_matrix = cosine_similarity(osdg_embeddings, sdg_embeddings)
    
    return similarity_matrix


def assign_labels_with_thresholds(similarity_matrix: np.ndarray,
                                primary_threshold: float = 0.4,
                                secondary_threshold: float = 0.3,
                                max_labels: int = 5) -> List[Dict]:
    """Assign multi-labels based on similarity thresholds."""
    
    results = []
    
    for i, similarities in enumerate(similarity_matrix):
        # Sort SDGs by similarity
        sdg_scores = [(j+1, similarities[j]) for j in range(17)]
        sdg_scores.sort(key=lambda x: x[1], reverse=True)
        
        primary_sdgs = []
        secondary_sdgs = []
        
        # Always assign at least one label if similarity is reasonable
        max_similarity = sdg_scores[0][1]
        
        for sdg_num, score in sdg_scores[:max_labels]:
            if score >= primary_threshold:
                primary_sdgs.append(sdg_num)
            elif score >= secondary_threshold:
                secondary_sdgs.append(sdg_num)
        
        all_sdgs = primary_sdgs + secondary_sdgs
        
        # If no labels assigned and max similarity is above a minimum threshold, assign the top one
        if not all_sdgs and max_similarity > 0.15:  # Relaxed minimum threshold
            all_sdgs = [sdg_scores[0][0]]
            secondary_sdgs = [sdg_scores[0][0]]
        
        # Create multi-hot vector
        multi_hot = [0] * 17
        for sdg in all_sdgs:
            multi_hot[sdg-1] = 1
        
        results.append({
            'primary_sdgs': primary_sdgs,
            'secondary_sdgs': secondary_sdgs,
            'all_sdgs': all_sdgs,
            'multi_hot_vector': multi_hot,
            'max_similarity': float(sdg_scores[0][1]),
            'num_labels': len(all_sdgs),
            'similarity_scores': {sdg: float(score) for sdg, score in sdg_scores}
        })
    
    return results


def create_dual_summarized_multilabel_dataset(osdg_file: str,
                                            sdg_paragraphs_file: str,
                                            summarization_method: str = 'facebook/bart-large-cnn',
                                            embedding_model: str = 'all-MiniLM-L6-v2',
                                            primary_threshold: float = 0.35,
                                            secondary_threshold: float = 0.25,
                                            max_labels: int = 5,
                                            test_subset: int = None) -> pd.DataFrame:
    """Create multi-label dataset using dual summarization + embeddings."""
    
    print("=== DUAL SUMMARIZATION + EMBEDDING APPROACH ===")
    print(f"Summarization method: {summarization_method}")
    print(f"Embedding model: {embedding_model}")
    print(f"Thresholds: primary={primary_threshold}, secondary={secondary_threshold}")
    print("Both SDG descriptions AND OSDG texts will be summarized before similarity calculation")
    
    if test_subset:
        print(f"TEST MODE: Using only first {test_subset} texts")
    print()
    
    # Load datasets
    print("Loading OSDG dataset...")
    osdg_df = pd.read_csv(osdg_file)
    
    # Apply test subset if specified
    if test_subset and test_subset < len(osdg_df):
        osdg_df = osdg_df.head(test_subset)
        print(f"Using subset of {len(osdg_df)} texts for testing")
    
    print(f"Loaded {len(osdg_df)} OSDG texts")
    
    print("Loading SDG paragraphs...")
    sdg_df = load_sdg_paragraphs(sdg_paragraphs_file)
    print(f"Loaded {len(sdg_df)} SDG descriptions")
    print()
    
    # Prepare OSDG texts
    osdg_texts = [str(text) for text in osdg_df['text']]
    
    # Create dual summarized descriptions
    start_time = time.time()
    summarized_sdgs, summarized_osdg_texts = create_dual_summarized_descriptions(
        sdg_df, osdg_texts, summarization_method
    )
    summarization_time = time.time() - start_time
    print(f"Dual summarization completed in {summarization_time:.2f} seconds")
    print()
    
    # Calculate similarities between summaries
    start_time = time.time()
    similarity_matrix = calculate_dual_summary_similarity(
        summarized_osdg_texts, summarized_sdgs, embedding_model
    )
    similarity_time = time.time() - start_time
    print(f"Similarity calculation completed in {similarity_time:.2f} seconds")
    print()
    
    # Assign labels
    print("Assigning multi-labels...")
    label_assignments = assign_labels_with_thresholds(
        similarity_matrix, primary_threshold, secondary_threshold, max_labels
    )
    
    # Create result dataframe
    result_data = []
    for i, (_, row) in enumerate(osdg_df.iterrows()):
        assignment = label_assignments[i]
        
        record = {
            'text_id': row['text_id'],
            'text': row['text'],
            'summarized_text': summarized_osdg_texts[i],
            'original_sdg_labels': row['sdg_labels'] if 'sdg_labels' in osdg_df.columns else None,
            'dual_summarized_assigned_sdgs': assignment['all_sdgs'],
            'primary_sdgs': assignment['primary_sdgs'],
            'secondary_sdgs': assignment['secondary_sdgs'],
            'multi_hot_vector': assignment['multi_hot_vector'],
            'max_similarity_score': assignment['max_similarity'],
            'num_assigned_labels': assignment['num_labels'],
            'similarity_scores': assignment['similarity_scores']
        }
        
        # Add individual SDG columns
        for sdg_num in range(1, 18):
            record[f'sdg_{sdg_num}'] = assignment['multi_hot_vector'][sdg_num-1]
        
        result_data.append(record)
    
    result_df = pd.DataFrame(result_data)
    
    # Add metadata
    result_df.attrs['metadata'] = {
        'approach': 'dual_summarization',
        'summarization_method': summarization_method,
        'embedding_model': embedding_model,
        'primary_threshold': primary_threshold,
        'secondary_threshold': secondary_threshold,
        'max_labels': max_labels,
        'summarization_time': summarization_time,
        'similarity_time': similarity_time,
        'total_processing_time': summarization_time + similarity_time,
        'summarized_sdg_descriptions': summarized_sdgs,
        'num_osdg_texts': len(osdg_texts),
        'num_sdg_descriptions': len(sdg_df)
    }
    
    return result_df


def save_dual_summarization_results(result_df: pd.DataFrame, output_dir: str, method_name: str):
    """Save the results with dual-summarization-specific naming."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clean method name for filename
    clean_method = method_name.replace('/', '_').replace('-', '_')
    
    # Save main dataset
    csv_path = output_path / f"dual_summarized_multilabel_{clean_method}_p{result_df.attrs['metadata']['primary_threshold']}_s{result_df.attrs['metadata']['secondary_threshold']}.csv"
    
    # Drop complex columns for CSV
    csv_df = result_df.drop(['similarity_scores'], axis=1, errors='ignore')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved dataset to: {csv_path}")
    
    # Save full results with metadata
    json_path = output_path / f"dual_summarized_multilabel_{clean_method}_full.json"
    
    full_data = {
        'metadata': result_df.attrs['metadata'],
        'data': result_df.to_dict('records')
    }
    
    with open(json_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    return csv_path, json_path


def evaluate_performance_metrics(result_df: pd.DataFrame) -> Dict:
    """Evaluate precision, recall, and F1 score if original labels are available."""
    
    if 'original_sdg_labels' not in result_df.columns or result_df['original_sdg_labels'].isna().all():
        print("No original labels available for evaluation")
        return {}
    
    from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
    
    # Convert original labels to binary matrix
    y_true = []
    y_pred = []
    
    for _, row in result_df.iterrows():
        # Parse original labels
        original_labels = str(row['original_sdg_labels'])
        if original_labels and original_labels != 'nan':
            # Assuming format like "[1, 3, 5]" or "1,3,5"
            try:
                if '[' in original_labels:
                    true_sdgs = eval(original_labels)
                else:
                    true_sdgs = [int(x.strip()) for x in original_labels.split(',') if x.strip()]
            except:
                true_sdgs = []
        else:
            true_sdgs = []
        
        # Create binary vectors
        true_binary = [0] * 17
        for sdg in true_sdgs:
            if 1 <= sdg <= 17:
                true_binary[sdg-1] = 1
        
        pred_binary = row['multi_hot_vector']
        
        y_true.append(true_binary)
        y_pred.append(pred_binary)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = {
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }
    
    return metrics


def print_dual_summary_statistics(result_df: pd.DataFrame):
    """Print summary statistics of the dual summarization results."""
    
    print("\n" + "="*70)
    print("DUAL SUMMARIZATION + EMBEDDING RESULTS SUMMARY")
    print("="*70)
    
    metadata = result_df.attrs['metadata']
    
    print(f"Approach: Dual Summarization (Both SDG and OSDG texts summarized)")
    print(f"Method: {metadata['summarization_method']} + {metadata['embedding_model']}")
    print(f"Thresholds: Primary {metadata['primary_threshold']}, Secondary {metadata['secondary_threshold']}")
    print(f"Processing time: {metadata['total_processing_time']:.2f}s (summarization: {metadata['summarization_time']:.2f}s)")
    print()
    
    # Text length comparison
    original_lengths = [len(str(text)) for text in result_df['text']]
    summary_lengths = [len(str(text)) for text in result_df['summarized_text']]
    
    print(f"Text length statistics:")
    print(f"  Original texts - Mean: {np.mean(original_lengths):.1f}, Median: {np.median(original_lengths):.1f}")
    print(f"  Summarized texts - Mean: {np.mean(summary_lengths):.1f}, Median: {np.median(summary_lengths):.1f}")
    print(f"  Compression ratio: {np.mean(summary_lengths) / np.mean(original_lengths) * 100:.1f}%")
    print()
    
    # Label distribution
    num_labels = result_df['num_assigned_labels']
    print(f"Label assignment statistics:")
    print(f"  Total texts: {len(result_df)}")
    print(f"  Average labels per text: {num_labels.mean():.2f}")
    print(f"  Texts with multiple labels: {(num_labels > 1).sum()} ({(num_labels > 1).mean()*100:.1f}%)")
    print(f"  Texts with single label: {(num_labels == 1).sum()} ({(num_labels == 1).mean()*100:.1f}%)")
    print(f"  Texts with no labels: {(num_labels == 0).sum()} ({(num_labels == 0).mean()*100:.1f}%)")
    print()
    
    # Similarity statistics
    max_sim = result_df['max_similarity_score']
    print(f"Similarity statistics (between summaries):")
    print(f"  Mean max similarity: {max_sim.mean():.3f}")
    print(f"  Median max similarity: {max_sim.median():.3f}")
    print(f"  Min similarity: {max_sim.min():.3f}")
    print(f"  Max similarity: {max_sim.max():.3f}")
    print()
    
    # SDG frequency
    print("SDG assignment frequency:")
    for sdg_num in range(1, 18):
        count = result_df[f'sdg_{sdg_num}'].sum()
        print(f"  SDG {sdg_num:2d}: {count:4d} texts ({count/len(result_df)*100:.1f}%)")
    
    # Performance metrics if available
    print()
    metrics = evaluate_performance_metrics(result_df)
    if metrics:
        print("Performance Metrics (if original labels available):")
        print(f"  Precision (micro): {metrics['precision_micro']:.3f}")
        print(f"  Recall (micro): {metrics['recall_micro']:.3f}")
        print(f"  F1 (micro): {metrics['f1_micro']:.3f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.3f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.3f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.3f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.3f}")


def compare_with_approach_1(result_df_approach2: pd.DataFrame, approach1_file: str = None):
    """Compare results with approach 1 if available."""
    
    if approach1_file and Path(approach1_file).exists():
        print(f"\n" + "="*70)
        print("COMPARISON WITH APPROACH 1")
        print("="*70)
        
        try:
            approach1_df = pd.read_csv(approach1_file)
            
            # Compare label distributions
            labels2 = result_df_approach2['num_assigned_labels']
            labels1 = approach1_df['num_assigned_labels']
            
            print(f"Average labels per text:")
            print(f"  Approach 1 (SDG summaries only): {labels1.mean():.2f}")
            print(f"  Approach 2 (Dual summaries): {labels2.mean():.2f}")
            print(f"  Difference: {labels2.mean() - labels1.mean():.2f}")
            print()
            
            # Compare similarity scores
            sim2 = result_df_approach2['max_similarity_score']
            sim1 = approach1_df['max_similarity_score']
            
            print(f"Maximum similarity scores:")
            print(f"  Approach 1 mean: {sim1.mean():.3f}")
            print(f"  Approach 2 mean: {sim2.mean():.3f}")
            print(f"  Difference: {sim2.mean() - sim1.mean():.3f}")
            
        except Exception as e:
            print(f"Could not compare with approach 1: {e}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Test dual summarization + embedding approach for SDG classification')
    
    parser.add_argument('--osdg-file', '-o',
                        default='data/processed/osdg_multilabel_threshold_0.6.csv',
                        help='OSDG dataset file')
    
    parser.add_argument('--sdg-paragraphs', '-s',
                        default='data/processed/sdg_paragraph_dataset.csv',
                        help='SDG paragraphs dataset file')
    
    parser.add_argument('--summarization-method', '-sm',
                        choices=[
                            'facebook/bart-large-cnn',    # Best general summarizer
                            'google/pegasus-xsum',        # Concise abstractive
                            'extractive',                 # Simple extractive
                            'keywords'                    # Keyword extraction
                        ],
                        default='facebook/bart-large-cnn',
                        help='Summarization method to use')
    
    parser.add_argument('--embedding-model', '-em',
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    
    parser.add_argument('--primary-threshold', '-p',
                        type=float,
                        default=0.35,  # Lowered from 0.4 to capture more labels
                        help='Primary similarity threshold')
    
    parser.add_argument('--secondary-threshold', '-t',
                        type=float,
                        default=0.25,  # Lowered from 0.3 to capture more labels
                        help='Secondary similarity threshold')
    
    parser.add_argument('--max-labels', '-m',
                        type=int,
                        default=5,
                        help='Maximum labels per text')
    
    parser.add_argument('--output', '-d',
                        default='data/processed',
                        help='Output directory')
    
    parser.add_argument('--compare-approach1', '-c',
                        help='Path to approach 1 results file for comparison')
    
    parser.add_argument('--test-subset', '-ts',
                        type=int,
                        help='Use only first N texts for testing (useful for development)')
    
    args = parser.parse_args()
    
    # Create dual summarized multi-label dataset
    result_df = create_dual_summarized_multilabel_dataset(
        args.osdg_file,
        args.sdg_paragraphs,
        args.summarization_method,
        args.embedding_model,
        args.primary_threshold,
        args.secondary_threshold,
        args.max_labels,
        test_subset=args.test_subset
    )
    
    # Save results
    csv_path, json_path = save_dual_summarization_results(result_df, args.output, args.summarization_method)
    
    # Print summary
    print_dual_summary_statistics(result_df)
    
    # Compare with approach 1 if specified
    if args.compare_approach1:
        compare_with_approach_1(result_df, args.compare_approach1)
    
    print(f"\n=== Dual Summarization + Embedding approach completed! ===")
    print(f"Results saved to: {args.output}")
    print(f"Key differences from Approach 1:")
    print(f"  - Both SDG descriptions AND OSDG texts are summarized")
    print(f"  - Similarity calculated between summaries (not original vs summary)")
    print(f"  - Should reduce noise and focus on core concepts")


if __name__ == "__main__":
    main()
