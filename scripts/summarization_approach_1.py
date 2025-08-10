#!/usr/bin/env python3
"""
Test summarization approaches for SDG classification improvement.

This script compares different summarization methods applied to SDG descriptions
before embedding-based similarity calculation to improve classification precision.
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


class SDGSummarizer:
    """Handle different summarization approaches for SDG descriptions."""
    
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
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize a single text using the specified method."""
        
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            if self.method == 'extractive':
                return self._extractive_summarize(text, max_length)
            
            elif self.method == 'keywords':
                return self._keyword_summarize(text)
            
            elif self.method in ['facebook/bart-large-cnn', 'google/pegasus-xsum']:
                # Abstractive summarization
                if len(text) < 100:  # Too short to summarize
                    return text
                    
                # Truncate if too long for model
                max_input_length = 1024
                if len(text.split()) > max_input_length:
                    text = ' '.join(text.split()[:max_input_length])
                
                summary = self.summarizer(text, max_length=max_length, min_length=min(30, max_length//3))
                return summary[0]['summary_text']
            
            else:
                return text
                
        except Exception as e:
            print(f"Summarization failed for text: {str(e)[:100]}...")
            return text  # Return original text if summarization fails
    
    def _extractive_summarize(self, text: str, max_length: int = 100) -> str:
        """Simple extractive summarization - select key sentences."""
        sentences = text.split('. ')
        
        # Score sentences by keyword importance
        key_terms = ['goal', 'target', 'sustainable', 'development', 'ensure', 'promote', 'achieve', 'access', 'reduce', 'improve']
        
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
        
        return summary.strip()
    
    def _keyword_summarize(self, text: str) -> str:
        """Extract keywords and create a keyword-based summary."""
        if self.summarizer is None:
            # Fallback to simple keyword extraction
            return self._simple_keyword_extract(text)
        
        try:
            keywords = self.summarizer.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english',
                top_k=8,
                use_maxsum=True
            )
            
            # Convert keywords to summary text
            keyword_phrases = [kw[0] for kw in keywords]
            return ", ".join(keyword_phrases)
            
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return self._simple_keyword_extract(text)
    
    def _simple_keyword_extract(self, text: str) -> str:
        """Simple keyword extraction as fallback."""
        # Common SDG-related important terms
        important_terms = [
            'poverty', 'hunger', 'health', 'education', 'gender', 'water', 'energy', 
            'work', 'industry', 'inequality', 'cities', 'consumption', 'climate', 
            'ocean', 'land', 'peace', 'partnership', 'sustainable', 'development',
            'access', 'quality', 'clean', 'affordable', 'decent', 'innovation',
            'reduced', 'responsible', 'action', 'life', 'justice', 'global'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in important_terms:
            if term in text_lower and term not in found_terms:
                found_terms.append(term)
        
        return ", ".join(found_terms[:8])


def load_sdg_paragraphs(file_path: str) -> pd.DataFrame:
    """Load the SDG paragraph dataset."""
    return pd.read_csv(file_path)


def create_summarized_sdg_descriptions(sdg_df: pd.DataFrame, summarization_method: str) -> Dict[int, str]:
    """Create summarized versions of SDG descriptions."""
    
    print(f"Creating summarized SDG descriptions using method: {summarization_method}")
    
    summarizer = SDGSummarizer(method=summarization_method)
    summarized_descriptions = {}
    
    for _, row in sdg_df.iterrows():
        sdg_num = row['sdg']
        original_text = row['text']
        
        print(f"Summarizing SDG {sdg_num}...")
        
        # Different summary lengths based on method
        if summarization_method == 'keywords':
            max_length = 50
        elif summarization_method == 'google/pegasus-xsum':
            max_length = 80
        else:
            max_length = 120
        
        summarized_text = summarizer.summarize_text(original_text, max_length=max_length)
        summarized_descriptions[sdg_num] = summarized_text
        
        print(f"  Original ({len(original_text)} chars): {original_text[:100]}...")
        print(f"  Summary ({len(summarized_text)} chars): {summarized_text}")
        print()
    
    return summarized_descriptions


def calculate_similarity_with_summaries(osdg_texts: List[str], 
                                      summarized_sdgs: Dict[int, str],
                                      embedding_model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Calculate similarity using summarized SDG descriptions."""
    
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    # Prepare SDG summaries in order
    sdg_texts = [summarized_sdgs[i] for i in range(1, 18)]
    
    print("Generating embeddings for OSDG texts...")
    osdg_embeddings = model.encode(osdg_texts, show_progress_bar=True, batch_size=32)
    
    print("Generating embeddings for summarized SDG descriptions...")
    sdg_embeddings = model.encode(sdg_texts, show_progress_bar=True, batch_size=32)
    
    print("Calculating cosine similarity...")
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
        
        for sdg_num, score in sdg_scores[:max_labels]:
            if score >= primary_threshold:
                primary_sdgs.append(sdg_num)
            elif score >= secondary_threshold:
                secondary_sdgs.append(sdg_num)
        
        all_sdgs = primary_sdgs + secondary_sdgs
        
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


def create_summarized_multilabel_dataset(osdg_file: str,
                                       sdg_paragraphs_file: str,
                                       summarization_method: str = 'facebook/bart-large-cnn',
                                       embedding_model: str = 'all-MiniLM-L6-v2',
                                       primary_threshold: float = 0.4,
                                       secondary_threshold: float = 0.3,
                                       max_labels: int = 5) -> pd.DataFrame:
    """Create multi-label dataset using summarization + embeddings."""
    
    print("=== Summarization + Embedding Approach ===")
    print(f"Summarization method: {summarization_method}")
    print(f"Embedding model: {embedding_model}")
    print(f"Thresholds: primary={primary_threshold}, secondary={secondary_threshold}")
    print()
    
    # Load datasets
    print("Loading OSDG dataset...")
    osdg_df = pd.read_csv(osdg_file)
    print(f"Loaded {len(osdg_df)} OSDG texts")
    
    print("Loading SDG paragraphs...")
    sdg_df = load_sdg_paragraphs(sdg_paragraphs_file)
    print(f"Loaded {len(sdg_df)} SDG descriptions")
    print()
    
    # Create summarized SDG descriptions
    start_time = time.time()
    summarized_sdgs = create_summarized_sdg_descriptions(sdg_df, summarization_method)
    summarization_time = time.time() - start_time
    print(f"Summarization completed in {summarization_time:.2f} seconds")
    print()
    
    # Prepare OSDG texts
    osdg_texts = [str(text) for text in osdg_df['text']]
    
    # Calculate similarities
    start_time = time.time()
    similarity_matrix = calculate_similarity_with_summaries(
        osdg_texts, summarized_sdgs, embedding_model
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
            'original_sdg_labels': row['sdg_labels'] if 'sdg_labels' in osdg_df.columns else None,
            'summarized_assigned_sdgs': assignment['all_sdgs'],
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
        'summarization_method': summarization_method,
        'embedding_model': embedding_model,
        'primary_threshold': primary_threshold,
        'secondary_threshold': secondary_threshold,
        'max_labels': max_labels,
        'summarization_time': summarization_time,
        'similarity_time': similarity_time,
        'total_processing_time': summarization_time + similarity_time,
        'summarized_descriptions': summarized_sdgs
    }
    
    return result_df


def save_results(result_df: pd.DataFrame, output_dir: str, method_name: str):
    """Save the results with method-specific naming."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clean method name for filename
    clean_method = method_name.replace('/', '_').replace('-', '_')
    
    # Save main dataset
    csv_path = output_path / f"summarized_multilabel_{clean_method}_p{result_df.attrs['metadata']['primary_threshold']}_s{result_df.attrs['metadata']['secondary_threshold']}.csv"
    
    # Drop complex columns for CSV
    csv_df = result_df.drop(['similarity_scores'], axis=1, errors='ignore')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved dataset to: {csv_path}")
    
    # Save full results with metadata
    json_path = output_path / f"summarized_multilabel_{clean_method}_full.json"
    
    full_data = {
        'metadata': result_df.attrs['metadata'],
        'data': result_df.to_dict('records')
    }
    
    with open(json_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    return csv_path, json_path


def print_summary_statistics(result_df: pd.DataFrame):
    """Print summary statistics of the results."""
    
    print("\n" + "="*60)
    print("SUMMARIZATION + EMBEDDING RESULTS SUMMARY")
    print("="*60)
    
    metadata = result_df.attrs['metadata']
    
    print(f"Method: {metadata['summarization_method']} + {metadata['embedding_model']}")
    print(f"Thresholds: Primary {metadata['primary_threshold']}, Secondary {metadata['secondary_threshold']}")
    print(f"Processing time: {metadata['total_processing_time']:.2f}s (summarization: {metadata['summarization_time']:.2f}s)")
    print()
    
    # Label distribution
    num_labels = result_df['num_assigned_labels']
    print(f"Total texts: {len(result_df)}")
    print(f"Average labels per text: {num_labels.mean():.2f}")
    print(f"Texts with multiple labels: {(num_labels > 1).sum()} ({(num_labels > 1).mean()*100:.1f}%)")
    print(f"Texts with single label: {(num_labels == 1).sum()} ({(num_labels == 1).mean()*100:.1f}%)")
    print(f"Texts with no labels: {(num_labels == 0).sum()} ({(num_labels == 0).mean()*100:.1f}%)")
    print()
    
    # Similarity statistics
    max_sim = result_df['max_similarity_score']
    print(f"Similarity statistics:")
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


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Test summarization + embedding approach for SDG classification')
    
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
                        default=0.4,
                        help='Primary similarity threshold')
    
    parser.add_argument('--secondary-threshold', '-t',
                        type=float,
                        default=0.3,
                        help='Secondary similarity threshold')
    
    parser.add_argument('--max-labels', '-m',
                        type=int,
                        default=5,
                        help='Maximum labels per text')
    
    parser.add_argument('--output', '-d',
                        default='data/processed',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create summarized multi-label dataset
    result_df = create_summarized_multilabel_dataset(
        args.osdg_file,
        args.sdg_paragraphs,
        args.summarization_method,
        args.embedding_model,
        args.primary_threshold,
        args.secondary_threshold,
        args.max_labels
    )
    
    # Save results
    csv_path, json_path = save_results(result_df, args.output, args.summarization_method)
    
    # Print summary
    print_summary_statistics(result_df)
    
    print(f"\n=== Summarization + Embedding approach completed! ===")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
