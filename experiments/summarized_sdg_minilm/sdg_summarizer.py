#!/usr/bin/env python3
"""
SDG Description Summarization Script
===================================

This script summarizes the long SDG descriptions to create more concise versions,
then tests if the summarized versions improve classification performance.

Approach:
1. Load original SDG descriptions
2. Summarize each SDG description using extractive/abstractive summarization
3. Create embeddings for summarized descriptions using all-MiniLM-L6-v2
4. Test classification performance with summarized descriptions vs original

Author: Experimental Framework
Date: August 22, 2025
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
import os

class SDGSummarizer:
    def __init__(self):
        """Initialize the SDG summarizer with models."""
        print("Loading models...")
        
        # Load sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load summarization model (using BART for abstractive summarization)
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if os.system("nvidia-smi") == 0 else -1  # Use GPU if available
        )
        
        print("Models loaded successfully!")
    
    def load_sdg_descriptions(self):
        """Load original SDG descriptions from the dataset."""
        print("Loading SDG descriptions...")
        
        # Load from the CSV file
        sdg_df = pd.read_csv('/Users/mahnoorzamir/Desktop/mitacs/project/data/processed/sdg_paragraph_dataset.csv')
        
        self.original_sdgs = {}
        self.summarized_sdgs = {}
        
        for _, row in sdg_df.iterrows():
            sdg_num = row['sdg']
            sdg_text = row['text']
            self.original_sdgs[sdg_num] = sdg_text
            
        print(f"Loaded {len(self.original_sdgs)} SDG descriptions")
        return self.original_sdgs
    
    def summarize_sdg_descriptions(self, max_length=150, min_length=50):
        """Summarize each SDG description."""
        print("Summarizing SDG descriptions...")
        
        for sdg_num, original_text in self.original_sdgs.items():
            print(f"Summarizing SDG {sdg_num}...")
            
            # Split long text into chunks if needed (BART has token limits)
            max_chunk_length = 1024
            if len(original_text) > max_chunk_length:
                # Take the first part of the description (usually contains the main goal)
                text_to_summarize = original_text[:max_chunk_length]
            else:
                text_to_summarize = original_text
            
            try:
                # Generate summary
                summary_result = self.summarizer(
                    text_to_summarize,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                
                summarized_text = summary_result[0]['summary_text']
                self.summarized_sdgs[sdg_num] = summarized_text
                
                print(f"SDG {sdg_num} summarized: {len(original_text)} -> {len(summarized_text)} chars")
                
            except Exception as e:
                print(f"Error summarizing SDG {sdg_num}: {e}")
                # Fallback: use first sentence
                sentences = original_text.split('. ')
                self.summarized_sdgs[sdg_num] = sentences[0] + '.'
        
        return self.summarized_sdgs
    
    def create_embeddings(self):
        """Create embeddings for both original and summarized SDG descriptions."""
        print("Creating embeddings...")
        
        # Original embeddings
        original_texts = [self.original_sdgs[i] for i in range(1, 18)]
        self.original_embeddings = self.embedding_model.encode(original_texts)
        
        # Summarized embeddings  
        summarized_texts = [self.summarized_sdgs[i] for i in range(1, 18)]
        self.summarized_embeddings = self.embedding_model.encode(summarized_texts)
        
        print(f"Created embeddings: Original shape {self.original_embeddings.shape}")
        print(f"Created embeddings: Summarized shape {self.summarized_embeddings.shape}")
        
        return self.original_embeddings, self.summarized_embeddings
    
    def save_summarized_descriptions(self):
        """Save the summarized descriptions to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save as CSV
        summary_data = []
        for sdg_num in range(1, 18):
            summary_data.append({
                'sdg': sdg_num,
                'original_text': self.original_sdgs[sdg_num],
                'summarized_text': self.summarized_sdgs[sdg_num],
                'original_length': len(self.original_sdgs[sdg_num]),
                'summarized_length': len(self.summarized_sdgs[sdg_num]),
                'compression_ratio': len(self.summarized_sdgs[sdg_num]) / len(self.original_sdgs[sdg_num])
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = f'results/sdg_summaries_{timestamp}.csv'
        summary_df.to_csv(csv_path, index=False)
        
        # Save as JSON for easy loading
        json_path = f'results/sdg_summaries_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'original_sdgs': self.original_sdgs,
                'summarized_sdgs': self.summarized_sdgs,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"Saved summarized descriptions to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        return csv_path, json_path
    
    def compare_descriptions(self):
        """Compare original vs summarized descriptions."""
        print("\n" + "="*60)
        print("SDG DESCRIPTION COMPARISON")
        print("="*60)
        
        for sdg_num in range(1, 18):
            original = self.original_sdgs[sdg_num]
            summarized = self.summarized_sdgs[sdg_num]
            
            print(f"\nSDG {sdg_num}:")
            print(f"Original ({len(original)} chars):")
            print(f"  {original[:200]}...")
            print(f"Summarized ({len(summarized)} chars):")
            print(f"  {summarized}")
            print(f"Compression: {len(summarized)/len(original):.2%}")

def main():
    """Main execution function."""
    print("="*60)
    print("SDG DESCRIPTION SUMMARIZATION EXPERIMENT")
    print("="*60)
    
    # Initialize summarizer
    summarizer = SDGSummarizer()
    
    # Load original descriptions
    original_sdgs = summarizer.load_sdg_descriptions()
    
    # Summarize descriptions
    summarized_sdgs = summarizer.summarize_sdg_descriptions()
    
    # Create embeddings
    original_embeddings, summarized_embeddings = summarizer.create_embeddings()
    
    # Save results
    csv_path, json_path = summarizer.save_summarized_descriptions()
    
    # Show comparison
    summarizer.compare_descriptions()
    
    print("\n" + "="*60)
    print("SUMMARIZATION COMPLETE!")
    print("="*60)
    print(f"Next step: Run classification experiments with summarized embeddings")
    print(f"Summarized descriptions saved to: {csv_path}")

if __name__ == "__main__":
    main()
