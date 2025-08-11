#!/usr/bin/env python3
"""
Test all-mpnet-base-v2 embedding model for SDG classification
Compares performance with the current all-MiniLM-L6-v2 baseline
"""

import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from pathlib import Path

def load_sdg_descriptions(sdg_file):
    """Load and format SDG descriptions"""
    with open(sdg_file, 'r') as f:
        sdg_data = json.load(f)
    
    sdg_descriptions = {}
    for sdg_num, sdg_info in sdg_data.items():
        # Create comprehensive description from name and first few targets
        description = f"SDG {sdg_num}: {sdg_info['name']}. "
        
        # Add first 3 targets for context
        targets = sdg_info['targets']
        for i, (target_key, target_data) in enumerate(targets.items()):
            if i >= 3:  # Limit to first 3 targets
                break
            description += f"Target {target_key}: {target_data['description']} "
        
        sdg_descriptions[sdg_num] = description.strip()
    
    return sdg_descriptions

def classify_with_embeddings(texts, sdg_descriptions, model_name, primary_threshold=0.4, secondary_threshold=0.3):
    """Classify texts using embedding similarity"""
    print(f"\n=== Testing {model_name} ===")
    
    # Load model
    start_time = time.time()
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create embeddings
    print("Creating embeddings...")
    start_time = time.time()
    
    # Embed texts
    text_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Embed SDG descriptions
    sdg_texts = list(sdg_descriptions.values())
    sdg_keys = list(sdg_descriptions.keys())
    sdg_embeddings = model.encode(sdg_texts, show_progress_bar=True, batch_size=32)
    
    embedding_time = time.time() - start_time
    print(f"Embeddings created in {embedding_time:.2f} seconds")
    
    # Calculate similarities
    print("Calculating similarities...")
    start_time = time.time()
    similarities = cosine_similarity(text_embeddings, sdg_embeddings)
    similarity_time = time.time() - start_time
    print(f"Similarities calculated in {similarity_time:.2f} seconds")
    
    results = []
    
    for i, text in enumerate(texts):
        text_similarities = similarities[i]
        max_similarity = np.max(text_similarities)
        
        # Find all SDGs above thresholds
        primary_sdgs = []
        secondary_sdgs = []
        
        for j, similarity in enumerate(text_similarities):
            sdg_num = sdg_keys[j]
            if similarity >= primary_threshold:
                primary_sdgs.append((sdg_num, similarity))
            elif similarity >= secondary_threshold:
                secondary_sdgs.append((sdg_num, similarity))
        
        # Sort by similarity score
        primary_sdgs.sort(key=lambda x: x[1], reverse=True)
        secondary_sdgs.sort(key=lambda x: x[1], reverse=True)
        
        # Combine assignments
        assigned_sdgs = [sdg for sdg, _ in primary_sdgs] + [sdg for sdg, _ in secondary_sdgs]
        
        results.append({
            'text': text,
            'assigned_sdgs': assigned_sdgs,
            'max_similarity_score': max_similarity,
            'primary_assignments': len(primary_sdgs),
            'secondary_assignments': len(secondary_sdgs),
            'total_assignments': len(assigned_sdgs)
        })
    
    total_time = load_time + embedding_time + similarity_time
    
    return results, {
        'model_name': model_name,
        'total_processing_time': total_time,
        'load_time': load_time,
        'embedding_time': embedding_time,
        'similarity_time': similarity_time,
        'embedding_dimensions': text_embeddings.shape[1],
        'total_texts': len(texts),
        'total_sdgs': len(sdg_descriptions)
    }

def compare_with_baseline(mpnet_results, baseline_file):
    """Compare mpnet results with baseline results"""
    # Load baseline results
    baseline_df = pd.read_csv(baseline_file)
    
    # Create comparison
    comparison = []
    
    for i, mpnet_result in enumerate(mpnet_results):
        if i < len(baseline_df):
            baseline_row = baseline_df.iloc[i]
            baseline_sdgs = str(baseline_row['similarity_assigned_sdgs']).strip('[]').replace("'", "").split(', ') if pd.notna(baseline_row['similarity_assigned_sdgs']) else []
            baseline_sdgs = [sdg.strip() for sdg in baseline_sdgs if sdg.strip()]
            
            mpnet_sdgs = mpnet_result['assigned_sdgs']
            
            # Calculate overlap
            baseline_set = set(baseline_sdgs)
            mpnet_set = set(mpnet_sdgs)
            
            overlap = len(baseline_set.intersection(mpnet_set))
            union = len(baseline_set.union(mpnet_set))
            jaccard = overlap / union if union > 0 else 0
            
            comparison.append({
                'text_index': i,
                'baseline_sdgs': baseline_sdgs,
                'mpnet_sdgs': mpnet_sdgs,
                'baseline_count': len(baseline_sdgs),
                'mpnet_count': len(mpnet_sdgs),
                'overlap_count': overlap,
                'jaccard_similarity': jaccard,
                'baseline_max_sim': baseline_row['max_similarity_score'],
                'mpnet_max_sim': mpnet_result['max_similarity_score']
            })
    
    return comparison

def main():
    # File paths
    data_dir = Path("data")
    sdg_file = data_dir / "sdg_structured.json"
    osdg_file = data_dir / "processed" / "osdg_multilabel_threshold_0.6.csv"
    baseline_file = data_dir / "processed" / "similarity_multilabel_embeddings_p0.4_s0.3.csv"
    
    # Load data
    print("Loading data...")
    sdg_descriptions = load_sdg_descriptions(sdg_file)
    osdg_df = pd.read_csv(osdg_file)
    
    # Test on a sample first (1000 texts for reasonable runtime)
    sample_size = 1000
    sample_texts = osdg_df['text'].head(sample_size).tolist()
    
    print(f"Testing on {sample_size} texts...")
    print(f"SDG descriptions loaded: {len(sdg_descriptions)}")
    
    # Test mpnet model
    mpnet_results, mpnet_stats = classify_with_embeddings(
        sample_texts, 
        sdg_descriptions, 
        'all-mpnet-base-v2',
        primary_threshold=0.4,
        secondary_threshold=0.3
    )
    
    # Calculate performance metrics
    total_assignments = sum(len(result['assigned_sdgs']) for result in mpnet_results)
    avg_assignments = total_assignments / len(mpnet_results)
    coverage = sum(1 for result in mpnet_results if len(result['assigned_sdgs']) > 0) / len(mpnet_results)
    avg_max_similarity = sum(result['max_similarity_score'] for result in mpnet_results) / len(mpnet_results)
    
    # Compare with baseline (if available)
    comparison = None
    if baseline_file.exists():
        print("\nComparing with baseline...")
        comparison = compare_with_baseline(mpnet_results, baseline_file)
        
        # Calculate comparison metrics
        avg_jaccard = sum(c['jaccard_similarity'] for c in comparison) / len(comparison)
        avg_baseline_sim = sum(c['baseline_max_sim'] for c in comparison) / len(comparison)
        avg_mpnet_sim = sum(c['mpnet_max_sim'] for c in comparison) / len(comparison)
        
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Average Jaccard similarity with baseline: {avg_jaccard:.3f}")
        print(f"Baseline avg max similarity: {avg_baseline_sim:.3f}")
        print(f"MPNet avg max similarity: {avg_mpnet_sim:.3f}")
        print(f"Similarity improvement: {(avg_mpnet_sim - avg_baseline_sim):.3f}")
    
    # Print results
    print(f"\n=== MPNET MODEL RESULTS ===")
    print(f"Model: {mpnet_stats['model_name']}")
    print(f"Embedding dimensions: {mpnet_stats['embedding_dimensions']}")
    print(f"Total processing time: {mpnet_stats['total_processing_time']:.2f} seconds")
    print(f"Average assignments per text: {avg_assignments:.2f}")
    print(f"Coverage (texts with assignments): {coverage:.1%}")
    print(f"Average max similarity score: {avg_max_similarity:.3f}")
    
    # Save results
    output_file = data_dir / "processed" / "mpnet_test_results_p0.4_s0.3.csv"
    results_df = pd.DataFrame([
        {
            'text': result['text'],
            'assigned_sdgs': result['assigned_sdgs'],
            'max_similarity_score': result['max_similarity_score'],
            'primary_assignments': result['primary_assignments'],
            'secondary_assignments': result['secondary_assignments'],
            'total_assignments': result['total_assignments']
        }
        for result in mpnet_results
    ])
    
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save statistics
    stats_file = data_dir / "processed" / "mpnet_test_stats.json"
    full_stats = {
        **mpnet_stats,
        'performance_metrics': {
            'average_assignments_per_text': avg_assignments,
            'coverage_percentage': coverage,
            'average_max_similarity': avg_max_similarity,
            'sample_size': sample_size,
            'primary_threshold': 0.4,
            'secondary_threshold': 0.3
        }
    }
    
    if comparison:
        full_stats['comparison_with_baseline'] = {
            'average_jaccard_similarity': avg_jaccard,
            'baseline_avg_max_similarity': avg_baseline_sim,
            'mpnet_avg_max_similarity': avg_mpnet_sim,
            'similarity_improvement': avg_mpnet_sim - avg_baseline_sim
        }
    
    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
