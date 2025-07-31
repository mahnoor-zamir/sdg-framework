
"""
Convert single-SDG benchmark dataset to multi-SDG label dataset.

This script processes the benchmark.csv file and creates a multi-label dataset where:
- Each text can have multiple SDG labels (multi-hot encoding)
- Labels are represented as binary vectors for all 17 SDGs
- Uses content analysis to identify potential cross-SDG relevance
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
import json

# SDG descriptions and keywords for cross-labeling
SDG_KEYWORDS = {
    1: ["poverty", "poor", "income", "extreme poverty", "social protection", "hunger", "basic services"],
    2: ["hunger", "food security", "nutrition", "agriculture", "sustainable agriculture", "food production"],
    3: ["health", "medical", "healthcare", "disease", "mortality", "wellbeing", "mental health", "maternal"],
    4: ["education", "learning", "school", "literacy", "skills", "training", "knowledge", "students"],
    5: ["gender", "women", "girls", "equality", "discrimination", "violence against women", "empowerment"],
    6: ["water", "sanitation", "hygiene", "clean water", "wastewater", "water resources", "drinking water"],
    7: ["energy", "renewable", "electricity", "power", "fuel", "energy efficiency", "clean energy"],
    8: ["employment", "jobs", "economic growth", "decent work", "productivity", "unemployment", "labour"],
    9: ["infrastructure", "industry", "innovation", "technology", "research", "development", "manufacturing"],
    10: ["inequality", "income distribution", "social inclusion", "discrimination", "migration", "refugees"],
    11: ["cities", "urban", "housing", "transport", "sustainable cities", "settlements", "public spaces"],
    12: ["consumption", "production", "waste", "resources", "sustainable consumption", "circular economy"],
    13: ["climate", "climate change", "emissions", "greenhouse gas", "adaptation", "mitigation", "global warming"],
    14: ["marine", "ocean", "sea", "fisheries", "coastal", "marine pollution", "aquatic", "maritime"],
    15: ["forest", "biodiversity", "ecosystem", "land", "desertification", "species", "conservation"],
    16: ["peace", "justice", "institutions", "governance", "corruption", "violence", "rule of law"],
    17: ["partnership", "cooperation", "development cooperation", "global partnership", "capacity building"]
}

def analyze_text_for_sdgs(text, current_sdg=None, current_label=None):
    """
    Analyze text content to identify potential relevance to multiple SDGs.
    
    Args:
        text (str): The text to analyze
        current_sdg (int): The currently assigned SDG
        current_label (bool): Whether the text is labeled as relevant to current_sdg
    
    Returns:
        dict: Dictionary with SDG numbers as keys and relevance scores as values
    """
    text_lower = text.lower()
    sdg_scores = {}
    
    for sdg_num, keywords in SDG_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Count keyword occurrences with some weighting
            occurrences = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
            score += occurrences
        
        # Normalize score by text length (to avoid bias toward longer texts)
        normalized_score = score / (len(text.split()) / 100)  # per 100 words
        sdg_scores[sdg_num] = normalized_score
    
    return sdg_scores

def create_multi_label_dataset(input_file, output_file, threshold=0.1):
    """
    Convert single-label dataset to multi-label dataset.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        threshold (float): Minimum score threshold for secondary SDG assignment
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} rows")
    print("SDG distribution:")
    print(df['sdg'].value_counts().sort_index())
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Create multi-label dataset structure
    multi_label_data = []
    text_to_sdgs = defaultdict(set)  # Track which SDGs each text is relevant to
    
    # Group by text to handle potential duplicates
    text_groups = df.groupby('text')
    
    print("\nProcessing texts and analyzing cross-SDG relevance...")
    
    for text, group in text_groups:
        # Get original SDG assignments for this text
        original_sdgs = set()
        for _, row in group.iterrows():
            if row['label']:  # Only include positive labels
                original_sdgs.add(row['sdg'])
        
        # Analyze text for potential relevance to other SDGs
        sdg_scores = analyze_text_for_sdgs(text)
        
        # Create multi-hot label vector
        sdg_labels = [0] * 17  # Initialize all SDGs as 0
        
        # Set original positive labels
        for sdg in original_sdgs:
            sdg_labels[sdg - 1] = 1
        
        # Add secondary SDG labels based on content analysis
        for sdg_num, score in sdg_scores.items():
            if sdg_num not in original_sdgs and score > threshold:
                # Only add if it's not already labeled and meets threshold
                sdg_labels[sdg_num - 1] = 1
        
        # Create row for multi-label dataset
        row_data = {
            'id': group.iloc[0]['id'],  # Use first ID if multiple
            'text': text,
            'original_sdgs': list(original_sdgs),
            'total_sdgs': sum(sdg_labels),
            'sdg_scores': dict(sdg_scores)
        }
        
        # Add individual SDG columns
        for i in range(17):
            row_data[f'sdg_{i+1}'] = sdg_labels[i]
        
        multi_label_data.append(row_data)
    
    # Create DataFrame
    multi_df = pd.DataFrame(multi_label_data)
    
    # Statistics
    print(f"\nMulti-label dataset created: {len(multi_df)} unique texts")
    print(f"Average SDGs per text: {multi_df['total_sdgs'].mean():.2f}")
    print(f"Texts with multiple SDGs: {len(multi_df[multi_df['total_sdgs'] > 1])}")
    
    # SDG frequency in multi-label dataset
    sdg_counts = {}
    for i in range(17):
        sdg_counts[i+1] = multi_df[f'sdg_{i+1}'].sum()
    
    print("\nSDG frequency in multi-label dataset:")
    for sdg, count in sorted(sdg_counts.items()):
        print(f"SDG {sdg}: {count}")
    
    # Save to CSV
    # Prepare columns for CSV output
    csv_columns = ['id', 'text', 'total_sdgs'] + [f'sdg_{i+1}' for i in range(17)]
    csv_df = multi_df[csv_columns].copy()
    
    print(f"\nSaving multi-label dataset to {output_file}...")
    csv_df.to_csv(output_file, index=False)
    
    # Save detailed version with scores
    detailed_output = output_file.replace('.csv', '_detailed.json')
    print(f"Saving detailed data with scores to {detailed_output}...")
    multi_df.to_json(detailed_output, orient='records', indent=2)
    
    return multi_df

def analyze_cross_sdg_patterns(df):
    """Analyze patterns in cross-SDG relationships."""
    print("\n" + "="*50)
    print("CROSS-SDG RELATIONSHIP ANALYSIS")
    print("="*50)
    
    # Co-occurrence matrix
    cooccurrence = np.zeros((17, 17))
    
    for _, row in df.iterrows():
        active_sdgs = [i for i in range(17) if row[f'sdg_{i+1}'] == 1]
        for i in active_sdgs:
            for j in active_sdgs:
                if i != j:
                    cooccurrence[i][j] += 1
    
    # Find most common SDG pairs
    sdg_pairs = []
    for i in range(17):
        for j in range(i+1, 17):
            if cooccurrence[i][j] > 0:
                sdg_pairs.append((i+1, j+1, int(cooccurrence[i][j])))
    
    sdg_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop 10 SDG co-occurrences:")
    for i, (sdg1, sdg2, count) in enumerate(sdg_pairs[:10]):
        print(f"{i+1:2d}. SDG {sdg1} & SDG {sdg2}: {count} times")
    
    # Texts with most SDGs
    max_sdgs = df['total_sdgs'].max()
    most_connected = df[df['total_sdgs'] == max_sdgs]
    
    print(f"\nTexts with most SDGs ({max_sdgs} SDGs):")
    for _, row in most_connected.head(3).iterrows():
        print(f"ID: {row['id']}")
        active_sdgs = [i+1 for i in range(17) if row[f'sdg_{i+1}'] == 1]
        print(f"SDGs: {active_sdgs}")
        print(f"Text preview: {row['text'][:150]}...")
        print()

def main():
    """Main execution function."""
    input_file = "benchmark.csv"
    output_file = "benchmark_multi_sdg.csv"
    
    print("Converting single-SDG dataset to multi-SDG label dataset")
    print("="*60)
    
    # Create multi-label dataset
    multi_df = create_multi_label_dataset(input_file, output_file, threshold=0.15)
    
    # Analyze patterns
    analyze_cross_sdg_patterns(multi_df)
    
    print(f"\n✅ Conversion complete!")
    print(f"📄 Multi-label dataset saved to: {output_file}")
    print(f"📊 Detailed data with scores saved to: benchmark_multi_sdg_detailed.json")
    print(f"📈 Summary statistics displayed above")

if __name__ == "__main__":
    main()
