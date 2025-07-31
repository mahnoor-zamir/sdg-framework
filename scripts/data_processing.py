import pandas as pd
import numpy as np

def create_multilabel_dataset():
    # Read the CSV file with tab separator
    df = pd.read_csv('osdg-community-dataset-v21-09-30.csv', sep='\t')
    
    # Calculate confidence scores for each text-SDG pair
    df['confidence'] = (df['labels_positive'] - df['labels_negative']) / (df['labels_positive'] + df['labels_negative'])
    
    # Use a more inclusive approach to handle the single-label nature of this dataset
    # Option 1: Lower confidence threshold to 0.3 to include more labels
    # Option 2: Include original SDG even if confidence is low, but mark high confidence ones
    
    # Filter labels with confidence >= 0.3 (more inclusive)
    df_filtered = df[df['confidence'] >= 0.3]
    
    # For texts that don't meet the 0.3 threshold, include them if confidence > 0
    # This ensures no text is left completely unlabeled
    df_backup = df[(df['confidence'] > 0) & (df['confidence'] < 0.3)]
    
    # Create a multi-hot vector for each text
    # First, get unique text IDs
    unique_texts = df['text_id'].unique()
    
    # Create empty array for multi-hot vectors (texts × 17 SDGs)
    multilabel_matrix = np.zeros((len(unique_texts), 17))
    texts = []
    
    # For each text, set 1s for SDGs that meet confidence threshold
    for idx, text_id in enumerate(unique_texts):
        text_data = df[df['text_id'] == text_id]
        texts.append(text_data['text'].iloc[0])  # Store the text
        
        # Get high-confidence SDGs for this text (>= 0.3)
        valid_sdgs = df_filtered[df_filtered['text_id'] == text_id]['sdg'].values
        
        # If no high-confidence SDGs, use backup (low but positive confidence)
        if len(valid_sdgs) == 0:
            valid_sdgs = df_backup[df_backup['text_id'] == text_id]['sdg'].values
        
        # Set 1s in the multi-hot vector for valid SDGs
        # SDGs are 1-indexed, so subtract 1 for array indexing
        for sdg in valid_sdgs:
            multilabel_matrix[idx, sdg-1] = 1
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'text': texts,
        'labels': [[int(val) for val in x] for x in multilabel_matrix]  # Convert to regular integers
    })
    
    # Save to CSV
    output_df.to_csv('data/multilabel_dataset.csv', index=False)
    print(f"Created dataset with {len(output_df)} examples")
    
    # Print some statistics
    total_positive_labels = multilabel_matrix.sum(axis=0)
    
    # Calculate text length statistics
    output_df['char_length'] = output_df['text'].str.len()
    output_df['word_length'] = output_df['text'].str.split().str.len()
    
    print(f"\nText length statistics:")
    print(f"Average characters per text: {output_df['char_length'].mean():.1f}")
    print(f"Average words per text: {output_df['word_length'].mean():.1f}")
    print(f"Min words: {output_df['word_length'].min()}")
    print(f"Max words: {output_df['word_length'].max()}")
    
    print("\nLabel distribution:")
    for sdg in range(17):
        count = total_positive_labels[sdg]
        print(f"SDG {sdg+1}: {int(count)} texts ({count/len(output_df)*100:.1f}%)")

if __name__ == "__main__":
    create_multilabel_dataset()
