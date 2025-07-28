import pandas as pd
import os
from pathlib import Path

def combine_sdg_csvs(folder_path, output_path):
    """
    Combine multiple SDG CSV files into one dataset for multi-label classification.
    
    Args:
        folder_path (str): Path to the folder containing SDG CSV files
        output_path (str): Path for the output combined CSV file
    """
    
    # Initialize list to store all data
    all_data = []
    
    # Get all CSV files in the folder
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    for csv_file in csv_files:
        # Extract SDG number from filename (e.g., "1.csv" -> 1)
        sdg_number = int(csv_file.split('.')[0])
        
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_cols = ['Title', 'Abstract']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {csv_file}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            # Extract only the required columns
            subset_df = df[['Title', 'Abstract']].copy()
            
            # Add SDG number column
            subset_df['SDG'] = sdg_number
            
            # Remove rows with missing title or abstract
            subset_df = subset_df.dropna(subset=['Title', 'Abstract'])
            
            all_data.append(subset_df)
            
            print(f"Processed {csv_file}: {len(subset_df)} records")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not all_data:
        print("No data was successfully processed!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by Title and Abstract to handle multi-label cases
    # Some papers might appear in multiple SDG files
    multi_label_df = combined_df.groupby(['Title', 'Abstract'])['SDG'].apply(list).reset_index()
    
    # Convert SDG lists to comma-separated strings for easier handling
    multi_label_df['SDG_Labels'] = multi_label_df['SDG'].apply(lambda x: ','.join(map(str, sorted(x))))
    
    # Create binary columns for each SDG (optional - useful for some ML frameworks)
    for i in range(1, 18):  # SDGs 1-17
        multi_label_df[f'SDG_{i}'] = multi_label_df['SDG'].apply(lambda x: 1 if i in x else 0)
    
    # Drop the intermediate SDG column
    multi_label_df = multi_label_df.drop('SDG', axis=1)
    
    # Save the combined dataset
    multi_label_df.to_csv(output_path, index=False)
    
    print(f"\nCombined dataset saved to: {output_path}")
    print(f"Total unique papers: {len(multi_label_df)}")
    print(f"Total original records: {len(combined_df)}")
    
    # Print some statistics
    print("\nSDG Distribution:")
    for i in range(1, 18):
        count = multi_label_df[f'SDG_{i}'].sum()
        print(f"SDG {i}: {count} papers")
    
    # Show multi-label statistics
    label_counts = multi_label_df['SDG_Labels'].value_counts()
    multi_label_papers = len(multi_label_df[multi_label_df['SDG_Labels'].str.contains(',')])
    print(f"\nPapers with multiple SDG labels: {multi_label_papers}")
    
    return multi_label_df

# Usage
if __name__ == "__main__":
    # Set your paths
    folder_path = "/Users/mahnoorzamir/Desktop/mitacs/project/scopus-test-set"
    output_path = "/Users/mahnoorzamir/Desktop/mitacs/project/combined_sdg_dataset.csv"
    
    # Combine the datasets
    combined_data = combine_sdg_csvs(folder_path, output_path)
    
    if combined_data is not None:
        print("\nFirst few rows of the combined dataset:")
        print(combined_data[['Title', 'Abstract', 'SDG_Labels']].head())
        
        print("\nDataset shape:", combined_data.shape)
        print("\nColumn names:", list(combined_data.columns))
