import pandas as pd

def analyze_sdg_distribution(file_path):
    """
    Analyze the distribution of single vs multiple SDG labels in the combined dataset.
    """
    # Read the combined dataset
    df = pd.read_csv(file_path)
    
    print(f"Total number of papers in the dataset: {len(df)}")
    print(f"Dataset shape: {df.shape}")
    print()
    
    # Analyze SDG_Labels column
    print("=== SDG Label Distribution Analysis ===")
    
    # Count papers with single vs multiple SDGs
    single_sdg = df[~df['SDG_Labels'].str.contains(',')].copy()
    multiple_sdg = df[df['SDG_Labels'].str.contains(',')].copy()
    
    print(f"Papers with single SDG: {len(single_sdg)} ({len(single_sdg)/len(df)*100:.2f}%)")
    print(f"Papers with multiple SDGs: {len(multiple_sdg)} ({len(multiple_sdg)/len(df)*100:.2f}%)")
    print()
    
    # Show distribution of number of SDGs per paper
    df['num_sdgs'] = df['SDG_Labels'].str.split(',').str.len()
    sdg_count_dist = df['num_sdgs'].value_counts().sort_index()
    
    print("=== Distribution by Number of SDGs per Paper ===")
    for num_sdgs, count in sdg_count_dist.items():
        percentage = (count / len(df)) * 100
        print(f"{num_sdgs} SDG(s): {count} papers ({percentage:.2f}%)")
    print()
    
    # Show most common single SDGs
    print("=== Most Common Single SDG Labels ===")
    single_sdg_counts = single_sdg['SDG_Labels'].value_counts()
    for sdg, count in single_sdg_counts.head(10).items():
        percentage = (count / len(single_sdg)) * 100
        print(f"SDG {sdg}: {count} papers ({percentage:.2f}% of single-SDG papers)")
    print()
    
    # Show most common multiple SDG combinations
    print("=== Most Common Multiple SDG Combinations ===")
    if len(multiple_sdg) > 0:
        multiple_sdg_counts = multiple_sdg['SDG_Labels'].value_counts()
        for sdg_combo, count in multiple_sdg_counts.head(10).items():
            percentage = (count / len(multiple_sdg)) * 100
            print(f"SDGs {sdg_combo}: {count} papers ({percentage:.2f}% of multi-SDG papers)")
    else:
        print("No papers with multiple SDG labels found.")
    print()
    
    # Calculate total SDG assignments (sum of all binary columns)
    binary_cols = [f'SDG_{i}' for i in range(1, 18)]
    total_assignments = df[binary_cols].sum().sum()
    print(f"Total SDG assignments across all papers: {total_assignments}")
    print(f"Average SDGs per paper: {total_assignments/len(df):.2f}")
    print()
    
    # Show SDG frequency distribution
    print("=== Individual SDG Frequency Distribution ===")
    sdg_frequencies = df[binary_cols].sum().sort_values(ascending=False)
    for sdg_col, count in sdg_frequencies.items():
        sdg_num = sdg_col.split('_')[1]
        percentage = (count / len(df)) * 100
        print(f"SDG {sdg_num}: {count} papers ({percentage:.2f}%)")
    
    return df

if __name__ == "__main__":
    file_path = "/Users/mahnoorzamir/Desktop/mitacs/project/combined_sdg_dataset.csv"
    df = analyze_sdg_distribution(file_path)
