# Notes
https://www.sciencedirect.com/science/article/pii/S0169023X24001290
https://www.researchgate.net/publication/369126974_Mapping_Research_to_the_Sustainable_Development_Goals_A_Contextualised_Approach
Synthetic Data Generation in Low-Resource Settings via Fine-Tuning of Large Language Models
https://arxiv.org/abs/2310.01119


## SDG Dataset Combination Analysis (July 28, 2025)

### Combined Dataset Creation Results:
```
Found 17 CSV files: ['1.csv', '10.csv', '11.csv', '12.csv', '13.csv', '14.csv', '15.csv', '16.csv', '17.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv']
Processed 1.csv: 45 records
Processed 10.csv: 4469 records
Processed 11.csv: 205 records
Processed 12.csv: 269 records
Processed 13.csv: 45 records
Processed 14.csv: 42 records
Processed 15.csv: 13 records
Processed 16.csv: 19 records
Processed 17.csv: 17 records
Processed 2.csv: 55 records
Processed 3.csv: 134 records
Processed 4.csv: 90 records
Processed 5.csv: 89 records
Processed 6.csv: 124 records
Processed 7.csv: 128 records
Processed 8.csv: 25 records
Processed 9.csv: 13 records

Combined dataset saved to: /Users/mahnoorzamir/Desktop/mitacs/project/combined_sdg_dataset.csv
Total unique papers: 5708
Total original records: 5782

SDG Distribution:
SDG 1: 45 papers
SDG 2: 55 papers
SDG 3: 133 papers
SDG 4: 90 papers
SDG 5: 89 papers
SDG 6: 124 papers
SDG 7: 128 papers
SDG 8: 25 papers
SDG 9: 13 papers
SDG 10: 4465 papers
SDG 11: 205 papers
SDG 12: 268 papers
SDG 13: 45 papers
SDG 14: 42 papers
SDG 15: 13 papers
SDG 16: 19 papers
SDG 17: 17 papers

Papers with multiple SDG labels: 63
```

### Key Dataset Insights:
- **Total papers**: 5,708 unique papers (74 duplicates removed from 5,782 original records)
- **Single SDG papers**: 5,645 (98.90%)
- **Multiple SDG papers**: 63 (1.10%)
- **SDG distribution breakdown**:
  - 1 SDG: 5,645 papers (98.90%)
  - 2 SDGs: 53 papers (0.93%)
  - 3 SDGs: 9 papers (0.16%)
  - 4 SDGs: 1 paper (0.02%)
- **Average SDGs per paper**: 1.01

### Most Common Single SDGs:
1. SDG 10 (Reduced Inequalities): 4,450 papers (78.83%)
2. SDG 12 (Responsible Production): 257 papers (4.55%)
3. SDG 11 (Sustainable Cities): 192 papers (3.40%)
4. SDG 6 (Clean Water): 118 papers (2.09%)
5. SDG 3 (Good Health): 116 papers (2.05%)

### Most Common Multi-label Combinations:
1. SDGs 3,5 (Health & Gender): 6 papers
2. SDGs 3,11 (Health & Cities): 4 papers
3. SDGs 6,7 (Water & Energy): 3 papers

### Dataset Structure:
- Columns: Title, Abstract, SDG_Labels (comma-separated), SDG_1 through SDG_17 (binary)
- Ready for multi-label classification tasks
- Highly imbalanced dataset with SDG 10 dominating (78.22% of all papers)

## Reflections & Next Steps
- The data extraction step is now robust for the provided CSV
- Combined SDG dataset created successfully for multi-label classification
- Dataset shows significant class imbalance - may need stratified sampling or balancing techniques
- Next: Manual/augmented QA dataset creation and continued research logging 
