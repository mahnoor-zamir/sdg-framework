# SDG Multi-Label Classification Dataset

A comprehensive project for creating and analyzing SDG (Sustainable Development Goals) multi-label classification datasets using text similarity and embeddings.

## Project Structure

```
project/
  data/
    raw/                              # Raw data files
      Global-Indicator-Framework-after-2025-review-English - A.RES.71.313 Annex.csv
      osdg-community-dataset-v21-09-30.csv
    processed/                        # Processed datasets
      osdg_multilabel_threshold_0.6.csv          # Filtered OSDG dataset
      sdg_paragraph_dataset.csv                  # Clean SDG descriptions
      similarity_multilabel_embeddings_p0.4_s0.3.csv  # Final multi-label dataset
    scopus-test-set/                  # Test data (17 CSV files)
    sdg_structured.json               # Hierarchical SDG data
  scripts/                            # Processing scripts
    data_collection.py                # Original data collection
    data_processing.py                # Multi-label conversion with filtering
    __pycache__/                      # Python cache
  research_log/
    notes.md                          # Research notes and progress
  analyze_dataset.py                  # Dataset analysis script
  create_sdg_paragraphs.py            # SDG paragraph dataset creation
  create_similarity_multilabel.py    # Similarity-based multi-label assignment
  extract_sdg_from_pdf.py            # PDF extraction for SDG data
  robust_model_comparison_results.json # Model comparison results
  requirements.txt
  README.md
```

## Project Overview

This project creates high-quality multi-label SDG classification datasets by combining:
- **OSDG Community Dataset**: Text excerpts with SDG annotations
- **Official SDG Documentation**: Clean paragraph descriptions from UN documents
- **Similarity-Based Labeling**: Using sentence embeddings for semantic matching

## Key Accomplishments

### 1. Data Processing & Filtering
- Processed 32,120 OSDG texts → filtered to 17,248 high-quality samples
- Applied agreement threshold ≥ 0.6 and positive > negative label filtering
- Extracted and cleaned SDG descriptions from official UN documents

### 2. Multi-Label Dataset Creation
- **Final Dataset**: `similarity_multilabel_embeddings_p0.4_s0.3.csv`
- **17,248 texts** with semantically-assigned SDG labels
- **Average 2.53 labels per text** (59% multi-label, 22% single-label)
- Used **all-MiniLM-L6-v2** embeddings for similarity calculation

### 3. Advanced Similarity Approach
- Dual-threshold system (primary: 0.4, secondary: 0.3)
- Cosine similarity between text embeddings and SDG descriptions
- Preserves semantic relationships better than keyword-based approaches

## Datasets

### Input Datasets
1. **OSDG Community Dataset** (`osdg-community-dataset-v21-09-30.csv`)
   - 32,120 text excerpts with SDG annotations
   - Agreement scores and positive/negative labels
   - Source: OSDG Community Project

2. **SDG Official Documentation**
   - Extracted from UN A.RES.71.313 Annex
   - Clean paragraph descriptions for all 17 SDGs
   - Comprehensive goal and target information

### Output Datasets
1. **Filtered OSDG Dataset** (`osdg_multilabel_threshold_0.6.csv`)
   - 17,248 high-quality texts (agreement ≥ 0.6)
   - Multi-hot vectors for original labels
   - Quality-filtered for reliable training data

2. **SDG Paragraph Dataset** (`sdg_paragraph_dataset.csv`)
   - 17 clean SDG descriptions
   - Combined goals and targets into coherent paragraphs
   - Reference dataset for similarity calculations

3. **Final Multi-Label Dataset** (`similarity_multilabel_embeddings_p0.4_s0.3.csv`)
   - 17,248 texts with embedding-based SDG labels
   - Primary and secondary label assignments
   - Detailed similarity scores and statistics

## Technical Implementation

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Dimensions**: 384
- **Size**: ~90MB
- **Performance**: Optimized for semantic similarity tasks

### Processing Pipeline
1. **Text Preprocessing**: Cleaning and standardization
2. **Embedding Generation**: Batch processing with progress tracking
3. **Similarity Calculation**: Cosine similarity between embeddings
4. **Multi-Label Assignment**: Dual-threshold classification
5. **Quality Analysis**: Comprehensive statistics and validation

### Key Scripts

#### `create_similarity_multilabel.py`
Main script for creating embedding-based multi-label datasets.
```bash
python create_similarity_multilabel.py --method embeddings --primary-threshold 0.4 --secondary-threshold 0.3
```

#### `scripts/data_processing.py`
Filters OSDG dataset and creates multi-hot vectors.
```bash
python scripts/data_processing.py
```

#### `create_sdg_paragraphs.py`
Extracts and cleans SDG descriptions from official documents.
```bash
python create_sdg_paragraphs.py
```

## Setup and Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn sentence-transformers
```

### Quick Start
1. **Filter OSDG Dataset**:
   ```bash
   python scripts/data_processing.py
   ```

2. **Create SDG Reference Dataset**:
   ```bash
   python create_sdg_paragraphs.py
   ```

3. **Generate Multi-Label Dataset**:
   ```bash
   python create_similarity_multilabel.py --method embeddings --primary-threshold 0.4 --secondary-threshold 0.3
   ```

### Customization
- **Adjust thresholds**: Change `--primary-threshold` and `--secondary-threshold` for different label distributions
- **Switch methods**: Use `--method tfidf` for TF-IDF-based similarity (fallback option)
- **Limit labels**: Use `--max-labels` to control maximum labels per text

## Results Summary

| Metric | Value |
|--------|-------|
| Total Texts | 17,248 |
| Avg Labels/Text | 2.53 |
| Multi-label Texts | 10,208 (59%) |
| Single-label Texts | 3,817 (22%) |
| No Labels | 3,223 (19%) |
| Avg Max Similarity | 0.395 |

## File Outputs

- **CSV Dataset**: Main dataset for machine learning workflows
- **JSON Dataset**: Full dataset with metadata and similarity scores
- **Statistics**: Detailed analysis of label distribution and quality metrics
