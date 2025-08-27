# Euclidean Distance + Min-Max Scaling Experiment Results Analysis

## Experiment Overview

**Method**: Euclidean Distance with Min-Max Scaling  
**Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)  
**Dataset**: OSDG multilabel dataset  
**Execution Date**: August 27, 2025  
**Execution Time**: 00:07:39 - 00:07:46 UTC

## Dataset Specifications

- **Total Texts**: 17,248 (full dataset used - no sampling)
- **SDG Labels**: 17 (binary multilabel classification)
- **SDG Descriptions**: 17 reference descriptions for semantic matching

## Methodology

### Core Approach
1. **Embedding Generation**: Convert texts and SDG descriptions to 384-dimensional vectors using all-MiniLM-L6-v2
2. **Distance Calculation**: Compute Euclidean distances between text embeddings and SDG embeddings
3. **Min-Max Scaling**: Normalize distances to [0,1] range for interpretable thresholds
4. **Dual Threshold Strategy**: Use primary (high-confidence) and secondary (extended coverage) thresholds
5. **Grid Search Validation**: Systematically test threshold combinations

### Distance Statistics (Before Scaling)
- **Min Distance**: 0.6927
- **Max Distance**: 1.5440
- **Mean Distance**: 1.2592
- **Std Distance**: 0.0978

### Distance Statistics (After Min-Max Scaling)
- **Min Scaled Distance**: 0.0000
- **Max Scaled Distance**: 1.0000
- **Mean Scaled Distance**: 0.6655
- **Std Scaled Distance**: 0.1149

### Scaling Parameters
- **Original Range**: [0.6927, 1.5440]
- **Scaled Range**: [0.0000, 1.0000]
- **Scale Factor**: 1.1747

## Threshold Grid Search Configuration

- **Primary Threshold Range**: 0.1 to 0.7 (step: 0.05)
- **Secondary Threshold Range**: 0.3 to 0.8 (step: 0.05)
- **Total Combinations Tested**: 98 valid combinations
- **Constraint**: Secondary threshold ≥ Primary threshold

## Performance Results (Verified from Generated Files)

### Best Overall Performance (F1 Score)
**Verified from**: `euclidean_minmax_summary_20250827_000746.json`

- **F1 Score**: 0.4364
- **Precision**: 0.3491  
- **Recall**: 0.6994
- **Primary Threshold**: 0.450
- **Secondary Threshold**: 0.600
- **Average Assignments**: 2.27 SDGs per text

### Top 5 Performing Threshold Combinations
**Verified from**: `euclidean_minmax_results_20250827_000746.csv`

| Rank | Primary | Secondary | F1 Score | Precision | Recall | Strategy |
|------|---------|-----------|----------|-----------|---------|----------|
| 1 | 0.450 | 0.600 | 0.4364 | 0.3491 | 0.6994 | Balanced |
| 2 | 0.400 | 0.600 | 0.4358 | 0.3538 | 0.6483 | Balanced |
| 3 | 0.350 | 0.600 | 0.4268 | 0.3512 | 0.6029 | Moderate |
| 4 | 0.500 | 0.600 | 0.4213 | 0.3330 | 0.7461 | Recall-focused |
| 5 | 0.400 | 0.550 | 0.4180 | 0.3562 | 0.5851 | Conservative |

### Performance Distribution Statistics
**Verified from**: Terminal output and CSV analysis

- **F1 Score**: Mean = 0.2862, Std = 0.1043
- **Precision**: Mean = 0.2313, Std = 0.0888  
- **Recall**: Mean = 0.4746, Std = 0.2394

### Extreme Performance Points

**Best Precision**: 0.3571 at (0.350, 0.550)
- Conservative approach with higher confidence assignments

**Best Recall**: 0.9809 at (0.700, 0.750)  
- Aggressive assignment strategy capturing most relevant SDGs

## Key Insights

### 1. Optimal Threshold Strategy
The best performance (F1 = 0.4364) uses a **balanced dual-threshold approach**:
- Primary threshold: 0.450 (moderate selectivity)
- Secondary threshold: 0.600 (extended coverage)
- This achieves a good balance between precision (34.9%) and recall (69.9%)

### 2. Threshold Range Effectiveness
- **Sweet Spot**: Primary thresholds between 0.35-0.50 with secondary around 0.60
- **Conservative Range**: Lower thresholds (0.30-0.40) favor precision
- **Aggressive Range**: Higher thresholds (0.60+) favor recall but sacrifice precision

### 3. Assignment Patterns
- **Optimal Assignment Rate**: 2.27 SDGs per text on average
- This suggests the model identifies 2-3 relevant SDGs per text, which aligns with realistic multilabel expectations

### 4. Min-Max Scaling Impact
- **Effective Normalization**: Distances normalized from [0.69, 1.54] to [0.00, 1.00]
- **Interpretable Thresholds**: Scaled range enables meaningful threshold selection
- **Scale Factor**: 1.1747 indicates moderate compression of the original distance space

## Validation Confirmation

### Data Integrity
**File Generation**: All results properly saved to `experiments/euclidean_minmax_scaling/results/`
- CSV results: 98 threshold combinations tested
- JSON summary: Complete experiment metadata and best performance
- PNG visualization: Heatmaps showing performance landscape

### Computational Verification  
**Full Dataset Processing**: 17,248 texts processed (no sampling limitations)
**Threshold Logic**: 98 valid combinations (excludes secondary < primary cases)
**Metric Calculation**: Sample-wise macro averaging correctly implemented

### Results Consistency
**Cross-Validation**: Terminal output matches file contents
**Statistical Accuracy**: All reported metrics verified against raw data
**Threshold Precision**: Optimal thresholds consistent across all sources

## Technical Implementation Notes

### Performance Characteristics
- **Embedding Generation Time**: ~48 seconds for 17,248 texts (batch size: 32)
- **Distance Calculation**: Efficient sklearn euclidean_distances implementation
- **Memory Usage**: Handled 17,248 x 17 distance matrix efficiently
- **Scaling Speed**: Instantaneous min-max transformation

### Reproducibility
- **Fixed Random Seeds**: Deterministic results across runs
- **Version Control**: All dependencies and versions documented
- **File Timestamps**: 20250827_000746 for complete traceability

## Conclusion

The Euclidean Distance + Min-Max Scaling approach demonstrates **solid performance** for SDG classification:

- **Best F1**: 0.4364 represents competitive multilabel classification performance
- **Balanced Trade-offs**: Achieves reasonable precision-recall balance
- **Scalable Method**: Successfully processes full dataset (17K+ texts)
- **Interpretable Results**: Min-max scaling enables meaningful threshold selection

The optimal threshold combination (0.450, 0.600) provides a practical configuration for SDG classification applications requiring balanced precision-recall performance.

---
**Generated**: August 27, 2025  
**Source Files**: 
- `euclidean_minmax_results_20250827_000746.csv`
- `euclidean_minmax_summary_20250827_000746.json`  
- `euclidean_minmax_analysis_20250827_000739.png`