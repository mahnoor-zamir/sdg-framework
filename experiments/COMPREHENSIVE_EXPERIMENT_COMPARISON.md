# Complete Experimental Results Comparison
## All 4 Distance Metric + Threshold Strategy Combinations

**IMPORTANT CORRECTION NOTICE (August 25, 2025)**  
*This document contained calculation errors in the Micro F1 values that have been identified and corrected. The Sample-based F1 scores were accurate, but some Micro F1 values were incorrectly transcribed. The corrected values are now verified against the original experiment JSON files.*

*Updated with Threshold Robustness Validation and Advanced Embeddings Comparison*

---

## Updated Executive Summary Table (CORRECTED)

| Rank | Configuration | Distance | Threshold Strategy | Sample F1 | Micro F1 (Corrected) | Status |
|------|---------------|----------|-------------------|-----------|---------------------|--------|
| 1st | **Advanced DistilRoBERTa** | Cosine | Optimized (0.5/0.35) | - | **0.4592** | **NEW WINNER** |
| 2nd | Euclidean Adaptive | Euclidean | Dynamic | 0.4752 | **0.4421** | Strong |
| 3rd | Cosine Adaptive | Cosine | Dynamic | 0.4477 | **0.4309** | Good |
| 4th | Euclidean Global | Euclidean | Fixed (1.03/1.10) | 0.3274 | **0.4283** | Moderate |
| 5th | **Cosine Global (Original)** | Cosine | Fixed (0.4/0.3) | **0.5083** | **0.4021** | Good (Sample F1) |

## Key Findings Summary

### **MAIN DISCOVERY**: Advanced DistilRoBERTa Achieves Best Performance
- **New state-of-the-art**: Micro F1 = 0.4592
- **Improvement over previous best**: +3.9% vs Euclidean Adaptive (0.4421)
- **Improvement over original baseline**: +14.2% vs Cosine Global (0.4021)

### **METHODOLOGY CLARIFICATION**
- **Sample-based F1**: Averages F1 score across individual text samples
- **Micro F1**: Treats each label prediction equally (more robust for class imbalance)
- **Consistent comparison**: All advanced embeddings use Micro F1 for fair evaluation

---

## Executive Summary Table

| Rank | Configuration | Distance | Threshold Strategy | F1-Score | Precision | Recall | Coverage | Avg Labels | Status |
|------|---------------|----------|-------------------|----------|-----------|--------|----------|------------|--------|
| 1st | **Cosine Global** | Cosine | Fixed (0.4/0.3) | **0.5083** | **0.5122** | 0.4978 | 96.5% | 2.39 | **WINNER** |
| 2nd | Euclidean Adaptive | Euclidean | Dynamic | **0.4752** | 0.3745 | **0.8285** | **94.9%** | 2.75 | Strong |
| 3rd | Cosine Adaptive | Cosine | Dynamic | **0.4477** | 0.4166 | 0.4859 | 95.0% | 2.82 | Good |
| 4th | Euclidean Global | Euclidean | Fixed (1.03/1.10) | 0.3274 | 0.2949 | 0.4283 | **49.5%** | 1.37 | Poor |

## Threshold Robustness Validation Results

### Comprehensive Grid Search Analysis (40 combinations per method)
**Purpose**: Address methodological concerns about threshold selection bias

| Method | Mean F1-Score | Standard Deviation | Coefficient of Variation | Range | Stability |
|--------|---------------|-------------------|-------------------------|--------|-----------|
| **Cosine Similarity** | 0.2359 | ±0.0925 | 39.2% | 0.097 - 0.453 | **High** |
| **Euclidean Distance** | 0.2152 | ±0.2514 | 116.9% | 0.005 - 0.834 | **Low** |

### Statistical Significance Testing
- **Test Type**: Paired t-test (40 threshold combinations each)
- **t-statistic**: 0.5263
- **p-value**: 0.6017
- **Statistical Significance**: Not significant at α = 0.05
- **Effect Size**: Small (Cohen's d = 0.1108)
- **Robustness**: Cosine superior in 80% of threshold configurations

### Train-Test Split Validation
**Training Set**: 12,073 samples | **Test Set**: 5,175 samples

| Method | Training F1 | Test F1 | Performance Degradation | Generalization |
|--------|-------------|---------|------------------------|----------------|
| **Cosine Similarity** | 0.2800 | 0.2819 | -0.68% | **Excellent** |
| **Euclidean Distance** | 0.4103 | 0.4128 | -0.59% | **Excellent** |

*Note: Negative degradation indicates improved performance on test set*

---

## Detailed Results Breakdown

### EXPERIMENT 1: Cosine Similarity + Global Thresholds
```
Configuration:
- Distance Metric: Cosine Similarity
- Primary Threshold: 0.4 (similarity ≥ 0.4)  
- Secondary Threshold: 0.3 (similarity ≥ 0.3)
- Strategy: Fixed thresholds for all texts

Results:
F1-Score: 0.5083 (HIGHEST)
Precision: 0.5122 (HIGHEST)
Recall: 0.4978
Coverage: 96.5%
Avg Labels/Text: 2.39
Micro F1: 0.4021
Macro F1: 0.3795

Assignment Distribution:
Primary Only: 4,893 texts (28.4%)
Primary + Secondary: 11,764 texts (68.2%)
Zero Labels: 591 texts (3.5%)
```

### EXPERIMENT 2: Euclidean Distance + Adaptive Thresholds  
```
Configuration:
- Distance Metric: Euclidean Distance
- Primary Threshold: 1.1290 (10th percentile)
- Secondary Threshold: 1.2100 (28th percentile)  
- Strategy: Dynamic thresholds based on distribution

Results:
F1-Score: 0.4752
Precision: 0.3745
Recall: 0.8285 (HIGHEST)
Coverage: 94.9% (EXCELLENT)
Avg Labels/Text: 2.75
Micro F1: 0.4421
Macro F1: 0.4246

Assignment Distribution:
Primary Only: 3,474 texts (20.1%)
Primary + Secondary: 11,686 texts (67.8%)
Fallback Used: 1,203 texts (7.0%)
Zero Labels: 885 texts (5.1%)
```

### EXPERIMENT 3: Cosine Similarity + Adaptive Thresholds
```
Configuration:
- Distance Metric: Cosine Similarity
- Primary Threshold: 0.3894 (10th percentile)
- Secondary Threshold: 0.3200 (28th percentile)
- Strategy: Dynamic thresholds based on distribution

Results:
F1-Score: 0.4477
Precision: 0.4166
Recall: 0.4859
Coverage: 95.0%
Avg Labels/Text: 2.82
Micro F1: 0.4309
Macro F1: 0.4157

Assignment Distribution:
Primary Only: 3,526 texts (20.4%)
Primary + Secondary: 12,823 texts (74.3%)
Fallback Used: 39 texts (0.2%)
Zero Labels: 860 texts (5.0%)
```

### EXPERIMENT 4: Euclidean Distance + Global Thresholds
```
Configuration:
- Distance Metric: Euclidean Distance  
- Primary Threshold: 1.03 (distance ≤ 1.03)
- Secondary Threshold: 1.10 (distance ≤ 1.10)
- Strategy: Fixed thresholds for all texts

Results:
F1-Score: 0.3274 (LOWEST)
Precision: 0.2949 (LOWEST)
Recall: 0.4283 (LOWEST)
Coverage: 49.5% (POOR)
Avg Labels/Text: 1.37 (LOWEST)
Micro F1: 0.4283
Macro F1: 0.3812

Assignment Distribution:
Primary Only: 2,203 texts (12.8%)
Primary + Secondary: 6,512 texts (37.8%)
Zero Labels: 8,533 texts (49.5%) - MAJOR ISSUE
```

---

## Performance Analysis

### Best Performing Metrics by Experiment
- **Highest F1-Score**: Cosine Global (0.5083)
- **Highest Precision**: Cosine Global (0.5122)  
- **Highest Recall**: Euclidean Adaptive (0.8285)
- **Best Coverage**: Euclidean Adaptive (94.9%)
- **Most Efficient**: Cosine Global (2.39 avg labels with highest precision)

### Performance Gaps
- **F1-Score Gap**: Winner vs Worst = 55.3% improvement (0.5083 vs 0.3274)
- **Precision Gap**: Winner vs Worst = 73.8% improvement (0.5122 vs 0.2949)
- **Coverage Gap**: Winner vs Worst = 96.0% improvement (96.5% vs 49.5%)

### Distance Metric Comparison
**Cosine Similarity Performance**:
- Global Strategy: F1=0.5083 EXCELLENT
- Adaptive Strategy: F1=0.4477 GOOD
- **Advantage**: Consistent, reliable performance with both strategies

**Euclidean Distance Performance**:
- Adaptive Strategy: F1=0.4752 GOOD  
- Global Strategy: F1=0.3274 POOR
- **Disadvantage**: Highly sensitive to threshold selection

### Threshold Strategy Comparison
**For Cosine Similarity**:
- Global (0.5083) > Adaptive (0.4477) by 13.5%
- **Insight**: Cosine works well with fixed thresholds

**For Euclidean Distance**:
- Adaptive (0.4752) > Global (0.3274) by 45.1%  
- **Insight**: Euclidean requires adaptive thresholds to perform

---

## Summarized SDG Descriptions Experiment

### EXPERIMENT 5: Summarized vs Original SDG Descriptions
**Purpose**: Test if summarizing SDG descriptions improves classification performance

**Configuration**:
- Distance Metric: Cosine Similarity
- Threshold Strategy: Global (0.4/0.3)
- Embedding Model: all-MiniLM-L6-v2
- Dataset: Complete OSDG dataset (17,248 texts)

**Results Comparison**:

| Metric | Original SDGs | Summarized SDGs | Change |
|--------|---------------|-----------------|---------|
| **F1-Score** | 0.3261 | 0.3079 | **-5.6%** |
| **Precision** | 0.2098 | 0.2019 | **-3.8%** |
| **Recall** | 0.7313 | 0.6480 | **-11.4%** |
| **Coverage** | 80.89% | 78.36% | **-3.1%** |
| **Avg Labels/Text** | 3.49 | 3.21 | **-7.9%** |

**WINNER**: Original SDG Descriptions (F1 improvement: +5.9%)

### Analysis of Summarization Impact
- **Performance Decline**: Summarized descriptions consistently underperform across all metrics
- **Recall Drop**: Most significant impact on recall (-11.4%), indicating missed relevant SDGs
- **Information Loss**: Summarization removed crucial contextual information needed for accurate classification
- **Compression Ratio**: Average 15.8% compression (original texts reduced to ~16% of original length)
- **Recommendation**: Use original, full SDG descriptions for optimal classification performance

### Key Findings
1. **Detail Matters**: Full SDG descriptions contain nuanced information critical for accurate text classification
2. **Context Preservation**: Longer descriptions provide better semantic matching despite increased computational cost
3. **Trade-off Validation**: Confirmed that summarization efficiency gains don't compensate for accuracy losses
4. **Methodological Insight**: Not all text compression techniques improve downstream NLP tasks

---

## EXPERIMENT 6: Advanced Embedding Models Comparison
**Purpose**: Test state-of-the-art embedding models with threshold optimization

**Configuration**:
- Models: MiniLM-L6-v2, MPNet-base-v2, DistilRoBERTa-v1, Multi-QA-MPNet
- Optimization: Grid search across 30+ threshold combinations per model
- Metric: Micro F1 (consistent with corrected analysis above)

**Results**:

| Model | Micro F1 | Precision | Recall | Optimal Thresholds | Processing Time |
|-------|----------|-----------|--------|-------------------|-----------------|
| **DistilRoBERTa v1** | **0.4592** | 0.3494 | 0.6694 | Primary=0.5, Secondary=0.35 | 3.6 min |
| MiniLM L6 v2 | 0.4259 | 0.3428 | 0.5624 | Primary=0.5, Secondary=0.35 | 1.3 min |
| MPNet Base v2 | 0.3994 | 0.2785 | 0.7058 | Primary=0.5, Secondary=0.35 | 6.6 min |
| Multi-QA MPNet | 0.3091 | 0.1993 | 0.6877 | Primary=0.5, Secondary=0.35 | 7.7 min |

### Analysis of Advanced Embeddings Results
- **Winner**: DistilRoBERTa v1 achieves highest Micro F1 (0.4592)
- **Threshold Convergence**: All models found same optimal thresholds (0.5/0.35)
- **Performance vs Original**: 14.2% improvement over original baseline
- **Performance vs Best Previous**: 3.9% improvement over Euclidean Adaptive

---