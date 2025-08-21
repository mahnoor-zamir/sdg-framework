# Cosine Similarity + MiniLM Experiments Summary

## Overview
This document summarizes the results of two systematic experiments comparing global vs adaptive thresholds for SDG classification using cosine similarity with the all-MiniLM-L6-v2 embedding model.

## Experimental Setup
- **Dataset**: 17,248 OSDG texts with multilabel SDG classifications
- **Embedding Model**: all-MiniLM-L6-v2 (sentence-transformers)
- **Distance Metric**: Cosine similarity
- **Date**: January 21, 2025

## Experiment 1: Global Fixed Thresholds
### Configuration
- Primary Threshold: 0.4
- Secondary Threshold: 0.3
- Max Labels per Text: 5
- Strategy: Traditional fixed thresholds

### Results
```
ASSIGNMENT STATISTICS:
  Average labels per text: 2.51
  Coverage rate: 80.9%
  Total assignments: 43,301
  Primary assignments: 17,872
  Secondary assignments: 27,064
  Zero-label texts: 3,296

EVALUATION METRICS:
Sample-based metrics:
  Precision: 0.4118
  Recall: 0.8725
  F1-Score: 0.5083
  Jaccard: 0.4118

Label-based metrics:
  Micro F1: 0.4021
  Macro F1: 0.3795

Conservation rate: 0.7058
Coverage rate: 0.8089
```

## Experiment 2: Adaptive Dynamic Thresholds
### Configuration
- Target Average Labels: 2.8
- Label Range: 1-4 per text
- Strategy: Dynamic percentile-based adaptive thresholds
- Coverage Guarantee: Yes
- Fallback Threshold: 0.25

### Calculated Thresholds
- Primary: 0.2679 (72nd percentile)
- Secondary: 0.2641 (71st percentile)
- Predicted avg labels: 2.84

### Results
```
ASSIGNMENT STATISTICS:
  Average labels per text: 2.76
  Coverage rate: 91.1%
  Total assignments: 47,527
  Primary assignments: 82,101
  Secondary assignments: 931
  Fallback assignments: 407
  Zero-label texts: 1,541

THRESHOLD USAGE:
  Primary Only: 14,420 texts (83.6%)
  Primary Secondary: 880 texts (5.1%)
  Fallback Used: 407 texts (2.4%)
  None: 1,541 texts (8.9%)

EVALUATION METRICS:
Sample-based metrics:
  Precision: 0.3452
  Recall: 0.8091
  F1-Score: 0.4477
  Jaccard: 0.3452
  Conservation: 0.8091

Label-based metrics:
  Micro F1: 0.4309
  Macro F1: 0.4157

Overall conservation rate: 0.8091
Coverage rate: 0.9107
```

## Comparative Analysis

### Key Metrics Comparison

| Metric | Global Threshold | Adaptive Threshold | Winner |
|--------|------------------|-------------------|---------|
| **F1-Score** | **0.5083** | 0.4477 | Global |
| **Precision** | **0.4118** | 0.3452 | Global |
| **Recall** | **0.8725** | 0.8091 | Global |
| **Micro F1** | 0.4021 | **0.4309** | Adaptive |
| **Macro F1** | 0.3795 | **0.4157** | Adaptive |
| **Coverage Rate** | 0.8089 | **0.9107** | Adaptive |
| **Conservation Rate** | 0.7058 | **0.8091** | Adaptive |
| **Avg Labels/Text** | 2.51 | **2.76** | Adaptive |

### Performance Trade-offs

#### Global Threshold Advantages:
1. **Higher Precision** (0.4118 vs 0.3452) - fewer false positives
2. **Higher Recall** (0.8725 vs 0.8091) - better at finding true positives  
3. **Superior F1-Score** (0.5083 vs 0.4477) - best overall balance
4. **More conservative** approach with fewer total assignments

#### Adaptive Threshold Advantages:
1. **Better Coverage** (91.1% vs 80.9%) - fewer unassigned texts
2. **Higher Conservation Rate** (80.9% vs 70.6%) - better multilabel performance
3. **Superior Label-wise F1** (Micro: 0.4309, Macro: 0.4157) - better per-SDG performance
4. **Controlled Assignment** - maintains target average labels (2.76 vs target 2.8)
5. **More sophisticated** threshold selection based on data distribution

## Conclusions

### Best Overall Approach: **Global Fixed Thresholds**
Despite being simpler, the global threshold approach (Experiment 1) achieves:
- **19% better F1-score** (0.5083 vs 0.4477) 
- **19% better precision** (0.4118 vs 0.3452)
- **8% better recall** (0.8725 vs 0.8091)

### Use Case Recommendations

**Choose Global Thresholds when:**
- Overall classification accuracy is priority
- Precision and recall balance is most important
- Simpler, more interpretable approach is preferred
- Processing efficiency is critical

**Choose Adaptive Thresholds when:**
- Coverage is critical (minimizing unassigned texts)
- Individual SDG performance matters more
- Consistent multilabel assignment is required
- Fine-tuning assignment distribution is needed

## Technical Insights

1. **Threshold Values**: Adaptive approach used much lower thresholds (0.27 vs 0.4/0.3), indicating the distribution-based approach found different optimal points
2. **Assignment Patterns**: Global approach relied heavily on secondary threshold (27K vs 18K assignments), while adaptive primarily used single threshold
3. **Coverage vs Precision**: Clear trade-off between covering more texts (adaptive) vs higher precision (global)

## Recommendations for Future Work

1. **Hybrid Approach**: Combine global threshold precision with adaptive coverage
2. **Threshold Optimization**: Use grid search to optimize global thresholds
3. **Model Comparison**: Test with different embedding models (e.g., sentence-BERT variations)
4. **Domain-Specific Tuning**: Adjust thresholds per SDG category
5. **Ensemble Methods**: Combine multiple threshold strategies

---
*Generated on January 21, 2025 - Clean Experimental Framework*
