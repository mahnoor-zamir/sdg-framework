# Advanced Embeddings Experiment Summary
**Date:** August 25, 2025  
**Experiment Type:** Multi-Model Embedding Comparison with Threshold Optimization  
**Dataset:** OSDG Multilabel Classification (17,248 texts, 17 SDG labels)

## Executive Summary

This experiment compared four state-of-the-art sentence embedding models for SDG (Sustainable Development Goals) text classification, using optimized dual-threshold assignment strategies. **DistilRoBERTa v1** emerged as the best performer with a Micro F1 score of **0.4592**, representing a **7.8% improvement** over the baseline MiniLM model.

## Methodology

### Models Tested
1. **all-MiniLM-L6-v2** (Baseline)
2. **all-mpnet-base-v2** (Advanced)
3. **all-distilroberta-v1** (Advanced)
4. **multi-qa-mpnet-base-dot-v1** (Advanced)

### Optimization Strategy
- **Dual-threshold system**: Primary threshold (strict matching) + Secondary threshold (relaxed matching)
- **Grid search optimization**: 30 parameter combinations per model
- **Threshold ranges**: Primary 0.2-0.6, Secondary 0.1-0.35 (step 0.05)
- **Maximum labels per text**: 5 (to prevent over-labeling)

## Results Summary

| Model | Micro F1 | Micro P | Micro R | Macro F1 | Macro P | Macro R | Coverage | Conservation | Time (min) |
|-------|----------|---------|---------|----------|---------|---------|----------|--------------|------------|
| **DistilRoBERTa v1** | **0.4592** | **0.3494** | 0.6694 | **0.4226** | **0.3731** | 0.6129 | 0.7347 | 0.6694 | 3.6 |
| MiniLM L6 v2 | 0.4259 | 0.3428 | 0.5624 | 0.3859 | 0.3538 | 0.5064 | 0.6556 | 0.5624 | **1.3** |
| MPNet Base v2 | 0.3994 | 0.2785 | **0.7058** | 0.3753 | 0.2921 | **0.6399** | **0.7721** | **0.7058** | 6.6 |
| Multi-QA MPNet | 0.3091 | 0.1993 | 0.6877 | 0.2890 | 0.2145 | 0.5797 | **0.8344** | 0.6877 | 7.7 |

## Key Findings

### Best Overall Performance: DistilRoBERTa v1
- **Micro F1**: 0.4592 (7.8% better than baseline)
- **Optimal thresholds**: Primary=0.5, Secondary=0.35
- **Balanced performance**: Strong in both precision and recall
- **Processing time**: 3.6 minutes (2.8x slower than baseline but reasonable)

### Speed Champion: MiniLM L6 v2
- **Processing time**: 1.3 minutes (fastest)
- **Solid performance**: 0.4259 Micro F1
- **Best efficiency**: Good performance-to-speed ratio

### Recall Champion: MPNet Base v2
- **Highest recall**: 0.7058 (captures most true positives)
- **High coverage**: 77.21% of texts received labels
- **Conservative**: Best at preserving original labels (70.58%)

### Coverage Champion: Multi-QA MPNet
- **Highest coverage**: 83.44% of texts labeled
- **Lowest precision**: 0.1993 (over-labels frequently)
- **Slowest**: 7.7 minutes processing time

## Performance Analysis

### Metric Explanations
- **Micro F1/P/R**: Overall performance treating each prediction equally
- **Macro F1/P/R**: Average performance across all 17 SDGs (treats each SDG equally)
- **Coverage**: Percentage of texts that received at least one SDG label
- **Conservation**: Percentage of original ground truth labels that were preserved

### Threshold Convergence
Interestingly, all models converged to the **same optimal thresholds**:
- **Primary threshold**: 0.5
- **Secondary threshold**: 0.35

This suggests these values represent a sweet spot for dual-threshold SDG classification across different embedding approaches.

## Model-Specific Insights

### DistilRoBERTa v1 (Winner)
- **Strengths**: Excellent balance of precision and recall, robust across metrics
- **Use case**: Best choice for production systems requiring high accuracy
- **Trade-off**: 2.3 minutes slower than baseline for 7.8% F1 improvement

### MiniLM L6 v2 (Baseline)
- **Strengths**: Fast processing, solid performance, good precision
- **Use case**: Best for real-time applications where speed matters
- **Trade-off**: Lower recall means some true positives are missed

### MPNet Base v2
- **Strengths**: Highest recall, excellent at finding relevant texts
- **Use case**: Best when missing relevant content is costly
- **Trade-off**: Lower precision means more false positives

### Multi-QA MPNet
- **Strengths**: Highest coverage, designed for question-answering tasks
- **Use case**: When broad coverage is more important than precision
- **Trade-off**: Slowest processing with lowest precision

## Recommendations

### For Production Use: **DistilRoBERTa v1**
- Best overall accuracy (0.4592 F1)
- Reasonable processing time (3.6 minutes)
- Balanced precision-recall profile

### For Real-Time Applications: **MiniLM L6 v2**
- Fastest processing (1.3 minutes)
- Good accuracy (0.4259 F1)
- Only 7.8% accuracy loss for 64% time savings

### For High Recall Scenarios: **MPNet Base v2**
- Best recall (0.7058)
- Good when missing content is expensive
- Moderate processing time (6.6 minutes)

## Technical Details

### Experimental Setup
- **Environment**: Python 3.x with sentence-transformers library
- **Hardware**: Standard CPU processing (no GPU acceleration noted)
- **Validation**: OSDG multilabel test dataset
- **Optimization**: Grid search across threshold combinations

### Processing Performance
- **Total experiment time**: 19.3 minutes
- **Embedding generation**: Dominant processing bottleneck
- **Similarity calculation**: Negligible time (<0.1 seconds)
- **Threshold optimization**: 30 combinations per model

## Future Work Suggestions

1. **Extended threshold ranges**: Test broader ranges (0.1-0.7) for potentially better optimization
2. **GPU acceleration**: Significantly reduce processing times
3. **Ensemble methods**: Combine predictions from top performers
4. **Per-SDG optimization**: Individual thresholds for each of the 17 SDGs
5. **Fine-tuning**: Domain-specific training on SDG texts

## Files Generated
- **Visualization**: `results/models_comparison_with_minilm_20250825_000904.png`
- **Summary CSV**: `results/models_comparison_summary_20250825_000905.csv`
- **Comprehensive JSON**: `results/comprehensive_comparison_20250825_000905.json`

---

**Conclusion**: This experiment successfully identified DistilRoBERTa v1 as the best-performing model for SDG text classification, with significant improvements over the baseline while maintaining reasonable processing times. The consistent optimal threshold values across models suggest robust parameter selection, and the comprehensive evaluation provides clear guidance for different use case scenarios.
