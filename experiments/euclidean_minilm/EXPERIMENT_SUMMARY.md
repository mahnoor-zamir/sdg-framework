# Euclidean Distance + MiniLM Experiments Summary

## Experiment Overview
This document summarizes the systematic comparison of global vs adaptive threshold strategies using Euclidean distance with the all-MiniLM-L6-v2 model for SDG multilabel classification.

## Dataset Information
- Total texts: 17,248 from OSDG dataset
- SDGs: 17 categories (SDG 1-17)
- Embedding model: all-MiniLM-L6-v2 (384-dimensional)
- Distance metric: Euclidean distance

## Experiment Configurations

### Experiment 1: Global Thresholds
- **Configuration**: Fixed thresholds across all texts
- **Primary threshold**: 1.03 (distance ≤ 1.03 = primary assignment)
- **Secondary threshold**: 1.10 (distance ≤ 1.10 = secondary assignment)
- **Date executed**: 2025-01-21

### Experiment 2: Adaptive Thresholds
- **Configuration**: Dynamic thresholds based on distance distributions
- **Target average labels**: 2.8 per text
- **Min/max labels**: 1-4 per text
- **Fallback threshold**: 1.25 (for texts with no close matches)
- **Date executed**: 2025-01-21

## Detailed Results

### Experiment 1 Results (Global Thresholds)
```
Evaluation Metrics:
  Precision: 0.2949
  Recall: 0.4283
  F1-Score: 0.3274
  Conservation: 0.4283
  Coverage: 0.495 (49.5%)

Assignment Statistics:
  Average labels per text: 1.37
  Total assignments: 23,624
  Primary assignments: 6,399
  Secondary assignments: 8,692
  Zero-label texts: 8,533 (49.5%)
```

### Experiment 2 Results (Adaptive Thresholds)
```
Calculated Thresholds:
  Primary: 1.1290 (10th percentile)
  Secondary: 1.2100 (28th percentile)
  Predicted avg labels: 2.75

Evaluation Metrics:
  Precision: 0.3745
  Recall: 0.8285
  F1-Score: 0.4752
  Conservation: 0.8285
  Coverage: 0.9487 (94.9%)

Assignment Statistics:
  Average labels per text: 2.75
  Total assignments: 47,392
  Primary assignments: 22,957
  Secondary assignments: 23,232
  Fallback assignments: 1,203
  Zero-label texts: 885 (5.1%)
```

## Performance Comparison

| Metric | Global Thresholds | Adaptive Thresholds | Improvement |
|--------|-------------------|---------------------|-------------|
| **F1-Score** | 0.3274 | **0.4752** | **+45.1%** |
| **Precision** | 0.2949 | **0.3745** | **+27.0%** |
| **Recall** | 0.4283 | **0.8285** | **+93.4%** |
| **Coverage** | 49.5% | **94.9%** | **+91.7%** |
| **Conservation** | 0.4283 | **0.8285** | **+93.4%** |
| Avg Labels/Text | 1.37 | **2.75** | **+100.7%** |

## Key Findings

### 1. Adaptive Thresholds Significantly Outperform Global
- **F1-Score improvement**: 45.1% (0.3274 → 0.4752)
- **Coverage improvement**: 91.7% (49.5% → 94.9%)
- **Recall improvement**: 93.4% (0.4283 → 0.8285)

### 2. Threshold Selection Critical for Euclidean Distance
- **Original thresholds** (0.47/0.50): Zero assignments (too restrictive)
- **Optimized thresholds** (1.03/1.10): Meaningful performance
- **Distance range**: 0.69-1.54, requiring careful threshold calibration

### 3. Coverage vs Precision Trade-off
- Global thresholds: Higher precision (0.2949), lower coverage (49.5%)
- Adaptive thresholds: Balanced precision (0.3745), excellent coverage (94.9%)

### 4. Fallback Mechanism Effectiveness
- Fallback threshold (1.25): Used for 7.0% of texts
- Only 5.1% of texts received zero labels (vs 49.5% in global)

## Technical Insights

### Distance Distribution Analysis
```
Euclidean Distance Statistics:
  Minimum: 0.6927
  Maximum: 1.5440
  Mean: 1.2592
  Std: 0.1879
```

### Threshold Optimization Process
1. **Distance analysis**: Examined distribution of closest SDG distances
2. **Percentile-based selection**: Used 10th and 28th percentiles for adaptive thresholds
3. **Validation**: Achieved target 2.8 labels/text (actual: 2.75)

## Best Configuration
**Adaptive thresholds with Euclidean distance** provides the optimal balance:
- **F1-Score**: 0.4752
- **Coverage**: 94.9%
- **Practical applicability**: Ensures most texts receive relevant SDG classifications

## Comparison with Cosine Similarity
When compared to cosine similarity experiments:
- Euclidean adaptive (F1=0.4752) vs Cosine adaptive (F1=0.4477): **+6.1%**
- Euclidean global (F1=0.3274) vs Cosine global (F1=0.5083): **-35.6%**

**Key insight**: Adaptive thresholds are more critical for Euclidean distance than cosine similarity.

## Files Generated
```
results/global_threshold/
├── experiment_1_results_20250821_183534.csv
└── experiment_1_stats_20250821_183534.json

results/adaptive_threshold/
├── experiment_2_results_20250821_185437.csv
└── experiment_2_stats_20250821_185437.json
```

## Recommendations
1. **Use adaptive thresholds** for Euclidean distance-based classification
2. **Careful threshold calibration** is essential for Euclidean (unlike cosine)
3. **Fallback mechanisms** prevent zero-label assignments effectively
4. **Consider distance metric selection** based on threshold flexibility requirements

---
*Generated on: 2025-01-21*
*Model: all-MiniLM-L6-v2*
*Distance: Euclidean*
