# Euclidean + MiniLM Experiments

This folder contains clean, organized experiments for SDG classification using Euclidean distance with the all-MiniLM-L6-v2 embedding model.

## Experiment Structure

### Experiment 1: Global Thresholds
- **File**: `experiment_1_global_threshold.py`
- **Approach**: Fixed global thresholds (optimized for Euclidean distance)
- **Purpose**: Baseline comparison with traditional threshold approach for Euclidean distance
- **Results**: Saved to `results/global_threshold/`

### Experiment 2: Adaptive Thresholds  
- **File**: `experiment_2_adaptive_threshold.py`
- **Approach**: Dynamic thresholds based on target metrics (2.8 avg labels)
- **Features**: 
  - Assignment control (1-4 labels per text)
  - Coverage guarantee (all texts get labels)
  - Fallback mechanism for difficult texts
  - Euclidean distance-specific threshold calculation
- **Results**: Saved to `results/adaptive_threshold/`

## Quick Start

Run experiments from the `euclidean_minilm` directory:

```bash
# Experiment 1 - Global Thresholds
python experiment_1_global_threshold.py

# Experiment 2 - Adaptive Thresholds  
python experiment_2_adaptive_threshold.py
```

## Key Differences from Cosine Similarity

### Distance Metric
- **Euclidean Distance**: Measures geometric distance between embedding vectors
- **Threshold Direction**: Lower distance = higher similarity (opposite of cosine)
- **Normalization**: May require different scaling compared to cosine similarity

### Threshold Values
- **Expected Range**: Typically different from cosine similarity thresholds
- **Optimization**: Euclidean-specific threshold tuning required
- **Previous Research**: Based on notes showing Euclidean tends to over-assign labels

## Output Files

Each experiment generates:
- **CSV Results**: Complete dataset with assignments and distance scores
- **JSON Stats**: Comprehensive metrics and configuration details
- **Console Report**: Real-time performance summary

## Key Metrics Tracked

- **Sample-based**: Precision, Recall, F1, Jaccard, Conservation Rate
- **Label-based**: Micro/Macro F1, Precision, Recall  
- **Assignment**: Coverage, Average labels per text, Threshold usage
- **Distance-specific**: Average Euclidean distances, distribution analysis
- **Performance**: Processing time, Memory usage

## Expected Results vs Cosine

Based on previous research findings:
- **Higher Recall**: Euclidean may achieve higher recall but at precision cost
- **Label Inflation**: Watch for over-assignment tendency (avg labels > 3.0)
- **F1 Comparison**: Expected F1 may be lower than cosine despite higher recall
- **Efficiency**: Quality vs quantity trade-off needs careful evaluation

## Research Context

From previous analysis:
- **Cosine Approach**: F1=0.5083, Precision=0.4118, Recall=0.8725, Avg=2.51 labels
- **Euclidean Historical**: F1=0.396, Precision=0.255, Recall=0.890, Avg=3.49 labels
- **Goal**: Optimize Euclidean to match cosine performance while leveraging its strengths

## Optimization Strategy

1. **Threshold Tuning**: Find optimal thresholds to avoid over-assignment
2. **Assignment Control**: Implement stricter label limits
3. **Quality Focus**: Prioritize precision-recall balance over raw recall
4. **Comparative Analysis**: Direct comparison with cosine results
