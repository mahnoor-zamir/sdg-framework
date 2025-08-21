# Cosine + MiniLM Experiments

This folder contains clean, organized experiments for SDG classification using cosine similarity with the all-MiniLM-L6-v2 embedding model.

## Experiment Structure

### Experiment 1: Global Thresholds
- **File**: `experiment_1_global_threshold.py`
- **Approach**: Fixed global thresholds (primary=0.4, secondary=0.3)
- **Purpose**: Baseline comparison with traditional threshold approach
- **Results**: Saved to `results/global_threshold/`

### Experiment 2: Adaptive Thresholds  
- **File**: `experiment_2_adaptive_threshold.py`
- **Approach**: Dynamic thresholds based on target metrics (2.8 avg labels)
- **Features**: 
  - Assignment control (1-4 labels per text)
  - Coverage guarantee (all texts get labels)
  - Fallback mechanism for difficult texts
- **Results**: Saved to `results/adaptive_threshold/`

## Quick Start

Run experiments from the `cosine_minilm` directory:

```bash
# Experiment 1 - Global Thresholds
python experiment_1_global_threshold.py

# Experiment 2 - Adaptive Thresholds  
python experiment_2_adaptive_threshold.py
```

## Output Files

Each experiment generates:
- **CSV Results**: Complete dataset with assignments and similarity scores
- **JSON Stats**: Comprehensive metrics and configuration details
- **Console Report**: Real-time performance summary

## Key Metrics Tracked

- **Sample-based**: Precision, Recall, F1, Jaccard, Conservation Rate
- **Label-based**: Micro/Macro F1, Precision, Recall  
- **Assignment**: Coverage, Average labels per text, Threshold usage
- **Performance**: Processing time, Memory usage

## Expected Results

**Experiment 1** (Global): ~72.5% conservation, ~2.5 avg labels, ~81% coverage
**Experiment 2** (Adaptive): ~80%+ conservation, ~2.8 avg labels, 100% coverage

The adaptive approach should demonstrate superior performance through controlled assignment and guaranteed coverage.
