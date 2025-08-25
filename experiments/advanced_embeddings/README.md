# Advanced Embeddings Models Experiment

This directory contains advanced embedding models for SDG classification with automatic threshold optimization and comprehensive performance comparison.

## 🎯 Overview

This experiment compares four state-of-the-art sentence transformer models:

0. **all-MiniLM-L6-v2** - Lightweight and efficient baseline model
1. **all-mpnet-base-v2** - High-performance transformer model
2. **all-distilroberta-v1** - Fast and efficient distilled model  
3. **multi-qa-mpnet-base-dot-v1** - Optimized for question-answering and similarity tasks

Each model automatically finds optimal primary and secondary thresholds through grid search and provides comprehensive evaluation metrics including detailed precision and recall analysis.

## 📁 Files Structure

```
advanced_embeddings/
├── README.md                        # This file
├── model_0_minilm_l6_v2.py         # MiniLM L6 v2 baseline model script
├── model_1_mpnet_base_v2.py        # MPNet Base v2 model script
├── model_2_distilroberta_v1.py     # DistilRoBERTa v1 model script  
├── model_3_multi_qa_mpnet.py       # Multi-QA MPNet model script
├── compare_all_models.py            # Compare all models performance
├── run_individual_model.py          # Run individual models
└── results/                         # Results directory (created automatically)
    ├── *_results_*.csv             # Detailed results for each model
    ├── *_optimization_*.csv        # Threshold optimization results
    ├── *_stats_*.json              # Model statistics and metrics
    ├── models_comparison_with_minilm_*.png # Performance comparison visualization
    ├── models_comparison_summary_*.csv # Comparison summary
    └── comprehensive_comparison_*.json # Complete comparison report
```

## 🚀 Usage

### Option 1: Run All Models and Compare (Recommended)

```bash
# Run all three models and generate comprehensive comparison
python compare_all_models.py
```

This will:
- Run all three models with threshold optimization
- Generate performance comparison visualizations
- Create comprehensive comparison reports
- Save all results in the `results/` directory

### Option 2: Run Individual Models

```bash
# List available models
python run_individual_model.py --list

# Run specific models
python run_individual_model.py --model 0  # MiniLM L6 v2 (baseline)
python run_individual_model.py --model 1  # MPNet Base v2
python run_individual_model.py --model 2  # DistilRoBERTa v1
python run_individual_model.py --model 3  # Multi-QA MPNet

# Or run directly
python model_0_minilm_l6_v2.py
python model_1_mpnet_base_v2.py
python model_2_distilroberta_v1.py  
python model_3_multi_qa_mpnet.py
```

## ⚙️ Configuration

Each model uses the following configuration:

- **Threshold Search Ranges:**
  - Primary: [0.2, 0.3, 0.4, 0.5, 0.6]
  - Secondary: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
- **Max Labels per Text:** 5
- **Distance Metric:** Cosine similarity
- **Batch Size:** 32 (for embedding generation)

## 📊 Output Metrics

Each model provides comprehensive evaluation metrics:

### Sample-based Metrics
- **Precision:** Average precision per text sample
- **Recall:** Average recall per text sample  
- **F1-Score:** Average F1 score per text sample
- **Jaccard:** Average Jaccard similarity per text sample

### Label-based Metrics
- **Micro F1/Precision/Recall:** Global scores across all predictions (treats each prediction equally)
- **Macro F1/Precision/Recall:** Average scores across all SDGs (treats each SDG equally)
- **Weighted F1/Precision/Recall:** Scores weighted by label support

### Additional Metrics
- **Conservation Rate:** Percentage of original labels preserved
- **Coverage Rate:** Percentage of texts receiving at least one label
- **Average Labels per Text:** Mean number of SDG labels assigned
- **Per-SDG Metrics:** Individual precision, recall, and F1 for each SDG

## 📈 Results Files

### Individual Model Results
- `{model}_results_{timestamp}.csv` - Detailed predictions and similarity scores
- `{model}_optimization_{timestamp}.csv` - Threshold optimization grid search results
- `{model}_stats_{timestamp}.json` - Complete model statistics and configuration

### Comparison Results
- `models_comparison_{timestamp}.png` - Performance comparison charts
- `models_comparison_summary_{timestamp}.csv` - Side-by-side metrics comparison
- `comprehensive_comparison_{timestamp}.json` - Complete comparison with recommendations

## 🔧 Requirements

Make sure you have the required dependencies:

```bash
pip install sentence-transformers
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
```

## 💡 Model Selection Guidance

- **For best accuracy:** Use the model with highest Micro F1 score
- **For fastest processing:** Check processing time in comparison results
- **For balanced performance:** Consider models with good F1 scores and reasonable processing time

## 📝 Customization

To modify the experiment:

1. **Change threshold ranges:** Edit the `threshold_search` in CONFIG
2. **Add new models:** Create new model scripts following the existing pattern
3. **Modify metrics:** Update the `calculate_evaluation_metrics()` function
4. **Change visualization:** Modify `create_comparison_visualizations()` in the comparison script

## 🎯 Expected Performance

Based on the models' characteristics:
- **MiniLM L6 v2:** Expected to be fastest with decent baseline performance
- **MPNet Base v2:** Expected to have highest accuracy but longer processing time
- **DistilRoBERTa v1:** Expected to balance speed and accuracy well
- **Multi-QA MPNet:** Expected to perform well on similarity tasks with moderate processing time

## 📋 Troubleshooting

**Common Issues:**
1. **Model download errors:** Ensure internet connection for downloading sentence transformer models
2. **Memory issues:** Reduce batch size in the embedding generation function
3. **Missing data files:** Ensure OSDG and SDG datasets are in `../../data/processed/`
4. **Permission errors:** Check write permissions for the `results/` directory

## 🏆 Using Results

After running experiments:
1. Check the comprehensive comparison report for the best performing model
2. Use the optimal thresholds found for your chosen model
3. Review the visualization to understand trade-offs between models
4. Consider both accuracy and processing time for production deployment
