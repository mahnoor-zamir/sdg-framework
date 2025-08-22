# Complete Experimental Results Comparison
## All 4 Distance Metric + Threshold Strategy Combinations

*Generated: August 21, 2025*

---

## Executive Summary Table

| Rank | Configuration | Distance | Threshold Strategy | F1-Score | Precision | Recall | Coverage | Avg Labels | Status |
|------|---------------|----------|-------------------|----------|-----------|--------|----------|------------|--------|
| 🥇 1st | **Cosine Global** | Cosine | Fixed (0.4/0.3) | **0.5083** | **0.5122** | 0.4978 | 96.5% | 2.39 | **WINNER** |
| 🥈 2nd | Euclidean Adaptive | Euclidean | Dynamic | **0.4752** | 0.3745 | **0.8285** | **94.9%** | 2.75 | Strong |
| 🥉 3rd | Cosine Adaptive | Cosine | Dynamic | **0.4477** | 0.4166 | 0.4859 | 95.0% | 2.82 | Good |
| 4th | Euclidean Global | Euclidean | Fixed (1.03/1.10) | 0.3274 | 0.2949 | 0.4283 | **49.5%** | 1.37 | Poor |

---

## Detailed Results Breakdown

### 🥇 EXPERIMENT 1: Cosine Similarity + Global Thresholds
```
Configuration:
- Distance Metric: Cosine Similarity
- Primary Threshold: 0.4 (similarity ≥ 0.4)  
- Secondary Threshold: 0.3 (similarity ≥ 0.3)
- Strategy: Fixed thresholds for all texts

Results:
✅ F1-Score: 0.5083 (HIGHEST)
✅ Precision: 0.5122 (HIGHEST)
• Recall: 0.4978
• Coverage: 96.5%
• Avg Labels/Text: 2.39
• Micro F1: 0.4885
• Macro F1: 0.4700

Assignment Distribution:
• Primary Only: 4,893 texts (28.4%)
• Primary + Secondary: 11,764 texts (68.2%)
• Zero Labels: 591 texts (3.5%)
```

### 🥈 EXPERIMENT 2: Euclidean Distance + Adaptive Thresholds  
```
Configuration:
- Distance Metric: Euclidean Distance
- Primary Threshold: 1.1290 (10th percentile)
- Secondary Threshold: 1.2100 (28th percentile)  
- Strategy: Dynamic thresholds based on distribution

Results:
• F1-Score: 0.4752
• Precision: 0.3745
✅ Recall: 0.8285 (HIGHEST)
✅ Coverage: 94.9% (EXCELLENT)
• Avg Labels/Text: 2.75
• Micro F1: 0.4421
• Macro F1: 0.4246

Assignment Distribution:
• Primary Only: 3,474 texts (20.1%)
• Primary + Secondary: 11,686 texts (67.8%)
• Fallback Used: 1,203 texts (7.0%)
• Zero Labels: 885 texts (5.1%)
```

### 🥉 EXPERIMENT 3: Cosine Similarity + Adaptive Thresholds
```
Configuration:
- Distance Metric: Cosine Similarity
- Primary Threshold: 0.3894 (10th percentile)
- Secondary Threshold: 0.3200 (28th percentile)
- Strategy: Dynamic thresholds based on distribution

Results:
• F1-Score: 0.4477
• Precision: 0.4166
• Recall: 0.4859
• Coverage: 95.0%
• Avg Labels/Text: 2.82
• Micro F1: 0.4185
• Macro F1: 0.4039

Assignment Distribution:
• Primary Only: 3,526 texts (20.4%)
• Primary + Secondary: 12,823 texts (74.3%)
• Fallback Used: 39 texts (0.2%)
• Zero Labels: 860 texts (5.0%)
```

### 4️⃣ EXPERIMENT 4: Euclidean Distance + Global Thresholds
```
Configuration:
- Distance Metric: Euclidean Distance  
- Primary Threshold: 1.03 (distance ≤ 1.03)
- Secondary Threshold: 1.10 (distance ≤ 1.10)
- Strategy: Fixed thresholds for all texts

Results:
• F1-Score: 0.3274 (LOWEST)
• Precision: 0.2949 (LOWEST)
• Recall: 0.4283 (LOWEST)
❌ Coverage: 49.5% (POOR)
• Avg Labels/Text: 1.37 (LOWEST)

Assignment Distribution:
• Primary Only: 2,203 texts (12.8%)
• Primary + Secondary: 6,512 texts (37.8%)
❌ Zero Labels: 8,533 texts (49.5%) - MAJOR ISSUE
```

---

## Performance Analysis

### Best Performing Metrics by Experiment
- **🏆 Highest F1-Score**: Cosine Global (0.5083)
- **🏆 Highest Precision**: Cosine Global (0.5122)  
- **🏆 Highest Recall**: Euclidean Adaptive (0.8285)
- **🏆 Best Coverage**: Euclidean Adaptive (94.9%)
- **🏆 Most Efficient**: Cosine Global (2.39 avg labels with highest precision)

### Performance Gaps
- **F1-Score Gap**: Winner vs Worst = 55.3% improvement (0.5083 vs 0.3274)
- **Precision Gap**: Winner vs Worst = 73.8% improvement (0.5122 vs 0.2949)
- **Coverage Gap**: Winner vs Worst = 96.0% improvement (96.5% vs 49.5%)

### Distance Metric Comparison
**Cosine Similarity Performance**:
- Global Strategy: F1=0.5083 ✅ EXCELLENT
- Adaptive Strategy: F1=0.4477 ✅ GOOD
- **Advantage**: Consistent, reliable performance with both strategies

**Euclidean Distance Performance**:
- Adaptive Strategy: F1=0.4752 ✅ GOOD  
- Global Strategy: F1=0.3274 ❌ POOR
- **Disadvantage**: Highly sensitive to threshold selection

### Threshold Strategy Comparison
**For Cosine Similarity**:
- Global (0.5083) > Adaptive (0.4477) by 13.5%
- **Insight**: Cosine works well with fixed thresholds

**For Euclidean Distance**:
- Adaptive (0.4752) > Global (0.3274) by 45.1%  
- **Insight**: Euclidean requires adaptive thresholds to perform

---

## Practical Recommendations

### 🏆 Production Recommendation
**Use: Cosine Similarity + Global Thresholds (0.4/0.3)**
- **Why**: Best F1-score (0.5083), highest precision (0.5122), excellent coverage (96.5%)
- **Reliability**: Consistent performance, predictable behavior
- **Efficiency**: Optimal precision-recall balance with reasonable label count

### 🔧 Alternative Options
1. **High Recall Priority**: Euclidean + Adaptive (Recall=0.8285)
2. **Balanced Approach**: Cosine + Adaptive (F1=0.4477)
3. **Avoid**: Euclidean + Global (poor coverage, lowest scores)

### 🎯 Key Takeaways
1. **Distance matters**: Cosine similarity is more robust and reliable
2. **Threshold strategy depends on distance**: Global works for cosine, adaptive needed for Euclidean
3. **Coverage critical**: Euclidean global fails to classify 49.5% of texts
4. **Precision-recall trade-off**: Higher recall often comes at precision cost

---

## Technical Notes

### Distance Ranges
- **Cosine Similarity**: 0.0-1.0 (normalized, bounded)
- **Euclidean Distance**: 0.69-1.54 (unbounded, requires calibration)

### Threshold Sensitivity
- **Cosine**: Robust to threshold selection
- **Euclidean**: Highly sensitive, requires careful tuning

### Computational Efficiency
All experiments completed in ~54-60 seconds on the same hardware, showing comparable computational cost.

---

## Methodological Concerns & Validation

### 🚨 Critical Issue: Threshold Selection Bias
**The Problem**: All results depend heavily on threshold selection, which could introduce bias favoring certain approaches.

**Why This Matters for Thesis Defense**:
- Reviewers will question if "best" performance is due to optimal thresholds rather than superior method
- Need to prove robustness across multiple threshold configurations
- Must demonstrate systematic, unbiased threshold selection methodology

### 📊 Proposed Validation Framework

#### 1. **Cross-Validation with Multiple Threshold Sets**
```
Test each approach with 5-10 different threshold combinations:
- Conservative: Higher precision, lower recall
- Moderate: Balanced precision-recall  
- Liberal: Higher recall, lower precision
- Quartile-based: 25th, 50th, 75th percentiles
- Standard deviations: μ±0.5σ, μ±1σ, μ±1.5σ
```

#### 2. **Grid Search Validation**
```
For each distance metric:
- Cosine: Test thresholds [0.2-0.6] in 0.05 steps (81 combinations)
- Euclidean: Test thresholds [0.8-1.4] in 0.05 steps (144 combinations)
- Report: Mean, std dev, confidence intervals across all combinations
```

#### 3. **Statistical Significance Testing**
```
- Paired t-tests between methods across threshold variations
- Bootstrap confidence intervals (1000 samples)
- Effect size calculations (Cohen's d)
- ANOVA to test if differences are statistically significant
```

#### 4. **Train-Test Split Validation**
```
Current Issue: Thresholds optimized on same data used for evaluation
Solution: 
- Split data 70% train / 30% test
- Optimize thresholds ONLY on training set
- Evaluate performance ONLY on test set
- Prevents overfitting to specific dataset characteristics
```

### 🛡️ Thesis Defense Strategies

#### **Acknowledge Limitations**:
"We recognize that threshold selection significantly impacts results. To address this methodological concern, we propose the following validation framework..."

#### **Demonstrate Robustness**:
"Our findings remain consistent across X different threshold configurations, with Method A showing superior performance in Y% of tested scenarios."

#### **Statistical Rigor**:
"Statistical significance testing (p<0.05) confirms that performance differences are not due to random threshold selection."

#### **Practical Justification**:
"In real-world applications, threshold selection would be domain-specific. Our systematic comparison provides guidelines for threshold selection across different use cases."

### 📝 Recommended Additional Experiments

#### **Experiment A: Threshold Robustness Analysis**
- Test each method with 20+ threshold combinations
- Plot performance curves across threshold ranges
- Calculate stability metrics (coefficient of variation)

#### **Experiment B: Cross-Dataset Validation**
- Train thresholds on OSDG dataset
- Test on independent SDG dataset (if available)
- Measure performance degradation/consistency

#### **Experiment C: Domain Expert Validation**
- Have SDG experts manually evaluate sample predictions
- Compare expert judgments vs. model predictions
- Calculate inter-rater agreement with automated classifications

### 🎯 Strengthened Conclusions

Instead of claiming "Method X is best," present:

1. **"Method X shows superior performance under balanced threshold conditions"**
2. **"Across 95% confidence intervals, Method X consistently outperforms alternatives"**  
3. **"Robustness analysis confirms Method X maintains performance across threshold variations"**
4. **"Statistical significance testing validates observed performance differences"**

### 📊 Validation Metrics to Include

- **Stability Score**: Performance variance across threshold ranges
- **Robustness Index**: Percentage of threshold configs where method wins
- **Statistical Significance**: p-values for pairwise method comparisons
- **Effect Size**: Magnitude of performance differences (small/medium/large)

---

*Dataset: 17,248 OSDG texts | Model: all-MiniLM-L6-v2 | SDGs: 17 categories*
*⚠️ Results require threshold robustness validation for thesis defense*
