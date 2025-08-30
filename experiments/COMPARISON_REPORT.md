
# Comparison Report

## Section 1: Euclidean Distance (Min-Max Scaling)
### 1.1 Global Thresholds
### 1.2 Adaptive Thresholds

## Section 2: Cosine Similarity
### 2.1 Global Thresholds
### 2.2 Adaptive Thresholds

## Section 3: Global vs Adaptive — Cross-Metric Comparison


## Results Summary (at ≈3 assignments/text)

| Metric         | Euclidean Global (0.50/0.60) | Euclidean Adaptive (2.75) | Cosine Global (0.4/0.3) | Cosine Adaptive (2.8) |
|----------------|-----------------------------|--------------------------|-------------------------|----------------------|
| Accuracy       | 74.6%                       | 82.85%                   | 87.2%                  | 80.9%                |
| Correct/Total  | 12,868 / 17,248             | ~14,292 / 17,248         | ~15,056 / 17,248       | ~13,950 / 17,248     |
| Avg Assignments| 2.90                        | 2.75                     | 2.51                   | 2.84                 |
| F1-Score       | 0.36                        | 0.4752                   | 0.5083                 | 0.4477               |
| Precision      | 0.27                        | 0.3745                   | 0.4118                 | 0.3452               |
| Coverage       | 86.5%                       | 94.9%                    | 80.9%                  | 91.0%                |

## Analysis

### Key Insights
- **Cosine global thresholding** achieves the highest F1 and accuracy at practical assignment levels, but with lower coverage than adaptive methods.
- **Euclidean adaptive thresholding** provides the best coverage and recall, and is the top performer for the Euclidean metric.
- **Cosine adaptive** offers a strong balance of coverage and F1, outperforming Euclidean global and nearly matching Euclidean adaptive in coverage.
- **Euclidean global** is the simplest but underperforms in all metrics compared to the others.

### Practical Recommendations
- For **maximum F1 and accuracy**: Use **cosine global** thresholds.
- For **maximum coverage and recall**: Use **euclidean adaptive** thresholds.
- For **balanced performance**: Cosine adaptive is also strong, especially if you want high coverage with cosine.
- Always monitor on new data and recalibrate as needed.

## Recommendation

**Summary Table:**
- **Cosine global**: Best F1/accuracy, moderate coverage.
- **Euclidean adaptive**: Best coverage/recall, strong F1.
- **Cosine adaptive**: High coverage, strong F1.
- **Euclidean global**: Simple, but lowest performance.

Choose based on your application’s priorities (F1, coverage, interpretability).

---
*Generated on: 2025-08-29*
*Model: all-MiniLM-L6-v2*
*Distance: Euclidean*
