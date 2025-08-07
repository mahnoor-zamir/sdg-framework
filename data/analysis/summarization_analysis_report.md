# Summarization Approach Analysis Report

## Executive Summary

Our evaluation of the BART summarization approach for SDG classification has revealed **significant performance degradation** compared to the baseline approach. The key findings are:

- **Overall Performance**: 62.4% vs 72.5% (baseline) - **10.1% decline**
- **Problematic SDGs**: Only 2 out of 4 showed improvement
- **Environmental SDGs**: Only 1 out of 5 showed improvement

## Detailed Results

### Overall Metrics Comparison

| Metric | Baseline | Summarization | Change |
|--------|----------|---------------|--------|
| Preservation Rate | 72.5% | 62.4% | **-10.1%** |
| Avg Labels/Text | 2.53 | 2.40 | -0.13 |
| Processing Time | ~73s | ~104s | +31s |

### Per-SDG Performance Analysis

#### Significant Improvements (>1%)
- **SDG 10** (Inequality): 60.9% → 62.9% (+2.1%)
- **SDG 15** (Life on Land): 72.1% → 73.4% (+1.3%)

#### Major Deteriorations (>10%)
- **SDG 2** (Zero Hunger): 71.5% → 24.1% (**-47.4%** - Critical)
- **SDG 3** (Health): 45.6% → 17.7% (**-27.9%** - Critical)  
- **SDG 14** (Life Below Water): 94.6% → 76.5% (**-18.1%**)
- **SDG 11** (Sustainable Cities): 68.6% → 52.5% (**-16.1%**)
- **SDG 4** (Education): 70.8% → 57.5% (**-13.3%**)
- **SDG 8** (Economic Growth): 56.0% → 43.2% (**-12.8%**)

### Problematic SDG Focus Analysis

Our hypothesis was that summarization would help with problematic SDGs (8, 9, 10, 12). Results:

| SDG | Description | Baseline | Summarized | Change | Assessment |
|-----|-------------|----------|------------|--------|------------|
| 8 | Economic Growth | 56.0% | 43.2% | **-12.8%** |  Worse |
| 9 | Industry/Innovation | 65.7% | 65.8% | +0.1% |  Neutral |
| 10 | Reduced Inequalities | 60.9% | 62.9% | +2.1% |  Better |
| 12 | Responsible Consumption | 84.5% | 78.4% | -6.1% |  Worse |

**Result**: Only 2/4 problematic SDGs improved, with minimal gains.

## Root Cause Analysis

### Why Summarization Failed

1. **Information Loss**: BART summarization reduced text from ~1000 words to ~250 words on average
   - Critical contextual information may have been lost
   - Nuanced SDG connections were simplified away

2. **Summarization Bias**: BART was trained on news articles and may not preserve scientific/policy content well
   - May favor certain topics over others
   - Could introduce systematic biases

3. **SDG Vocabulary Mismatch**: 
   - **SDG 2 (Hunger)**: Lost 47.4% performance - likely food security context was over-simplified
   - **SDG 3 (Health)**: Lost 27.9% performance - medical terminology may have been generalized
   - **SDG 14 (Oceans)**: Lost 18.1% performance - marine science terms may have been simplified

4. **Compression vs. Context Trade-off**:
   - Shorter summaries → less vocabulary overlap with SDG descriptions
   - Original texts had richer semantic content for embedding matching

### Environmental vs Social SDG Performance

- **Environmental SDGs** (6, 7, 13, 14, 15): 1/5 improved
  - These rely on specific technical terminology that summarization may have removed
- **Social SDGs** (1, 2, 3, 4, 5): Mixed results, but several major declines

## Alternative Approaches

Given the failure of BART summarization, consider:

### 1. Extractive Summarization
- Keep original sentences intact
- Preserve technical terminology
- Select most relevant sentences rather than rewriting

### 2. Domain-Specific Summarization
- Use models trained on scientific/policy documents
- Consider models like SciBERT or specialized sustainability models

### 3. Selective Summarization
- Only summarize very long texts (>500 words)
- Keep technical documents unsummarized
- Apply different strategies per SDG domain

### 4. Hybrid Approaches
- Use both original and summarized versions
- Ensemble predictions from multiple text representations
- Weight by confidence scores

## Recommendations

### Immediate Actions
1. **Abandon BART summarization approach** - Performance is significantly worse
2. **Use baseline similarity approach** - 72.5% preservation is better than 62.4%
3. **Investigate alternative improvements**:
   - Better embedding models (larger, domain-specific)
   - Improved similarity thresholds
   - Ensemble methods

### Future Research Directions

1. **Domain-Specific Models**: 
   - Train or fine-tune embeddings on sustainability corpus
   - Use models like ClimateBERT or sustainability-focused transformers

2. **Multi-Modal Approaches**:
   - Combine text embeddings with keyword matching
   - Use both dense and sparse retrieval methods

3. **Active Learning**:
   - Use human feedback to improve difficult cases
   - Focus on problematic SDGs with targeted improvements

## Conclusion

The BART summarization approach **failed to improve SDG classification** and should be **discontinued**. The 10.1% performance decline is too significant to justify, and the approach particularly hurt performance on several key SDGs.

**Key Lesson**: Summarization can hurt performance when domain-specific terminology and context are critical for classification. The rich semantic content in full texts was more valuable than the focused content in summaries.

**Next Steps**: Focus on improving the baseline approach through better embeddings, ensemble methods, or domain-specific models rather than text preprocessing techniques.

---
*Analysis Date: August 7, 2025*  
*Dataset: 17,248 OSDG texts*  
*Methods: BART + sentence-transformers vs sentence-transformers only*
