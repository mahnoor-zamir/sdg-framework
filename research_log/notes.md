# Supervisor Summary (Executive Overview)

Purpose
- Build an accurate, scalable multi-label SDG classifier using semantic similarity.

Data
- Ground truth: 17,248 texts filtered from OSDG community dataset (agreement ≥ 0.6).
- SDG references: 17 paragraph descriptions extracted from Agenda 2030.

Methods Tested
- Approach 1: Original Text vs Original SDG descriptions (embeddings + cosine similarity).
- Approach 2: Original Text vs Summarized SDG descriptions (BART summaries).
- Approach 3: Summarized Text vs Summarized SDG descriptions (dual summarization).
- **Approach 4: Euclidean Distance** - Original Text vs Original SDG descriptions (embeddings + Euclidean distance).

**CRITICAL REASSESSMENT: Euclidean Distance Findings**
- **Initial Impression**: 88.97% preservation rate appeared superior to cosine similarity's 75.57%.
- **Critical Discovery**: Higher preservation is largely due to **label inflation** - assigning more labels per text.
- **Key Issue**: Euclidean assigns 1.36x more labels (3.49 vs 2.57 avg/text) but with lower quality.

**Precision-Recall Analysis Reveals**:
- **Cosine Precision**: 0.294 (29.4% of assigned labels are correct)
- **Euclidean Precision**: 0.255 (25.5% of assigned labels are correct)
- **F1 Score**: Cosine (0.423) > Euclidean (0.396)
- **Labeling Efficiency**: Only 14.6% of Euclidean's extra labels are actually correct

**🏆 FINAL VERDICT: COSINE SIMILARITY REMAINS CHAMPION**
- Better precision-recall balance (higher F1 score)
- More efficient and reliable label assignments
- Less prone to over-labeling issues

Why Euclidean Distance Misleads
- Higher recall comes at an unacceptable precision cost
- Label inflation creates false impression of improvement
- Over-assigns labels with 85.4% of extra labels being incorrect

Key Results (corrected analysis)
- **Cosine (Best)**: F1=0.423; Precision=0.294; Recall=0.756; Avg labels=2.57
- **Euclidean (Over-labels)**: F1=0.396; Precision=0.255; Recall=0.890; Avg labels=3.49
- **Lesson**: Preservation rate alone is misleading - must consider precision-recall trade-off

**FINAL SYSTEMATIC COMPARISON - ALL EXPERIMENTS (2025-01-21)**

## Complete Distance Metric + Threshold Strategy Comparison

| Configuration | Distance | Threshold | F1 Score | Precision | Recall | Coverage | Avg Labels | Best? |
|---------------|----------|-----------|----------|-----------|---------|----------|------------|--------|
| **🏆 Cosine Global** | Cosine | 0.4/0.3 | **0.5083** | **0.5122** | 0.4978 | 96.5% | 2.39 | **WINNER** |
| Cosine Adaptive | Cosine | Dynamic | 0.4477 | 0.4166 | 0.4859 | 95.0% | 2.82 | Strong |
| Euclidean Adaptive | Euclidean | Dynamic | 0.4752 | 0.3745 | 0.8285 | 94.9% | 2.75 | Good |
| Euclidean Global | Euclidean | 1.03/1.10 | 0.3274 | 0.2949 | 0.4283 | 49.5% | 1.37 | Poor |

## Key Findings from Systematic Experiments

### 1. **Distance Metric Impact**
- **Cosine similarity**: More stable, less sensitive to threshold selection
- **Euclidean distance**: Requires careful threshold calibration, more volatile

### 2. **Threshold Strategy Impact**  
- **Cosine**: Global thresholds (F1=0.5083) > Adaptive (F1=0.4477) by 13.5%
- **Euclidean**: Adaptive thresholds (F1=0.4752) > Global (F1=0.3274) by 45.1%

### 3. **Overall Champion**
**🏆 Cosine Similarity + Global Thresholds (0.4/0.3)**
- **F1-Score**: 0.5083 (highest overall)
- **Precision**: 0.5122 (best precision-recall balance) 
- **Coverage**: 96.5% (excellent)
- **Reliability**: Consistent, predictable performance

### 4. **Critical Insights**
- **Threshold sensitivity**: Euclidean requires adaptive thresholds, cosine works well with global
- **Coverage vs precision**: All methods achieve >94% coverage except Euclidean global
- **Precision priority**: Cosine global maintains >50% precision while others are 30-40%

## Historical Context

| Earlier Experiments | F1 Score | Notes |
|---------------------|----------|-------|
| Baseline (Cosine 0.4/0.3) | 0.423 | Original implementation |
| Per-SDG thresholds + fallback | 0.423 | Balanced approach |  
| Euclidean (0.47/0.45) | 0.396 | Over-labeling issues |

Per‑SDG Highlights (Euclidean Distance)
- **Outstanding preservation**: SDG 14 (98.9%), SDG 6 (98.4%), SDG 4 (96.0%), SDG 7 (96.6%)
- **Excellent performance**: SDG 1 (94.5%), SDG 12 (93.5%), SDG 11 (89.3%), SDG 13 (88.5%)
- **Good performance**: 12/15 original SDGs achieve >80% preservation
- **High precision SDGs**: SDG 5 (84.6%), SDG 3 (54.1%), SDG 2 (41.5%), SDG 6 (41.9%)

Files of Record
- **BEST RESULT**: `data/processed/similarity_multilabel_euclidean_embeddings_p0.47_s0.45_fbt10.42_t20.44.csv`
- Euclidean analysis: `data/analysis/sdg_overlap_analysis.*` (latest)
- Previous cosine results: `data/processed/similarity_multilabel_embeddings_p0.4_s0.3_fbt10.28_t20.3_psdg.csv`

Recommendation
- **🏆 ADOPT COSINE SIMILARITY with per‑SDG thresholds + fallback** for production deployment.
- **Configuration**: Primary/Secondary 0.4/0.3, per-SDG F1-optimal thresholds, fallback 0.28/0.30.
- **Rationale**: Best precision-recall balance (F1=0.423), efficient labeling, avoids over-assignment.
- **Key Lesson**: Preservation rate alone is misleading - must evaluate precision-recall trade-off.

Next Steps
- Deploy production system using **cosine similarity** approach (not Euclidean).
- Focus on precision-optimized thresholds for critical SDGs if needed.
- **Important**: Always evaluate using F1 score, not just preservation/recall metrics.

---

# SDG Multi-Label Classification Research Notes

## Project Overview
This project focuses on creating and analyzing multi-label SDG classification datasets using text similarity approaches with embeddings.

## Experimental Methodology

### Experiment 1: Baseline Dataset Filtering
- **Objective**: Create high-quality baseline dataset from OSDG community data
- **Source**: OSDG community dataset with agreement-based filtering
- **Filtering criteria**: Agreement score ≥ 0.6 and positive labels > negative labels
- **Result**: 17,248 high-quality texts from original 32,120 entries
- **Format**: Multi-label with individual SDG binary columns
- **Purpose**: Establish ground truth for comparison with similarity-based approaches

### Experiment 2: Original Text vs Original SDG Descriptions (Baseline Similarity)
- **Objective**: Create similarity-based multi-label dataset using original SDG descriptions
- **SDG Reference Creation**:
  - **Source**: Agenda 2030 PDF extraction
  - **Processing**: Combined SDG goals and targets into paragraph format
  - **Result**: 17 clean SDG descriptions for similarity matching
- **Method**: 
  - Sentence transformer embeddings (`all-MiniLM-L6-v2`)
  - Cosine similarity between original OSDG texts and original SDG descriptions
  - Thresholds: Primary (0.4) and Secondary (0.3) similarity thresholds
- **Results**: 
  - Multi-label assignments with 2.53 average labels per text
  - 72.50% preservation rate of original labels
  - 81.3% similarity coverage

### Experiment 3: Original Text vs Summarized SDG Descriptions (Approach 1)
- **Objective**: Test if summarizing SDG descriptions improves similarity matching
- **Method**:
  - Summarized the 17 original SDG descriptions using BART-large-CNN
  - Maintained original OSDG dataset texts
  - Applied same embedding model (`all-MiniLM-L6-v2`) and similarity thresholds
- **Hypothesis**: Shorter, more focused SDG descriptions might improve precision
- **Key Difference**: SDG reference texts are condensed while maintaining core semantic meaning

### Experiment 4: Summarized Text vs Summarized SDG Descriptions (Approach 2)
- **Objective**: Test dual summarization approach for both texts and SDG descriptions
- **Method**:
  - Summarized both the OSDG dataset texts AND the SDG descriptions using BART-large-CNN
  - Applied same embedding and similarity calculation pipeline
  - Maintained identical thresholds for fair comparison
- **Hypothesis**: Dual summarization might reduce noise and improve semantic alignment
- **Key Difference**: Both input texts and reference descriptions are summarized

## Experimental Results Comparison

### Approach Performance Summary

| Approach | Description | Preservation Rate | Average Labels | Coverage | Processing Time |
|----------|-------------|------------------|---------------|----------|----------------|
| **Baseline** | Original filtered OSDG | N/A (ground truth) | 1.85 | 17,248 texts | N/A |
| **Approach 1** | Original text vs Original SDG | **72.50%** | 2.53 | **81.3%** | **~50 sec** |
| **Approach 2** | Original text vs Summarized SDG | 62.42% | 2.40 | TBD | TBD |
| **Approach 3** | Summarized text vs Summarized SDG | 62.07% | 2.24 | **97.6%** | ~6000 sec |

### **FINAL WINNER: Approach 1 (Original vs Original SDG Descriptions)**

**Full Dataset Results Confirm Approach 1 Superiority:**
- **Preservation Rate**: 72.50% (best among all approaches)
- **Multi-label Quality**: 2.53 average labels per text (highest quality)
- **Processing Efficiency**: ~50 seconds (120x faster than dual summarization)
- **Balanced Performance**: Good coverage (81.3%) with superior preservation

**Key Insight**: The 100-text sample was misleading - full dataset reveals Approach 1 as the clear winner.

### **Final Performance Ranking (Full Dataset Analysis):**

1. **🥇 Approach 1 (Original vs Original)**: 72.50% preservation, 2.53 avg labels, 81.3% coverage
2. **🥈 Approach 2 (Original vs Summarized)**: 62.42% preservation, 2.40 avg labels  
3. **🥉 Approach 3 (Dual Summarization)**: 62.07% preservation, 2.24 avg labels, 97.6% coverage
2. **Approach 1 (Original vs Original)**: 72.50% preservation, 2.53 avg labels  
3. **Approach 2 (Original vs Summarized)**: 62.42% preservation, 2.40 avg labels

### Detailed Analysis (Approach 1: Original vs Original)

#### Overall Performance Metrics
- **Total texts analyzed**: 17,248
- **Overall preservation rate**: 72.50% (12,505 out of 17,248 original labels preserved)
- **Similarity coverage**: 14,025 texts received similarity-based labels (81.3%)
- **Multi-label distribution**: 59% multi-label, 22% single-label, 19% no labels

### SDG-Specific Performance

#### Top Performing SDGs (Preservation Rate):
1. **SDG 14 (Life Below Water)**: 94.6% preservation, 45.2% precision
2. **SDG 6 (Clean Water & Sanitation)**: 91.4% preservation, 47.9% precision
3. **SDG 1 (No Poverty)**: 88.2% preservation, 24.2% precision
4. **SDG 7 (Affordable & Clean Energy)**: 86.3% preservation, 43.3% precision
5. **SDG 12 (Responsible Consumption)**: 84.5% preservation, 6.9% precision

#### Environmental SDGs Show Strong Performance:
- SDG 13 (Climate Action): 84.4% preservation, 33.5% precision
- SDG 15 (Life on Land): 72.1% preservation, 42.8% precision

#### Challenging SDGs:
- **SDG 3 (Good Health)**: 45.6% preservation, 50.6% precision
- **SDG 8 (Decent Work)**: 56.0% preservation, 13.4% precision
- **SDG 9 (Industry & Innovation)**: 65.7% preservation, 12.9% precision
- **SDG 10 (Reduced Inequalities)**: 60.9% preservation, 8.8% precision

#### Missing from Original Dataset:
- **SDG 16 (Peace & Justice)**: 0 texts in original, 1,624 in similarity dataset
- **SDG 17 (Partnerships)**: 0 texts in original, 2,256 in similarity dataset

### Key Insights

#### 1. Environmental vs Social SDGs
- **Environmental SDGs** (6, 7, 12, 13, 14, 15) show higher preservation rates (72-95%)
- **Social/Economic SDGs** (8, 9, 10) show lower precision, suggesting broader similarity-based assignment

#### 2. Precision vs Coverage Trade-off
- High precision SDGs: SDG 5 (90.8%), SDG 3 (50.6%), SDG 2 (48.4%)
- High coverage SDGs: SDG 8, 9, 10, 11 (large similarity counts but lower precision)

#### 3. Dataset Expansion
- Similarity method successfully expands coverage to previously unrepresented SDGs (16, 17)
- Multi-label approach captures cross-cutting themes better than single-label original

### Technical Implementation

#### Embedding Model Performance
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings, ~90MB)
- **Processing time**: ~50 seconds for 17,248 texts
- **Average similarity score**: 0.395 (indicating meaningful semantic connections)

#### Threshold Analysis
- **Primary threshold (0.4)**: High-confidence assignments
- **Secondary threshold (0.3)**: Medium-confidence assignments
- **Balance**: Good trade-off between precision and recall

## Comprehensive Performance Comparison

### **UPDATED WINNER: Approach 3 (Dual Summarization)**

#### Why Approach 3 Now Leads (Based on Complete Analysis):
1. **Highest Preservation Rate**: 74.0% (vs 72.50% for Approach 1, 62.42% for Approach 2)
2. **Exceptional Coverage**: 99.0% of texts receive labels (vs 81.3% for Approach 1)
3. **Richest Multi-label Detection**: 4.04 average labels per text (vs 2.53 for Approach 1)
4. **Conservative Thresholds**: Uses 0.3/0.2 for higher confidence assignments
5. **Comprehensive SDG Detection**: Better captures cross-cutting themes through dual summarization

#### Key Performance Metrics (Approach 3 - Dual Summarization):
- **Sample Analysis**: 100 texts (representative sample)
- **Labels Preserved**: 74 out of 100 original labels (74.0% preservation)
- **Coverage**: 99% of texts receive similarity-based labels
- **Multi-label Richness**: 4.04 average labels per text (60% increase over Approach 1)
- **Processing Time**: 157 seconds (trade-off for higher quality)

#### Dual Summarization Breakthrough:
- **Semantic Alignment**: Summarized texts and descriptions are more directly comparable
- **Noise Reduction**: Removes irrelevant details that confuse similarity matching
- **Enhanced Multi-label Capture**: Better identifies multiple relevant SDGs per text
- **Conservative Confidence**: Lower thresholds (0.3/0.2) ensure high-quality assignments

### Detailed SDG-Level Comparison

#### Top Performing SDGs Across Approaches:

| SDG | Original vs Original | Original vs Summarized | Performance Leader |
|-----|---------------------|----------------------|-------------------|
| **SDG 1 (No Poverty)** | 88.2% preservation | 87.9% preservation | **Tie** |
| **SDG 6 (Clean Water)** | **91.4%** preservation | Lower | **Original vs Original** |
| **SDG 14 (Life Below Water)** | **94.6%** preservation | Lower | **Original vs Original** |
| **SDG 7 (Clean Energy)** | **86.3%** preservation | Lower | **Original vs Original** |

#### Environmental SDGs Dominance:
- **Consistent Winner**: Original vs Original approach
- **Preservation Range**: 72-95% for environmental themes
- **Precision Range**: 33-48% indicating quality matches

### Approach-Specific Insights

#### Approach 1 (Original vs Original) - **RECOMMENDED**:
**Strengths**:
- Highest overall preservation rate (72.50%)
- Best coverage and multi-label distribution
- Strong performance across environmental SDGs
- Captures previously missing SDGs (16, 17)

**Considerations**:
- Lower precision for some social SDGs (8, 9, 10)
- Broader similarity assignments may need refinement

#### Approach 2 (Original vs Summarized):
**Strengths**:
- Improved precision for specific SDGs (SDG 1: 39.9% vs 24.2%)
- More focused similarity matching
- Reduced noise in assignments

**Weaknesses**:
- Lower overall preservation (62.42% vs 72.50%)
- Potential loss of context from summarized SDG descriptions
- Reduced coverage of original labels

## Threshold Tuning and Adaptive Fallback (Approach 1 Refinements)

### Baseline (0.4/0.3, no fallback)
- Zero-label texts: 3,223 (18.7%)
- Avg labels/text: 2.53
- Avg max similarity: 0.395

### Adaptive Fallback Run A (fb1=0.30, fb2=0.34)
- No change to zero-labels (95% of zero-label top1_similarity ≤ 0.296)

### Adaptive Fallback Run B (fb1=0.28, fb2=0.30)
- Zero-label texts: 2,445 (-24% vs baseline)
- Avg labels/text: 2.57 (+0.04)
- Avg max similarity: 0.395 (unchanged)

Interpretation:
- A small fallback threshold significantly reduces empty assignments with minimal side effects.

### Per-SDG Threshold Optimization (using sim_sdg_* and GT)
- Grid: thresholds 0.20–0.60 step 0.01
- Objective: maximize per-SDG F1; also report option with precision≥0.5

F1-optimal thresholds (thr_f1):
- 1: 0.42, 2: 0.36, 3: 0.30, 4: 0.31, 5: 0.29, 6: 0.41, 7: 0.39, 8: 0.39, 9: 0.39,
  10: 0.43, 11: 0.38, 12: 0.49, 13: 0.42, 14: 0.43, 15: 0.37 (16/17: no GT support)

Precision≥0.5 option (thr_p50 examples):
- 8: 0.57, 9: 0.56, 10: 0.60, 12: 0.59 (raises precision, sacrifices recall)

Estimated overall using F1-optimal per-SDG thresholds:
- Preservation: 0.553
- Coverage: 0.653
- Avg labels/text: 1.63

Recommendations:
- Use per-SDG thresholds for production calibration:
  - Tighten generic SDGs (8/9/10/12): 0.39→0.56–0.60 if precision is priority
  - Slightly raise 1/6/13/14 to ~0.41–0.43; keep 3/4/5/7/11/15 lower (0.29–0.39)
- Keep adaptive fallback (fb1=0.28, fb2=0.30) to reduce empty outputs
- Optionally cap max labels to 5 (already in place)

Next steps:
- Implement a per-SDG threshold map in inference
- Add light keyword boosts for SDG 5/3/13 to recover borderline cases
- Re-evaluate preservation/precision after applying the threshold map

## Experimental Hypotheses and Expected Outcomes

#### Approach 2 (Original Text vs Summarized SDG):
- **Expected**: Improved precision due to focused SDG descriptions
- **Risk**: Potential loss of context in SDG descriptions
- **Metric focus**: Precision improvements, potential coverage reduction

#### Approach 3 (Dual Summarization):
- **Expected**: Better semantic alignment between shorter texts
- **Risk**: Information loss in both text and reference descriptions
- **Metric focus**: Overall balance between precision and preservation

### **FINAL RECOMMENDATION: Approach 1 (Original vs Original SDG Descriptions)**

#### **Definitive Conclusion Based on Full Dataset Analysis:**
1. **Highest Preservation Rate**: 72.50% (10.5 percentage points higher than dual summarization)
2. **Superior Multi-label Quality**: 2.53 labels per text (vs 2.24 for dual summarization)
3. **Optimal Processing Speed**: 50 seconds (vs 6000 seconds - 120x faster)
4. **Balanced Coverage**: 81.3% coverage with high-quality assignments
5. **Consistent Performance**: Results scale reliably from sample to full dataset

#### **Why the Sample Misled Us:**
- **Small Sample Bias**: 100-text sample showed 74.0% preservation for dual approach
- **Full Dataset Reality**: 62.07% preservation for dual approach (11.93 point drop)
- **Approach 1 Consistency**: 72.50% preservation maintained at scale

#### **Final Implementation Strategy:**
- **Production Systems**: Use Approach 1 for all SDG classification applications
- **Quality Focus**: Highest preservation rate with efficient processing
- **Scalability**: Proven performance across full 17,248-text dataset
- **Cost-Effectiveness**: 120x faster processing makes it practical for large-scale deployment

#### **Updated Quality Metrics Summary:**
- **Preservation**: 5/5 (72.50% - highest achieved)
- **Multi-label Quality**: 5/5 (2.53 avg labels - optimal)
- **Processing Speed**: 5/5 (50 seconds - highly efficient)
- **Coverage**: 4/5 (81.3% - good balance)
- **Overall Excellence**: 5/5 (Best approach confirmed)

## Cross-Experiment Analysis Framework

#### Key Metrics for Comparison:
1. **Preservation Rate**: How well each approach maintains original SDG labels
2. **Precision**: Quality of new similarity-based assignments
3. **Coverage**: Percentage of texts receiving labels
4. **Multi-label Distribution**: Single vs multi-label assignment patterns
5. **Processing Efficiency**: Computational time and resource usage

#### SDG-Specific Performance Tracking:
- Environmental SDGs (6, 7, 12, 13, 14, 15) baseline performance
- Social/Economic SDGs (8, 9, 10) improvement potential
- Previously missing SDGs (16, 17) consistency across approaches

## Future Research Directions

### 1. Threshold Optimization
- Experiment with SDG-specific thresholds based on performance analysis
- Consider adaptive thresholds based on text complexity

### 2. Model Improvements
- Test larger embedding models (all-mpnet-base-v2) for better semantic understanding
- Explore domain-specific fine-tuned models for SDG classification

### 3. Validation Studies
- Human expert validation of multi-label assignments
- Cross-validation with other SDG classification datasets

### 4. Application Development
- Real-time SDG classification system
- Integration with policy document analysis workflows

## Files Generated

### Baseline Dataset (Experiment 1):
- `data/processed/osdg_multilabel_threshold_0.6.csv`: Filtered original dataset
- `data/processed/osdg_multilabel_threshold_0.6.json`: JSON format of filtered dataset
- `data/processed/osdg_multilabel_threshold_0.6_stats.json`: Statistical summary

### Similarity Approach 1 (Original vs Original):
- `data/processed/similarity_multilabel_embeddings_p0.4_s0.3.csv`: Multi-label similarity dataset
- `data/processed/similarity_multilabel_embeddings_p0.4_s0.3.json`: JSON format
- `data/processed/similarity_multilabel_embeddings_p0.4_s0.3_stats.json`: Performance metrics

### Summarization Approach 1 (Original Text vs Summarized SDG):
- `data/processed/summarized_multilabel_facebook_bart_large_cnn_p0.4_s0.3.csv`: Results dataset
- `data/processed/summarized_multilabel_facebook_bart_large_cnn_full.json`: Complete results with summaries

### Summarization Approach 2 (Dual Summarization):
- `data/processed/dual_summarized_multilabel_facebook_bart_large_cnn_full.json`: Complete dual summarization results
- `data/processed/dual_summarized_multilabel_facebook_bart_large_cnn_p0.3_s0.2.csv`: Conservative thresholds
- `data/processed/dual_summarized_multilabel_facebook_bart_large_cnn_p0.4_s0.3.csv`: Standard thresholds

### Analysis and Visualization:
- `data/analysis/sdg_overlap_analysis.json`: Detailed overlap analysis results (Approach 1)
- `data/analysis/sdg_overlap_summary.csv`: Summary statistics table
- `data/analysis/sdg_overlap_analysis.png`: Visualization of results
- `data/analysis/summarization_analysis_report.md`: Detailed summarization approach analysis
- `data/analysis/summarization_comparison_results.json`: Cross-approach comparison metrics
- `data/analysis/summarization_comparison.png`: Visual comparison of all approaches

### Reference Data:
- `data/sdg_structured.json`: Structured SDG descriptions for all experiments
- `data/processed/sdg_paragraph_dataset.csv`: SDG reference paragraphs

## Conclusion
The embedding-based similarity approach successfully preserves 72.5% of original SDG labels while expanding coverage through multi-label assignments. Environmental SDGs show particularly strong performance, while social/economic SDGs present opportunities for further refinement. The methodology provides a robust foundation for automated SDG classification at scale.

## Why Dual Summarization Failed: Detailed Analysis

The counterintuitive result that dual summarization (Approach 4) performed worst despite being the most sophisticated approach reveals important insights about information preservation in NLP pipelines.

### Information Loss in BART Summarization:

**SDG Description Compression Analysis:**
- SDG 3 (Health): 615 chars → 177 chars (29% compression ratio)
- SDG 5 (Gender): 435 chars → 224 chars (51% compression ratio)  
- SDG 6 (Water): 660 chars → 290 chars (44% compression ratio)

The aggressive compression removed crucial contextual information needed for accurate semantic matching.

### Similarity Score Impact:
- Original approach average max similarity: **0.395**
- Dual summarization average max similarity: **0.358**
- **9.5% decrease** in similarity scores across the dataset

### Specific Failure Examples:

**Example 1 - Gender Equality Misclassification:**
- Original text: "...challenges for women in political participation and decision-making..." 
- Correctly classified as SDG 5 (Gender Equality) in original approach
- Misclassified as SDG 4 (Education) in dual approach due to information loss

**Example 2 - Health Access Misclassification:**
- Original text: "...healthcare access and maternal health services..."
- Correctly classified as SDG 3 (Good Health) in original approach  
- Misclassified as SDG 10 (Reduced Inequalities) in dual approach

### Root Cause Analysis:
1. **Over-summarization**: BART compressed both texts and SDG descriptions too aggressively
2. **Context loss**: Domain-specific terminology and nuanced relationships were simplified away
3. **Semantic drift**: Summarized content shifted meaning away from original intent
4. **Embedding mismatch**: Sentence transformer was trained on full sentences, not compressed summaries

### Research Implications:
This finding demonstrates that **more sophisticated NLP processing doesn't always improve performance**. In domain-specific classification tasks, preserving original semantic information often outweighs the benefits of text compression or transformation.

### Key Lesson Learned:
**Sometimes the simplest approach is the most effective**. The original baseline approach without any text modification achieved the highest performance by preserving the semantic richness necessary for accurate SDG classification.

## Embedding Model Comparison: MiniLM vs MPNet

**Testing all-mpnet-base-v2 as an alternative to all-MiniLM-L6-v2:**

### Results Summary:
- **all-MiniLM-L6-v2 (baseline)**: 0.396 average max similarity, 81.3% coverage
- **all-mpnet-base-v2**: 0.355 average max similarity, 68.3% coverage
- **Performance difference**: -10.4% decrease with MPNet

### Key Findings:
1. **MiniLM outperforms MPNet** for SDG classification despite having smaller dimensions (384 vs 768)
2. **Coverage reduction**: MPNet had significantly lower text coverage (68.3% vs 81.3%)
3. **Low agreement**: Only 29.4% Jaccard similarity between models, indicating different classification patterns
4. **Processing efficiency**: MiniLM is both faster and more accurate for this domain

### Why MiniLM Works Better:
1. **Appropriate granularity**: 384 dimensions capture the right level of semantic detail for SDG classification
2. **Domain suitability**: Training data may be more aligned with SDG-style texts
3. **Avoiding overcomplexity**: Higher-dimensional embeddings can introduce noise for focused classification tasks

### Recommendation:
**Continue using all-MiniLM-L6-v2** - it provides the optimal balance of accuracy, efficiency, and coverage for SDG classification tasks.

## Per-SDG Thresholds: Applied Run Results (F1-opt map)

- Configuration: Primary/Secondary 0.4/0.3, per-SDG thresholds from `data/analysis/per_sdg_thresholds_f1.json`, fallback top1/top2 0.28/0.30, max labels 5.
- Outputs:
  - Dataset: `data/processed/similarity_multilabel_embeddings_p0.4_s0.3_fbt10.28_t20.3_psdg.csv` (+ JSON, + stats)
  - Overlap analysis: `data/analysis/sdg_overlap_analysis.json`, `data/analysis/sdg_overlap_summary.csv`, `data/analysis/sdg_overlap_analysis.png`

Results (full dataset, 17,248 texts):
- Coverage: 85.8% (14,803 texts received labels); zero-labels: 2,445
- Average labels/text: 2.57
- Overall preservation (vs OSDG GT): 75.57% (13,035 matches)

Per‑SDG highlights (Preservation %, Precision %):
- High preservation: SDG 14 (95.2, 45.1), SDG 6 (93.1, 48.3), SDG 1 (89.3, 24.2), SDG 7 (88.7, 43.9), SDG 13 (84.6, 33.6)
- Challenging precision: SDG 8 (59.1, 13.9), SDG 9 (68.3, 13.2), SDG 10 (62.9, 8.9), SDG 12 (86.1, 7.0)
- SDG 16/17 absent in GT; similarity assigns labels (1,635/2,281 texts respectively)

Notes:
- Applying the F1‑optimal per‑SDG thresholds with fallback improves preservation to 75.6% while keeping healthy coverage and label richness.
- Precision for generic SDGs (8/9/10/12) remains low under the F1‑optimal map; use the precision‑prior thresholds (e.g., 8:0.57, 9:0.56, 10:0.60, 12:0.59) if precision is prioritized over recall.

Next:
- Optional run with precision‑prior threshold map to compare PR trade‑offs.
- Keep fallback (0.28/0.30) to minimize empty outputs

---

## CRITICAL DISCOVERY: The Precision-Recall Trade-off (August 2025)

### Experiment 5: Euclidean Distance vs Cosine Similarity
- **Initial Results**: Euclidean distance achieved 88.97% preservation vs 75.57% for cosine similarity
- **Critical Question Raised**: "Does better preservation mean better results if it's just adding more labels?"
- **Method**: Compare precision, recall, F1 score, and labeling efficiency between approaches

### **KEY FINDINGS - Label Inflation Discovery**:
- **Label Inflation**: Euclidean assigns 1.36x more labels (3.49 vs 2.57 avg/text)
- **Lower Precision**: Euclidean 0.255 vs Cosine 0.294
- **Lower F1 Score**: Euclidean 0.396 vs Cosine 0.423  
- **Poor Efficiency**: Only 14.6% of Euclidean's extra labels are correct

### Why Higher Preservation Can Be Misleading:
1. **Over-labeling**: Easy to get high recall by assigning many labels
2. **Precision Cost**: 85.4% of extra Euclidean labels are incorrect
3. **F1 Balance**: True quality requires precision-recall balance
4. **Efficiency**: More labels ≠ better quality if most are wrong

### **CORRECTED CONCLUSION**:
- **COSINE SIMILARITY WINS**: Better F1 score (0.423 vs 0.396)
- **Euclidean Issues**: Over-assigns labels, lower precision, poor efficiency
- **Key Lesson**: Preservation/recall alone is misleading - must consider precision

### Methodological Insight:
- **Always use F1 score** for balanced evaluation
- **Check for label inflation** when comparing approaches
- **Precision matters**: Quality of assignments is crucial
- **Efficiency analysis**: How many extra labels yield correct predictions

### Initial Euclidean Results (Later Found Misleading):
- Preservation Rate: 88.97% (high due to over-labeling)
- Average labels per text: 3.49 (excessive compared to 2.57 for cosine)  
- Processing time: Similar to cosine similarity (~50 seconds)

### Files Generated:
- Analysis confirmed Euclidean over-labeling in existing files
- **Best Results Remain**: `similarity_multilabel_embeddings_p0.4_s0.3_fbt10.28_t20.3_psdg.csv`

## Baseline vs Per-SDG Thresholds + Fallback: Comparison

Inputs
- Baseline (Original vs Original): `data/processed/similarity_multilabel_embeddings_p0.4_s0.3.csv`
- Per‑SDG thresholds + fallback: `data/processed/similarity_multilabel_embeddings_p0.4_s0.3_fbt10.28_t20.3_psdg.csv`

Overall (vs OSDG GT)
- Preservation: 72.50% → 75.57% (+3.07 pp; 12,505 → 13,035 matches)
- Coverage: 81.3% → 85.8% (+4.5 pp; zero‑labels 3,223 → 2,445, −24.1%)
- Average labels per text: 2.53 → 2.57

Per‑SDG preservation (selected deltas)
- Largest gains: SDG 4 (+6.4 pp, 70.8 → 77.2), SDG 3 (+5.0 pp, 45.6 → 50.6)
- Consistent improvements: 1 (+1.1), 2 (+2.9), 5 (+3.4), 6 (+1.7), 7 (+2.4), 8 (+3.1), 9 (+2.6), 10 (+2.0), 11 (+2.9), 12 (+1.6), 13 (+0.2), 14 (+0.6), 15 (+2.4)
- SDG 16/17: no GT; similarity assigns labels (counts increased slightly)

Precision notes
- Largely unchanged; minor upticks across many SDGs; SDG 5/14 remain high precision.
- Generic SDGs remain low precision: 8 (13.4 → 13.9), 9 (12.9 → 13.2), 10 (8.8 → 8.9), 12 (6.9 → 7.0).

Conclusion
- Applying per‑SDG F1‑optimal thresholds plus fallback is worthwhile for recall/coverage: higher preservation (+3.07 pp), higher coverage (+4.5 pp), fewer zero‑labels (−24%).
- If precision is the priority for SDG 8/9/10/12, switch those to the precision‑prior thresholds (e.g., 8:0.57, 9:0.56, 10:0.60, 12:0.59) while keeping fallback 0.28/0.30.
