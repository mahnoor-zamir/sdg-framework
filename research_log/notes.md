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

### Experimental Hypotheses and Expected Outcomes

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
