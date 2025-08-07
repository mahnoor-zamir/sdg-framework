# SDG Multi-Label Classification Research Notes

## Project Overview
This project focuses on creating and analyzing multi-label SDG classification datasets using text similarity approaches with embeddings.

## Dataset Creation Process

### 1. Original OSDG Dataset Processing
- **Source**: OSDG community dataset with agreement-based filtering
- **Filtering criteria**: Agreement score ≥ 0.6 and positive labels > negative labels
- **Result**: 17,248 high-quality texts from original 32,120 entries
- **Format**: Multi-label with individual SDG binary columns

### 2. SDG Reference Dataset Creation
- **Source**: Agenda 2030 PDF extraction
- **Processing**: Combined SDG goals and targets into paragraph format
- **Result**: 17 clean SDG descriptions for similarity matching
- **Purpose**: Reference text for semantic similarity calculations

### 3. Similarity-Based Multi-Label Dataset
- **Method**: Sentence transformer embeddings (`all-MiniLM-L6-v2`)
- **Similarity calculation**: Cosine similarity between OSDG texts and SDG descriptions
- **Thresholds**: Primary (0.4) and Secondary (0.3) similarity thresholds
- **Result**: Multi-label assignments with 2.53 average labels per text

## Analysis Results (SDG Label Overlap Analysis)

### Overall Performance Metrics
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

## Summarization Experiment Results (August 7, 2025)

### BART Summarization + Embedding Approach
**EXECUTIVE SUMMARY: FAILED EXPERIMENT - SIGNIFICANT PERFORMANCE DECLINE**

#### Key Metrics:
- **Overall preservation rate**: 62.4% (vs 72.5% baseline) - **10.1% decline**
- **Processing time**: 104s (vs 73s baseline) - 31s slower
- **Average labels per text**: 2.40 (vs 2.53 baseline)

#### Major Findings:

##### Critical Performance Degradations:
- **SDG 2 (Zero Hunger)**: 71.5% → 24.1% (**-47.4%** - Critical failure)
- **SDG 3 (Good Health)**: 45.6% → 17.7% (**-27.9%** - Critical failure)
- **SDG 14 (Life Below Water)**: 94.6% → 76.5% (**-18.1%**)
- **SDG 11 (Sustainable Cities)**: 68.6% → 52.5% (**-16.1%**)
- **SDG 4 (Quality Education)**: 70.8% → 57.5% (**-13.3%**)
- **SDG 8 (Decent Work)**: 56.0% → 43.2% (**-12.8%**)

##### Minimal Improvements:
- **SDG 10 (Reduced Inequalities)**: 60.9% → 62.9% (+2.1%)
- **SDG 15 (Life on Land)**: 72.1% → 73.4% (+1.3%)

##### Problematic SDG Analysis:
- Hypothesis: Summarization would help SDGs 8, 9, 10, 12
- **Reality**: Only 2/4 improved, with SDG 8 and 12 getting significantly worse
- **Conclusion**: Hypothesis rejected

#### Root Cause Analysis:

1. **Information Loss**: BART compression (1000→250 words avg) removed critical context
2. **Domain Mismatch**: BART trained on news; poor for scientific/policy content
3. **Vocabulary Simplification**: Technical terms essential for SDG matching were generalized
4. **Semantic Dilution**: Rich contextual information was summarized away

#### Lessons Learned:

1. **Summarization Paradox**: More focused text ≠ better classification when domain expertise matters
2. **Context is King**: Full text provides richer semantic signals than summaries
3. **Domain Specificity**: General summarization models can hurt specialized tasks
4. **Vocabulary Preservation**: Technical terminology is crucial for accurate SDG classification

### Recommendations Moving Forward:

#### ❌ Discontinued Approaches:
- BART or similar abstractive summarization for SDG classification
- General-purpose summarization models for domain-specific tasks

#### ✅ Validated Approaches:
- Baseline similarity approach with full text (72.5% preservation)
- Sentence transformer embeddings on complete documents

#### 🔬 Future Research Directions:
1. **Domain-Specific Models**: ClimateBERT, SustainabilityBERT
2. **Ensemble Methods**: Combine multiple embedding models
3. **Hybrid Approaches**: Dense + sparse retrieval
4. **Selective Processing**: Context-aware text preprocessing
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
- `data/processed/osdg_multilabel_threshold_0.6.csv`: Filtered original dataset
- `data/processed/similarity_multilabel_embeddings_p0.4_s0.3.csv`: Similarity-based multi-label dataset
- `data/analysis/sdg_overlap_analysis.json`: Detailed overlap analysis results
- `data/analysis/sdg_overlap_summary.csv`: Summary statistics table
- `data/analysis/sdg_overlap_analysis.png`: Visualization of results

## Conclusion
The embedding-based similarity approach successfully preserves 72.5% of original SDG labels while expanding coverage through multi-label assignments. Environmental SDGs show particularly strong performance, while social/economic SDGs present opportunities for further refinement. The methodology provides a robust foundation for automated SDG classification at scale.
