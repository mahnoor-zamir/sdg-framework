# SDG QA System

## Project Structure

```
project/
  data/
    sdg_raw/           # Raw SDG docs, PDFs, indicator lists
    sdg_structured.json  # Hierarchical SDG data (Goal→Target→Indicator)
    qa_pairs.json      # Manual and augmented QA pairs
  scripts/             # All scripts (processing, modeling, etc.)
    data_collection.py
    data_processing.py
    qa_dataset.py
    augment.py
    train_qa.py
    retrieval.py
    pipeline.py
    evaluation.py
    webapp.py
  models/              # Saved/fine-tuned models
  outputs/             # Results, logs, metrics
  notebooks/           # For exploration and prototyping
  requirements.txt
  README.md
```

## Workflow Overview

1. **Data Collection & Structuring**: Gather and structure SDG data into hierarchical JSON.
2. **QA Dataset Creation**: Manually and automatically generate QA pairs.
3. **Model Training & Retrieval**: Fine-tune QA models and build retrieval system.
4. **End-to-End Pipeline**: Integrate retrieval and QA for query answering.
5. **Evaluation**: Implement F1, EM, BLEU, and error analysis.
6. **Web Interface**: User-friendly interface for querying the system.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Follow scripts in `scripts/` for each workflow step.

## Contribution
- Please follow modular code practices.
- Document all new scripts and data sources.

## License
See LICENSE file.
