#!/usr/bin/env python3
"""
Euclidean Distance + Min-Max Scaling with SDG Summarization Experiment
=====================================================================

This script evaluates whether using summaries of SDG goal+targets as reference improves classification.

- Summarizes each SDG using facebook/bart-large-cnn (HuggingFace Transformers)
- Runs the same threshold validation pipeline as the main experiment
- All other logic is identical to euclidean_minmax_threshold_validation.py

Author: Research Team
Date: August 29, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class EuclideanMinMaxSummarizationExperiment:
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2', summarizer_model='facebook/bart-large-cnn'):
        self.data_path = data_path
        self.model_name = model_name
        self.summarizer_model = summarizer_model
        self.model = None
        self.texts = None
        self.true_labels = None
        self.text_embeddings = None
        self.sdg_summaries = None
        self.sdg_embeddings = None
        self.raw_distances = None
        self.scaled_distances = None
        self.scaler = MinMaxScaler()

    def load_data_and_model(self):
        # Load OSDG dataset
        osdg_path = os.path.join(self.data_path, 'data', 'processed', 'osdg_multilabel_threshold_0.6.csv')
        if not os.path.exists(osdg_path):
            raise FileNotFoundError(f"OSDG dataset not found at {osdg_path}")
        df = pd.read_csv(osdg_path)
        self.texts = df['text'].tolist()
        label_columns = [col for col in df.columns if col.startswith('sdg_') and col != 'sdg_labels']
        self.true_labels = df[label_columns].values.astype(int)

        # Load SDG descriptions
        sdg_data_path = os.path.join(self.data_path, 'data', 'processed', 'sdg_paragraph_dataset.csv')
        if not os.path.exists(sdg_data_path):
            raise FileNotFoundError(f"SDG dataset not found at {sdg_data_path}")
        sdg_df = pd.read_csv(sdg_data_path)
        self.sdg_texts = sdg_df['text'].tolist()[:17]

        # Load embedding model
        self.model = SentenceTransformer(self.model_name)

    def generate_sdg_summaries(self, max_length=60, min_length=20):
        print(f"Summarizing SDGs using {self.summarizer_model}...")
        summarizer = pipeline('summarization', model=self.summarizer_model)
        self.sdg_summaries = []
        for text in self.sdg_texts:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            self.sdg_summaries.append(summary)
        print("SDG summarization completed!")

    def generate_embeddings(self):
        print("Generating text embeddings...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, batch_size=32)
        print("Generating SDG summary embeddings...")
        self.sdg_embeddings = self.model.encode(self.sdg_summaries, show_progress_bar=True)

    def calculate_euclidean_distances(self):
        self.raw_distances = euclidean_distances(self.text_embeddings, self.sdg_embeddings)

    def apply_minmax_scaling(self):
        distances_reshaped = self.raw_distances.reshape(-1, 1)
        distances_scaled_reshaped = self.scaler.fit_transform(distances_reshaped)
        self.scaled_distances = distances_scaled_reshaped.reshape(self.raw_distances.shape)

    def evaluate_single_threshold(self, primary_thresh, secondary_thresh=None):
        predictions = []
        for i in range(len(self.scaled_distances)):
            text_distances = self.scaled_distances[i]
            assigned_labels = []
            primary_matches = np.where(text_distances <= primary_thresh)[0]
            assigned_labels.extend(primary_matches)
            if secondary_thresh is not None and secondary_thresh > primary_thresh:
                secondary_matches = np.where((text_distances <= secondary_thresh) & (text_distances > primary_thresh))[0]
                assigned_labels.extend(secondary_matches[:2])
            binary_pred = np.zeros(len(self.sdg_summaries))
            for label_idx in assigned_labels:
                binary_pred[label_idx] = 1
            predictions.append(binary_pred)
        predictions = np.array(predictions)
        macro_f1 = f1_score(self.true_labels, predictions, average='macro', zero_division=0)
        accuracy = np.mean([
            np.where(self.true_labels[i] == 1)[0][0] in np.where(predictions[i] == 1)[0]
            for i in range(len(predictions))
        ])
        avg_assignments = np.mean([pred.sum() for pred in predictions])
        return {
            'primary_threshold': primary_thresh,
            'secondary_threshold': secondary_thresh,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'avg_assignments': avg_assignments
        }

    def sweep_thresholds(self):
        # Only test (0.5, 0.6) as primary/secondary thresholds
        thresholds = [(0.5, 0.6)]
        results = []
        for primary_thresh, secondary_thresh in thresholds:
            result = self.evaluate_single_threshold(primary_thresh, secondary_thresh)
            results.append(result)
        self.results = pd.DataFrame(results)
        return self.results

    def run_complete_experiment(self):
        self.load_data_and_model()
        # --- With Summarization ---
        self.generate_sdg_summaries()
        self.generate_embeddings()
        self.calculate_euclidean_distances()
        self.apply_minmax_scaling()
        results_summarized = self.sweep_thresholds()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.data_path, 'experiments', 'euclidean_minmax_scaling', 'summarization_experiment', 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_path_sum = os.path.join(results_dir, f'summarization_experiment_results_{timestamp}_summarized.csv')
        results_summarized.to_csv(csv_path_sum, index=False)
        print(f"Results with summarization saved to: {csv_path_sum}")

        # --- Without Summarization (use original SDG texts) ---
        self.sdg_summaries = self.sdg_texts  # Use full SDG text as 'summary'
        self.generate_embeddings()
        self.calculate_euclidean_distances()
        self.apply_minmax_scaling()
        results_full = self.sweep_thresholds()
        csv_path_full = os.path.join(results_dir, f'summarization_experiment_results_{timestamp}_fulltext.csv')
        results_full.to_csv(csv_path_full, index=False)
        print(f"Results with full SDG text saved to: {csv_path_full}")
        return {'summarized': results_summarized, 'fulltext': results_full}

def main():
    data_path = "/Users/mahnoorzamir/Desktop/mitacs/project"
    experiment = EuclideanMinMaxSummarizationExperiment(data_path)
    experiment.run_complete_experiment()

if __name__ == "__main__":
    main()
