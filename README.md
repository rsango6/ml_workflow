# Clinical Data Analysis Pipeline
A code sample of the ML pipeline run on clinical samples for classification purposes

## Overview
This repository contains a robust, modular machine learning pipeline designed to predict clinical outcomes (Label) from high-dimensional patient data. 

Refactored from exploratory notebooks into a production-ready Object-Oriented framework, this tool standardizes the workflow from raw data ingestion to model evaluation. It is designed to ensure reproducibility and prevent common pitfalls like data leakage in imbalanced datasets.

## Key Features
* **Reproducible Architecture:** Uses a centralized `Config` class to manage paths, hyperparameters, and random seeds, ensuring every run is deterministic.
* **Leakage-Free Preprocessing:** Implements `imblearn.pipeline.Pipeline` to ensure SMOTE (oversampling) is applied *only* during training folds, while validation is performed on the original real-world distribution.
* **Automated EDA:** Automatically generates and saves distribution plots for target and categorical features to disk.
* **Production Logging:** Replaces standard print statements with Python's `logging` module for timestamped, graded execution tracking.
* **Model Agnostic:** seamlessly switches between Random Forest and XGBoost via command-line arguments.

## Technical Approach
The pipeline handles the end-to-end machine learning lifecycle:
1.  **Data Loading:** Ingests raw clinical CSVs.
2.  **Preprocessing:** * *Categorical:* Mode imputation + One-Hot Encoding (with unknown category handling).
    * *Imbalance:* SMOTE oversampling (applied strictly within training folds).
3.  **Modeling:** Supports RandomForest and XGBoost classifiers.
4.  **Evaluation:** Stratified K-Fold Cross-Validation reporting AUC, F1, and MCC scores.

## Installation

Ensure you have Python 3.8+ and the following dependencies installed:

```bash
uv venv
source .venv/bin/activate
uv pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
