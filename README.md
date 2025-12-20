# Emotion Recognition and Emotion-Dynamics Analysis Pipeline

This repository contains the complete codebase used for training emotion recognition models, performing cross-corpus evaluations, and computing emotion-dynamic features for depression analysis.

The project explores two modeling pipelines—a Multi-Layer Perceptron (MLP) and a Recurrent Neural Network (RNN)—and evaluates their performance across multiple datasets and experimental settings.

---

## Repository Overview

This repository includes code for:

- Grid search for hyperparameter optimization  
- Model training for MLP- and RNN-based architectures  
- Within-dataset and cross-corpus evaluation  
- Inference on a depression dataset (E-DAIC)  
- Emotion-dynamic feature computation and statistical analysis (R)  

---

## Datasets Used

The following datasets are used in this project:

- **CREMA-D** – Used for training and evaluation  
- **RAVDESS** – Used for training and evaluation  
- **COMBINED dataset** – A merged version of CREMA-D and RAVDESS  
- **E-DAIC** – Used only for inference and downstream emotion-dynamics analysis  

Models are trained on:
- CREMA-D only  
- RAVDESS only  
- The COMBINED dataset  

---

## Modeling Pipelines

Two emotion recognition pipelines are explored:

### 1. MLP-Based Pipeline
- Frame-level emotion classification  
- Hyperparameters selected via grid search  
- Best model selected based on validation performance  

### 2. RNN-Based Pipeline
- Sequence-based emotion modeling  
- Uses LSTM / GRU architectures  
- Hyperparameters selected via grid search  
- Best model selected based on sequence-level performance  

---

## Hyperparameter Optimization

- Grid search is performed separately for:
  - CREMA-D
  - RAVDESS
  - COMBINED dataset
- Best hyperparameter configurations for both MLP and RNN models are selected based on validation accuracy.
- Poorly performing feature combinations and architectures are pruned in later grid searches to reduce computational cost.

---

## Evaluation Strategy

The best-performing models are evaluated using:

### Within-Dataset Evaluation
- Training and testing on the same dataset (e.g., CREMA-D → CREMA-D)

### Cross-Corpus Evaluation
- Training on one dataset and testing on another:
  - CREMA-D → RAVDESS
  - RAVDESS → CREMA-D

This setup evaluates model generalization across datasets.

---

## Inference on E-DAIC

- The final best-performing models trained on the COMBINED dataset are used to perform emotion inference on the E-DAIC dataset.
- Predicted per-frame emotion probabilities are saved and used as input for emotion-dynamics analysis.

---

## Emotion-Dynamics Analysis

- Emotion-dynamic features computed include:
  - Emotional intensity  
  - Variability  
  - Inertia  
  - Instability  
  - Cross-lagged dependencies (emotion network density)  

- Statistical analyses are implemented in **R**.

The R scripts for emotion-dynamics computation and analysis are provided in the folder "Organized_Codes"

## Technologies Used

- **Python (PyTorch)** – Model training, evaluation, and inference  
- **R** – Statistical analysis and emotion-dynamics modeling  
- **CUDA-enabled GPUs** – Used for large-scale training and grid search  

---

## Notes

- This repository contains research code developed for a thesis project.
- The models operate on **privacy-preserving facial feature representations** rather than raw images or videos.
- Emotion-dynamic features are **not intended as diagnostic tools**, but as research markers for studying affective patterns.
