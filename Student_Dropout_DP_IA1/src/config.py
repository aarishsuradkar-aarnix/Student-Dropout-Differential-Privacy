"""
config.py
Configuration module for the Student Dropout Prediction project with Differential Privacy.
Defines all global constants, model hyperparameters, privacy settings, and file paths.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASET_FILENAME = "data.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
UCI_DOI = "10.24432/C5MC89"

TARGET_COLUMN = "Target"
CLASS_MAPPING = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
NUM_CLASSES = 3

# Reproducibility Seed
RANDOM_SEED = 42

# Train / Test Splitting
TEST_SIZE = 0.20
STRATIFY = True

# Neural Network Architecture
INPUT_DIM = 36          # 36 predictor features in the dataset
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32
OUTPUT_DIM = 3          # 3 target classes: Dropout, Enrolled, Graduate
DROPOUT_RATE = 0.15     # Regularization dropout inside MLP

# Training Hyperparameters (Shared for Baseline and DP models)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE_BASELINE = 0.005
LEARNING_RATE_DP = 0.015     # DP-SGD often benefits from slightly larger LR due to noise regularization
WEIGHT_DECAY = 1e-4

# Differential Privacy Settings (Opacus DP-SGD)
MAX_GRAD_NORM = 1.0          # L2 gradient clipping threshold C
TARGET_DELTA = 1e-5          # δ condition: δ < 1/N (N_train ≈ 3539, so 1e-5 << 2.8e-4)
TARGET_EPSILONS = [1.0, 3.0, 5.0, 10.0]  # Standard privacy budgets for privacy-utility exploration
