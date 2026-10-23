"""
baseline_model.py
Defines the standard (non-private) Multi-Layer Perceptron (MLP) architecture
and training pipeline for Student Dropout Prediction.
Serves as the empirical upper-bound benchmark for classification performance.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT_RATE,
        EPOCHS, LEARNING_RATE_BASELINE, WEIGHT_DECAY, RANDOM_SEED
    )
except ImportError:
    from config import (
        INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT_RATE,
        EPOCHS, LEARNING_RATE_BASELINE, WEIGHT_DECAY, RANDOM_SEED
    )


class StudentDropoutMLP(nn.Module):
    """
    Multi-Layer Perceptron for Student Dropout and Academic Success prediction.
    
    Architecture:
      Input (36 features)
        -> Linear(36, hidden_dim1) -> ReLU -> Dropout(dropout_rate)
        -> Linear(hidden_dim1, hidden_dim2) -> ReLU -> Dropout(dropout_rate * 0.7)
        -> Linear(hidden_dim2, 3) -> Logits for {0: Dropout, 1: Enrolled, 2: Graduate}
        
    Design Rationale for Privacy Compatibility:
      Avoids Batch Normalization layers. Batch Normalization couples gradients across
      different samples in a mini-batch, violating sample independence required for
      per-sample gradient clipping in Differentially Private SGD (DP-SGD).
      Linear + ReLU + Dropout architectures are natively compatible with Opacus.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim1=HIDDEN_DIM1,
                 hidden_dim2=HIDDEN_DIM2, output_dim=OUTPUT_DIM,
                 dropout_rate=DROPOUT_RATE):
        super(StudentDropoutMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed=RANDOM_SEED):
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE_BASELINE,
                         weight_decay=WEIGHT_DECAY, device="cpu", verbose=True):
    """
    Trains the baseline (non-private) neural network.

    Parameters
    ----------
    model : nn.Module
        An instance of StudentDropoutMLP.
    train_loader : DataLoader
        DataLoader providing training batches (X_batch, y_batch).
    epochs : int, optional
        Number of training epochs (default 25).
    lr : float, optional
        Learning rate (default 0.005).
    weight_decay : float, optional
        L2 regularization weight decay (default 1e-4).
    device : str, optional
        Execution device ('cpu' or 'cuda').
    verbose : bool, optional
        Whether to print epoch progress.

    Returns
    -------
    model : nn.Module
        Trained model.
    loss_history : list of float
        Average training loss per epoch.
    training_time : float
        Elapsed training wall-clock time in seconds.
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    start_time = time.perf_counter()

    if verbose:
        print("\n" + "="*50)
        print("Training Baseline (Non-Private) Model")
        print("="*50)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        epoch_loss = total_loss / total_samples
        loss_history.append(epoch_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs or epoch == 1):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Training Loss: {epoch_loss:.4f}")

    elapsed_time = time.perf_counter() - start_time
    if verbose:
        print(f"Baseline Training Completed in {elapsed_time:.2f} seconds.")

    return model, loss_history, elapsed_time


if __name__ == "__main__":
    from preprocessing import get_preprocessed_pipeline
    pipeline = get_preprocessed_pipeline()
    model = StudentDropoutMLP()
    trained_model, losses, t = train_baseline_model(model, pipeline["train_loader"], epochs=5)
    print("Baseline test run successful! Final loss:", losses[-1])
