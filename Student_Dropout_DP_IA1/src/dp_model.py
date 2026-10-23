"""
dp_model.py
Implements Differentially Private training for Student Dropout Prediction
using DP-SGD (Differentially Private Stochastic Gradient Descent) via Opacus.

Key DP-SGD Mathematical Mechanisms:
  1. Per-Sample Gradient Clipping:
     Each individual sample's gradient g_i is bounded:
       g_i_clipped = g_i / max(1, ||g_i||_2 / C)
     where C is the maximum gradient norm threshold (sensitivity bound Δ = C).
  2. Calibrated Gaussian Noise Addition:
     Noise sampled from N(0, σ^2 * C^2 * I) is added to the batch gradient sum:
       g_private = (1 / B) * ( Σ g_i_clipped + N(0, σ^2 * C^2 * I) )
  3. Privacy Accounting (Rényi Differential Privacy / RDP):
     Opacus tracks cumulative privacy expenditure across all training steps to compute
     the exact (ε, δ)-DP guarantee.
"""

import time
import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

# Suppress minor non-fatal Opacus experimental RNG warnings during academic testing
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")

try:
    from .config import (
        EPOCHS, LEARNING_RATE_DP, MAX_GRAD_NORM, TARGET_DELTA,
        TARGET_EPSILONS, WEIGHT_DECAY, RANDOM_SEED
    )
    from .baseline_model import StudentDropoutMLP, set_seed
except ImportError:
    from config import (
        EPOCHS, LEARNING_RATE_DP, MAX_GRAD_NORM, TARGET_DELTA,
        TARGET_EPSILONS, WEIGHT_DECAY, RANDOM_SEED
    )
    from baseline_model import StudentDropoutMLP, set_seed


def train_dp_model_with_target_epsilon(
    model,
    train_loader,
    target_epsilon,
    target_delta=TARGET_DELTA,
    epochs=EPOCHS,
    lr=LEARNING_RATE_DP,
    max_grad_norm=MAX_GRAD_NORM,
    device="cpu",
    verbose=True
):
    """
    Trains a Differentially Private model using Opacus DP-SGD calibrated to reach
    a specified target epsilon (ε) at target delta (δ) after the specified epochs.

    Parameters
    ----------
    model : nn.Module
        Base StudentDropoutMLP model architecture.
    train_loader : DataLoader
        PyTorch DataLoader with training data.
    target_epsilon : float
        Target privacy budget (e.g. 1.0, 3.0, 5.0, 10.0).
    target_delta : float, optional
        Target failure probability δ (default 1e-5).
    epochs : int, optional
        Training epochs (default 25).
    lr : float, optional
        Learning rate for optimizer (default 0.015).
    max_grad_norm : float, optional
        L2 clipping norm C (default 1.0).
    device : str, optional
        Compute device ('cpu' or 'cuda').
    verbose : bool, optional
        Whether to print epoch logs.

    Returns
    -------
    model : nn.Module
        Trained differentially private model.
    achieved_epsilon : float
        The actual achieved epsilon calculated by the privacy accountant.
    achieved_delta : float
        The privacy parameter delta used for accounting.
    loss_history : list of float
        Average training loss per epoch.
    training_time : float
        Elapsed training time in seconds.
    noise_multiplier : float
        The calibrated noise scale σ chosen by the privacy engine.
    """
    set_seed(RANDOM_SEED)
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    # DP-SGD performs best with standard SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    privacy_engine = PrivacyEngine()

    # Opacus make_private_with_epsilon wraps module, optimizer, and data_loader
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        loss_reduction="mean"
    )

    noise_multiplier = dp_optimizer.noise_multiplier
    loss_history = []
    start_time = time.perf_counter()

    if verbose:
        print("\n" + "="*60)
        print(f"Training Differentially Private Model (Target eps={target_epsilon}, delta={target_delta})")
        print(f"Calibrated Noise Multiplier sigma: {noise_multiplier:.4f} | Clipping Norm C: {max_grad_norm}")
        print("="*60)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in dp_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            dp_optimizer.zero_grad()
            outputs = dp_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            dp_optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        loss_history.append(epoch_loss)

        # Query accountant for current spent privacy budget
        current_eps = privacy_engine.get_epsilon(delta=target_delta)

        if verbose and (epoch % 5 == 0 or epoch == epochs or epoch == 1):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Accumulated (eps = {current_eps:.2f}, delta = {target_delta})")

    elapsed_time = time.perf_counter() - start_time
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)

    if verbose:
        print(f"DP Training Complete: Achieved eps = {final_epsilon:.2f}, delta = {target_delta} in {elapsed_time:.2f}s")

    return dp_model, final_epsilon, target_delta, loss_history, elapsed_time, noise_multiplier


def run_privacy_budget_sweep(train_loader, test_loader, epsilons=TARGET_EPSILONS,
                             target_delta=TARGET_DELTA, epochs=EPOCHS, lr=LEARNING_RATE_DP,
                             max_grad_norm=MAX_GRAD_NORM, device="cpu", verbose=True):
    """
    Iterates across a series of target privacy budgets (e.g. ε = 1, 3, 5, 10),
    training a separate model for each budget and collecting outputs for privacy-utility trade-off analysis.

    Parameters
    ----------
    train_loader : DataLoader
    test_loader : DataLoader
    epsilons : list of float
        Target epsilon values.
    target_delta : float
    epochs : int
    lr : float
    max_grad_norm : float
    device : str
    verbose : bool

    Returns
    -------
    dict of dicts
        Contains trained models, privacy guarantees, losses, and timings keyed by target epsilon.
    """
    sweep_results = {}

    for target_eps in epsilons:
        # Initialize fresh model architecture with fixed seed for fair comparison
        model = StudentDropoutMLP()
        dp_model, final_eps, final_delta, losses, t_time, noise_mult = train_dp_model_with_target_epsilon(
            model=model,
            train_loader=train_loader,
            target_epsilon=target_eps,
            target_delta=target_delta,
            epochs=epochs,
            lr=lr,
            max_grad_norm=max_grad_norm,
            device=device,
            verbose=verbose
        )

        sweep_results[target_eps] = {
            "model": dp_model,
            "target_epsilon": target_eps,
            "achieved_epsilon": final_eps,
            "achieved_delta": final_delta,
            "noise_multiplier": noise_mult,
            "loss_history": losses,
            "training_time": t_time
        }

    return sweep_results
