"""
Temporal model: a small GRU over per-participant behaviour sequences -> PHQ-8.

Runs in two modes from the same code:
    - raw:          uses sequences as-is
    - personalised: uses deviation-from-own-baseline sequences

Training both and comparing them (and comparing against the averaging baseline)
is the disciplined benchmark the project calls for. The trained model and its
test predictions are saved to results/ for the uncertainty and explainability
steps that follow.

Run from the project root, e.g.:
    python -m src.model --mode personalised
    python -m src.model --mode raw
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.personalise import build_sequences, FEATURES

RESULTS_DIR = Path("results")
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tensors(sequences, info, split_name):
    """Stack the sequences for one split into padded tensors.

    All synthetic sequences share the same length, so we can stack directly.
    Returns X (n, frames, features), y (n,), and the matching info rows.
    """
    mask = (info["split"] == split_name).to_numpy()
    idx = np.where(mask)[0]
    X = np.stack([sequences[i] for i in idx]).astype("float32")
    y = info.loc[mask, "phq8"].to_numpy().astype("float32")
    rows = info.loc[mask].reset_index(drop=True)
    return torch.from_numpy(X), torch.from_numpy(y), rows


class GRURegressor(nn.Module):
    """Small GRU followed by a linear head predicting a single PHQ-8 value.

    Dropout is included so the same architecture supports MC-dropout uncertainty
    in the next step.
    """
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]          # final time-step summary
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def standardise(X_train, *others):
    """Standardise features using training-set statistics only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(0) + 1e-6
    out = [(X_train - mean) / std]
    for X in others:
        out.append((X - mean) / std)
    return out, (mean, std)


def train(mode, epochs=60, lr=1e-2):
    set_seed()
    personalise = (mode == "personalised")
    sequences, info = build_sequences(personalise=personalise)

    X_train, y_train, _ = make_tensors(sequences, info, "train")
    X_val, y_val, _ = make_tensors(sequences, info, "val")
    X_test, y_test, test_rows = make_tensors(sequences, info, "test")

    (X_train, X_val, X_test), _ = standardise(X_train, X_val, X_test)

    model = GRURegressor(n_features=len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()             # mean absolute error, robust and interpretable

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Final test evaluation.
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).numpy()
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    pred_label = (test_pred >= 10).astype(int)
    true_label = (y_test.numpy() >= 10).astype(int)
    acc = (pred_label == true_label).mean()

    print(f"=== Temporal GRU ({mode}) ===")
    print(f"best val MAE={best_val:.2f}")
    print(f"test:  MAE={mae:5.2f}   RMSE={rmse:5.2f}   depression-accuracy={acc:.2%}")

    # Save model + test artifacts for the uncertainty/explainability steps.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / f"model_{mode}.pt")
    np.savez(
        RESULTS_DIR / f"test_{mode}.npz",
        X_test=X_test.numpy(),
        y_test=y_test.numpy(),
        pred=test_pred,
        participant_id=test_rows["participant_id"].to_numpy(),
        gender=test_rows["gender"].to_numpy(),
    )
    print(f"Saved model to results/model_{mode}.pt and test data to results/test_{mode}.npz")
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "personalised"],
                        default="personalised")
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()