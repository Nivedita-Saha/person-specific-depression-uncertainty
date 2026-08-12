"""
Sanity check for the synthetic data: confirm the depression signal is present.

Compares average positive-affect action-unit activity during the interaction
phase between depressed and non-depressed participants, and saves a simple
figure. Run from the project root:
    python -m src.inspect_data
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # save figures without opening a window
import matplotlib.pyplot as plt

DATA_DIR = Path("data/synthetic")
RESULTS_DIR = Path("results")
POSITIVE_AUS = ["AU06_cheek_raiser", "AU12_lip_corner"]


def main():
    features = pd.read_csv(DATA_DIR / "features.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")

    # Keep only the interaction phase, where the reaction signal lives.
    interaction = features[features["phase"] == "interaction"].copy()

    # Average each participant's positive-affect AUs across the session.
    per_person = (
        interaction
        .groupby("participant_id")[POSITIVE_AUS]
        .mean()
        .reset_index()
    )
    per_person["positive_affect"] = per_person[POSITIVE_AUS].mean(axis=1)

    # Attach the depression label.
    merged = per_person.merge(
        labels[["participant_id", "depressed", "phq8"]],
        on="participant_id",
    )

    grp = merged.groupby("depressed")["positive_affect"].agg(["mean", "std", "count"])
    print("Average positive-affect activity (interaction phase):")
    print(grp.to_string())
    print()

    corr = merged["positive_affect"].corr(merged["phq8"])
    print(f"Correlation between positive-affect activity and PHQ-8: {corr:.3f}")
    print("(A clear negative value means higher PHQ-8 -> less positive expression, as intended.)")

    # Save a figure: positive affect vs PHQ-8.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["phq8"], merged["positive_affect"], alpha=0.6)
    ax.set_xlabel("PHQ-8 score")
    ax.set_ylabel("Positive-affect activity (interaction)")
    ax.set_title("Synthetic data: depression signal check")
    fig.tight_layout()
    out_path = RESULTS_DIR / "signal_check.png"
    fig.savefig(out_path, dpi=120)
    print("Figure saved to:", out_path.resolve())


if __name__ == "__main__":
    main()