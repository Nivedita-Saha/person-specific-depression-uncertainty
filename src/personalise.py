"""
Person-specific transform: reference each session to the individual's own baseline.

For every participant we take the calibration-phase frames (their neutral opening
segment), compute a per-feature baseline, and express the whole session as
deviations from that baseline. The model then sees how each person departs from
their *own* norm rather than from a population average - a lightweight, honest
operationalisation of the person-specific cognition idea behind the project.

This module exposes functions used by the temporal model; it can also be run
directly to print a quick before/after comparison:
    python -m src.personalise
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/synthetic")
FEATURES = [
    "AU06_cheek_raiser", "AU12_lip_corner", "AU01_inner_brow",
    "AU04_brow_lowerer", "gaze_x", "gaze_y", "pose_Rx", "pose_Ry",
]


def load_features_labels():
    features = pd.read_csv(DATA_DIR / "features.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    return features, labels


def participant_baseline(session_df):
    """Mean of each feature over the participant's calibration frames."""
    calib = session_df[session_df["phase"] == "calibration"]
    return calib[FEATURES].mean()


def to_sequence(session_df, personalise):
    """Return an (n_frames x n_features) array for one participant.

    If personalise is True, subtract the participant's calibration baseline so
    the sequence expresses deviation from their own norm. We keep only the
    interaction frames, which is where the reaction signal lives.
    """
    session_df = session_df.sort_values("frame")
    interaction = session_df[session_df["phase"] == "interaction"]
    seq = interaction[FEATURES].to_numpy(dtype=float)
    if personalise:
        base = participant_baseline(session_df).to_numpy(dtype=float)
        seq = seq - base
    return seq


def build_sequences(personalise):
    """Build per-participant sequences plus aligned labels and metadata.

    Returns:
        sequences: list of (n_frames x n_features) arrays, one per participant
        info: DataFrame with participant_id, phq8, depressed, gender, split
    """
    features, labels = load_features_labels()
    sequences = []
    ordered_ids = []
    for pid, session_df in features.groupby("participant_id"):
        sequences.append(to_sequence(session_df, personalise))
        ordered_ids.append(pid)

    info = (
        pd.DataFrame({"participant_id": ordered_ids})
        .merge(labels, on="participant_id")
        .reset_index(drop=True)
    )
    return sequences, info


def _demo():
    """Print a small before/after comparison for a few participants."""
    features, labels = load_features_labels()
    print("Effect of person-specific referencing on the two positive-affect AUs")
    print("(mean over interaction frames):\n")
    print(f"{'pid':>4} {'phq8':>5} {'raw_pos':>9} {'personalised_pos':>18}")
    for pid in [0, 1, 2, 3, 4]:
        session = features[features["participant_id"] == pid]
        raw = to_sequence(session, personalise=False)[:, :2].mean()
        pers = to_sequence(session, personalise=True)[:, :2].mean()
        phq = int(labels.loc[labels["participant_id"] == pid, "phq8"].iloc[0])
        print(f"{pid:>4} {phq:>5} {raw:>9.3f} {pers:>18.3f}")

    print("\nSanity: personalised values center reactions around each person's own")
    print("baseline, so between-person differences reflect reaction change, not")
    print("just naturally high or low resting expression.")


if __name__ == "__main__":
    _demo()