"""Shared evaluation: synthetic is the positive class; scores are P(synthetic)."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix, roc_curve


def evaluate(y, probability):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid aligned labels/probabilities")
    predicted = (p > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    fnr = 1 - tpr
    delta = fnr - fpr
    right = int(np.flatnonzero(delta <= 0)[0])
    if delta[right] == 0 or right == 0:
        eer = float(fpr[right])
    else:
        left = right - 1
        weight = delta[left] / (delta[left] - delta[right])
        eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
    legacy_fpr, legacy_tpr, _ = roc_curve(y, p)
    legacy = float(legacy_fpr[np.argmin(np.abs(1 - legacy_tpr - legacy_fpr))])
    return {
        "n": int(len(y)), "correct": int((predicted == y).sum()),
        "accuracy": float(accuracy_score(y, predicted)),
        "auc": float(roc_auc_score(y, p)), "eer": eer,
        "eer_legacy": legacy, "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
