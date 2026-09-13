"""Freeze model/VAD and select only a decision threshold for balanced accuracy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

from behavior_features import extract_model_features
from vad import turns_from_wav


def scores(y, p, threshold):
    prediction = p > threshold
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "correct": int((prediction == y).sum()), "n": len(y),
            "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


def main():
    path = Path("artifacts/model.joblib")
    bundle = joblib.load(path)
    model_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if bundle["trained_split"] != "train":
        raise RuntimeError("Expected the existing train-only fitted model")
    manifest = pd.read_csv("manifest.csv")
    val = manifest.loc[manifest.split == "val"].copy()
    assert len(val) == 71
    rows = []
    for item in val.itertuples():
        payload = turns_from_wav(Path("audio") / (item.anon_id + ".wav"), **bundle["vad_config"])
        features = extract_model_features(payload, bundle.get("feature_blocks", ()))
        rows.append([features.get(name, np.nan) for name in bundle["feats"]])
    with threadpool_limits(limits=1):
        probability = bundle["model"].predict_proba(np.asarray(rows))[:, 1]
    y = (val.label == "synthetic").to_numpy().astype(int)
    unique = np.unique(probability)
    candidates = np.unique(np.r_[0., .5, 1., (unique[:-1] + unique[1:]) / 2])
    measurements = [scores(y, probability, value) for value in candidates]
    # Exact integer numerator for (TP/Nsynthetic + TN/Nhuman)/2.
    def rank(record):
        tn, fp = record["confusion_matrix"][0]
        fn, tp = record["confusion_matrix"][1]
        return (tp * (tn + fp) + tn * (tp + fn), -abs(record["threshold"] - .5))
    best = max(measurements, key=rank)
    before = scores(y, probability, .5)
    changed = best["balanced_accuracy"] > before["balanced_accuracy"]
    selected = best if changed else before
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_sha256": model_sha, "feature_count": len(bundle["feats"]),
              "feature_blocks": bundle.get("feature_blocks", []), "model_retrained": False,
              "selection_split": "val", "selection_note": "Threshold selected on the same 71 validation calls; this is not an independent estimate of its gain.",
              "comparison": ">", "candidate_count": len(candidates), "changed": changed,
              "before": before, "after": selected, "candidates": measurements}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/threshold.json").write_text(json.dumps(report, indent=2) + "\n")
    policy = {"model_sha256": model_sha, "threshold": selected["threshold"],
              "comparison": ">", "objective": "balanced_accuracy", "selection_split": "val"}
    Path("artifacts/decision_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    val["p_synthetic"] = probability
    val["is_synthetic"] = probability > selected["threshold"]
    val.to_csv("reports/threshold_predictions_val.csv", index=False)
    print(json.dumps({"before": before, "after": selected, "changed": changed}, indent=2), flush=True)
    longest = manifest.loc[manifest.duration_s.idxmax()]
    print("Longest call:", longest.anon_id, float(longest.duration_s), longest.split, flush=True)


if __name__ == "__main__":
    main()
