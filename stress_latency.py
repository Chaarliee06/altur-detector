"""Frozen-model sensitivity to faster synthetic responses; no val fitting.

The requested latency-only intervention is not a physical new engine. A second
test shifts caller VAD turns as a whole and recomputes all timing features, so
the selected silence features do not stay artificially fixed in that test.
Neither test generates audio or estimates hidden-set performance.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from behavior_features import extract_blocks, extract_model_features, response_series
from features import _stats, merge
from metrics import evaluate


def shift_latency_features(payload, blocks, seconds):
    f = extract_model_features(payload, blocks)
    _, latency = response_series(merge(payload["turns"], 0), merge(payload["turns"], 1))
    shifted = latency - seconds
    # Preserve the original observations/pairings: no refiltering below -2 s,
    # no clipping negative latencies to zero. All non-latency variables freeze.
    _stats(shifted, "lat", f)
    if len(shifted):
        f.update(lat_under_05=float((shifted < .5).mean()),
                 lat_under_10=float((shifted < 1).mean()),
                 lat_over_25=float((shifted > 2.5).mean()),
                 lat_iqr=float(np.percentile(shifted, 75) - np.percentile(shifted, 25)),
                 lat_cv=float(shifted.std() / max(abs(shifted.mean()), 1e-6)))
    for key, value in extract_blocks(payload, blocks, latency_override=shifted).items():
        if key.startswith("lat_"):
            f[key] = value
    return f


def shift_caller_turns(payload, seconds):
    shifted = []
    for turn in payload["turns"]:
        item = dict(turn)
        if item["channel"] == 0:
            item["start"] = max(0., item["start"] - seconds)
            item["end"] = max(0., item["end"] - seconds)
        if item["end"] > item["start"]:
            shifted.append(item)
    return {"turns": sorted(shifted, key=lambda t: t["start"])}


def main():
    phase3 = json.loads(Path("reports/phase3.json").read_text())
    if not phase3.get("selected_metrics", {}).get("offline_equivalent"):
        raise RuntimeError("Phase 3 HTTP gate must pass before stress evaluation")
    if hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest() != phase3["selected_model_sha256"]:
        raise RuntimeError("Selected model changed after validation")
    val = pd.read_csv("manifest.csv").query("split == 'val'")
    assert len(val) == 71
    payloads = {r.anon_id: json.loads((Path(phase3["vad_cache"]) / (r.anon_id + ".json")).read_text())
                for r in val.itertuples()}
    y = (val.label == "synthetic").to_numpy().astype(int)
    output, prediction_rows = [], []
    for name, path in (("baseline", "artifacts/baseline_model.joblib"), ("selected", "artifacts/model.joblib")):
        bundle = joblib.load(path)
        blocks = bundle.get("feature_blocks", ())
        clean = [extract_model_features(payloads[r.anon_id], blocks) for r in val.itertuples()]
        with threadpool_limits(limits=1):
            original_p = bundle["model"].predict_proba(np.asarray([[f.get(k, np.nan) for k in bundle["feats"]] for f in clean]))[:, 1]
        original = evaluate(y, original_p)
        expected = phase3["baseline"] if name == "baseline" else phase3["selected_metrics"]
        assert original["correct"] == expected["correct"]
        assert abs(original["brier"] - expected["brier"]) < 1e-12
        for mode in ("latency_only", "caller_turn_shift"):
            for seconds in (0., 1., 1.5):
                features = []
                for i, row in enumerate(val.itertuples()):
                    payload = payloads[row.anon_id]
                    if row.label != "synthetic" or seconds == 0:
                        f = clean[i]
                    elif mode == "latency_only":
                        f = shift_latency_features(payload, blocks, seconds)
                    else:
                        f = extract_model_features(shift_caller_turns(payload, seconds), blocks)
                    features.append(f)
                with threadpool_limits(limits=1):
                    matrix = np.asarray([[f.get(k, np.nan) for k in bundle["feats"]] for f in features])
                    probability = bundle["model"].predict_proba(matrix)[:, 1]
                assert np.array_equal(probability[y == 0], original_p[y == 0]), "Human predictions must not change"
                scores = evaluate(y, probability)
                scores.update(model=name, mode=mode, acceleration_s=seconds,
                              accuracy_drop_pp=100 * (original["accuracy"] - scores["accuracy"]),
                              synthetic_recall=float(np.mean(probability[y == 1] > .5)),
                              model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                output.append(scores)
                for row, p in zip(val.itertuples(), probability):
                    prediction_rows.append({"anon_id": row.anon_id, "label": row.label, "model": name,
                                            "mode": mode, "acceleration_s": seconds, "p_synthetic": p})
                print(f"{name} / {mode} / -{seconds:g}s: {scores['correct']}/71; drop {scores['accuracy_drop_pp']:.2f} pp", flush=True)
    report = {
        "n": 71, "human_unchanged": 37, "synthetic_modified": 34,
        "latency_only": "Subtract delta from every originally valid signed latency; recompute lat statistics, thresholds, CV and selected latency entropy/trend/autocorrelation. Preserve observations and agent pairings. Freeze non-latency variables, including overlap, barge-in and silence fills.",
        "caller_turn_shift": "Move all synthetic caller VAD intervals earlier by delta, clip at zero, keep the agent fixed and recompute every model feature, including overlaps and silence fills. Reassociation/filtering can change which latencies remain; this is a separate, broader counterfactual.",
        "limitations": "Feature/turn-space stress only, not modified WAVs through HTTP, real faster-engine audio or hidden-set accuracy. Negative latencies are allowed. No retraining, calibration or block selection uses the stress results.",
        "results": output,
    }
    Path("reports/stress_latency.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(prediction_rows).to_csv("reports/stress_latency_predictions.csv", index=False)


if __name__ == "__main__":
    main()
