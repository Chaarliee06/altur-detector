"""Train the supplied HGB on train only; never include val in model fitting."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from metrics import evaluate

MODEL_PARAMETERS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                        l2_regularization=1.0, random_state=0)


def fit_and_evaluate(frame, feature_names, output=None, vad_config=None):
    train = frame.loc[frame.split == "train"]
    val = frame.loc[frame.split == "val"]
    assert len(train) == 282 and len(val) == 71, "Unexpected official split sizes"
    assert not set(train.anon_id) & set(val.anon_id), "Call ID overlap"
    # Official splits are speaker-disjoint. No random-fold generalization claims.
    with threadpool_limits(limits=1):
        model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
        model.fit(train[feature_names].to_numpy(), (train.label == "synthetic").to_numpy())
        probability = model.predict_proba(val[feature_names].to_numpy())[:, 1]
    scores = evaluate((val.label == "synthetic").astype(int).to_numpy(), probability)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feats": list(feature_names),
                     "vad_config": vad_config, "trained_split": "train",
                     "train_calls": 282, "nan_policy": "native_hgb",
                     "decision_threshold": 0.5,
                     "score_semantics": "P(synthetic)", "metrics": scores}, output)
    return scores, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="cache/features_vad.csv")
    parser.add_argument("--vad-config", default="artifacts/vad_config.json")
    parser.add_argument("--output", default="artifacts/model.joblib")
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    names = [c for c in frame if c not in ("anon_id", "label", "split")]
    config = json.loads(Path(args.vad_config).read_text())
    scores, _ = fit_and_evaluate(frame, names, args.output, config)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
