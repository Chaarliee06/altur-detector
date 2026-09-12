"""Evaluate the official val split via the same POST /detect used by judging.

Dataset audio may only go to loopback. For a Docker service, publish its port
locally. Public deployment smoke checks use generated, non-dataset audio.
"""
import argparse
import base64
import ipaddress
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd
from metrics import evaluate


def require_local_url(url):
    parsed = urlparse(url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme != "http" or not loopback or parsed.username or parsed.password:
        raise ValueError("Dataset evaluation requires a numeric HTTP loopback address; audio must stay local")


def evaluate_http(url, data_dir=Path("."), output=Path("reports/phase2_http.json"),
                  offline=Path("reports/phase1_predictions_val.csv")):
    require_local_url(url)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    val = manifest.loc[manifest.split == "val"].copy()
    assert len(val) == 71 and val.anon_id.is_unique
    rows = []
    with httpx.Client(base_url=url.rstrip("/"), timeout=60, trust_env=False, follow_redirects=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        for item in val.itertuples():
            audio = base64.b64encode((data_dir / "audio" / f"{item.anon_id}.wav").read_bytes()).decode("ascii")
            start = time.perf_counter()
            response = client.post("/detect", json={"audio": audio, "format": "wav"})
            elapsed = time.perf_counter() - start
            response.raise_for_status()
            result = response.json()
            assert type(result["is_synthetic"]) is bool
            confidence = float(result["confidence"])
            assert np.isfinite(confidence) and 0.5 <= confidence <= 1.0
            probability = confidence if result["is_synthetic"] else 1 - confidence
            rows.append({"anon_id": item.anon_id, "label": item.label,
                         "is_synthetic": result["is_synthetic"], "confidence": confidence,
                         "p_synthetic": probability, "latency_s": elapsed})
            if len(rows) % 10 == 0:
                print(f"HTTP val: {len(rows)}/71", flush=True)
    frame = pd.DataFrame(rows)
    scores = evaluate((frame.label == "synthetic").astype(int), frame.p_synthetic)
    scores.update({"transport": "HTTP loopback", "model_sha256": health.json()["model_sha256"],
                   "pipeline_sha256": health.json()["pipeline_sha256"],
                   "abstentions": int((frame.confidence == 0.5).sum()),
                   "latency_p50_ms": float(frame.latency_s.quantile(0.5) * 1000),
                   "latency_p95_ms": float(frame.latency_s.quantile(0.95) * 1000),
                   "latency_scope": "Sequential, warm server; includes JSON encoding, HTTP and inference, excludes WAV read/base64"})
    if offline and offline.exists():
        reference = pd.read_csv(offline)
        joined = frame.merge(reference[["anon_id", "p_synthetic"]], on="anon_id", validate="one_to_one", suffixes=("", "_offline"))
        assert len(joined) == 71
        error = float(np.max(np.abs(joined.p_synthetic - joined.p_synthetic_offline)))
        scores["max_probability_delta_offline"] = error
        scores["offline_equivalent"] = error < 1e-12
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_name(output.stem + "_predictions.csv"), index=False)
    output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))
    if scores.get("offline_equivalent") is False:
        raise SystemExit("HTTP differs from offline predictions; phase gate stays closed")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/phase2_http.json"))
    parser.add_argument("--offline", type=Path, default=Path("reports/phase1_predictions_val.csv"))
    args = parser.parse_args()
    evaluate_http(args.url, args.data_dir, args.output, args.offline)
