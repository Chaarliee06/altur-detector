"""HTTP contract/edge checks using generated tones only, never dataset audio."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

import httpx
import numpy as np
import soundfile as sf
from provenance import pipeline_fingerprint


def wav_bytes(seconds=8.0, channels=2, sr=8000, subtype="PCM_16", pattern="silence"):
    data = np.zeros((int(seconds * sr), channels))
    if pattern != "silence":
        intervals = [(0, 0.5, 1.0), (min(1, channels - 1), 2.0, 2.5)]
        if pattern == "many":
            intervals += [(0, 3.5, 4.0), (min(1, channels - 1), 5.0, 5.5), (0, 6.5, 7.0)]
        if pattern == "caller_only":
            intervals = [(0, a, b) for _, a, b in intervals]
        for channel, start, end in intervals:
            left, right = int(start * sr), min(len(data), int(end * sr))
            if right > left:
                data[left:right, channel] = .2 * np.sin(2 * np.pi * (350 + 100 * channel) * np.arange(right - left) / sr)
    result = io.BytesIO()
    sf.write(result, data, sr, format="WAV", subtype=subtype)
    return result.getvalue()


def payload(raw):
    return {"audio": base64.b64encode(raw).decode("ascii"), "format": "wav"}


def check_endpoint(url, model_path=Path("artifacts/model.joblib"),
                   output=Path("reports/phase2_edges.json")):
    cases = [
        ("stereo_silence", payload(wav_bytes()), 200, True),
        ("mono", payload(wav_bytes(channels=1, pattern="many")), 200, True),
        ("short_stereo", payload(wav_bytes(seconds=1.0, pattern="many")), 200, True),
        ("empty_valid_wav", payload(wav_bytes(seconds=0)), 200, True),
        ("fewer_than_four_turns", payload(wav_bytes(pattern="few")), 200, True),
        ("caller_only", payload(wav_bytes(pattern="caller_only")), 200, True),
        ("invalid_base64", {"audio": "%%%"}, 422, False),
        ("not_wav", payload(b"not a WAV file"), 422, False),
        ("truncated_wav", payload(wav_bytes()[:100]), 422, False),
        ("wrong_sample_rate", payload(wav_bytes(sr=16000)), 422, False),
        ("wrong_sample_width", payload(wav_bytes(subtype="PCM_24")), 422, False),
        ("three_channels", payload(wav_bytes(channels=3)), 422, False),
        ("wrong_format_field", {"audio": "", "format": "mp3"}, 422, False),
        ("missing_audio", {}, 422, False),
        ("null_audio", {"audio": None}, 422, False),
        ("generated_multiturn", payload(wav_bytes(pattern="many")), 200, False),
    ]
    records = []
    with httpx.Client(base_url=url.rstrip("/"), trust_env=False, follow_redirects=False, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        model_hash = health.json()["model_sha256"]
        expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert model_hash == expected_hash, "Deployed artifact differs from the evaluated local model"
        assert health.json()["pipeline_sha256"] == pipeline_fingerprint(), "Deployed inference code or dependencies differ from the local pipeline"
        for name, body, expected_status, abstention in cases:
            start = time.perf_counter()
            result = client.post("/detect", json=body)
            assert result.status_code == expected_status, (name, result.status_code, result.text[:300])
            if expected_status == 200:
                answer = result.json()
                assert set(answer) == {"is_synthetic", "confidence"}
                assert type(answer["is_synthetic"]) is bool
                assert np.isfinite(answer["confidence"]) and .5 <= answer["confidence"] <= 1
                if abstention:
                    assert answer == {"is_synthetic": False, "confidence": .5}, name
            records.append({"case": name, "status": result.status_code, "passed": True,
                            "latency_ms": (time.perf_counter() - start) * 1000})
        assert client.post("/detect", content=b"{", headers={"content-type": "application/json"}).status_code == 422
        records.append({"case": "malformed_json", "status": 422, "passed": True})
        assert client.get("/health").json()["ok"] is True
    report = {"url": url, "model_sha256": model_hash, "passed": True,
              "pipeline_sha256": pipeline_fingerprint(),
              "n": len(records), "audio_source": "Generated zeros/tones; no challenge audio", "checks": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HTTP edge cases: {len(records)}/{len(records)} passed; model SHA-256 matched", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase2_edges.json"))
    args = parser.parse_args()
    check_endpoint(args.url, args.model, args.output)
