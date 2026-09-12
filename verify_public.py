"""Verify a deployed HTTPS endpoint using generated audio and a 15-minute soak.

This is availability/contract verification, not a remote dataset accuracy test.
The no-external-audio restriction keeps all 71-call evaluation on loopback.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
from check_endpoint import check_endpoint, payload, wav_bytes
from provenance import pipeline_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--seconds", type=float, default=900)
    args = parser.parse_args()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase2_public.json").write_text(json.dumps({"passed": False, "status": "pending", "url": args.url}) + "\n")
    if urlparse(args.url).scheme != "https":
        raise SystemExit("A stable public HTTPS URL is required")
    if args.seconds < 900:
        raise SystemExit("The deployment gate requires at least 900 seconds of verification")
    # The user explicitly deferred Docker. Keep the local HTTP and public gates.
    local_report = Path("reports/phase2_http.json")
    local = json.loads(local_report.read_text()) if local_report.exists() else {}
    if not local.get("offline_equivalent") or local.get("pipeline_sha256") != pipeline_fingerprint():
        raise SystemExit("Evaluate the current pipeline locally via HTTP before public verification")
    edge_report = check_endpoint(args.url, output=Path("reports/phase2_public_edges.json"))
    body = payload(wav_bytes(pattern="many"))
    started = time.monotonic()
    observations = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=30, trust_env=False, follow_redirects=False) as client:
        expected = client.post("/detect", json=body)
        expected.raise_for_status()
        while True:
            elapsed = time.monotonic() - started
            health = client.get("/health")
            health.raise_for_status()
            assert health.json()["model_sha256"] == edge_report["model_sha256"]
            assert health.json()["pipeline_sha256"] == edge_report["pipeline_sha256"]
            response = client.post("/detect", json=body)
            response.raise_for_status()
            assert response.json() == expected.json(), "Inconsistent deployment response"
            observations.append({"elapsed_s": elapsed, "healthy": True})
            print(f"Public service healthy: {elapsed:.0f}/{args.seconds:.0f} s", flush=True)
            if elapsed >= args.seconds:
                break
            time.sleep(min(10, args.seconds - elapsed))
    report = {"passed": True, "url": args.url, "duration_s": time.monotonic() - started,
              "model_sha256": edge_report["model_sha256"], "checks": observations,
              "pipeline_sha256": edge_report["pipeline_sha256"],
              "docker": "Deferred by user", "scope": "Generated tones only; dataset metric reproduced over local HTTP"}
    Path("reports/phase2_public.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
