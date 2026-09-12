"""Start a real HTTP server, evaluate all val calls and check edge cases."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
from check_endpoint import check_endpoint
from evaluate_http import evaluate_http


def wait_for_health(url, process=None, seconds=30):
    deadline = time.monotonic() + seconds
    with httpx.Client(trust_env=False, timeout=1, follow_redirects=False) as client:
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError("Service exited before readiness")
            try:
                response = client.get(url + "/health")
                if response.status_code == 200 and response.json().get("ok"):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.1)
    raise RuntimeError("Service did not become ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Service Python; use a fresh environment to check dependencies")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--report-prefix", default="phase2")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--offline", type=Path, default=Path("reports/phase1_predictions_val.csv"))
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "MODEL_PATH": str(args.model.resolve()), "PORT": str(args.port), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen([args.python, "serve.py"], env=env)
    try:
        wait_for_health(url, process)
        evaluate_http(url, output=Path(f"reports/{args.report_prefix}_http.json"), offline=args.offline)
        check_endpoint(url, model_path=args.model, output=Path(f"reports/{args.report_prefix}_edges.json"))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
