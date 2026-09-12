"""Verify the promoted model through HTTP, optionally in a clean runtime."""
import argparse
import json
from pathlib import Path
from phase3 import candidate_http


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", help="Server interpreter (e.g. a runtime-only virtualenv)")
    args = parser.parse_args()
    report = json.loads(Path("reports/phase3.json").read_text())
    selected = report["selected_name"]
    offline = Path(f"reports/phase3_{selected}_offline.csv") if selected != "baseline" else Path("reports/phase1_predictions_val.csv")
    candidate_http("final", Path("artifacts/model.joblib"), offline, edges=True, server_python=args.python)
