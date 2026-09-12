"""Build the isolated runtime and evaluate through its locally published port."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from check_endpoint import check_endpoint
from evaluate_http import evaluate_http
from verify_local import wait_for_health


def main():
    report_path = Path("reports/phase2_docker.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps({"passed": False, "status": "running"}) + "\n")
    if not shutil.which("docker"):
        report_path.write_text(json.dumps({"passed": False, "status": "blocked", "reason": "Docker executable and daemon are unavailable"}, indent=2) + "\n")
        raise SystemExit("Phase 2 blocked: Docker is unavailable. Dockerfile creation does not verify the container.")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    image = "altur-detector:phase2"
    subprocess.run(["docker", "build", "--pull", "-t", image, "."], check=True)
    name = "altur-check-" + uuid.uuid4().hex[:10]
    subprocess.run(["docker", "run", "--rm", "--detach", "--name", name,
                    "--publish", "127.0.0.1:8001:8000", image], check=True)
    try:
        url = "http://127.0.0.1:8001"
        wait_for_health(url, seconds=60)
        scores = evaluate_http(url, output=Path("reports/phase2_docker_http.json"))
        edges = check_endpoint(url, output=Path("reports/phase2_docker_edges.json"))
        image_id = subprocess.check_output(["docker", "inspect", "--format", "{{.Image}}", name], text=True).strip()
        report_path.write_text(json.dumps({"passed": scores["offline_equivalent"] and edges["passed"],
                                          "image_id": image_id, "metrics": scores}, indent=2) + "\n")
    finally:
        subprocess.run(["docker", "stop", name], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
