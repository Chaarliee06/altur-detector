"""Fingerprint the actual inference pipeline without exposing source or audio."""
import hashlib
from pathlib import Path


def pipeline_fingerprint(root=None, model_path=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("app.py", "features.py", "behavior_features.py", "vad.py", "provenance.py", "requirements.lock", "artifacts/model.joblib"):
        digest.update(name.encode("utf-8") + b"\0")
        path = Path(model_path) if name == "artifacts/model.joblib" and model_path else root / name
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
