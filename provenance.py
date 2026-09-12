"""Fingerprint the actual inference pipeline without exposing source or audio."""
import hashlib
from pathlib import Path


def pipeline_fingerprint(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("app.py", "features.py", "vad.py", "provenance.py", "requirements.lock", "artifacts/model.joblib"):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
