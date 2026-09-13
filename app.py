"""The judging endpoint. Audio is processed in memory and is never stored."""
import base64
import binascii
from contextlib import asynccontextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from behavior_features import BLOCKS, extract_model_features
from provenance import pipeline_fingerprint
from vad import energy_profile, read_wav, turns_from_profiles

MAX_AUDIO_SECONDS = 900
MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_BASE64_LENGTH = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
MAX_REQUEST_BYTES = MAX_BASE64_LENGTH + 4096
MIN_AUDIO_SECONDS = 3.0
ROOT = Path(__file__).resolve().parent


def abstain():
    return {"is_synthetic": False, "confidence": 0.5}


class DetectAlways200:
    """Normalize parsing, routing and response failures before sending headers."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "").rstrip("/") != "/detect":
            return await self.app(scope, receive, send)
        messages = []

        async def capture(message):
            messages.append(message)

        try:
            await self.app(scope, receive, capture)
            start = next(m for m in messages if m["type"] == "http.response.start")
            body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
            # CORS preflights can have a plain-text 200 response.
            if scope["method"] != "OPTIONS":
                result = json.loads(body)
                assert type(result["is_synthetic"]) is bool
                assert math.isfinite(result["confidence"]) and 0 <= result["confidence"] <= 1
            assert start["status"] == 200
        except Exception:
            cors = {}
            for message in messages:
                for name, value in message.get("headers", []):
                    if name.lower().startswith(b"access-control-") or name.lower() == b"vary":
                        cors[name.decode("latin1")] = value.decode("latin1")
            if not cors and any(k.lower() == b"origin" for k, _ in scope.get("headers", [])):
                # Keep the public, credential-free CORS policy on fallback.
                cors["Access-Control-Allow-Origin"] = "*"
            return await JSONResponse(abstain(), status_code=200, headers=cors)(scope, receive, send)
        for message in messages:
            await send(message)


class RequestBodyLimit:
    """Bound JSON bodies before parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                response = JSONResponse(abstain(), status_code=200)
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


@asynccontextmanager
async def lifespan(app):
    path = Path(os.environ.get("MODEL_PATH", str(ROOT / "artifacts/model.joblib")))
    # Only load our own trusted artifact; joblib is not an upload format.
    bundle = joblib.load(path)
    if bundle.get("trained_split") != "train" or bundle.get("nan_policy") != "native_hgb":
        raise RuntimeError("Model must use train-only fitting and native missing values")
    if not bundle.get("vad_config"):
        raise RuntimeError("A frozen VAD configuration is required")
    if set(bundle.get("feature_blocks", ())) - set(BLOCKS):
        raise RuntimeError("Model requires unsupported behavior blocks")
    app.state.bundle = bundle
    app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    threshold = float(bundle["decision_threshold"])
    policy_path = ROOT / "artifacts/decision_policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if policy["model_sha256"] == app.state.model_sha256:
            threshold = float(policy["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError("Invalid decision threshold")
    app.state.decision_threshold = threshold
    app.state.pipeline_sha256 = pipeline_fingerprint(model_path=path)
    with threadpool_limits(limits=1):
        bundle["model"].predict_proba(np.zeros((1, len(bundle["feats"]))))
        yield


app = FastAPI(title="Altur · Detector conversacional", version="0.4.0", lifespan=lifespan)
app.add_middleware(RequestBodyLimit)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(DetectAlways200)


class DetectRequest(BaseModel):
    # call_id, sample_rate, channels, format and other metadata are ignored.
    model_config = ConfigDict(extra="ignore")
    audio_base64: str | None = Field(default=None, max_length=MAX_BASE64_LENGTH)
    audio: str | None = Field(default=None, max_length=MAX_BASE64_LENGTH)


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the returned class; 0.5 is abstention")


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return JSONResponse(abstain(), status_code=200)


@app.post("/detect/", response_model=DetectResponse, include_in_schema=False)
@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        audio = req.audio_base64 or req.audio
        if not audio:
            return abstain()
        return classify(audio)
    except Exception:
        return abstain()


def classify(audio):
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "audio must be valid base64") from None
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "WAV exceeds 32 MiB")
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise HTTPException(422, "audio must contain a WAV file")
    if struct.unpack_from("<I", raw, 4)[0] + 8 > len(raw):
        raise HTTPException(422, "Truncated WAV")
    try:
        info = sf.info(io.BytesIO(raw))
    except (sf.LibsndfileError, ValueError):
        raise HTTPException(422, "Unreadable WAV") from None
    if info.channels == 1:
        return abstain()
    if info.channels != 2 or info.samplerate != 8000 or info.subtype != "PCM_16":
        raise HTTPException(422, "Expected stereo PCM16 WAV at 8000 Hz")
    if info.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(413, "WAV exceeds 900 seconds")
    if info.duration < MIN_AUDIO_SECONDS:
        return abstain()
    try:
        data, sr = read_wav(raw)
    except (sf.LibsndfileError, ValueError):
        raise HTTPException(422, "Unreadable WAV samples") from None
    if not np.any(data):
        return abstain()
    bundle = app.state.bundle
    config = bundle["vad_config"]
    profiles = [energy_profile(data[:, ch], sr, config["frame_ms"]) for ch in (0, 1)]
    turns = turns_from_profiles(profiles, **config)
    if len(turns["turns"]) < 4:
        return abstain()
    features = extract_model_features(turns, bundle.get("feature_blocks", ()))
    if not features or features["n_caller"] + features["n_agent"] < 4:
        return abstain()
    row = np.asarray([[features.get(name, np.nan) for name in bundle["feats"]]], dtype=float)
    p = float(bundle["model"].predict_proba(row)[0, 1])
    if not math.isfinite(p) or not 0 <= p <= 1:
        return abstain()
    is_synthetic = bool(p > app.state.decision_threshold)
    # Preserve full precision so HTTP AUC/Brier reproduce the offline evaluation.
    return {"is_synthetic": bool(is_synthetic), "confidence": p if is_synthetic else 1 - p}


@app.get("/health")
def health():
    return {"ok": True, "model_sha256": app.state.model_sha256,
            "pipeline_sha256": app.state.pipeline_sha256,
            "features": len(app.state.bundle["feats"]),
            "feature_blocks": app.state.bundle.get("feature_blocks", []),
            "decision_threshold": app.state.decision_threshold,
            "contract_version": "judge-audio-base64-v1",
            "audio_fields": ["audio_base64", "audio"], "detect_error_status": 200}
