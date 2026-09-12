"""POST /detect  ->  {"is_synthetic": bool, "confidence": float}
Ejecutar:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64, json, tempfile, os
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from features import extract
from vad import turns_from_wav

app = FastAPI(title="Altur - Detector de llamante sintetico")
BUNDLE = joblib.load("model.joblib")   # {"model":..., "feats":[...], "medians":{...}}

class Req(BaseModel):
    audio: str                 # WAV estereo 8 kHz en base64
    format: str = "wav"

def featurize(turns: dict) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(turns, f); tmp = f.name
    try:
        f = extract(tmp)
    finally:
        os.unlink(tmp)
    row = [f.get(k, np.nan) for k in BUNDLE["feats"]]
    row = [BUNDLE["medians"].get(k, 0.0) if (v is None or (isinstance(v, float) and np.isnan(v)))
           else v for k, v in zip(BUNDLE["feats"], row)]
    return np.asarray(row, dtype=float).reshape(1, -1)

@app.post("/detect")
def detect(req: Req):
    try:
        raw = base64.b64decode(req.audio)
        turns = turns_from_wav(raw)
        if len(turns["turns"]) < 4:
            # audio inutilizable: no adivinar con confianza alta
            return {"is_synthetic": False, "confidence": 0.5}
        p = float(BUNDLE["model"].predict_proba(featurize(turns))[0, 1])
        return {"is_synthetic": bool(p > 0.5), "confidence": round(p if p > .5 else 1 - p, 4)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health(): return {"ok": True}
