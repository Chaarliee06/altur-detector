"""Deriva turnos desde el WAV estereo. CRITICO: en la evaluacion NO les dan
turns/*.json, solo el audio. Esta funcion tiene que reproducir una segmentacion
parecida a la del dataset o las features de latencia se descalibran.

VALIDAR ASI: correr sobre los WAV de train y comparar contra turns/*.json.
Ajustar los parametros hasta que las features de latencia coincidan."""
import numpy as np

def vad_energy(x, sr, frame_ms=30, thresh_db=-38, min_speech=0.2, min_sil=0.25):
    """VAD por energia. Simple y sin dependencias. Suficiente para telefonia
    donde los canales estan separados y hay poco cruce entre ellos."""
    n = int(sr * frame_ms / 1000)
    if len(x) < n: return []
    nf = len(x) // n
    fr = x[:nf * n].reshape(nf, n)
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.abs(x).max() + 1e-12) + 1e-12)

    # umbral adaptativo: piso de ruido + margen, acotado por thresh_db
    floor = np.percentile(db, 10)
    thr = max(thresh_db, floor + 12)
    act = db > thr

    # cierra huecos cortos, luego descarta islas cortas
    segs, i = [], 0
    while i < nf:
        if act[i]:
            j = i
            while j < nf and act[j]: j += 1
            segs.append([i * frame_ms / 1000, j * frame_ms / 1000])
            i = j
        else: i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_sil: merged[-1][1] = s[1]
        else: merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_speech]

def turns_from_wav(path_or_bytes, sr_expected=8000):
    """Devuelve el mismo formato que turns/*.json: {"turns":[{channel,start,end}]}"""
    import soundfile as sf, io
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    data, sr = sf.read(src, always_2d=True)
    if data.shape[1] < 2:
        raise ValueError("Se esperaba audio estereo: canal 0 caller, canal 1 agente")
    out = []
    for ch in (0, 1):
        for s, e in vad_energy(data[:, ch], sr):
            out.append({"channel": ch, "start": round(s, 2), "end": round(e, 2)})
    return {"turns": sorted(out, key=lambda t: t["start"])}
