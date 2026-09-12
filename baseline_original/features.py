"""Extrae features de turn-taking desde turns/*.json. Sin tocar el audio."""
import json, csv, glob, os
import numpy as np

CALLER, AGENT = 0, 1

def _stats(x, prefix, out):
    if len(x) == 0:
        for s in ["mean","med","std","min","max","p10","p90"]:
            out[f"{prefix}_{s}"] = np.nan
        return
    a = np.asarray(x, dtype=float)
    out[f"{prefix}_mean"] = a.mean()
    out[f"{prefix}_med"]  = np.median(a)
    out[f"{prefix}_std"]  = a.std()
    out[f"{prefix}_min"]  = a.min()
    out[f"{prefix}_max"]  = a.max()
    out[f"{prefix}_p10"]  = np.percentile(a, 10)
    out[f"{prefix}_p90"]  = np.percentile(a, 90)

def merge(turns, ch, gap=0.25):
    """Une segmentos contiguos del mismo canal en 'enunciados'."""
    t = sorted([x for x in turns if x["channel"] == ch], key=lambda x: x["start"])
    if not t: return []
    out = [[t[0]["start"], t[0]["end"]]]
    for s in t[1:]:
        if s["start"] - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], s["end"])
        else:
            out.append([s["start"], s["end"]])
    return out

def extract(path):
    turns = json.load(open(path))["turns"]
    C = merge(turns, CALLER)
    A = merge(turns, AGENT)
    f = {}
    if not C or not A:
        return f

    total = max(max(e for _, e in C), max(e for _, e in A))
    f["total_s"] = total
    f["n_caller"] = len(C); f["n_agent"] = len(A)
    f["caller_speech_s"] = sum(e - s for s, e in C)
    f["agent_speech_s"]  = sum(e - s for s, e in A)
    f["caller_ratio"] = f["caller_speech_s"] / max(total, 1e-6)
    f["speech_ratio"] = f["caller_speech_s"] / max(f["agent_speech_s"], 1e-6)
    f["turns_per_min"] = len(C) / max(total / 60, 1e-6)

    # --- LATENCIA DE RESPUESTA: el rasgo estrella ---
    lat, overlap_starts = [], 0
    for cs, ce in C:
        prev = [(s, e) for s, e in A if s < cs]
        if not prev: continue
        _, ae = max(prev, key=lambda x: x[1])
        gap = cs - ae
        if gap < 0: overlap_starts += 1
        if -2.0 < gap < 12.0: lat.append(gap)
    _stats(lat, "lat", f)
    f["lat_n"] = len(lat)
    f["barge_in_rate"] = overlap_starts / max(len(C), 1)
    if lat:
        a = np.asarray(lat)
        f["lat_under_05"] = float((a < 0.5).mean())
        f["lat_under_10"] = float((a < 1.0).mean())
        f["lat_over_25"]  = float((a > 2.5).mean())
        f["lat_iqr"] = float(np.percentile(a, 75) - np.percentile(a, 25))
        f["lat_cv"] = float(a.std() / max(abs(a.mean()), 1e-6))

    # --- DURACION DE ENUNCIADOS ---
    cd = [e - s for s, e in C]; ad = [e - s for s, e in A]
    _stats(cd, "cdur", f); _stats(ad, "adur", f)
    f["cdur_cv"] = float(np.std(cd) / max(np.mean(cd), 1e-6))

    # --- PAUSAS INTERNAS (ritmo) ---
    raw = sorted([x for x in turns if x["channel"] == CALLER], key=lambda x: x["start"])
    ip = [raw[i+1]["start"] - raw[i]["end"] for i in range(len(raw)-1)
          if 0 < raw[i+1]["start"] - raw[i]["end"] < 1.5]
    _stats(ip, "ipause", f)
    f["ipause_per_min"] = len(ip) / max(total / 60, 1e-6)
    f["frag_ratio"] = len(raw) / max(len(C), 1)   # fragmentacion del habla

    # --- SOLAPAMIENTO Y REACCION A INTERRUPCION ---
    ov = 0.0; yields = []
    for cs, ce in C:
        for as_, ae in A:
            o = min(ce, ae) - max(cs, as_)
            if o > 0:
                ov += o
                if as_ > cs:           # el agente interrumpe al caller
                    yields.append(ce - as_)   # cuanto tarda en callarse
    f["overlap_s"] = ov
    f["overlap_ratio"] = ov / max(f["caller_speech_s"], 1e-6)
    _stats(yields, "yield", f)
    f["interrupted_n"] = len(yields)

    # --- SILENCIOS DEL AGENTE: el caller los llena? ---
    fills = 0; chances = 0
    for i in range(len(A) - 1):
        hole = A[i+1][0] - A[i][1]
        if hole > 3.0:
            chances += 1
            if any(A[i][1] < cs < A[i+1][0] for cs, _ in C): fills += 1
    f["fill_rate"] = fills / max(chances, 1)
    f["fill_chances"] = chances
    return f

if __name__ == "__main__":
    rows = list(csv.DictReader(open("manifest.csv")))
    data = []
    for r in rows:
        p = f"turns/{r['anon_id']}.json"
        if not os.path.exists(p): continue
        f = extract(p)
        f.update(anon_id=r["anon_id"], label=r["label"], split=r["split"])
        data.append(f)
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv("features.csv", index=False)
    print(df.shape, "->", "features.csv")
