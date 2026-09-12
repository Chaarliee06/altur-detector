"""The supplied energy VAD, parameterized for a train-only grid search."""
import io
import numpy as np

DEFAULT_CONFIG = {"frame_ms": 30, "thresh_db": -38.0, "min_speech": 0.2,
                  "min_sil": 0.25, "noise_margin": 12.0}


def read_wav(path_or_bytes, sr_expected=8000):
    import soundfile as sf
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    data, sr = sf.read(source, always_2d=True, dtype="float64")
    if sr != sr_expected:
        raise ValueError(f"Expected {sr_expected} Hz, received {sr}")
    if data.shape[1] != 2:
        raise ValueError("Expected stereo WAV: caller on channel 0, agent on channel 1")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite audio samples")
    return data, sr


def energy_profile(x, sr, frame_ms=30):
    n = int(sr * frame_ms / 1000)
    if len(x) < n:
        return np.empty(0)
    nf = len(x) // n
    frames = x[:nf * n].reshape(nf, n)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms / (np.abs(x).max() + 1e-12) + 1e-12)


def segments_from_profile(db, frame_ms=30, thresh_db=-38, min_speech=0.2,
                          min_sil=0.25, noise_margin=12):
    if len(db) == 0:
        return []
    threshold = max(thresh_db, float(np.percentile(db, 10)) + noise_margin)
    active = db > threshold
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1) * frame_ms / 1000
    ends = np.flatnonzero(edges == -1) * frame_ms / 1000
    merged = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] < min_sil:
            merged[-1][1] = float(end)
        else:
            merged.append([float(start), float(end)])
    return [s for s in merged if s[1] - s[0] >= min_speech]


def turns_from_profiles(profiles, **config):
    turns = []
    for channel in (0, 1):
        for start, end in segments_from_profile(profiles[channel], **config):
            turns.append({"channel": channel, "start": round(start, 2), "end": round(end, 2)})
    return {"turns": sorted(turns, key=lambda turn: turn["start"])}


def vad_energy(x, sr, frame_ms=30, thresh_db=-38, min_speech=0.2,
               min_sil=0.25, noise_margin=12):
    return segments_from_profile(energy_profile(x, sr, frame_ms), frame_ms=frame_ms,
        thresh_db=thresh_db, min_speech=min_speech, min_sil=min_sil, noise_margin=noise_margin)


def turns_from_wav(path_or_bytes, sr_expected=8000, **config):
    config = {**DEFAULT_CONFIG, **config}
    data, sr = read_wav(path_or_bytes, sr_expected)
    profiles = [energy_profile(data[:, channel], sr, config["frame_ms"]) for channel in (0, 1)]
    return turns_from_profiles(profiles, **config)
