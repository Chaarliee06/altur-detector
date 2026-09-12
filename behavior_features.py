"""Additional VAD-only behavior blocks. Constants are fixed before ablation.

Missing opportunities are NaN, not evidence of a failed reaction. Agent silence
and interrupted/resumed speech are timing proxies; no semantic claim is made.
"""
import numpy as np
from features import extract_turns, merge

BLOCKS = ("interruption_recovery", "silence_recovery", "consistency", "drift", "autocorrelation")
LATENCY_BINS = (-np.inf, -1, 0, .5, 1, 1.5, 2.5, 4, 6, 12, np.inf)
DURATION_BINS = (0, .25, .5, 1, 2, 3, 5, 8, 12, 20, np.inf)


def response_series(caller, agent):
    """Exactly the baseline latency matching/filter, retaining caller indices."""
    indices, values = [], []
    for index, (start, _) in enumerate(caller):
        previous = [end for astart, end in agent if astart < start]
        if previous:
            gap = start - max(previous)
            if -2 < gap < 12:
                indices.append(index)
                values.append(gap)
    return np.asarray(indices, dtype=float), np.asarray(values, dtype=float)


def summary(values, prefix, out, statistics=("med", "p90", "std")):
    a = np.asarray(values, dtype=float)
    for statistic in statistics:
        out[f"{prefix}_{statistic}"] = (float({"med": np.median, "p90": lambda x: np.percentile(x, 90),
                                               "std": np.std}[statistic](a)) if len(a) else np.nan)


def entropy(values, bins):
    if not len(values):
        return np.nan
    counts = np.histogram(values, bins=bins)[0]
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(len(bins) - 1))


def interruption_recovery(caller, agent):
    """One event per caller utterance: first agent onset inside that utterance.

    Resume means the next caller utterance starts before the next agent turn,
    with a maximum 12 s after the interrupting agent's end. Clip the observation
    window at the last observed speech. Unobservable windows are excluded.
    """
    stop, stopped_during, observed, resumed = [], [], 0, 0
    waits, relative, durations = [], [], []
    end_of_observation = max([e for _, e in caller + agent], default=0)
    for ci, (cs, ce) in enumerate(caller):
        hits = [(ai, astart, ae) for ai, (astart, ae) in enumerate(agent) if cs < astart < ce]
        if not hits:
            continue
        ai, astart, ae = hits[0]
        stop.append(ce - astart)
        stopped_during.append(ce <= ae)
        next_agent = agent[ai + 1][0] if ai + 1 < len(agent) else end_of_observation
        window_end = min(ae + 12, next_agent, end_of_observation)
        if window_end <= ce:
            continue
        observed += 1
        if ci + 1 < len(caller) and caller[ci + 1][0] < window_end:
            start, end = caller[ci + 1]
            resumed += 1
            waits.append(start - ce)
            relative.append(start - ae)
            durations.append(end - start)
    f = {
        "recovery_stop_within_05": float(np.mean(np.asarray(stop) <= .5)) if stop else np.nan,
        "recovery_stop_within_10": float(np.mean(np.asarray(stop) <= 1)) if stop else np.nan,
        "recovery_stop_before_agent_end": float(np.mean(stopped_during)) if stop else np.nan,
        "recovery_resume_observable_rate": observed / len(stop) if stop else np.nan,
        "recovery_resume_rate": resumed / observed if observed else np.nan,
    }
    summary(waits, "recovery_resume_wait", f, ("med", "p90"))
    summary(relative, "recovery_resume_from_agent_end", f, ("med", "p90"))
    summary(durations, "recovery_resume_duration", f, ("med",))
    # Baseline yield_* already measures the distribution of time to stop.
    return f


def silence_recovery(caller, agent):
    waits, lengths, whole_turns, wait_fractions, occupied = [], [], [], [], []
    for (_, end), (next_start, _) in zip(agent, agent[1:]):
        gap = next_start - end
        if gap <= 3:
            continue
        starts = [(cs, ce) for cs, ce in caller if end < cs < next_start]
        if not starts:
            continue
        cs, ce = starts[0]
        waits.append(cs - end)
        lengths.append(min(ce, next_start) - cs)
        whole_turns.append(ce - cs)
        wait_fractions.append((cs - end) / gap)
        occupied.append(sum(min(e, next_start) - s for s, e in starts) / gap)
    f = {}
    summary(waits, "silence_fill_wait", f)
    summary(lengths, "silence_fill_duration", f)
    summary(whole_turns, "silence_fill_whole_turn", f, ("med",))
    summary(wait_fractions, "silence_wait_fraction", f, ("med",))
    summary(occupied, "silence_occupied_fraction", f, ("med",))
    # Baseline fill_rate/fill_chances already encode whether silence was filled.
    return f


def consistency(caller, agent, latencies):
    f = {"lat_entropy": entropy(latencies, LATENCY_BINS)}
    for name, turns in (("cdur", caller), ("adur", agent)):
        a = np.asarray([end - start for start, end in turns])
        f[name + "_iqr"] = float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) else np.nan
        f[name + "_entropy"] = entropy(a, DURATION_BINS)
        if name == "adur":
            f["adur_cv"] = float(a.std() / max(a.mean(), 1e-6)) if len(a) else np.nan
    # lat std/IQR/CV, both duration stds and caller duration CV already exist.
    return f


def drift(indices, latencies):
    slope, r2 = np.nan, np.nan
    if len(latencies) >= 3:
        x = indices - indices.mean()
        y = latencies - latencies.mean()
        if np.dot(x, x) > 0:
            slope = float(np.dot(x, y) / np.dot(x, x))
            r2 = float(np.dot(x, y) ** 2 / (np.dot(x, x) * np.dot(y, y))) if np.dot(y, y) > 1e-12 else 0.
    return {"lat_trend_slope": slope, "lat_trend_r2": r2}


def autocorrelation(latencies):
    f = {}
    for lag in (1, 2):
        value = np.nan
        if len(latencies) - lag >= 3:
            left, right = latencies[:-lag], latencies[lag:]
            if left.std() > 1e-6 and right.std() > 1e-6:
                value = float(np.corrcoef(left, right)[0, 1])
        f[f"lat_autocorr_{lag}"] = value
    return f


def extract_blocks(payload, blocks=BLOCKS, latency_override=None):
    turns = payload["turns"] if isinstance(payload, dict) else payload
    caller, agent = merge(turns, 0), merge(turns, 1)
    indices, latencies = response_series(caller, agent)
    if latency_override is not None:
        latencies = np.asarray(latency_override, dtype=float)
        if len(latencies) != len(indices):
            raise ValueError("Counterfactual must preserve the observed latency series")
    f = {}
    for block in blocks:
        if block == "interruption_recovery":
            f.update(interruption_recovery(caller, agent))
        elif block == "silence_recovery":
            f.update(silence_recovery(caller, agent))
        elif block == "consistency":
            f.update(consistency(caller, agent, latencies))
        elif block == "drift":
            f.update(drift(indices, latencies))
        elif block == "autocorrelation":
            f.update(autocorrelation(latencies))
        else:
            raise ValueError(f"Unknown feature block: {block}")
    return f


def extract_model_features(payload, blocks=()):
    f = extract_turns(payload)
    if f and blocks:
        extra = extract_blocks(payload, blocks)
        if set(f) & set(extra):
            raise ValueError("Behavior block duplicates baseline feature names")
        f.update(extra)
    return f
