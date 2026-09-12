"""Blocking VAD gate. Selection uses train correlations, never validation labels."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline_original import features as original_features
from baseline_original import vad as original_vad
from features import extract, extract_turns
from train import fit_and_evaluate
from vad import DEFAULT_CONFIG, energy_profile, read_wav, turns_from_profiles, turns_from_wav

PRIORITY = ["lat_med", "lat_mean", "barge_in_rate", "overlap_ratio", "n_caller"]
GRID = {
    "frame_ms": [20, 30], "thresh_db": [-46, -42, -38, -34, -30, -26],
    "min_speech": [0.1, 0.2, 0.3], "min_sil": [0.1, 0.2, 0.25, 0.3, 0.4],
    "noise_margin": [6, 12, 18],
}
PROFILE_CACHE = {}
TRAIN_ROWS = []
TRAIN_REFERENCE = None
FEATURE_NAMES = []


def correlation_table(reference, observed, names):
    records = []
    for name in names:
        x = reference[name].to_numpy(dtype=float)
        y = observed[name].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        count = int(mask.sum())
        r = float(np.corrcoef(x[mask], y[mask])[0, 1]) if count > 2 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0 else None
        records.append({"feature": name, "pearson_r": r, "n_valid": count,
                        "n_missing": int(len(x) - count),
                        "mae": float(np.abs(x[mask] - y[mask]).mean()) if count else None})
    return records


def dataset_frame(rows, vectors):
    frame = pd.DataFrame(vectors).reindex(columns=FEATURE_NAMES)
    for key in ["anon_id", "label", "split"]:
        frame[key] = [row[key] for row in rows]
    return frame


def evaluate_grid_config(config):
    vectors = [extract_turns(turns_from_profiles(PROFILE_CACHE[row['anon_id']][config['frame_ms']], **config)) for row in TRAIN_ROWS]
    frame = pd.DataFrame(vectors).reindex(columns=FEATURE_NAMES)
    records = correlation_table(TRAIN_REFERENCE, frame, PRIORITY)
    corr = {r['feature']: r['pearson_r'] for r in records}
    return {**config, **{name: corr[name] for name in PRIORITY},
            'min_valid': min(r['n_valid'] for r in records),
            'mean_priority_r': float(np.mean([corr[name] or -1 for name in PRIORITY]))}


def bleed_diagnostic(data, sr, turns):
    # Training diagnostic only. These reference masks are never used at inference.
    masks = np.zeros((len(data), 2), dtype=bool)
    for turn in turns:
        start = max(0, int(turn['start'] * sr))
        end = min(len(data), int(turn['end'] * sr))
        masks[start:end, turn['channel']] = True
    only_agent = masks[:, 1] & ~masks[:, 0]
    x = data[only_agent, 1]
    y = data[only_agent, 0]
    if len(x) < sr or np.dot(x, x) < 1e-10:
        return {'beta': 0.0, 'correlation': 0.0, 'agent_only_seconds': len(x) / sr}
    beta = float(np.dot(x, y) / np.dot(x, x))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-10 else 0.0
    return {'beta': beta, 'correlation': corr, 'agent_only_seconds': len(x) / sr}


def main():
    global PROFILE_CACHE, TRAIN_ROWS, TRAIN_REFERENCE, FEATURE_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    start_time = time.perf_counter()
    for directory in ['cache', 'reports', 'artifacts']:
        Path(directory).mkdir(exist_ok=True)
    rows = pd.read_csv('manifest.csv').to_dict('records')
    assert len(rows) == 353 and len({r['anon_id'] for r in rows}) == 353
    TRAIN_ROWS = [r for r in rows if r['split'] == 'train']
    assert len(TRAIN_ROWS) == 282
    reference_vectors = []
    default_vectors = []
    bleed = []
    with threadpool_limits(limits=1):
        for index, row in enumerate(rows):
            call_id = row['anon_id']
            ref_path = Path('turns') / f'{call_id}.json'
            payload = json.loads(ref_path.read_text())
            reference = original_features.extract(ref_path)
            current = extract(ref_path)
            assert list(reference) == list(current)
            np.testing.assert_allclose(list(reference.values()), list(current.values()), rtol=0, atol=0, equal_nan=True)
            reference_vectors.append(reference)
            wav_path = Path('audio') / f'{call_id}.wav'
            supplied_turns = original_vad.turns_from_wav(wav_path)
            optimized_turns = turns_from_wav(wav_path)
            assert supplied_turns == optimized_turns, f'Default VAD changed: {call_id}'
            default_vectors.append(extract_turns(supplied_turns))
            data, sr = read_wav(wav_path)
            if row['split'] == 'train':
                PROFILE_CACHE[call_id] = {
                    frame: [energy_profile(data[:, ch], sr, frame) for ch in (0, 1)]
                    for frame in GRID['frame_ms']
                }
                bleed.append({'anon_id': call_id, **bleed_diagnostic(data, sr, payload['turns'])})
            if (index + 1) % 50 == 0:
                print(f'Prepared {index+1}/353 WAV; original feature and VAD equivalence checked.', flush=True)
    FEATURE_NAMES = list(pd.DataFrame(reference_vectors).columns)
    reference = dataset_frame(rows, reference_vectors)
    default = dataset_frame(rows, default_vectors)
    reference.to_csv('cache/features_reference.csv', index=False)
    default.to_csv('cache/features_default_vad.csv', index=False)
    TRAIN_REFERENCE = reference[reference.split == 'train'].reset_index(drop=True)
    default_train = default[default.split == 'train'].reset_index(drop=True)
    default_correlations = correlation_table(TRAIN_REFERENCE, default_train, FEATURE_NAMES)
    pd.DataFrame(default_correlations).to_csv('reports/default_vad_correlations_train.csv', index=False)
    pd.DataFrame(bleed).to_csv('reports/bleed_train.csv', index=False)
    reference_scores, _ = fit_and_evaluate(reference, FEATURE_NAMES)
    default_scores, _ = fit_and_evaluate(default, FEATURE_NAMES)
    preparation = {'features': len(FEATURE_NAMES), 'reference_metrics': reference_scores,
                   'default_vad_metrics': default_scores,
                   'default_priority_correlations': [r for r in default_correlations if r['feature'] in PRIORITY],
                   'bleed_calls_abs_correlation_over_06': sum(abs(r['correlation']) > 0.6 and abs(r['beta']) > 0.003 for r in bleed)}
    Path('reports/phase1_preparation.json').write_text(json.dumps(preparation, indent=2))
    print(json.dumps(preparation, indent=2), flush=True)

    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f'Starting {len(configs)} configurations on TRAIN ONLY ({len(TRAIN_ROWS)} calls).', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        for result in pool.map(evaluate_grid_config, configs, chunksize=4):
            results.append(result)
            if len(results) % 30 == 0:
                print(f'Grid {len(results)}/{len(configs)}; best train lat_med r={max(r["lat_med"] or -1 for r in results):.6f}', flush=True)
                pd.DataFrame(results).to_csv('reports/vad_grid_train.csv', index=False)
    grid = pd.DataFrame(results).sort_values(['lat_med', 'mean_priority_r'], ascending=False, kind='stable')
    grid.to_csv('reports/vad_grid_train.csv', index=False)
    best = grid.iloc[0]
    config = {name: int(best[name]) if name == 'frame_ms' else float(best[name]) for name in GRID}
    # Freeze the config before computing selected-model validation predictions.
    Path('artifacts/vad_config.json').write_text(json.dumps(config, indent=2))
    print('Config selected using train correlation only: '+json.dumps(config), flush=True)
    vectors = [extract_turns(turns_from_wav(Path('audio') / f'{r["anon_id"]}.wav', **config)) for r in rows]
    selected = dataset_frame(rows, vectors)
    selected.to_csv('cache/features_vad.csv', index=False)
    train_corr = correlation_table(TRAIN_REFERENCE, selected[selected.split == 'train'].reset_index(drop=True), FEATURE_NAMES)
    val_corr = correlation_table(reference[reference.split == 'val'].reset_index(drop=True), selected[selected.split == 'val'].reset_index(drop=True), FEATURE_NAMES)
    pd.DataFrame(train_corr).to_csv('reports/vad_correlations_train.csv', index=False)
    pd.DataFrame(val_corr).to_csv('reports/vad_correlations_val.csv', index=False)
    scores, probability = fit_and_evaluate(selected, FEATURE_NAMES, 'artifacts/model.joblib', config)
    val = selected[selected.split == 'val'][['anon_id', 'label']].copy()
    val['p_synthetic'] = probability
    val['predicted_synthetic'] = probability > 0.5
    val.to_csv('reports/phase1_predictions_val.csv', index=False)
    r_lat = next(r['pearson_r'] for r in train_corr if r['feature'] == 'lat_med')
    passed = r_lat is not None and r_lat > .9 and scores['accuracy'] >= .958 - .03
    report = {**preparation, 'grid': GRID, 'grid_configurations': len(configs), 'selected_config': config,
              'selection_split': 'train', 'selected_vad_metrics': scores,
              'priority_correlations_train': [r for r in train_corr if r['feature'] in PRIORITY],
              'priority_correlations_val': [r for r in val_corr if r['feature'] in PRIORITY],
              'gate': {'lat_med_r_train': r_lat, 'minimum_accuracy': .928, 'passed': passed},
              'elapsed_s': time.perf_counter() - start_time,
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'soundfile', 'joblib']},
              'features_sha256': hashlib.sha256(Path('features.py').read_bytes()).hexdigest()}
    Path('reports/phase1.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit('PHASE 1 GATE NOT PASSED. Do not advance to phase 2.')


if __name__ == '__main__':
    main()
