"""Fit source-depot models and save strictly held-out predictions for offline scoring."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from common import DROP, HIST, SEED, split_holdout
from features import shape
from score import true_cost, vehicle_cost
from v2_params import PARAMS
from v5_percost_cfg import CM

SCENARIOS = ("source", "offset_only", "partial_scale", "full_scale")
SCALE_FACTORS = (1.0, 1.6, 2.5)
SENSOR_POLICIES = ("A0", "A1", "A2")
DRIFT = {"s120", "s122", "s126", "s128"}
INPUT_HASH = "6e88bb69b36a916f84fb50b38a0da53d5856815acdfc07b6b0cff7e18bb5297b"


def fingerprint(archive):
    h = hashlib.sha256()
    with archive.open("rb") as data:
        archive_hash = hashlib.file_digest(data, "sha256").hexdigest()
    if archive_hash != INPUT_HASH:
        raise ValueError("The input archive does not match the verified competition ZIP.")
    paths = [Path(__file__), *(SRC / f"{n}.py" for n in
             ("common", "features", "score", "v2_params", "v5_percost_cfg"))]
    for path in paths:
        h.update(path.read_text(encoding="utf-8").encode("utf-8"))
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    xgboost=xgb.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    h.update(json.dumps(versions, sort_keys=True).encode())
    h.update(archive_hash.encode())
    return h.hexdigest(), versions


def load_data(archive):
    with ZipFile(archive) as data:
        train = pd.read_csv(data.open("train.csv"))
        target = pd.read_csv(data.open("test.csv"), usecols=["id", "s120", "s122", "s126"])
        example = pd.read_csv(data.open("sample_submission.csv"), usecols=["id"])
    if len(train) != 56000 or not train.id.is_unique or len(target) != 20000:
        raise ValueError("Unexpected data shape.")
    if not target.id.is_unique or set(example.id) != set(target.id) or set(target.id) & set(train.id):
        raise ValueError("Invalid vehicle IDs.")
    dev, holdout = split_holdout(train)
    return dev, holdout, target


def severity_features(frame, sensor_names):
    keep = [c for c in sensor_names if c not in DRIFT]
    return pd.concat([frame[keep], shape(frame)], axis=1).to_numpy(dtype=np.float32)


def batch_parameters(source, target):
    def intercept(frame):
        d = frame[["s120", "s126"]].dropna()
        design = np.column_stack([d.s126, np.ones(len(d))])
        return np.linalg.lstsq(design, d.s120, rcond=None)[0][1]

    def relative_level(frame):
        d = frame.loc[frame.s126 > 0, ["s122", "s126"]].dropna()
        return (d.s122 / d.s126).median()

    return intercept(target) - intercept(source), relative_level(target) / relative_level(source)


def target_layout(frame):
    # Generate RAW target readings, then pass through the original alignment exactly once.
    raw = frame.copy()
    raw[HIST[1:]] = frame[HIST[:-1]].to_numpy()
    raw[HIST[0]] = np.where(frame[HIST[0]].isna(), np.nan, 0.0)
    raw[["s056", "s063"]] = raw[["s056", "s063"]].round(-1)
    raw[HIST[:-1]] = raw[HIST[1:]].to_numpy()
    raw[HIST[-1]] = np.nan
    return raw


def wear_groups(values, cutoff):
    return np.where(values.isna(), 2, np.where(values > cutoff, 1, 0)).astype(np.int8)


def fit_pair(x, cost_x, y, cost, rows, threads):
    classifier = xgb.XGBClassifier(**{**PARAMS, "n_jobs": threads})
    classifier.fit(x[rows], y[rows])
    failures = rows[y[rows] == 1]
    regressor = xgb.XGBRegressor(**{**CM, "n_estimators": 1000, "max_depth": 2, "n_jobs": threads})
    regressor.fit(cost_x[failures], np.log(cost[failures]))
    return classifier, regressor


def fit_fold(train, outer_train, valid, target, x, cost_x, sensors, threads):
    train_ids = train.id.to_numpy()[outer_train]
    assert not set(train_ids).intersection(valid.id)
    y, cost = train.failure.to_numpy(), true_cost(train)
    assert np.isfinite(cost).all() and np.all(cost[y == 1] > 0)
    cal_y = y[outer_train]
    cal_raw = np.full(len(cal_y), np.nan)
    cal_sev = np.full(len(cal_y), np.nan)
    inner_fold = np.full(len(cal_y), -1, dtype=np.int8)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for i, (fit, predict) in enumerate(inner.split(outer_train, cal_y)):
        assert not np.intersect1d(fit, predict).size
        assert np.all(inner_fold[predict] == -1)
        clf, reg = fit_pair(x, cost_x, y, cost, outer_train[fit], threads)
        rows = outer_train[predict]
        cal_raw[predict] = clf.predict_proba(x[rows])[:, 1]
        cal_sev[predict] = reg.predict(cost_x[rows])
        inner_fold[predict] = i
    assert (inner_fold >= 0).all() and np.isfinite(cal_raw).all() and np.isfinite(cal_sev).all()
    failures = cal_y == 1
    resid = np.log(cost[outer_train][failures]) - cal_sev[failures]

    clf, reg = fit_pair(x, cost_x, y, cost, outer_train, threads)
    valid_x = valid[sensors].to_numpy(dtype=np.float32)
    native = clf.predict_proba(valid_x)[:, 1]
    offset, scale = batch_parameters(train.iloc[outer_train], target)
    simulated = target_layout(valid[sensors])
    pred_sev = np.stack([reg.predict(severity_features(view, sensors))
                         for view in [valid, simulated]])
    raw = np.empty((len(SCENARIOS), len(SENSOR_POLICIES), len(valid)), dtype=np.float32)
    raw[0] = native   # All three sensor policies use the original readings at the old depot.
    offset_channel, scale_channel = sensors.index("s120"), sensors.index("s122")
    for scenario, factor in enumerate(SCALE_FACTORS, 1):
        shifted = simulated.to_numpy(dtype=np.float32, copy=True)
        shifted[:, offset_channel] += 16024000.0
        shifted[:, scale_channel] *= factor
        raw[scenario, 0] = clf.predict_proba(shifted)[:, 1]
        shifted[:, offset_channel] -= offset
        raw[scenario, 1] = clf.predict_proba(shifted)[:, 1]
        shifted[:, scale_channel] /= scale
        raw[scenario, 2] = clf.predict_proba(shifted)[:, 1]
    cutoff = train.iloc[outer_train].s126.quantile(0.95)
    target_mix = np.bincount(wear_groups(target.s126, cutoff), minlength=3) / len(target)
    return dict(ids=valid.id.to_numpy(dtype=str), y=valid.failure.to_numpy(), cost=true_cost(valid),
                wear=wear_groups(valid.s126, cutoff), target_mix=target_mix, wear_cutoff=cutoff,
                offset=offset, scale=scale, raw=raw, severity=pred_sev, resid=resid,
                cal_ids=np.asarray(train_ids, dtype=str), cal_y=cal_y, cal_raw=cal_raw,
                cal_severity=cal_sev, inner_fold=inner_fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--holdout", action="store_true", help="One evaluation fit on the entire dev set.")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    assert vehicle_cost([400, 0, 1200, 0], [1, 1, 0, 0], [300, 50, 0, 0]).sum() == 1241
    signature, versions = fingerprint(args.zip)
    dev, holdout, target = load_data(args.zip)
    sensors = [f"s{i:03d}" for i in range(170) if f"s{i:03d}" not in DROP]
    x = dev[sensors].to_numpy(dtype=np.float32)
    cost_x = severity_features(dev, sensors)
    args.cache.mkdir(parents=True, exist_ok=True)
    print(f"Validated archive and official example. Development: {len(dev)} rows, "
          f"{dev.failure.sum()} failures. Versions: {versions}", flush=True)
    for seed in ([SEED] if args.holdout else args.seeds):
        if args.holdout:
            splits = [(np.arange(len(dev)), None)]
        else:
            splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(x, dev.failure)
        for fold, (a, b) in enumerate(splits, 1):
            name = "holdout" if args.holdout else f"dev_{seed}_{fold}"
            path = args.cache / f"{name}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as previous:
                    if str(previous["signature"]) != signature:
                        raise ValueError(f"Cache mismatch at {path}. Use a new cache directory.")
                print(f"{name} already complete", flush=True)
                continue
            start = time.monotonic()
            valid = holdout if b is None else dev.iloc[b]
            print(f"Fitting {name} ({len(a)} train, {len(valid)} validation)", flush=True)
            out = fit_fold(dev, a, valid, target, x, cost_x, sensors, args.threads)
            tmp = path.with_suffix(".part.npz")
            np.savez_compressed(tmp, **out, signature=signature, seed=seed, fold=fold,
                                versions=json.dumps(versions), scenarios=SCENARIOS, policies=SENSOR_POLICIES)
            tmp.replace(path)
            print(f"Saved {name} after {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
