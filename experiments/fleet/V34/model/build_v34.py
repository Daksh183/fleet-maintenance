"""Build the frozen V34 budget policy from the official ZIP and local V33 files."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import KFold

from hist_gp import GaussianSeverity, extract, select

ZIP_HASH = "6e88bb69b36a916f84fb50b38a0da53d5856815acdfc07b6b0cff7e18bb5297b"
V33_HASH = "eed2288b3309d637fab1eae2975ceee67a07d6bfda5c39f1d24c9ebabe981583"
MISSINGNESS_SENSORS = [
    f"s{i:03d}" for i in range(170) if i not in {47, 63, 120, 122, 146}
]
USAGE_SENSORS = [f"s{i:03d}" for i in range(80, 90)]


def context_features(frame):
    wear = np.log1p(frame.s126.to_numpy(float))
    usage = np.log1p(frame[USAGE_SENSORS].sum(axis=1, min_count=10).to_numpy(float))
    return np.column_stack([
        wear, usage, wear - usage, np.log1p(frame.s123), np.log1p(frame.s149),
        frame[MISSINGNESS_SENSORS].isna().mean(axis=1),
    ])


def fit_context(dev, test, q):
    features = context_features(dev)
    low, median, high = np.nanquantile(features, [0.01, 0.5, 0.99], axis=0)

    def filled(values):
        return np.clip(np.where(np.isfinite(values), values, median), low, high)

    train_values = filled(features)
    center = train_values.mean(axis=0)
    scale = np.maximum(train_values.std(axis=0), 0.001)

    def design(values, transformed):
        return np.column_stack([
            np.ones(len(values)), (transformed - center) / scale, np.isnan(values[:, :2]),
        ])

    x = design(features, train_values)
    offset = logit(np.clip(q, 1e-6, 1 - 1e-6))
    outcome = dev.failure.to_numpy()

    def loss(coef):
        score = offset + x @ coef
        value = np.sum(np.logaddexp(0, score) - outcome * score) + 8 * (coef @ coef)
        gradient = x.T @ (expit(score) - outcome) + 16 * coef
        return value, gradient

    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Failure-odds fit did not converge: {result.message}")
    test_values = context_features(test)
    return np.clip(design(test_values, filled(test_values)) @ result.x, -1.5, 1.5)


def quantile_budget(probability, mean, residuals, spread=1.0):
    result = np.zeros(len(probability))
    positive = probability > 1 / 11
    levels = 1 - 1 / (11 * probability[positive])
    step = np.broadcast_to(spread, probability.shape)[positive]
    result[positive] = np.exp(mean[positive] + step * np.quantile(residuals, levels))
    return result


def estimate_severity(train, test):
    features = extract(train)
    outcome = np.log(train.repair_cost.to_numpy())
    mean = np.full(len(train), np.nan)
    spread = np.full(len(train), np.nan)
    for fit, held in KFold(5, shuffle=True, random_state=42).split(outcome):
        model = GaussianSeverity().fit(select(features, fit), outcome[fit])
        mean[held], spread[held] = model.predict(select(features, held))
    supported = np.isfinite(mean)
    assert supported.sum() == 383 and (spread[supported] > 0).all()
    test_mean, test_spread = GaussianSeverity().fit(features, outcome).predict(extract(test))
    residuals = outcome[supported] - mean[supported]
    return test_mean, test_spread, residuals, residuals / spread[supported]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zip", "v33", "runtime", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline", action="store_true", help="Rebuild the archived V33 for comparison.")
    args = parser.parse_args()
    sys.path.insert(0, str((args.runtime / "src").resolve()))
    from calib import make
    from common import HIST, split_holdout
    from decide import decide_pv
    from score import vehicle_cost

    old_path = args.v33 / "submission_v33_gp_budget.csv"
    assert file_hash(args.zip) == ZIP_HASH and file_hash(old_path) == V33_HASH
    assert vehicle_cost([400, 0, 1200, 0], [1, 1, 0, 0], [300, 50, 0, 0]).sum() == 1241
    with ZipFile(args.zip) as source:
        train = pd.read_csv(source.open("train.csv"))
        test = pd.read_csv(source.open("test.csv"))
        sample = pd.read_csv(source.open("sample_submission.csv"), usecols=["id"])
    assert len(sample) == len(test) == 20000 and sample.id.is_unique and test.id.is_unique
    assert set(sample.id) == set(test.id)
    old = pd.read_csv(old_path).set_index("id", verify_integrity=True).loc[test.id].reset_index()
    test[HIST[:-1]] = test[HIST[1:]].to_numpy()
    test[HIST[-1]] = np.nan
    failures = train.loc[train.failure == 1]
    assert len(failures) == 397
    mean, sigma, residuals, standardized = estimate_severity(failures, test)
    supported = np.isfinite(mean)

    with np.load(args.v33 / "model/baseline.npz", allow_pickle=False) as cached:
        assert np.array_equal(cached["ids"], test.id)
        q, tree, tree_residuals = (cached[key] for key in ("q", "tree", "residual"))
    p = 3 * q / (1 - q + 3 * q)
    flags, old_raw = decide_pv(p, tree, tree_residuals)
    inspected = old.flag.to_numpy() == 1
    assert np.array_equal(flags, old.flag) and inspected.sum() == 1780
    old_gp = inspected & supported & (old_raw > 0)
    old_raw[old_gp] = quantile_budget(p[old_gp], mean[old_gp], residuals)
    rebuilt = np.where(inspected, 1.25 * old_raw, old_raw).round(2)
    np.testing.assert_array_equal(rebuilt, old.budget)

    budgets = old.budget.to_numpy().copy()
    if args.baseline:
        budgets = rebuilt
    else:
        dev, _ = split_holdout(train)
        raw = np.load(args.runtime / "reports/p_oof_v2.npy", allow_pickle=False)
        assert raw.shape == (44800,) and len(dev) == len(raw) and np.isfinite(raw).all()
        platt = make("platt").fit(raw, dev.failure.to_numpy())
        correction = fit_context(dev, test, platt.predict(raw))
        budget_p = expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + correction)
        active = inspected & supported
        fallback = inspected & ~supported
        budgets[active] = (1.25 * quantile_budget(
            budget_p[active], mean[active], standardized, sigma[active]
        )).round(2)
        budgets[fallback] = (1.25 * quantile_budget(
            budget_p[fallback], tree[fallback], tree_residuals
        )).round(2)
    np.testing.assert_array_equal(budgets[~inspected], old.budget.to_numpy()[~inspected])

    frame = pd.DataFrame({"id": test.id, "flag": old.flag, "budget": budgets})
    frame = frame.set_index("id").loc[sample.id].reset_index()
    assert np.isfinite(budgets).all() and (budgets >= 0).all()
    args.out.mkdir(parents=True, exist_ok=True)
    name = old_path.name if args.baseline else "submission_v34_budget.csv"
    dest = args.out / name
    temporary = dest.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, float_format="%.2f")
    pd.testing.assert_frame_equal(pd.read_csv(temporary), frame)
    if args.baseline:
        assert file_hash(temporary) == V33_HASH
    temporary.replace(dest)

    previous = old.budget.to_numpy()
    audit = {
        "rows": len(frame), "inspections": int(inspected.sum()), "changed_flags_vs_v33": 0,
        "increased_inspected_budgets": int((inspected & (budgets > previous)).sum()),
        "reduced_inspected_budgets": int((inspected & (budgets < previous)).sum()),
        "zero_to_positive": int((inspected & (previous == 0) & (budgets > 0)).sum()),
        "positive_to_zero": int((inspected & (previous > 0) & (budgets == 0)).sum()),
        "total_parts_budget": round(float(budgets[inspected].sum()), 2),
        "gp_residuals": len(standardized), "v33_rebuild_exact": True,
        "sha256": file_hash(dest),
    }
    (args.out / "build_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
