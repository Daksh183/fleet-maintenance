"""Validate and build the frozen v32 fleet-maintenance policy."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

HISTOGRAMS = ((37, 47), (54, 63), (80, 90))
SEEDS = (42, 100, 101)
FACTORS = (1.0, 1.25)
CANDIDATE = "submission_v32_hist_kernel.csv"
INCUMBENT = "submission_v17_bud125.csv"
HISTORICAL_BLOBS = {
    "reports/p_oof_v2.npy": "a27fd683fed99551fa6011715dc70466cef7e0b1",
    "reports/p_te_v2.npy": "c11ef66cc11b512e4a148d6835aa0df7c2491426",
    "reports/v10_posterior.npy": "90d5ff72bdcd1fed4a492c2d20aa3f3df1fa74a9",
    f"submissions/{INCUMBENT}": "a0817001e8606c2c9a6accce313eadbc0136765c",
}


def histogram_blocks(frame):
    blocks = []
    for start, stop in HISTOGRAMS:
        values = frame[[f"s{col:03d}" for col in range(start, stop)]].to_numpy(float)
        total = values.sum(axis=1)
        complete = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (total > 0)
        denominator = np.where(complete, total, np.nan)
        blocks.append(values[:, :-1].cumsum(axis=1) / denominator[:, None])
    blocks.append(np.log1p(denominator)[:, None])
    return np.column_stack(blocks)


class HistogramKernel:
    blocks = (slice(0, 9), slice(9, 17), slice(17, 26), slice(26, 27))

    @staticmethod
    def distance(left, right):
        return cdist(left, right, metric="cityblock") / left.shape[1]

    def fit(self, features, target):
        complete = np.isfinite(features).all(axis=1)
        self.reference = features[complete]
        self.model = None
        self.scales = {}
        if len(self.reference) < 6:
            return self
        pairs = np.triu_indices(len(self.reference), k=1)
        distance = np.zeros((len(self.reference), len(self.reference)))
        for index, block in enumerate(self.blocks):
            positions = self.reference[:, block]
            block_distance = self.distance(positions, positions)
            scale = np.median(block_distance[pairs])
            if scale > 0:
                self.scales[index] = scale
                distance += block_distance / scale
        if not self.scales:
            return self
        distance /= len(self.scales)
        observed = target[complete]
        self.center = observed.mean()
        self.model = KernelRidge(alpha=0.25, kernel="precomputed").fit(
            np.exp(-distance), observed - self.center
        )
        np.fill_diagonal(distance, np.inf)
        self.radius = np.quantile(np.partition(distance, 4, axis=1)[:, 4], 0.95)
        return self

    def predict(self, features, batch_size=2000):
        result = np.full(len(features), np.nan)
        if self.model is None:
            return result
        complete = np.flatnonzero(np.isfinite(features).all(axis=1))
        for start in range(0, len(complete), batch_size):
            rows = complete[start:start + batch_size]
            distance = np.zeros((len(rows), len(self.reference)))
            for index, scale in self.scales.items():
                block = self.blocks[index]
                distance += self.distance(features[rows, block], self.reference[:, block]) / scale
            distance /= len(self.scales)
            supported = np.partition(distance, 4, axis=1)[:, 4] <= self.radius
            if supported.any():
                result[rows[supported]] = self.model.predict(np.exp(-distance[supported])) + self.center
        return result


def blended_residuals(actual_log, tree_oof, kernel_oof):
    active = np.isfinite(kernel_oof)
    return actual_log[active] - 0.5 * (tree_oof[active] + kernel_oof[active])


def actions(probability, tree, tree_residuals, kernel, blend_residuals):
    baseline = decide_pv(probability, tree, tree_residuals)
    flags, budgets = (value.copy() for value in baseline)
    active = np.isfinite(kernel) & (len(blend_residuals) >= 20)
    if active.any():
        mean = 0.5 * (tree[active] + kernel[active])
        flags[active], budgets[active] = decide_pv(probability[active], mean, blend_residuals)
    assert np.array_equal(flags[~active], baseline[0][~active])
    assert np.array_equal(budgets[~active], baseline[1][~active])
    return baseline, (flags, budgets), active


def rounded_budget(flags, raw, factor):
    return np.where(flags == 1, raw * factor, raw).round(2)


def fixed_odds(probability, multiplier):
    return multiplier * probability / (1 - probability + multiplier * probability)


def historical_odds(probability, posterior):
    grid, weights = posterior
    return sum(weight * fixed_odds(probability, odds) for odds, weight in zip(grid, weights))


def read_archive(archive):
    with ZipFile(archive) as zip_file:
        train = pd.read_csv(zip_file.open("train.csv"))
        test = pd.read_csv(zip_file.open("test.csv"))
        sample = pd.read_csv(zip_file.open("sample_submission.csv"), usecols=["id"])
    assert len(train) == 56000 and train.id.is_unique
    assert len(test) == len(sample) == 20000 and test.id.is_unique and sample.id.is_unique
    assert set(test.id) == set(sample.id) and set(test.id).isdisjoint(train.id)
    return train, test, sample


def verify_archive_files(repo):
    for filename, expected in HISTORICAL_BLOBS.items():
        content = (repo / filename).read_bytes()
        if filename.endswith(".csv"):
            content = content.replace(b"\r\n", b"\n")
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual != expected:
            raise ValueError(f"The historical artifact differs from the tagged file: {filename}")


def summary_row(ids, actual, reference, candidate, active, factor, population):
    base_flags, base_raw = reference
    new_flags, new_raw = candidate
    base_cost, base_parts = accounting(actual, base_flags, rounded_budget(base_flags, base_raw, factor))
    new_cost, new_parts = accounting(actual, new_flags, rounded_budget(new_flags, new_raw, factor))
    delta = new_cost - base_cost
    added = new_flags > base_flags
    dropped = new_flags < base_flags
    result = {
        "rows": len(ids), "kernel_active": int(active.sum()),
        "costly_failures": int((actual >= 1000).sum()),
        "costly_failures_kernel_active": int((active & (actual >= 1000)).sum()),
        "flags_base": int(base_flags.sum()), "flags_v32": int(new_flags.sum()),
        "flags_added": int(added.sum()), "flags_dropped": int(dropped.sum()),
        "base_actual_total": base_cost.sum(), "v32_actual_total": new_cost.sum(),
        "base_cost": population.normal @ base_cost, "v32_cost": population.normal @ new_cost,
        "base_wear_cost": population.standard @ base_cost, "v32_wear_cost": population.standard @ new_cost,
        "added_actual_delta": delta[added].sum(), "dropped_actual_delta": delta[dropped].sum(),
        "same_flag_actual_delta": delta[~(added | dropped)].sum(),
    }
    for component in base_parts:
        result[f"base_{component}"] = population.normal @ base_parts[component]
        result[f"v32_{component}"] = population.normal @ new_parts[component]
        result[f"wear_delta_{component}"] = population.standard @ (new_parts[component] - base_parts[component])
    return {**result, **population.influence(delta, ids)}, delta


def validate_fold(data, raw_train, native, shifted, posterior, signature):
    if str(data["signature"]) != signature:
        raise ValueError("Fold cache fingerprint does not match the current inputs and environment.")
    assert tuple(data["scenarios"]) == SCENARIOS
    assert data["policies"][0] == "A0"
    index = raw_train.index
    ids = data["ids"]
    train = index.get_indexer(data["cal_ids"])
    valid = index.get_indexer(ids)
    assert train.min() >= 0 and valid.min() >= 0
    assert len(set(train)) == len(train) and len(set(valid)) == len(valid)
    assert not np.intersect1d(train, valid).size
    actual = data["cost"]
    assert np.array_equal(raw_train.failure.to_numpy()[valid], data["y"])
    assert np.array_equal(raw_train.failure.to_numpy()[train], data["cal_y"])
    assert np.array_equal(raw_train.repair_cost.fillna(0).to_numpy()[valid], actual)
    failures = data["cal_y"] == 1
    failure_rows = train[failures]
    failure_log = np.log(raw_train.repair_cost.to_numpy()[failure_rows])
    tree_oof = data["cal_severity"][failures]
    np.testing.assert_allclose(failure_log - tree_oof, data["resid"])
    kernel_oof = np.full(len(failure_rows), np.nan)
    inner = data["inner_fold"][failures]
    assert set(inner) == set(range(5))
    for fold in range(5):
        held = inner == fold
        model = HistogramKernel().fit(native[failure_rows[~held]], failure_log[~held])
        kernel_oof[held] = model.predict(native[failure_rows[held]])
    residual = blended_residuals(failure_log, tree_oof, kernel_oof)
    model = HistogramKernel().fit(native[failure_rows], failure_log)
    kernel = (model.predict(native[valid]), model.predict(shifted[valid]))
    platt = make("platt").fit(data["cal_raw"], data["cal_y"])
    probabilities = [platt.predict(raw.astype(float)) for raw in data["raw"][:, 0]]
    rows = []
    base = {"seed": int(data["seed"]), "fold": int(data["fold"]), "active_inner_residuals": len(residual)}
    for odds_method, odds in (("fixed", 1), ("fixed", 3), ("historical", 3)):
        pop = Population(data["wear"], data["target_mix"], data["y"], odds)
        stress = {factor: [] for factor in FACTORS}
        for index, scenario in enumerate(SCENARIOS):
            q = probabilities[index]
            probability = fixed_odds(q, odds) if odds_method == "fixed" else historical_odds(q, posterior)
            layout = int(index > 0)
            reference, candidate, active = actions(
                probability, data["severity"][layout].astype(float), data["resid"], kernel[layout], residual
            )
            for factor in FACTORS:
                row, delta = summary_row(ids, actual, reference, candidate, active, factor, pop)
                rows.append({**base, "odds_method": odds_method, "odds": odds,
                             "budget_factor": factor, "scenario": scenario, **row})
                if layout:
                    stress[factor].append((row, delta))
        for factor, results in stress.items():
            stress_delta = np.mean([delta for _, delta in results], axis=0)
            first = results[0][0]
            influence = pop.influence(stress_delta, ids)
            mean = {key: np.mean([row[key] for row, _ in results]) for key in first
                    if key not in influence}
            rows.append({**base, "odds_method": odds_method, "odds": odds,
                         "budget_factor": factor, "scenario": "stress_mean", **mean,
                         **influence})
    return rows


def repeat_summary(folds):
    summaries = []
    by = ["seed", "odds_method", "odds", "budget_factor", "scenario"]
    influence_fields = {"influential_id", "wear_influential_id", "influence", "wear_influence",
                        "worst_removal", "wear_worst_removal"}
    for _, group in folds.groupby(by, sort=True):
        result = {key: group.iloc[0][key] for key in by}
        result["folds"] = len(group)
        for key in group.columns:
            if key in by or key in influence_fields or key == "fold":
                continue
            count_or_total = key in {"rows", "kernel_active", "costly_failures", "costly_failures_kernel_active",
                                    "flags_base", "flags_v32", "flags_added", "flags_dropped", "realized_delta"}
            result[key] = group[key].sum() if count_or_total or "actual_" in key else group[key].mean()
        for prefix in ("", "wear_"):
            effect = f"{prefix}influence"
            row = group.loc[group[effect].abs().idxmax()]
            result[f"{prefix}influential_id"] = row[f"{prefix}influential_id"]
            result[f"{prefix}delta_after_most_influential"] = (
                result[f"{prefix}delta"] + row[effect] / len(group)
            )
            result[f"{prefix}delta_after_least_favorable_deletion"] = (
                result[f"{prefix}delta"] + group[f"{prefix}worst_removal"].max() / len(group)
            )
        summaries.append(result)
    return pd.DataFrame(summaries)


def primary_results(report):
    common = report[(report.odds_method == "fixed") & (report.budget_factor == 1.25)]
    return (common[(common.odds == 1) & (common.scenario == "source")],
            common[(common.odds == 3) & (common.scenario == "stress_mean")])


def validate(args, train, path_signature, enabled_holdout):
    dev, holdout = split_holdout(train)
    native = histogram_blocks(train)
    hist_cols = [f"s{col:03d}" for col in range(54, 64)]
    shifted = native.copy()
    # The target layout only changes H54 among the kernel inputs.
    shifted_counts = target_layout(train[hist_cols])[hist_cols[:-1]].to_numpy(float)
    total = shifted_counts.sum(axis=1)
    complete = np.isfinite(shifted_counts).all(axis=1) & (shifted_counts >= 0).all(axis=1) & (total > 0)
    shifted[:, 9:17] = shifted_counts[:, :-1].cumsum(axis=1) / np.where(complete, total, np.nan)[:, None]
    posterior = np.load(args.repo / "reports/v10_posterior.npy")
    indexed = train.set_index("id")
    paths = [(42, [args.cache / "holdout.npz"])] if enabled_holdout else [
        (seed, [args.cache / f"dev_{seed}_{fold}.npz" for fold in range(1, 6)]) for seed in SEEDS
    ]
    output = []
    expected_ids = set(holdout.id if enabled_holdout else dev.id)
    for seed, fold_paths in paths:
        seen = set()
        for fold_num, path in enumerate(fold_paths, start=1):
            with np.load(path, allow_pickle=False) as stored:
                data = dict(stored)
            valid_ids = set(data["ids"])
            assert data["seed"] == seed and data["fold"] == fold_num
            assert not seen.intersection(valid_ids)
            assert set(data["cal_ids"]) == set(dev.id) - valid_ids
            seen.update(valid_ids)
            output.extend(validate_fold(data, indexed, native, shifted, posterior, path_signature))
        assert seen == expected_ids
        print(f"Scored {'holdout' if enabled_holdout else f'development seed {seed}'}", flush=True)
    report = repeat_summary(pd.DataFrame(output))
    stage = "holdout" if enabled_holdout else "development"
    report.to_csv(args.out / f"{stage}_paired_results.csv", index=False)
    source, stress = primary_results(report)
    resilient = ((stress.wear_delta < 0) & (stress.wear_delta_after_most_influential < 0)).sum()
    decision = {
        "native_cost_tree": float(source.base_cost.mean()),
        "native_cost_v32": float(source.v32_cost.mean()),
        "stress_wear_cost_tree": float(stress.base_wear_cost.mean()),
        "stress_wear_cost_v32": float(stress.v32_wear_cost.mean()),
        "stress_wear_delta_after_most_influential": float(stress.wear_delta_after_most_influential.mean()),
        "favorable_stress_repeats_after_deletion": int(resilient),
    }
    if enabled_holdout:
        decision["supports_trial"] = bool(source.delta.iloc[0] < 0 and stress.wear_delta.iloc[0] < 0)
    else:
        decision["passes_development"] = bool(
            resilient >= 2 and stress.wear_delta.mean() < 0 and source.delta.mean() <= 0
        )
    (args.out / f"{stage}_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


def submission_frame(ids, policy):
    flags, budgets = policy
    return pd.DataFrame({"id": ids, "flag": flags, "budget": rounded_budget(flags, budgets, 1.25)})


def validate_submission(frame, ids):
    assert list(frame.columns) == ["id", "flag", "budget"]
    assert len(frame) == 20000 and frame.id.is_unique
    assert np.array_equal(frame.id, ids)
    assert pd.api.types.is_integer_dtype(frame.flag) and frame.flag.isin([0, 1]).all()
    assert np.isfinite(frame.budget).all() and (frame.budget >= 0).all()


def difference(old, new):
    assert np.array_equal(old.id, new.id)
    added = (new.flag == 1) & (old.flag == 0)
    dropped = (new.flag == 0) & (old.flag == 1)
    common = (new.flag == 1) & (old.flag == 1)
    return {
        "flags_added": int(added.sum()), "flags_dropped": int(dropped.sum()),
        "budgets_changed_on_common_flags": int((common & (old.budget != new.budget)).sum()),
        "allocated_budget_delta": round(float((new.flag * new.budget - old.flag * old.budget).sum()), 2),
        "identical": bool(old.equals(new)),
    }


def build(args, train, test, sample):
    ids = test.id.to_numpy()
    archived = pd.read_csv(args.repo / "submissions" / INCUMBENT)
    validate_submission(archived, ids)
    dev, _ = split_holdout(train)
    raw_oof = np.load(args.repo / "reports/p_oof_v2.npy")
    raw_test = np.load(args.repo / "reports/p_te_v2.npy")
    posterior = np.load(args.repo / "reports/v10_posterior.npy")
    assert raw_oof.shape == (len(dev),) and raw_test.shape == (len(test),)
    assert np.isfinite(raw_oof).all() and np.isfinite(raw_test).all()
    q = make("platt").fit(raw_oof, dev.failure.to_numpy()).predict(raw_test)
    test[HIST[:-1]] = test[HIST[1:]].to_numpy()
    test[HIST[-1]] = np.nan
    failures = train.loc[train.failure == 1]
    assert len(failures) == 397
    outcome = np.log(failures.repair_cost.to_numpy())
    sensors = [col for col in train if col.startswith("s") and col not in DROP and col not in DRIFT]
    x = pd.concat([failures[sensors], shape(failures)], axis=1)
    target_x = pd.concat([test[sensors], shape(test)], axis=1)
    hist = histogram_blocks(failures)
    params = {**CM, "n_estimators": 1000, "max_depth": 2, "n_jobs": args.threads}
    tree_oof = np.empty(len(failures))
    kernel_oof = np.full(len(failures), np.nan)
    for training, held in KFold(5, shuffle=True, random_state=42).split(x):
        regressor = XGBRegressor(**params).fit(x.iloc[training], outcome[training])
        tree_oof[held] = regressor.predict(x.iloc[held])
        kernel_oof[held] = HistogramKernel().fit(hist[training], outcome[training]).predict(hist[held])
    residual = outcome - tree_oof
    kernel_residual = blended_residuals(outcome, tree_oof, kernel_oof)
    tree = XGBRegressor(**params).fit(x, outcome).predict(target_x)
    test_hist = histogram_blocks(test)
    kernel = HistogramKernel().fit(hist, outcome).predict(test_hist)
    probability = fixed_odds(q, 3)
    reference, candidate, active = actions(probability, tree, residual, kernel, kernel_residual)
    historical = submission_frame(ids, decide_pv(historical_odds(q, posterior), tree, residual))
    baseline = submission_frame(ids, reference)
    result = submission_frame(ids, candidate)
    disabled = actions(probability, tree, residual, kernel, np.empty(0))[1]
    pd.testing.assert_frame_equal(baseline, submission_frame(ids, disabled))
    pd.testing.assert_frame_equal(baseline.loc[~active], result.loc[~active])
    ordered = result.set_index("id").loc[sample.id].reset_index()
    path = args.out / CANDIDATE
    tmp = path.with_suffix(".part")
    ordered.to_csv(tmp, index=False, float_format="%.2f")
    reopened = pd.read_csv(tmp)
    validate_submission(reopened, sample.id)
    pd.testing.assert_frame_equal(ordered, reopened)
    tmp.replace(path)
    np.savez_compressed(args.cache / "v32_final_tree.npz", ids=np.asarray(ids, dtype=str),
                        tree=tree, residual=residual, q=q, active=active)
    audit = {
        "rows": len(reopened), "inspection_flags": int(result.flag.sum()),
        "total_allocated_budget": round(float((result.flag * result.budget).sum()), 2),
        "eligible_test_histograms": int(np.isfinite(test_hist).all(axis=1).sum()),
        "kernel_active_test_rows": int(active.sum()), "kernel_residual_count": len(kernel_residual),
        "archive_vs_historical_replay": difference(archived, historical),
        "historical_replay_vs_fixed_odds_tree": difference(historical, baseline),
        "fixed_odds_tree_vs_v32": difference(baseline, result),
        "archive_vs_v32": difference(archived, result),
        "matches_control_when_disabled": True,
        "unsupported_rows_match_control": True,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.out / "submission_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


def self_check():
    test = np.array([[1., 1.], [0., 1.], [0., 0.]])
    distances = HistogramKernel.distance(test[:1], test)[0]
    np.testing.assert_array_equal(distances, [0, .5, 1])
    assert np.exp(-distances[0]) == 1
    assert vehicle_cost([400, 0, 1200, 0], [1, 1, 0, 0], [300, 50, 0, 0]).sum() == 1241


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "holdout", "build"))
    for name in ("repo", "zip", "cache", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    load_helpers(args.repo)
    verify_archive_files(args.repo)
    self_check()
    signature, _ = fingerprint(args.zip)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    train, test, sample = read_archive(args.zip)
    if args.stage == "build":
        build(args, train, test, sample)
    else:
        if args.stage == "holdout":
            decision = json.loads((args.out / "development_decision.json").read_text(encoding="utf-8"))
            if not decision["passes_development"]:
                raise ValueError("Development did not pass the frozen holdout gate.")
        validate(args, train, signature, args.stage == "holdout")


def load_helpers(repo):
    global DRIFT, DROP, HIST, CM, SCENARIOS, fingerprint, split_holdout, target_layout
    global shape, make, decide_pv, Population, accounting, vehicle_cost
    sys.path.insert(0, str((repo / "validation").resolve()))
    from fit_folds import DRIFT, DROP, HIST, CM, SCENARIOS, fingerprint, split_holdout, target_layout
    from features import shape
    from calib import make
    from decide import decide_pv
    from metrics import Population, accounting
    from score import vehicle_cost


if __name__ == "__main__":
    main()
