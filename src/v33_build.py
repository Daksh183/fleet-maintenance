"""Build the frozen V33 parts-budget candidate using the team's competition-data cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from v33_hist_gp import GaussianSeverity, extract, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zip", "base", "runtime", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str((args.runtime / "src").resolve()))
    from common import HIST
    from decide import decide_pv
    from score import vehicle_cost

    with args.zip.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == "6e88bb69b36a916f84fb50b38a0da53d5856815acdfc07b6b0cff7e18bb5297b"
    assert vehicle_cost([400, 0, 1200, 0], [1, 1, 0, 0], [300, 50, 0, 0]).sum() == 1241
    with ZipFile(args.zip) as archive:
        train = pd.read_csv(archive.open("train.csv"))
        test = pd.read_csv(archive.open("test.csv"))
        sample = pd.read_csv(archive.open("sample_submission.csv"))
    test[HIST[:-1]] = test[HIST[1:]].to_numpy()
    test[HIST[-1]] = np.nan
    failures = train.loc[train.failure == 1]
    assert len(failures) == 397
    y = np.log(failures.repair_cost.to_numpy())
    features = extract(failures)
    oof = np.full(len(y), np.nan)
    for fit, held in KFold(5, shuffle=True, random_state=42).split(y):
        model = GaussianSeverity().fit(select(features, fit), y[fit])
        oof[held] = model.predict(select(features, held))
    eligible = np.isfinite(oof)
    residual = y[eligible] - oof[eligible]
    severity = GaussianSeverity().fit(features, y).predict(extract(test))
    with np.load(args.base, allow_pickle=False) as cache:
        assert np.array_equal(cache["ids"], test.id) and np.isfinite(cache["q"]).all()
        q = cache["q"]
        p = 3*q / (1-q+3*q)
        flags, old_budget = decide_pv(p, cache["tree"], cache["residual"])
    updated = old_budget.copy()
    active = (flags == 1) & (old_budget > 0) & np.isfinite(severity)
    _, updated[active] = decide_pv(p[active], severity[active], residual)
    amount = np.where(flags == 1, 1.25*updated, updated).round(2)
    old_amount = np.where(flags == 1, 1.25*old_budget, old_budget).round(2)
    assert np.array_equal(amount[~active], old_amount[~active])
    frame = pd.DataFrame({"id": test.id, "flag": flags, "budget": amount})
    frame = frame.set_index("id").loc[sample.id].reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    filename = args.out / "submission_v33_gp_budget.csv"
    temporary = filename.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, float_format="%.2f", lineterminator="\r\n")
    check = pd.read_csv(temporary)
    pd.testing.assert_frame_equal(frame, check)
    assert list(check.columns) == ["id", "flag", "budget"]
    assert len(check) == 20000 and check.id.is_unique and check.id.equals(sample.id)
    assert check.flag.isin([0,1]).all() and np.isfinite(check.budget).all() and (check.budget >= 0).all()
    temporary.replace(filename)
    audit = dict(rows=len(check), flags=int(check.flag.sum()), eligible_residuals=int(eligible.sum()),
                 supported_test_rows=int(np.isfinite(severity).sum()), changed_flags_vs_local_control=0,
                 changed_charged_budgets=int((amount != old_amount).sum()),
                 allocated_budget=round(float((check.flag*check.budget).sum()), 2),
                 sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    print(json.dumps(audit, indent=2))
    (args.out/"build_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
