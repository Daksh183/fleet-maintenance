"""Shared data loading, fixed folds/hold-out, and the experiment logger.

Every version (v0..v8) uses the same hold-out split and the same 5 folds,
so results in experiments.csv are directly comparable.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = ROOT / "reports" / "experiments.csv"
SEED = 42
N_FOLDS = 5


HIST = [f"s{i:03d}" for i in range(54, 64)]  # 10-bin histogram, shifted by 1 bin in test (v1)
DROP = ["s047", "s146"]  # s047: 100% NaN in test; s146: constant in train


def load(raw=False):
    """Load train/test. Unless raw=True, apply the v1 fixes:
    - realign the s054-s063 histogram: test bin k+1 == train bin k (KS <= 0.07 per bin),
      so test s055..s063 -> s054..s062; train s063 has no test counterpart -> NaN in test.
    - drop s047 (absent in test) and s146 (constant).
    """
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    if not raw:
        te[HIST[:-1]] = te[HIST[1:]].to_numpy()
        te[HIST[-1]] = np.nan
        tr, te = tr.drop(columns=DROP), te.drop(columns=DROP)
    sensors = [c for c in tr.columns if c.startswith("s")]
    return tr, te, sensors


def split_holdout(tr: pd.DataFrame):
    """Locked 20% hold-out, stratified on failure. Touch only at v0, v4, v8."""
    dev_idx, ho_idx = train_test_split(
        np.arange(len(tr)), test_size=0.2, stratify=tr["failure"], random_state=SEED
    )
    return tr.iloc[dev_idx].reset_index(drop=True), tr.iloc[ho_idx].reset_index(drop=True)


def folds(y):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def fold_costs(c, flag, budget, fold_list):
    """Per-vehicle cost in each validation fold -> (mean, std)."""
    from score import per_vehicle

    vals = [per_vehicle(c[va], flag[va], budget[va]) for _, va in fold_list]
    return float(np.mean(vals)), float(np.std(vals))


def write_submission(te, flag, budget, version):
    """Write submissions/submission_<version>.csv with the format checks the grader applies."""
    flag = np.asarray(flag).astype(int)
    budget = np.maximum(np.asarray(budget, float), 0.0)
    assert len(te) == 20000 and te["id"].is_unique and set(flag) <= {0, 1} and np.isfinite(budget).all()
    path = ROOT / "submissions" / f"submission_{version}.csv"
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": te["id"], "flag": flag, "budget": budget.round(2)}).to_csv(path, index=False)
    print(f"wrote {path.name}: flagged {flag.mean():.2%} of test vehicles ({flag.sum()})")
    return path


def log_result(version, change, cv_mean, cv_std, holdout=None, public_lb=None, notes=""):
    LOG.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": version, "change": change,
        "cv_mean": round(cv_mean, 4), "cv_std": round(cv_std, 4),
        "holdout": None if holdout is None else round(holdout, 4),
        "public_lb": public_lb, "notes": notes,
    }])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"[{version}] {change}: CV {cv_mean:.4f} ± {cv_std:.4f}"
          + ("" if holdout is None else f" | holdout {holdout:.4f}"))
