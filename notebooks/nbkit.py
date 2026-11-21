"""Helpers for building the per-version notebooks. Every notebook is SELF-CONTAINED: the shared prelude
below is pasted into each one, so a notebook runs on its own (as a Kaggle notebook must)."""
import nbformat as nbf

SETUP = r'''
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_predict

DATA, OUT = Path("../data"), Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42
INSPECT, UNDER, OVER = 10.0, 0.20, 0.02          # inspection cost; shortage cost / unit; unused-budget cost / unit
Q_STAR = UNDER / (UNDER + OVER)                  # newsvendor critical ratio 0.909
HISTB = [f"s{i:03d}" for i in range(54, 64)]     # 10-bin histogram that is shifted by one bin in test

def load(raw=False):
    """train/test with the v1 fixes: realign s054-s063 (test bin k+1 == train bin k), drop s047 (all NaN in
    test) and s146 (constant). raw=True returns the files untouched."""
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    if not raw:
        te[HISTB[:-1]] = te[HISTB[1:]].to_numpy(); te[HISTB[-1]] = np.nan
        tr, te = tr.drop(columns=["s047", "s146"]), te.drop(columns=["s047", "s146"])
    return tr, te, [c for c in tr.columns if c.startswith("s")]

tr, te, S = load()
dev_idx, ho_idx = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=tr.failure, random_state=SEED)
dev, ho = tr.iloc[dev_idx].reset_index(drop=True), tr.iloc[ho_idx].reset_index(drop=True)   # ho: locked 20% hold-out
y, c = dev.failure.to_numpy(), dev.repair_cost.fillna(0).to_numpy() * dev.failure.to_numpy()
y_ho, c_ho = ho.failure.to_numpy(), ho.repair_cost.fillna(0).to_numpy() * ho.failure.to_numpy()
c_all = tr.repair_cost.fillna(0).to_numpy() * tr.failure.to_numpy()
F = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))   # the 5 fixed folds
print(f"train {tr.shape}  test {te.shape}  sensors {len(S)}  dev {len(dev)} / hold-out {len(ho)}  failures {int(y.sum())}")
'''

FUNCS = r'''
def vehicle_cost(c, flag, b):
    return np.where(flag == 1, INSPECT + UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0), c)

def per_vehicle(c, flag, b):
    return float(vehicle_cost(np.asarray(c, float), np.asarray(flag), np.asarray(b, float)).mean())

fold_cost = lambda flag, b: np.array([per_vehicle(c[v], flag[v], b[v]) for _, v in F])

def write_submission(ids, flag, budget, name):
    flag, budget = np.asarray(flag).astype(int), np.maximum(np.asarray(budget, float), 0)
    assert len(ids) == 20000 and ids.is_unique and set(flag) <= {0, 1} and np.isfinite(budget).all()
    pd.DataFrame({"id": ids, "flag": flag, "budget": budget.round(2)}).to_csv(OUT / f"submission_{name}.csv", index=False)
    print(f"wrote outputs/submission_{name}.csv: flagged {flag.mean():.2%} of test vehicles")

def decide_global(p, fail_costs):
    """Same repair-cost distribution F (empirical failure costs) for every vehicle."""
    p = np.clip(np.asarray(p, float), 0, 1); fc = np.sort(np.asarray(fail_costs, float)); mu = fc.mean()
    b = np.zeros_like(p); need = p > 1 - Q_STAR
    b[need] = np.quantile(fc, 1 - (1 - Q_STAR) / p[need])                       # newsvendor budget
    fail = (UNDER * np.maximum(fc[None, :] - b[:, None], 0) + OVER * np.maximum(b[:, None] - fc[None, :], 0)).mean(1)
    return ((INSPECT + p * fail + (1 - p) * OVER * b) < p * mu).astype(int), b   # flag iff cheaper than not flagging

def decide_pv(p, m, resid, fail_costs=None, K=99, scale=None):
    """Per-vehicle cost distribution: log c | fail ~ m_i + scale_i * residuals (out-of-fold residuals).
    fail_costs given -> budget from that GLOBAL distribution (ablation); scale -> per-vehicle spread (v9)."""
    p, m = np.clip(np.asarray(p, float), 0, 1), np.asarray(m, float)
    sc = np.ones_like(m) if scale is None else np.asarray(scale, float)
    M = np.exp(m[:, None] + sc[:, None] * np.quantile(resid, (np.arange(K) + 0.5) / K)[None, :]); mu = M.mean(1)
    b = np.zeros_like(p); need = p > 1 - Q_STAR; level = 1 - (1 - Q_STAR) / p[need]
    b[need] = (np.exp(m[need] + sc[need] * np.quantile(resid, level)) if fail_costs is None
               else np.quantile(np.asarray(fail_costs, float), level))
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return ((INSPECT + p * fail + (1 - p) * OVER * b) < p * mu).astype(int), b

class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(p), y); return self
    def predict(self, p): return self.m.predict_proba(self._x(p))[:, 1]
    @staticmethod
    def _x(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))[:, None]

class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y); return self
    def predict(self, p): return self.m.predict(p)

make_cal = lambda kind: {"platt": Platt, "iso": Iso}[kind]()

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=1, n_jobs=-1, random_state=SEED)                         # failure classifier (v2..)
CM = dict(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
          min_child_weight=5, n_jobs=-1, random_state=SEED)                             # repair-cost model (v5)
inner = KFold(5, shuffle=True, random_state=SEED)

def cost_model(params, Xa, lc, Xb):
    """XGBoost on log repair cost (failures only). Returns predictions for Xb and OUT-OF-FOLD residuals of Xa."""
    resid = lc - cross_val_predict(xgb.XGBRegressor(**params), Xa, lc, cv=inner)
    return xgb.XGBRegressor(**params).fit(Xa, lc).predict(Xb), resid

def fit_oof(X, params=PARAMS):
    """5-fold out-of-fold failure probabilities on dev, and probabilities for test from a model fit on all train."""
    o = np.zeros(len(dev))
    for a, b in F:
        o[b] = xgb.XGBClassifier(**params).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    return o

shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))       # prior shift by odds factor k
'''

SHAPE = r'''
HIST = {a: [f"s{i:03d}" for i in range(a, a + 10)] for a in (23, 37, 54, 80, 129, 154)}   # six hidden 10-bin histograms
HCOLS = {col for g in HIST.values() for col in g}

def shape(df):
    """Histogram-shape features: per-bin shares, mean/std bin, entropy, low/high-3 tail shares, log usage."""
    out = {"logU": np.log1p(df[HIST[54][:9]].sum(axis=1, min_count=9))}
    for a, g in HIST.items():
        g = g[:9] if a == 54 else g                        # test s063 is NaN after realignment
        x = df[g].to_numpy(float); tot = x.sum(1, keepdims=True)
        sh = np.where(tot > 0, x / np.where(tot > 0, tot, 1), np.nan); k = np.arange(len(g)); mean = (sh * k).sum(1)
        out.update({f"h{a}_sh{i}": sh[:, i] for i in range(len(g))})
        out[f"h{a}_mean"] = np.where(tot[:, 0] > 0, mean, np.nan)
        out[f"h{a}_std"] = np.where(tot[:, 0] > 0, np.sqrt(np.maximum((sh * k ** 2).sum(1) - mean ** 2, 0)), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"h{a}_ent"] = np.where(tot[:, 0] > 0, -np.nansum(np.where(sh > 0, sh * np.log(sh), 0), 1), np.nan)
        out[f"h{a}_lo3"], out[f"h{a}_hi3"] = sh[:, :3].sum(1), sh[:, -3:].sum(1)
    return pd.DataFrame(out, index=df.index)

def ratios(df, cols):
    U = df[HIST[54][:9]].sum(axis=1, min_count=9).replace(0, np.nan)
    return df[[col for col in cols if col not in HCOLS]].div(U, axis=0).add_suffix("_perU")
'''

EM = r'''
def em_odds_factor(p, pi_tr, it=300):
    """Saerens-Latinne-Decaestecker EM: odds factor k by which the test failure odds exceed the training odds."""
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * p
        pi = (a / (a + (1 - pi) / (1 - pi_tr) * (1 - p))).mean()
    return (pi / (1 - pi)) / (pi_tr / (1 - pi_tr))
'''


class NB:
    def __init__(self, title, sub=None):
        self.cells, self.title, self.sub = [], title, sub

    def md(self, s): self.cells.append(nbf.v4.new_markdown_cell(s.strip())); return self
    def code(self, s): self.cells.append(nbf.v4.new_code_cell(s.strip())); return self

    def header(self, version, sub_line, body):
        self.md(f"# {self.title}\n\n**Version:** {version}  \n**Kaggle submission:** {sub_line}\n\n{body}\n\n"
                "*Self-contained: needs only `../data/train.csv` and `../data/test.csv`. Part of the Fleet Maintenance "
                "project (Kaggle `mlprojectfleet2`); the cost, decision rule and validation protocol are derived in "
                "`fleet_maintenance.ipynb`.*")
        self.code(SETUP); self.code(FUNCS); return self

    def save(self, path):
        import os
        only = os.environ.get("ONLY")                    # ONLY=a.ipynb,b.ipynb rebuilds just those
        if only and path not in only.split(","): return
        nb = nbf.v4.new_notebook(); nb.cells = self.cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, path); print("wrote", path, len(self.cells), "cells")
