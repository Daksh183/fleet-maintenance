"""(Adversarial validation was dropped: apart from s120/s122 and column rounding, train and test are indistinguishable, AUC 0.51.)
v67: several validation methods side by side, BACKTESTED against ranking changes whose public result we know.
Candidates (public change vs the v12 classifier, flag count matched; lower is better):
  base (v12 classifier)  0      nos122 (v22) +333    ipcw (v18 ensemble, 4 members) +784    pu (v24) +735
  bag5 (v13; also bagged cost model) +1135    drop4 (v3b: drop the four drifted sensors; worse than v3 on public)
and the new ones: aft (ensemble), aftd6, clf+aft.
Validation methods (all on the 80% dev split):
  plain       3x5 stratified CV
  wear        same predictions, validation weighted to the test depot's wear (s126)
  complete    same, complete cases only with inverse-censoring weights (v18's judge)
  instrument  CV predictions on validation vehicles given the test depot's readings (s122 x2.55, s120 + 16.0M)
  cluster     leave-one-vehicle-cluster-out (k-means on histogram shapes), clusters weighted to their test share
  ho-wear     train below the 80th percentile of s126, score above
  ho-usage    train below the 80th percentile of total usage, score above
Measure: cost per vehicle at each candidate's own best flag level (ranking quality only), plus weighted AP."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
t = np.maximum(dev.followup_days.to_numpy(float), 0.5); qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
NO = [s for s in S if s not in ("s120", "s122")]; DR = [s for s in S if s not in ("s120", "s122", "s126", "s128")]
P = {**PARAMS, "n_jobs": 2}
n = len(dev); base_dev = dev; _sh = dev.copy(); _sh["s122"] = _sh.s122 * 2.55; _sh["s120"] = _sh.s120 + 16.0e6
dev = pd.concat([dev, _sh], ignore_index=True)          # rows n.. are the same vehicles with the test depot's readings
def km_censor(tt, ev, H=365):
    cens = (ev == 0) & (tt < H); G = 1.0; out = {}
    for s in np.unique(tt[cens]):
        G *= 1 - ((tt == s) & cens).sum() / (tt >= s).sum(); out[s] = G
    k = np.array(list(out)); v = np.array(list(out.values()))
    return lambda u: np.where(u <= k[0], 1.0, v[np.searchsorted(k, u, side="right") - 1].clip(min=1e-3))
def ipcw_w(idx):
    G = km_censor(t[idx], y[idx]); cm = (y[idx] == 1) | (t[idx] >= 365); w = np.zeros(len(idx))
    w[cm] = 1 / G(np.where(y[idx] == 1, np.minimum(t[idx], 365) - 1, 364)[cm]); return w
def xclf(a, b, cols, w=None, **kw):
    k = np.ones(len(a), bool) if w is None else w > 0
    return xgb.XGBClassifier(**{**P, **kw}).fit(dev[cols].iloc[a[k]], y[a[k]], sample_weight=None if w is None else w[k]).predict_proba(dev[cols].iloc[b])[:, 1]
def aft(a, b, scale=1.0, **kw):
    d = xgb.DMatrix(dev[NO].iloc[a]); d.set_float_info("label_lower_bound", t[a]); d.set_float_info("label_upper_bound", np.where(y[a] == 1, t[a], np.inf))
    pr = dict(max_depth=4, eta=0.02, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, nthread=2, seed=42, objective="survival:aft",
              aft_loss_distribution="normal", aft_loss_distribution_scale=scale, base_score=365.0); pr.update(kw)
    r = -xgb.train(pr, d, 1000).predict(xgb.DMatrix(dev[NO].iloc[b]), output_margin=True) / scale
    return 1 / (1 + np.exp(-np.clip(r + 6, -13, 13)))
def pu(a, b):
    cens = (y[a] == 0) & (t[a] < 365); A = xgb.XGBClassifier(**P).fit(dev[S].iloc[a[~cens]], y[a[~cens]])
    pi = A.predict_proba(dev[S].iloc[a[cens]])[:, 1]; X = pd.concat([dev[S].iloc[a[~cens]], dev[S].iloc[a[cens]], dev[S].iloc[a[cens]]])
    Y = np.r_[y[a[~cens]], np.ones(cens.sum()), np.zeros(cens.sum())]; W = np.r_[np.ones((~cens).sum()), pi, 1 - pi]
    return xgb.XGBClassifier(**P).fit(X, Y, sample_weight=W).predict_proba(dev[S].iloc[b])[:, 1]
D6 = dict(max_depth=6, min_child_weight=5); SL = dict(n_estimators=1000, learning_rate=0.02, colsample_bytree=0.5)
# unit fits; candidates are averages of units
UNIT = {"base": lambda a, b: xclf(a, b, S), "nos122": lambda a, b: xclf(a, b, NO), "drop4": lambda a, b: xclf(a, b, DR), "pu": pu,
        **{f"seed{i}": (lambda a, b, i=i: xclf(a, b, S, random_state=i)) for i in (1, 2)},
        "ip0": lambda a, b: xclf(a, b, S, ipcw_w(a)), "ip1": lambda a, b: xclf(a, b, DR, ipcw_w(a)), "ip2": lambda a, b: xclf(a, b, S, ipcw_w(a), **D6),
        "aft1": aft, "aft2": lambda a, b: aft(a, b, 2.0), "aftd6": lambda a, b: aft(a, b, max_depth=6, min_child_weight=5)}
CAND = {"base": ["base"], "nos122": ["nos122"], "ipcw": ["ip0", "ip1", "ip2"], "pu": ["pu"], "bag3": ["base", "seed1", "seed2"],
        "drop4": ["drop4"], "aft": ["aft1", "aft2", "aftd6"], "aftd6": ["aftd6"], "clf+aft": ["nos122", "aft1", "aft2", "aftd6"]}
PUBLIC = {"nos122": 333, "ipcw": 784, "pu": 735, "bag3": 1135, "drop4": None}
L = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))); sig = lambda v: 1 / (1 + np.exp(-v))
def combine(pred, parts):          # probability average for the classifier ensembles (as uploaded), logit average where AFT is involved
    if any(p.startswith("aft") for p in parts): return sig(np.mean([L(pred[p]) for p in parts], 0))
    return np.mean([pred[p] for p in parts], 0)
if __name__ == "__main__":
    # ---- splits
    dev_all = dev
    HIST = (23, 37, 54, 80, 129, 154)
    def shares(d):
        out = []
        for h in HIST:
            x = d[[f"s{i:03d}" for i in range(h, h + (9 if h == 54 else 10))]].to_numpy(float); s_ = x.sum(1, keepdims=True); out.append(np.sqrt(np.nan_to_num(x / np.where(s_ > 0, s_, 1))))
        u = np.log1p(np.nan_to_num(d[[f"s{i:03d}" for i in range(54, 63)]].sum(axis=1).to_numpy()))
        return np.column_stack(out + [(u - 10) / 5])
    km = KMeans(8, random_state=0, n_init=4).fit(shares(base_dev)); cl = km.labels_; ct = km.predict(shares(te))
    W_cl = (np.bincount(ct, minlength=8) / len(te) / (np.bincount(cl, minlength=8) / n))[cl]
    print("clusters: dev share / test share / failures:", [(round(float((cl == k).mean()), 2), round(float((ct == k).mean()), 2), int(y[cl == k].sum())) for k in range(8)], flush=True)
    e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
    h_ = lambda v: np.histogram(np.log1p(v[np.isfinite(v)]), e)[0] / np.isfinite(v).sum()
    ratio = np.clip(h_(te.s126.to_numpy()) / np.maximum(h_(tr.s126.to_numpy()), 1e-9), 0, 5)
    W_wear = ratio[np.clip(np.searchsorted(e, np.log1p(base_dev.s126.fillna(base_dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
    W_comp = ipcw_w(np.arange(n))
    x = base_dev.s126.fillna(base_dev.s126.median()).to_numpy(); u = base_dev[[f"s{i:03d}" for i in range(54, 63)]].sum(axis=1, min_count=1).fillna(0).to_numpy()
    HOLD = {"ho-wear": x >= np.quantile(x, 0.8), "ho-usage": u >= np.quantile(u, 0.8)}
    CV = [(r, a, b) for r in range(2) for a, b in StratifiedKFold(5, shuffle=True, random_state=300 + r).split(np.zeros(len(y)), y)]
    LCO = [(np.where(cl != k)[0], np.where(cl == k)[0]) for k in range(8)]
    jobs = [("cv", i, a, np.r_[b, b + n]) for i, (r, a, b) in enumerate(CV)] + [("lco", i, a, b) for i, (a, b) in enumerate(LCO)]
    for nm, mask in HOLD.items():
        lo, hi = np.where(~mask)[0], np.where(mask)[0]
        jobs += [(nm, i, lo[a], lo[b]) for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(lo, y[lo]))] + [(nm, 5, lo, hi)]
        print(nm, f"train failures {int(y[lo].sum())}, scored {len(hi)} vehicles with {int(y[hi].sum())} failures", flush=True)
    tasks = [(u_, j) for u_ in UNIT for j in range(len(jobs))]
    R = Parallel(n_jobs=4, batch_size=1)(delayed(lambda u_, j: UNIT[u_](jobs[j][2], jobs[j][3]))(u_, j) for u_, j in tasks)
    R = dict(zip(tasks, R)); print("fits done", flush=True)
    PRED = [{u_: R[(u_, j)] for u_ in UNIT} for j in range(len(jobs))]
    np.savez_compressed(ROOT / "reports/v67_units.npz", **{f"{u_}|{j}": R[(u_, j)] for u_, j in tasks})
    LAM = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
    def costs(q, idx):
        out = []
        for l in LAM:
            qq = l * q / (1 - q) / (1 + l * q / (1 - q)); fl, bu = decide_pv(qq, md[idx], resid)
            out.append(np.where(fl == 1, INSPECT + UNDER * np.maximum(c[idx] - bu, 0) + OVER * np.maximum(bu - c[idx], 0), c[idx]))
        return np.array(out)
    RES = {}; PV = {}                                          # PV: per-vehicle cost at the chosen level, for bootstrap
    for cn, parts in CAND.items():
        def put(method, folds, W):
            # folds: list of (cost matrix [lam, n], idx, q); one level per method and candidate
            C = np.concatenate([f[0] for f in folds], 1); idx = np.concatenate([f[1] for f in folds]); q = np.concatenate([f[2] for f in folds]); w = W[idx]
            tot = (C * w[None, :]).sum(1) / w.sum(); k = int(np.argmin(tot)); RES[(method, cn)] = (tot[k], average_precision_score(y[idx][w > 0], q[w > 0], sample_weight=w[w > 0]))
            per = np.zeros(len(y)); den = np.zeros(len(y)); np.add.at(per, idx, C[k] * w); np.add.at(den, idx, w); PV[(method, cn)] = (per, den)
        cvf = []; insf = []
        for r in range(2):
            o = np.zeros(len(y)); os_ = np.zeros(len(y)); ids = [i for i, (rr, _, _) in enumerate(CV) if rr == r]
            for i in ids:
                both = combine(PRED[i], parts); m_ = len(CV[i][2]); o[CV[i][2]] = both[:m_]; os_[CV[i][2]] = both[m_:]
            for i in ids:
                _, a, b = CV[i]; cal = make("platt").fit(o[a], y[a]); q = cal.predict(o[b]); cvf.append((costs(q, b), b, q))
                qs = cal.predict(os_[b]); insf.append((costs(qs, b), b, qs))
        for method, W in (("plain", np.ones(len(y))), ("wear", W_wear), ("complete", W_comp)): put(method, cvf, W)
        put("instrument", insf, np.ones(len(y)))
        o = np.zeros(len(y))
        for i, (a, b) in enumerate(LCO): o[b] = combine(PRED[len(CV) + i], parts)
        lf = []
        for a, b in LCO:
            q = make("platt").fit(o[a], y[a]).predict(o[b]); lf.append((costs(q, b), b, q))
        put("cluster", lf, W_cl)
        for nm, mask in HOLD.items():
            ids = [j for j, jb in enumerate(jobs) if jb[0] == nm]; oo = np.zeros(len(y))
            for j in ids[:5]: oo[jobs[j][3]] = combine(PRED[j], parts)
            lo, hi = np.where(~mask)[0], jobs[ids[5]][3]; q = make("platt").fit(oo[lo], y[lo]).predict(combine(PRED[ids[5]], parts))
            put(nm, [(costs(q, hi), hi, q)], np.ones(len(y)))
    methods = ["plain", "wear", "complete", "instrument", "cluster", "ho-wear", "ho-usage"]
    rng = np.random.default_rng(0); I = rng.integers(0, len(y), (1000, len(y))); rows = []
    for m in methods:
        pb, db = PV[(m, "base")]
        for cn in CAND:
            p, d = PV[(m, cn)]; boot = (p[I].sum(1) / d[I].sum(1)) / (pb[I].sum(1) / db[I].sum(1)) - 1
            rows.append(dict(method=m, cand=cn, cost=RES[(m, cn)][0], pct=100 * (RES[(m, cn)][0] / RES[(m, "base")][0] - 1), AP=RES[(m, cn)][1], p_better=float((boot < 0).mean())))
    O = pd.DataFrame(rows); O.to_csv(ROOT / "reports/v67_valmethods.csv", index=False)
    for col, title in (("pct", "cost change vs v12 classifier, % (negative = better)"), ("p_better", "bootstrap probability of beating v12"), ("AP", "average precision")):
        print(f"\n{title}"); print(O.pivot(index="cand", columns="method", values=col)[methods].loc[list(CAND)].round(2 if col != "AP" else 3).to_string())
    print("\nbacktest on candidates that were WORSE than v12 on the public board (nos122, ipcw, pu, bag3, drop4): how many does each method also call worse?")
    for m in methods:
        o = O[(O.method == m)].set_index("cand").pct; known = ["nos122", "ipcw", "pu", "bag3", "drop4"]
        ks = ["nos122", "ipcw", "pu", "bag3"]; corr = np.corrcoef([o[k] for k in ks], [PUBLIC[k] for k in ks])[0, 1]
        print(f"  {m:12s} calls worse {int(sum(o[k] > 0 for k in known))}/5; mean predicted change {np.mean([o[k] for k in known]):+.2f}%; correlation with public size {corr:+.2f}; says aft {o['aft']:+.2f}%")
