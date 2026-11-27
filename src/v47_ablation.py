"""v47: ablation of the GP repair-cost model's inputs and form, against v42 (v39 inputs, 3-band spread).
Each variant changes ONE thing. Criterion: honest cross-fitted budget loss over the flagged-test level mix (v42 flags), on the same
383 eligible failures, several fold seeds. All new input groups are chosen a priori (no selection on the repair costs).
Usage: python src/v47_ablation.py [seeds...]"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, START, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy()); c = np.exp(y)
WEAR = ["s126", "s149", "s123"]; SENS = ["s116", "s142", "s105", "s104", "s078"]
HCOLS = [f"s{i:03d}" for a in START for i in range(a, a + (9 if a == 54 else 10))]
OTHER = [s for s in S if s not in HCOLS and s not in WEAR and s not in SENS]
def z(cols, fill=None):
    X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
    return X if fill is None else np.nan_to_num(X, nan=fill)
H = extract(f); HG, EXP = H[:6], H[6]; ZW, ZS = z(WEAR), z(SENS); BASE = HG + [EXP, ZW, ZS]
valid = sum(np.isfinite(x).all(1).astype(int) for x in HG) >= 3           # v42's eligibility, held fixed for every variant
class GP(GaussianSeverity):
    """predict() without the row filter, so dropping a histogram does not change which vehicles are scored"""
    def predict(self, features):
        return np.clip(np.einsum("i,ijk->jk", self.amplitude, self.blocks(features)) @ self.alpha + self.center, *self.bounds)
class Matern(GP):
    """Matern-3/2 kernels in place of the Gaussian ones (rougher functions)"""
    def blocks(self, features):
        out = []
        for x, base, scale in zip(features, self.training, self.scale):
            r = np.sqrt(3 * cdist(x, base, "sqeuclidean") / scale); out.append(np.nan_to_num((1 + r) * np.exp(-r), nan=0.))
        return np.asarray(out)
class Cauchy(GP):
    """rational quadratic (alpha=1): a mixture of Gaussian widths"""
    def blocks(self, features):
        return np.asarray([np.nan_to_num(1 / (1 + cdist(x, b, "sqeuclidean") / (2 * s)), nan=0.) for x, b, s in zip(features, self.training, self.scale)])
def hraw(a):
    cols = [f"s{i:03d}" for i in range(a, a + (9 if a == 54 else 10))]; return z(cols, 0.0)
RAW = [hraw(a) for a in START]
Zo = z(OTHER, 0.0); PC = PCA(10, random_state=0).fit_transform(np.clip(Zo, -5, 5))
MISS = np.log1p(f[S].isna().sum(1).to_numpy(float))[:, None]; MISS = (MISS - MISS.mean()) / MISS.std()
# variants: name -> (features, model class, target transform)
V = {"base (v42)": (BASE, GP, "log")}
for i, a in enumerate(START): V[f"drop hist s{a:03d}"] = ([b for j, b in enumerate(BASE) if j != i], GP, "log")
for i, nm in ((6, "exposure"), (7, "wear"), (8, "sensors")): V[f"drop {nm}"] = ([b for j, b in enumerate(BASE) if j != i], GP, "log")
V["impute wear/sensor NaN=0"] = (HG + [EXP, z(WEAR, 0.0), z(SENS, 0.0)], GP, "log")
V["+ all other sensors"] = (BASE + [Zo], GP, "log")
V["+ PCA10 of other sensors"] = (BASE + [PC], GP, "log")
V["+ missing count"] = (BASE + [MISS], GP, "log")
V["+ raw bin levels (6 groups)"] = (BASE + RAW, GP, "log")
V["+ raw bin levels (1 group)"] = (BASE + [np.column_stack(RAW)], GP, "log")
V["raw levels replace shares"] = (RAW + [EXP, ZW, ZS], GP, "log")
V["sensors one kernel each"] = (HG + [EXP, ZW] + [ZS[:, [j]] for j in range(ZS.shape[1])], GP, "log")
V["wear one kernel each"] = (HG + [EXP] + [ZW[:, [j]] for j in range(3)] + [ZS], GP, "log")
V["+ wear x sensors joint"] = (BASE + [np.column_stack([ZW, ZS])], GP, "log")
V["Matern 3/2"] = (BASE, Matern, "log")
V["rational quadratic"] = (BASE, Cauchy, "log")
V["target cost^0.25"] = (BASE, GP, "q")
V["target cost^0.5"] = (BASE, GP, "h")
V["blend 0.8 GP + 0.2 tree"] = (BASE, GP, "blend2")
V["blend 0.6 GP + 0.4 tree"] = (BASE, GP, "blend4")
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v42_gp_bands.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1); k = lev > 0; p, lev = p[k], lev[k]
FWD = {"log": (np.log, np.exp), "q": (lambda v: v ** 0.25, lambda t: np.maximum(t, 0) ** 4), "h": (lambda v: v ** 0.5, lambda t: np.maximum(t, 0) ** 2)}
def run(name, seed):
    F, cls, tt = V[name]; fwd, inv = FWD.get(tt, FWD["log"]); t = fwd(c); o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y):
        o[b] = cls().fit(select(F, a), t[a]).predict(select(F, b))
        if tt.startswith("blend"):
            w = int(tt[-1]) / 10
            tree = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.02, subsample=0.8, colsample_bytree=0.5, n_jobs=1,
                                    random_state=42).fit(f[S].iloc[a], y[a])
            o[b] = (1 - w) * o[b] + w * tree.predict(f[S].iloc[b])
    mu, cc, tt_ = o[valid], c[valid], t[valid]; r = tt_ - mu; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed + 1000).split(mu)): fold[b] = j
    band = np.searchsorted(np.quantile(mu, [1 / 3, 2 / 3]), mu); Q = np.zeros((len(lev), len(mu)))
    for j in range(5):
        for kk in range(3):
            src = (fold != j) & (band == kk); idx = (fold == j) & (band == kk)
            if idx.any(): Q[:, idx] = np.quantile(r[src], lev)[:, None]
    B = np.minimum(inv(mu[None, :] + Q), 8000)
    return name, seed, float((p * (UNDER * np.maximum(cc[None, :] - B, 0) + OVER * np.maximum(B - cc[None, :], 0)).mean(1)).sum())
if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 7, 123, 11, 99]
    print(f"{valid.sum()} eligible failures, {len(V)} variants, seeds {seeds}", flush=True)
    R = Parallel(n_jobs=6, verbose=0)(delayed(run)(n, s) for s in seeds for n in V)
    T = pd.DataFrame(R, columns=["variant", "seed", "loss"]).pivot(index="variant", columns="seed", values="loss")
    D = 100 * (T / T.loc["base (v42)"] - 1); D["mean"] = D[seeds].mean(1); D["better"] = (D[seeds] < 0).sum(1)
    D = D.sort_values("mean"); D.to_csv(ROOT / "reports/v47_ablation.csv"); print(D.round(2).to_string())
