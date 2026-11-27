"""v55: confirm v54's two small winners on ten fresh seeds (601-610): prior variance on the GP log-parameters, residual spread /
level. 'level +x' adds x to every log budget (what a repair-cost multiplier does) to see whether 'spread x1.1' is just a level effect."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import v54_gpsettings as G
FIT = {"base": {}, "prior var 0.5": dict(pv=0.5), "prior var 1": dict(pv=1.0), "prior var 2": dict(pv=2.0)}
def run(s):
    out = {}; O = {n: G.oof(kw, s) for n, kw in FIT.items()}
    for n, o in O.items(): out[n] = G.loss(o, s)
    for sc in (1.05, 1.1, 1.2): out[f"spread x{sc}"] = G.loss(O["base"], s, sc=sc)
    for lv in (1.05, 1.1, 1.19): out[f"level x{lv}"] = G.loss(O["base"] , s) if False else level(O["base"], s, lv)
    out["prior var 1 + spread x1.1"] = G.loss(O["prior var 1"], s, sc=1.1)
    out["prior var 1 + level x1.19"] = level(O["prior var 1"], s, 1.19)
    return out
def level(o, s, mult):
    """budgets x mult after the usual 3-band quantiles (loss recomputed with the multiplier inside the cap)"""
    A = G.A; keep = A.np.minimum
    A.np.minimum = lambda a, b: keep(a * mult, b) if np.isscalar(b) and b == 8000 else keep(a, b)
    try: return G.loss(o, s)
    finally: A.np.minimum = keep
if __name__ == "__main__":
    R = Parallel(n_jobs=4)(delayed(run)(s) for s in range(601, 611)); rows = []
    for n in R[0]:
        d = np.array([100 * (r[n].sum() / r["base"].sum() - 1) for r in R]); L = np.mean([r[n] for r in R], 0); L0 = np.mean([r["base"] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, mean=d.mean(), seeds_better=int((d < 0).sum()), boot_sd=boot.std(), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values("mean"); O.to_csv(G.A.ROOT / "reports/v55_confirm.csv", index=False); print(O.round(2).to_string(index=False))
