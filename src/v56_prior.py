"""v56: how far does the prior effect go, and which parameter carries it? Seeds 701-708. 'amp'/'noise': tight prior (variance 0.5) on only
the kernel amplitudes or only the noise. Level multipliers on top of base and of prior var 0.5."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import v54_gpsettings as G, v55_confirm as C
FIT = {"base": {}, "prior var 1": dict(pv=1.0), "prior var 0.5": dict(pv=0.5), "prior var 0.25": dict(pv=0.25), "prior var 0.1": dict(pv=0.1),
       "prior var 0.05": dict(pv=0.05), "tight amp only": dict(pv=(0.5, 4.0)), "tight noise only": dict(pv=(4.0, 0.5))}
def run(s):
    out = {}
    for n, kw in FIT.items():
        o = G.oof(kw, s); out[n] = G.loss(o, s)
        if n in ("base", "prior var 0.5"):
            for lv in (1.1, 1.19, 1.3, 1.45, 1.6): out[f"{n}, level x{lv}"] = C.level(o, s, lv)
    return out
if __name__ == "__main__":
    R = Parallel(n_jobs=4)(delayed(run)(s) for s in range(701, 709)); rows = []
    for n in R[0]:
        d = np.array([100 * (r[n].sum() / r["base"].sum() - 1) for r in R]); L = np.mean([r[n] for r in R], 0); L0 = np.mean([r["base"] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, mean=d.mean(), seeds_better=int((d < 0).sum()), boot_sd=boot.std(), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values("mean"); O.to_csv(G.A.ROOT / "reports/v56_prior.csv", index=False); print(O.round(2).to_string(index=False))
