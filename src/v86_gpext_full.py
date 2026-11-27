"""v86: v85's promising variants re-scored with the FULL expected cost per flagged vehicle: p * (shortfall + over on failure) +
(1 - p) * 0.02 * budget (unused budget when it does not fail), which v54/v85's loss left out. Same seeds and folds."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import v47_ablation as A, v54_gpsettings as V, v85_gpext as X
def full(fn):
    def g(*a, **k):
        # recompute budgets the same way, then add the healthy term: call fn on a fake cost vector of zeros to get the over-budget part
        L = fn(*a, **k); cc = A.c.copy(); A.c[:] = 0.0; Lh = fn(*a, **k); A.c[:] = cc
        # with c = 0 the loss is p * 0.02 * B; the healthy term is (1-p) * 0.02 * B, i.e. Lh scaled per level: (1-p)/p
        return L, Lh
    return g
def split(L, Lh):
    return L
def run(s):
    out = {}
    base = V.oof({}, s); o_b, sd_b = X.oof({}, s); o_l, sd_l = X.oof(dict(lin=True), s)
    def tot(fn, *a):
        A.p_save = A.p; L = fn(*a)
        cc = A.c.copy(); A.c[:] = 0.0; pp = A.p.copy(); A.p[:] = 1.0; Bsum = fn(*a) / A.OVER; A.c[:] = cc; A.p[:] = pp      # sum over levels of budgets
        # Bsum = sum_levels B (per vehicle) with p = 1, OVER cancelled; healthy term needs per-level (1-p): recompute explicitly below
        return L
    def full_loss(budgets):
        cc = A.c[A.valid]
        fail = (A.p[:, None] * (A.UNDER * np.maximum(cc[None, :] - budgets, 0) + A.OVER * np.maximum(budgets - cc[None, :], 0))).sum(0)
        healthy = ((1 - A.p)[:, None] * A.OVER * budgets).sum(0)
        return fail, healthy
    # capture budgets by monkeypatching np.minimum inside the loss functions is fragile; instead recompute budgets here
    from sklearn.model_selection import KFold
    def budgets(o, sd, scheme):
        mu, t = o[A.valid], A.y[A.valid]; sg = sd[A.valid] if sd is not None else np.ones(len(mu)); fold = np.zeros(len(mu), int)
        for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s + 1000).split(mu)): fold[b] = j
        r = (t - mu) / (sg if scheme != "bands" else 1.0); nb = 1 if scheme == "pvar" else 3
        band = np.searchsorted(np.quantile(mu, np.linspace(0, 1, nb + 1)[1:-1]), mu); Q = np.zeros((len(A.lev), len(mu)))
        for j in range(5):
            for k in range(nb):
                src = (fold != j) & (band == k); idx = (fold == j) & (band == k)
                if idx.any(): Q[:, idx] = np.quantile(r[src], A.lev)[:, None] * (sg[idx][None, :] if scheme != "bands" else 1.0)
        return np.minimum(np.exp(mu[None, :] + Q), 8000)
    for nm, (o, sd, sc) in {"base": (base, None, "bands"), "same | pvar+bands": (o_b, sd_b, "pvar+bands"), "same | pvar": (o_b, sd_b, "pvar"),
                            "lin": (o_l, None, "bands"), "lin | pvar+bands": (o_l, sd_l, "pvar+bands"), "lin | pvar": (o_l, sd_l, "pvar")}.items():
        out[nm] = full_loss(budgets(o, sd, sc))
    return out
if __name__ == "__main__":
    seeds = list(range(501, 509)); R = Parallel(n_jobs=4)(delayed(run)(s) for s in seeds); rows = []
    chk = np.mean([100 * (r["base"][0].sum() / V.loss(V.oof({}, 501), 501).sum() - 1) for r in R[:1]])
    print(f"check: failure part of base reproduces v54's loss within {chk:+.3f}%")
    for n in R[0]:
        fail = np.mean([r[n][0].sum() for r in R]); hl = np.mean([r[n][1].sum() for r in R])
        tot = np.array([(r[n][0] + r[n][1]).sum() / (r["base"][0] + r["base"][1]).sum() - 1 for r in R]) * 100
        L = np.mean([r[n][0] + r[n][1] for r in R], 0); L0 = np.mean([r["base"][0] + r["base"][1] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, failure_part=fail, healthy_part=hl, total_change_pct=tot.mean(), seeds_better=int((tot < 0).sum()), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values("total_change_pct"); O.to_csv(A.ROOT / "reports/v86_gpext_full.csv", index=False); print(O.round(2).to_string(index=False))
