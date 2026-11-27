"""v48: confirm v47's small winners alone and combined on TEN FRESH fold seeds (not used to pick them)."""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
import v47_ablation as A
H, EXP, ZW, ZS = A.HG, A.EXP, A.ZW, A.ZS
def F(drop=(), exp=True): return [h for a, h in zip(A.START, H) if a not in drop] + ([EXP] if exp else []) + [ZW, ZS]
V = {"base (v42)": (F(), A.GP, "log"),
     "drop exposure": (F(exp=False), A.GP, "log"),
     "drop s129": (F((129,)), A.GP, "log"),
     "drop s023": (F((23,)), A.GP, "log"),
     "drop exp+s129": (F((129,), False), A.GP, "log"),
     "drop exp+s023": (F((23,), False), A.GP, "log"),
     "drop s129+s023": (F((129, 23)), A.GP, "log"),
     "drop exp+s129+s023": (F((129, 23), False), A.GP, "log"),
     "blend 0.8": (F(), A.GP, "blend2"),
     "drop exp + blend 0.8": (F(exp=False), A.GP, "blend2"),
     "drop exp+s129+s023 + blend 0.8": (F((129, 23), False), A.GP, "blend2")}
A.V.clear(); A.V.update(V)
def run(n, s, v):
    A.V[n] = v; return A.run(n, s)
if __name__ == "__main__":
    seeds = list(range(201, 211))
    R = Parallel(n_jobs=6)(delayed(run)(n, s, v) for s in seeds for n, v in V.items())
    T = pd.DataFrame(R, columns=["variant", "seed", "loss"]).pivot(index="variant", columns="seed", values="loss")
    D = 100 * (T / T.loc["base (v42)"] - 1)
    O = pd.DataFrame({"mean_%": D.mean(1), "se": D.std(1) / np.sqrt(len(seeds)), "better_of_10": (D < 0).sum(1), "worst_%": D.max(1)}).sort_values("mean_%")
    D.to_csv(A.ROOT / "reports/v48_combo.csv"); print(O.round(2).to_string())
