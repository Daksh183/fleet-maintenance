"""Check that every notebook regenerates exactly the submission file that was (or will be) uploaded."""
import numpy as np, pandas as pd
from pathlib import Path
rows = [("v2", 1), ("v3", 2), ("v3b", 3), ("v4_platt", 4), ("v5_raw", 5), ("v5_platt", 6),
        ("v4_iso", None), ("v5_iso", None), ("v7_k4", None), ("v7_em", None), ("v8_k4", None),
        ("v9_k1", None), ("v9_k4", None), ("v9_em", None), ("v10_bayes", None), ("v10_k2", None), ("v12_robust", 12), ("v13_bag", 13), ("v14_map", 14), ("v15_map_v12", 15)]
ok = True
for name, n in rows:
    a, b = Path(f"outputs/submission_{name}.csv"), Path(f"../submissions/submission_{name}.csv")
    if not a.exists(): print(f"MISSING  {name}"); ok = False; continue
    x, y = pd.read_csv(a), pd.read_csv(b)
    same = x.id.equals(y.id) and (x.flag == y.flag).all() and np.allclose(x.budget, y.budget, atol=0.015)
    ok &= same
    print(f"{'MATCH' if same else 'DIFFER'}  {name:10s} flags differ {int((x.flag != y.flag).sum()):4d}  max |budget diff| {np.abs(x.budget - y.budget).max():.2f}  {'(uploaded as #%d)' % n if n else ''}")
print("ALL MATCH" if ok else "SOME DIFFER")
