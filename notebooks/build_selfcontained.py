"""Turn final_report.ipynb into a self-contained Final_Code.ipynb for the course ZIP.

The saved artifacts that Models 2 and 3 load from ../reports and ../submissions are embedded in the notebook as one
compressed base64 blob, so the notebook needs only the competition's train.csv and test.csv. The check against the
uploaded files compares a fingerprint (ids, flags, budgets to the cent) instead of reading ../submissions.

Usage: python build_selfcontained.py OUT.ipynb   (run from notebooks/, after final_report.ipynb is built)
"""
import base64, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP, SUB = ROOT / "reports", ROOT / "submissions"
out_path = Path(sys.argv[1])

# ---- the artifacts, packed into one compressed npz -------------------------------------------------------------
NPY = ["v18_qm", "v18_resid", "v19_q", "v21_H", "v22_q", "v24_q", "v24_pu_x2_q", "v18_ens10_q", "v18_ens_q", "p_oof_v2"]
art = {k: np.load(REP / f"{k}.npy") for k in NPY}
b33 = np.load(REP / "v33_baseline.npz")
art.update({"v33_ids": b33["ids"], "v33_q": b33["q"], "v33_tree": b33["tree"], "v33_residual": b33["residual"]})
v33 = pd.read_csv(SUB / "submission_v33_gp_budget.csv")
art["v33_sub_ids"], art["v33_sub_budget"] = v33.id.to_numpy().astype("U6"), v33.budget.to_numpy(float)
art["v51_post_csv"] = np.frombuffer((REP / "v51_post.csv").read_bytes(), dtype=np.uint8)


def fingerprint(df):
    h = hashlib.sha256()
    h.update("\n".join(df.id.astype(str)).encode()); h.update(df.flag.to_numpy(np.int64).tobytes())
    h.update(np.round(df.budget.to_numpy(float) * 100).astype(np.int64).tobytes())
    return h.hexdigest()


FP = {n: fingerprint(pd.read_csv(SUB / f"submission_{n}.csv")) for n in ("v3", "v34_budget", "v51_gp_reduced")}
buf = io.BytesIO(); np.savez_compressed(buf, **art)
blob = base64.b64encode(buf.getvalue()).decode()
blob_lines = "\n".join(blob[i:i + 120] for i in range(0, len(blob), 120))

ART_CELL = f'''# Saved artifacts used by Models 2 and 3, embedded so that this notebook needs only train.csv and test.csv.
# Contents (each produced by the script named in the repository Daksh183/fleet-maintenance):
#   v33_*        v12 test probabilities q, tree log-cost and its residuals (src/v33_build.py) -> Model 2 flags
#   v33_sub_*    budgets of submission_v33_gp_budget.csv, kept by v34 for unflagged vehicles (ignored by the scorer)
#   p_oof_v2     out-of-fold classifier probabilities on the development split (Platt fit)
#   v18_*, v19_q, v21_H, v22_q, v24_*   test-set rankings used as hypotheses by the harness (src/v18..v24 builders)
#   v51_post_csv the posterior over test-depot hypotheses fitted to 26 public scores (src/v51_harness.py)
import base64, io, zlib
_BLOB = """
{blob_lines}
"""
ART = dict(np.load(io.BytesIO(base64.b64decode("".join(_BLOB.split())))))
print(f"{{len(ART)}} embedded artifacts loaded")
'''

nb = json.loads((Path(__file__).resolve().parent / "final_report.ipynb").read_text())
cells = nb["cells"]


def src(c): return "".join(c["source"])


def setsrc(c, s): c["source"] = s.splitlines(keepends=True)


def sub(c, old, new):
    s = src(c); assert old in s, old[:80]; setsrc(c, s.replace(old, new))


for c in cells:
    if c["cell_type"] != "code": continue
    s = src(c)
    if 'DATA, OUT = Path("../data")' in s:
        sub(c, 'DATA, OUT = Path("../data"), Path("outputs"); OUT.mkdir(exist_ok=True)',
            '# train.csv and test.csv from the Kaggle competition: looked for next to this notebook, in data/, ../data/,\n'
            '# or the Kaggle input folder; set FLEET_DATA to point elsewhere.\n'
            'import os\n'
            '_CAND = [os.environ.get("FLEET_DATA", ""), ".", "data", "../data", "/kaggle/input/mlprojectfleet2"]\n'
            'DATA = next((Path(d) for d in _CAND if d and (Path(d) / "train.csv").exists() and (Path(d) / "test.csv").exists()), None)\n'
            'assert DATA is not None, "put train.csv and test.csv from the competition next to this notebook"\n'
            'OUT = Path("outputs"); OUT.mkdir(exist_ok=True)')
    if 'REP = Path("../reports")\n' in s and "quantile_budget" in s:
        sub(c, 'REP = Path("../reports")\n', '')
        sub(c, 'base = np.load(REP / "v33_baseline.npz"); assert np.array_equal(base["ids"], te.id)',
            'base = {k: ART[f"v33_{k}"] for k in ("ids", "q", "tree", "residual")}; assert np.array_equal(base["ids"], te.id)')
        sub(c, 'p_oof = np.load(REP / "p_oof_v2.npy")', 'p_oof = ART["p_oof_v2"]')
        sub(c, 'v33file = pd.read_csv(Path("../submissions/submission_v33_gp_budget.csv")).set_index("id").loc[te.id]',
            'v33file = pd.DataFrame({"budget": ART["v33_sub_budget"]}, index=ART["v33_sub_ids"]).loc[te.id]')
    if 'REP, SUB = Path("../reports"), Path("../submissions")' in s:
        sub(c, 'REP, SUB = Path("../reports"), Path("../submissions")\n', '')
        s2 = re.sub(r'np\.load\(REP / "([A-Za-z0-9_]+)\.npy"\)', r'ART["\1"]', src(c)); setsrc(c, s2)
        assert "REP" not in src(c), src(c)
    if 'pd.read_csv(REP / path)' in s:
        sub(c, 'P = pd.read_csv(REP / path)', 'P = pd.read_csv(io.BytesIO(ART[path.replace(".csv", "_csv")].tobytes()))')
    if 'Path("../submissions") / f"submission_{name}.csv"' in s:
        sub(c, '''for name in ("v3", "v34_budget", "v51_gp_reduced"):
    a, b = pd.read_csv(OUT / f"submission_{name}.csv"), pd.read_csv(Path("../submissions") / f"submission_{name}.csv")
    same = a.id.equals(b.id) and (a.flag == b.flag).all() and np.allclose(a.budget, b.budget, atol=0.015)
    print(f"{'MATCH ' if same else 'DIFFER'} {name}: flags differ {int((a.flag != b.flag).sum())}, max budget difference {np.abs(a.budget - b.budget).max():.2f}")''',
            f'''import hashlib
UPLOADED = {json.dumps(FP, indent=4)}   # fingerprints of the files uploaded to Kaggle (ids, flags, budgets to the cent)
def fingerprint(df):
    h = hashlib.sha256()
    h.update("\\n".join(df.id.astype(str)).encode()); h.update(df.flag.to_numpy(np.int64).tobytes())
    h.update(np.round(df.budget.to_numpy(float) * 100).astype(np.int64).tobytes())
    return h.hexdigest()
for name in ("v3", "v34_budget", "v51_gp_reduced"):
    same = fingerprint(pd.read_csv(OUT / f"submission_{{name}}.csv")) == UPLOADED[name]
    print(f"{{'MATCH ' if same else 'DIFFER'}} {{name}}: regenerated file {{'identical to' if same else 'differs from'}} the uploaded submission")

# the final model's predictions, in the file name the course asks for
import shutil; shutil.copy(OUT / "submission_v51_gp_reduced.csv", "Inference_Test.csv")
print("wrote Inference_Test.csv (Model 3, v51)")''')
    assert "../" not in src(c) or "_CAND" in src(c), src(c)[:200]

# embedded-artifact cell right after the setup/function cells (before the data-analysis section)
idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "def vehicle_cost" in src(c))
cells.insert(idx + 1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ART_CELL.splitlines(keepends=True)})

# markdown: describe the self-contained layout
for c in cells:
    if c["cell_type"] != "markdown": continue
    s = src(c)
    s = s.replace("""**To run:** put `train.csv` and `test.csv` from the competition in `../data/` and run from the `notebooks/` folder. Models 2 and 3
load saved artifacts from `../reports/` (probabilities, out-of-fold predictions, public-score posteriors), as the course asked on 26 Nov
("save your trained artifacts"); each artifact is listed with the script in `../src/` that produced it.""",
                  """**To run:** put `train.csv` and `test.csv` from the competition next to this notebook (or in `data/`) and run all cells
(under a minute on a laptop). Models 2 and 3 use saved artifacts (probabilities, out-of-fold predictions, the public-score posterior),
as the course asked on 26 Nov ("save your trained artifacts"); they are **embedded in the notebook** (the cell after the helper
functions), so nothing else is needed. The last code cell of Section 8 also writes `Inference_Test.csv` (Model 3). File names such as
`src/...` and `reports/...` below refer to the GitHub repository, where the scripts that produced each artifact live.""")
    s = s.replace("in\n`outputs/` and checks each one against the uploaded file.", "in\n`outputs/` and checks each one against the uploaded file.")
    s = s.replace("**Artifacts:** `../reports/v33_baseline.npz`", "**Artifacts** (embedded): `v33_baseline.npz`")
    s = s.replace("`../reports/p_oof_v2.npy`", "`p_oof_v2.npy`").replace("`../submissions/submission_v33_gp_budget.csv`", "`submission_v33_gp_budget.csv`")
    s = s.replace("**Artifacts** (`../reports/`):", "**Artifacts** (embedded, from the repository's `reports/`):")
    s = s.replace("`../reports/findings.md`", "`reports/findings.md` in the repository")
    setsrc(c, s)
    assert "../" not in s, s[:300]

for c in cells:
    if c["cell_type"] == "code": c["outputs"], c["execution_count"] = [], None
out_path.write_text(json.dumps(nb, indent=1))
print(f"wrote {out_path}: {len(cells)} cells, artifact blob {len(blob) / 1e6:.2f} MB")
