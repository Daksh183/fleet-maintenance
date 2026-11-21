"""v0: trivial baselines. Floor/ceiling for every later version."""
import numpy as np

from common import folds, load, log_result, split_holdout
from score import per_vehicle, true_cost

tr, te, S = load()
dev, ho = split_holdout(tr)
c_dev, c_ho = true_cost(dev), true_cost(ho)
F = folds(dev["failure"])
med = float(np.median(c_dev[c_dev > 0]))

policies = {
    "flag nobody": lambda c: (np.zeros(len(c)), np.zeros(len(c))),
    "flag all, budget 0": lambda c: (np.ones(len(c)), np.zeros(len(c))),
    f"flag all, budget median ({med:.0f})": lambda c: (np.ones(len(c)), np.full(len(c), med)),
    "oracle (perfect foresight)": lambda c: ((c > 0).astype(int), c),
}
for name, pol in policies.items():
    vals = [per_vehicle(c_dev[va], *pol(c_dev[va])) for _, va in F]
    log_result("v0", name, np.mean(vals), np.std(vals), holdout=per_vehicle(c_ho, *pol(c_ho)))
