"""Cost accounting and paired influence diagnostics for the fleet experiment."""
import numpy as np

from score import INSPECT, OVER, UNDER, vehicle_cost


def accounting(actual, flag, budget):
    parts = {
        "inspection": INSPECT * flag,
        "shortage": UNDER * flag * np.maximum(actual - budget, 0),
        "excess": OVER * flag * np.maximum(budget - actual, 0),
        "missed": (1 - flag) * actual,
    }
    realized = vehicle_cost(actual, flag, budget)
    np.testing.assert_allclose(sum(parts.values()), realized, rtol=1e-12, atol=1e-10)
    return realized, parts


class Population:
    def __init__(self, wear, mix, outcomes, odds):
        self.groups = wear
        self.mix = mix
        self.w = np.where(outcomes == 1, odds, 1.0)
        self.sums = np.bincount(wear, weights=self.w, minlength=3)
        assert np.all(self.sums > 1) and np.isclose(mix.sum(), 1)
        self.normal = self.w / self.w.sum()
        self.standard = mix[wear] * self.w / self.sums[wear]
        assert np.isclose(self.standard.sum(), 1.0)

    def influence(self, d, ids):
        # Exact change in a fold mean when one row is removed, renormalizing its group.
        group_means = np.bincount(self.groups, weights=self.w * d, minlength=3) / self.sums
        mean = self.normal @ d
        change = self.w * (mean - d) / (self.w.sum() - self.w)
        wear_change = self.standard * self.sums[self.groups] * (
            group_means[self.groups] - d) / (self.sums[self.groups] - self.w)
        i = np.abs(change).argmax()
        j = np.abs(wear_change).argmax()
        return dict(delta=mean, wear_delta=self.standard @ d, realized_delta=d.sum(),
                    influence=change[i], influential_id=ids[i],
                    wear_influence=wear_change[j], wear_influential_id=ids[j],
                    worst_removal=change.max(), wear_worst_removal=wear_change.max())
