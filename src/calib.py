"""Probability calibrators (Platt / isotonic), shared by v4+."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
        return self

    def predict(self, p):
        return self.m.predict_proba(logit(p)[:, None])[:, 1]


class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p):
        return self.m.predict(p)


def make(kind):
    return {"platt": Platt, "iso": Iso}[kind]()
