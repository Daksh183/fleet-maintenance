"""Classifier settings used since v2 (single source of truth)."""
from common import SEED

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=1, n_jobs=-1, random_state=SEED)
