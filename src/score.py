"""Exact competition cost function.

cost_i = 10 + 0.20*max(c-b, 0) + 0.02*max(b-c, 0)   if flagged
       = c                                          if not flagged
c = repair_cost if the vehicle failed, else 0.
"""
import numpy as np
import pandas as pd

INSPECT, UNDER, OVER = 10.0, 0.20, 0.02


def true_cost(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["failure"].to_numpy() == 1, df["repair_cost"].fillna(0).to_numpy(float), 0.0)


def vehicle_cost(c, flag, budget) -> np.ndarray:
    c = np.asarray(c, float)
    b = np.maximum(np.asarray(budget, float), 0.0)
    flagged = INSPECT + UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0)
    return np.where(np.asarray(flag).astype(bool), flagged, c)


def per_vehicle(c, flag, budget) -> float:
    """Mean cost per vehicle, comparable between train (56k) and test (20k)."""
    return float(vehicle_cost(c, flag, budget).mean())


if __name__ == "__main__":
    # Worked example from the problem statement: total 1,241.
    assert vehicle_cost([400, 0, 1200, 0], [1, 1, 0, 0], [300, 50, 0, 0]).sum() == 1241.0
    print("score.py OK (worked example = 1241)")
