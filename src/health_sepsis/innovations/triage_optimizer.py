from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def optimize_triage(
    probs: np.ndarray,
    capacity: int,
    benefit_tp: float = 1.0,
    cost_fp: float = 0.25,
    cost_fn: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Choose up to K cases to alert to maximize expected utility.
    Utility per alert = p*benefit_tp - (1-p)*cost_fp.
    If capacity < count of positive-utility cases, take top-K by utility.
    Returns a boolean mask of chosen alerts and the implied probability threshold.
    """
    utility = probs*benefit_tp - (1-probs)*cost_fp
    idx_sorted = np.argsort(-utility)
    chosen_idx = idx_sorted[utility[idx_sorted] > 0]
    if capacity < len(chosen_idx):
        chosen_idx = chosen_idx[:capacity]
    mask = np.zeros_like(probs, dtype=bool)
    mask[chosen_idx] = True
    thr = float(np.min(probs[chosen_idx])) if np.any(mask) else 1.0
    return mask, thr

def summarize(mask: np.ndarray, y_true: np.ndarray, probs: np.ndarray) -> pd.Series:
    tp = int(((mask) & (y_true==1)).sum())
    fp = int(((mask) & (y_true==0)).sum())
    prec = tp/max(tp+fp, 1)
    cov  = int(mask.sum())/len(mask)
    return pd.Series({"selected": int(mask.sum()), "precision": prec, "coverage": cov, "min_p": probs[mask].min() if mask.any() else 1.0})
