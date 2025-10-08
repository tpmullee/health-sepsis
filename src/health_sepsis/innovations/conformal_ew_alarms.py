from __future__ import annotations
import numpy as np, pandas as pd
from typing import Optional, Dict

def mondrian_thresholds(
    df: pd.DataFrame,
    prob_col: str = "p_hat",
    label_col: str = "y",
    group_col: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute group-wise (Mondrian) thresholds that control patient-level false alert rate.
    For negatives, take the max probability across time; threshold = quantile_{1-alpha} of that max.
    If no group_col, returns a single threshold with key '*'.
    """
    if group_col is None:
        neg_max = df[df[label_col]==0].groupby("patient_id")[prob_col].max()
        thr = float(np.quantile(neg_max, 1 - alpha)) if len(neg_max) else 1.0
        return {"*": thr}

    out: Dict[str, float] = {}
    for g, gdf in df.groupby(group_col, dropna=False):
        neg_max = gdf[gdf[label_col]==0].groupby("patient_id")[prob_col].max()
        thr = float(np.quantile(neg_max, 1 - alpha)) if len(neg_max) else 1.0
        out[str(g)] = thr
    return out

def apply_alerts(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    prob_col: str = "p_hat",
    label_col: str = "y",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Mark first alert per patient when probability crosses its group's threshold.
    """
    df = df.sort_values(["patient_id","time"])
    def row_thr(row):
        key = str(row[group_col]) if group_col else "*"
        return thresholds.get(key, thresholds.get("*", 1.0))
    df["thr"] = df.apply(row_thr, axis=1)
    df["alert"] = (df[prob_col] >= df["thr"]).astype(int)
    # keep only first alert per patient
    first = df[df["alert"]==1].groupby("patient_id")["time"].min().rename("first_alert_time")
    return df.merge(first, on="patient_id", how="left")
