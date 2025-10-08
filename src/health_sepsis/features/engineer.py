from __future__ import annotations
import pandas as pd
from typing import Tuple, List

CATEGORICAL = ["sex","unit"]
NUMERIC     = ["age","hr","map","wbc","lactate","creatinine","temp"]

def basic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = df.copy()
    # Simple missing indicators (if any) and caps already applied upstream.
    y = df["y_sepsis"].astype(int)
    X = df[CATEGORICAL + NUMERIC]
    return X, y, NUMERIC, CATEGORICAL
