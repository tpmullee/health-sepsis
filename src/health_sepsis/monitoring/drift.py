from __future__ import annotations
import numpy as np, pandas as pd

def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index across quantile buckets."""
    cuts = np.quantile(expected.dropna(), np.linspace(0,1,buckets+1))
    e = pd.cut(expected, bins=np.unique(cuts), include_lowest=True).value_counts(normalize=True).sort_index()
    a = pd.cut(actual,   bins=np.unique(cuts), include_lowest=True).value_counts(normalize=True).sort_index()
    a = a.reindex(e.index).fillna(0.0001); e = e.reindex(e.index).fillna(0.0001)
    return float(((a - e) * np.log(a / e)).sum())
