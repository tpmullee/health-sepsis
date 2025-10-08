from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

def platt_sigmoid(model, X, y, cv=3):
    """Wrap model with Platt scaling (sigmoid) using out-of-fold CalibratedClassifierCV."""
    cvobj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return CalibratedClassifierCV(model, method="sigmoid", cv=cvobj)
