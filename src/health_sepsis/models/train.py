from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

def make_models() -> Dict[str, object]:
    # Reasonable defaults; tune later if needed
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss"
    )
    return {"logreg": logreg, "xgb": xgb}

def build_pipelines(preproc: ColumnTransformer, models: Dict[str, object]) -> Dict[str, Pipeline]:
    return {name: Pipeline([("prep", preproc), ("clf", model)]) for name, model in models.items()}
