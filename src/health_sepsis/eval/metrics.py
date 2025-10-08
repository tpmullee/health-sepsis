from __future__ import annotations
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def classification_report_dict(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(np.mean(y_true)),
        "precision_at_thr": float(np.sum((y_pred==1)&(y_true==1))/max(np.sum(y_pred==1),1)),
        "recall_at_thr": float(np.sum((y_pred==1)&(y_true==1))/max(np.sum(y_true==1),1)),
        "specificity_at_thr": float(np.sum((y_pred==0)&(y_true==0))/max(np.sum(y_true==0),1)),
    }

def reliability(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {"calib_prob_true": prob_true.tolist(), "calib_prob_pred": prob_pred.tolist()}
