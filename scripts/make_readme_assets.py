import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from health_sepsis.data.loaders import load_synthetic_cohort
from health_sepsis.features.engineer import basic_features
import joblib

ART = Path("artifacts"); DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)


# 1) Load model + fresh test data
df = load_synthetic_cohort(seed=17)
X, y, *_ = basic_features(df)
model = joblib.load(ART/"model.joblib")
p = model.predict_proba(X)[:,1]

# 2) Metrics
fpr, tpr, _ = roc_curve(y, p)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y, p)
ap = average_precision_score(y, p)
brier = brier_score_loss(y, p)
prev = float(np.mean(y))

# 3) Save metrics json
metrics = {"auc_roc": float(roc_auc), "auc_pr": float(ap), "brier": float(brier), "prevalence": prev}
DOCS.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2))

# 4) ROC
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(DOCS/"roc.png", dpi=160); plt.close()

# 5) PRC
plt.figure()
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
plt.legend(loc="upper right")
plt.tight_layout(); plt.savefig(DOCS/"pr.png", dpi=160); plt.close()

# 6) Calibration (quantile bins)
bins = pd.qcut(p, q=10, duplicates="drop")
dfc = pd.DataFrame({"y":y, "p":p, "bin":bins}).groupby("bin", observed=False).agg(obs=("y","mean"), pred=("p","mean"))
plt.figure()
plt.plot([0,1],[0,1],"--", linewidth=1)
plt.scatter(dfc["pred"], dfc["obs"])
plt.xlabel("Mean Predicted"); plt.ylabel("Observed")
plt.title("Calibration (quantile bins)")
plt.tight_layout(); plt.savefig(DOCS/"calibration.png", dpi=160); plt.close()

print("Wrote:", [str(p) for p in (DOCS/"roc.png", DOCS/"pr.png", DOCS/"calibration.png", DOCS/"metrics.json")])
