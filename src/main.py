from __future__ import annotations
import json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd, typer
from health_sepsis.data.loaders import load_synthetic_cohort
from health_sepsis.features.engineer import basic_features, NUMERIC, CATEGORICAL
from health_sepsis.models.train import build_preprocessor, make_models, build_pipelines
from health_sepsis.eval.metrics import classification_report_dict, reliability
from health_sepsis.innovations.triage_optimizer import optimize_triage, summarize
app = typer.Typer(add_completion=False)

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

@app.command()
def demo_train(seed: int = 7):
    """Train baseline models on synthetic cohort; save best pipeline."""
    df = load_synthetic_cohort(seed=seed)
    X, y, num, cat = basic_features(df)
    # split
    msk = np.random.RandomState(seed).rand(len(df)) < 0.8
    X_train, y_train = X[msk], y[msk]
    X_test,  y_test  = X[~msk], y[~msk]

    pre = build_preprocessor(num, cat)
    pipes = build_pipelines(pre, make_models())

    scores = {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:,1]
        scores[name] = classification_report_dict(y_test, p)
        joblib.dump(pipe, ART/f"model_{name}.joblib")
    best = max(scores.items(), key=lambda kv: kv[1]["auc_pr"])[0]
    print("Model leaderboard:", json.dumps(scores, indent=2))
    # Save winner + columns
    Path(ART).mkdir(exist_ok=True)
    joblib.dump(pipes[best], ART/"model.joblib")
    with open(ART/"columns.json","w") as f: json.dump(list(X.columns), f)
    # Write quick reliability
    with open(ART/"reliability.json","w") as f: json.dump(reliability(y_test, p), f)
    print(f"Saved {best} to artifacts/model.joblib")

@app.command()
def optimize(capacity: int = 100):
    """Run capacity-aware triage optimization on a held-out sample."""
    df = load_synthetic_cohort(seed=9)  # new draw
    X, y, *_ = basic_features(df)
    model = joblib.load(ART/"model.joblib")
    p = model.predict_proba(X)[:,1]
    mask, thr = optimize_triage(p, capacity=capacity, benefit_tp=1.0, cost_fp=0.25)
    print({"threshold": thr})
    print(summarize(mask, y.values, p).to_dict())

@app.command()
def serve(host: str="127.0.0.1", port: int=8000):
    """Start FastAPI server (uvicorn)."""
    os.execvp("uvicorn", ["uvicorn", "health_sepsis.serving.app:app", "--host", host, "--port", str(port)])

if __name__ == "__main__":
    app()
