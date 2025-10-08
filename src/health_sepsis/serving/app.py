from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import os, joblib, numpy as np, pandas as pd
from ._state import get_model, get_columns

app = FastAPI(title="health-sepsis-model")

class Patient(BaseModel):
    age: float; sex: str; unit: str
    hr: float; map: float; wbc: float; lactate: float; creatinine: float; temp: float

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    cols_path  = os.getenv("COLUMNS_PATH", "artifacts/columns.json")
    if not (os.path.exists(model_path) and os.path.exists(cols_path)):
        return JSONResponse(status_code=500, content={"status":"error","detail":"artifacts missing"})
    try:
        _ = get_columns()
        return {"status":"ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})

@app.post("/score")
def score(p: Patient):
    model = get_model()  # lazy-load on first call
    cols  = get_columns()
    df = pd.DataFrame([{**p.dict()}])[cols]
    prob = float(model.predict_proba(df)[:,1][0])
    return {"prob_sepsis": prob}
