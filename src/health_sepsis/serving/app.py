from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
from ._state import get_model, get_columns

app = FastAPI(title="health-sepsis-model")

class Patient(BaseModel):
    age: float; sex: str; unit: str
    hr: float; map: float; wbc: float; lactate: float; creatinine: float; temp: float

@app.on_event("startup")
def startup():
    get_model()  # warm

@app.post("/score")
def score(p: Patient):
    model = get_model()
    cols  = get_columns()
    df = pd.DataFrame([{**p.dict()}])[cols]
    prob = float(model.predict_proba(df)[:,1][0])
    return {"prob_sepsis": prob}
