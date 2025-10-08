from __future__ import annotations
import numpy as np, pandas as pd



def load_synthetic_cohort(n_patients: int = 8000, seed: int = 7) -> pd.DataFrame:
    """One-row-per-patient synthetic cohort with plausible vitals/labs and sepsis label."""
    rng = np.random.default_rng(seed)
    age = rng.normal(62, 15, n_patients).clip(18, 95)
    sex = rng.choice(["F","M"], size=n_patients, p=[0.52,0.48])
    unit = rng.choice(["ICU","StepDown","MedSurg"], size=n_patients, p=[0.35,0.25,0.40])
    # Baseline vitals/labs
    hr = rng.normal(86, 16, n_patients).clip(40, 180)
    map_ = rng.normal(78, 14, n_patients).clip(40, 120)
    wbc = rng.normal(9.5, 4.0, n_patients).clip(1, 40)
    lactate = np.abs(rng.normal(1.8, 0.9, n_patients))
    creat = np.abs(rng.normal(1.2, 0.5, n_patients))
    temp = rng.normal(37.0, 0.8, n_patients)

    # non-linear risk function (latent logit)
    logits = (
        -4.0
        + 0.018*(age-60)
        + 0.9*(unit=="ICU")
        + 0.35*((hr-85)/15)
        - 0.4*((map_-75)/12)
        + 0.22*((wbc-10)/5)
        + 0.50*((lactate-2.0)/1.0)
        + 0.30*((creat-1.0)/0.5)
        + 0.18*((temp-37.0)/0.8)
        + 0.15*((sex=="M"))
        + rng.normal(0, 0.5, n_patients)
    )
    p = 1/(1+np.exp(-logits))
    y = (rng.uniform(0,1,n_patients) < p).astype(int)  # ~10–20% positive

    df = pd.DataFrame({
        "patient_id": np.arange(n_patients),
        "age": age, "sex": sex, "unit": unit,
        "hr": hr, "map": map_, "wbc": wbc,
        "lactate": lactate, "creatinine": creat, "temp": temp,
        "y_sepsis": y
    })
    return df
