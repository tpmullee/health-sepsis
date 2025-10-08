import joblib, os, json
_model = None
_columns = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    return _model

def get_columns():
    global _columns
    if _columns is None:
        with open(os.getenv("COLUMNS_PATH", "artifacts/columns.json")) as f:
            _columns = json.load(f)
    return _columns
