"""
FastAPI app for predicting formation energy from chemical formula.
Uses a Random Forest model trained on element fraction descriptors.
"""

import os
import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from jarvis.ai.descriptors.elemental import get_element_fraction_desc
import numpy as np

app = FastAPI(title="Materials Formation Energy Predictor")

# Load model at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "rf_form_energy_model.pkl")
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Train the model first with train.py")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/predict")
def predict(formula: str = "Si"):
    if model is None:
        return {"error": "Model not loaded. Run train.py first."}
    desc = get_element_fraction_desc(formula)
    pred = model.predict([desc])[0]
    return {
        "formula": formula,
        "predicted_formation_energy_eV_per_atom": round(float(pred), 6),
    }


class BatchRequest(BaseModel):
    formulas: list[str]


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if model is None:
        return {"error": "Model not loaded. Run train.py first."}
    descs = [get_element_fraction_desc(f) for f in req.formulas]
    preds = model.predict(descs)
    return {
        "results": [
            {"formula": f, "predicted_formation_energy_eV_per_atom": round(float(p), 6)}
            for f, p in zip(req.formulas, preds)
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
