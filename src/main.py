from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------
# INITIALIZE APPLICATION
# ------------------------------------------------

app = FastAPI(
    title="Solar Power Monitoring API",
    description="ML-based Solar Power Prediction and Performance Deviation Analysis",
    version="1.0"
)

# ------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "solar_model.pkl")

config = joblib.load(MODEL_DIR / "model_config.pkl")

FEATURES = config["features"]

THRESHOLD = config["deviation_threshold"]

# ------------------------------------------------
# INPUT SCHEMA
# ------------------------------------------------

class SolarInput(BaseModel):

    timestamp: datetime

    plant_id: int = Field(ge=1, le=2)

    dc_power: float = Field(ge=0)

    ambient_temperature: float

    module_temperature: float

    irradiation: float = Field(ge=0)

    actual_ac_power: float | None = Field(default=None, ge=0)


# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------

def prepare_features(data):

    timestamp = data.timestamp

    values = {
        "DC_POWER": data.dc_power,
        "AMBIENT_TEMPERATURE": data.ambient_temperature,
        "MODULE_TEMPERATURE": data.module_temperature,
        "IRRADIATION": data.irradiation,
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_year": timestamp.timetuple().tm_yday,
        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
        "plant_id": data.plant_id
    }

    return pd.DataFrame([values])[FEATURES]


# ------------------------------------------------
# HOME ENDPOINT
# ------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Solar Power Monitoring API is running",
        "status": "Active",
        "model": config["model_name"]
    }


# ------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model_name": config["model_name"],
        "target": config["target"],
        "features": FEATURES,
        "deviation_threshold": THRESHOLD
    }


# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------

@app.post("/predict")
def predict(data: SolarInput):

    X = prepare_features(data)

    prediction = float(model.predict(X)[0])

    return {
        "plant_id": data.plant_id,
        "predicted_ac_power": round(prediction, 3),
        "unit": "Dataset AC_POWER units",
        "model": config["model_name"]
    }


# ------------------------------------------------
# PERFORMANCE ANALYSIS ENDPOINT
# ------------------------------------------------

@app.post("/analyze")
def analyze(data: SolarInput):

    if data.actual_ac_power is None:

        return {
            "error": "actual_ac_power is required for performance analysis"
        }

    X = prepare_features(data)

    prediction = float(model.predict(X)[0])

    residual = data.actual_ac_power - prediction

    absolute_residual = abs(residual)

    status = (
        "Investigate"
        if absolute_residual > THRESHOLD
        else "Normal"
    )

    return {
        "plant_id": data.plant_id,
        "actual_ac_power": data.actual_ac_power,
        "predicted_ac_power": round(prediction, 3),
        "residual": round(residual, 3),
        "absolute_residual": round(absolute_residual, 3),
        "deviation_threshold": round(THRESHOLD, 3),
        "status": status,
        "interpretation": (
            "Unusual performance deviation detected. "
            "Further investigation is recommended."
            if status == "Investigate"
            else "No significant deviation detected."
        )
    }