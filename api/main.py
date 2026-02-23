from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class Input(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: Input):
    # Replace with real model loading + prediction
    return {"prediction": 0, "note": "connect your model here"}