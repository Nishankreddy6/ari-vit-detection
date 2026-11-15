"""Web app: backend + page in one.   Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core import load_checkpoint, predict_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="ARI Detection")
state = {"model": None, "classes": None}


@app.on_event("startup")
def load_model():
    try:
        state["model"], state["classes"] = load_checkpoint("checkpoints/best.pt", DEVICE)
        print("Model loaded.")
    except FileNotFoundError:
        print("No checkpoints/best.pt yet. Run train.py first.")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"model_loaded": state["model"] is not None, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "No trained model found. Run train.py first.")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read that image. Use a JPG or PNG.")
    return predict_rgb(state["model"], state["classes"], cv2.cvtColor(img, cv2.COLOR_BGR2RGB), DEVICE)
