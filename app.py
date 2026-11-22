"""Clean FastAPI entry point for Emotion Classification with Attention."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from emotion_engine import (
    EMOTION_META,
    EmotionEngine,
    load_benchmark,
    load_per_class_report,
)

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Emotion Classification with Attention")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

engine: EmotionEngine | None = None
load_error: str | None = None

try:
    engine = EmotionEngine(BASE_DIR / "saved_emotion_model")
except Exception as exc:
    load_error = str(exc)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def serve_home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/dashboard")
def api_dashboard():
    return {
        "ready": engine is not None,
        "model_name": engine.model_name if engine else "Not Loaded",
        "error": load_error,
        "emotions": EMOTION_META,
        "benchmark": load_benchmark(BASE_DIR / "saved_emotion_model"),
        "per_class": load_per_class_report(BASE_DIR / "saved_emotion_model"),
    }


@app.post("/api/predict")
def api_predict(payload: TextRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Please enter text to analyze.")
    return engine.predict(payload.text.strip())


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7861, reload=False)
