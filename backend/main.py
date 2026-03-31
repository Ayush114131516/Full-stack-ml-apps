import io
import os
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MNIST Digit Recogniser", version="1.0")

# CORS 
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    os.getenv("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup 
print("Loading model …")
model = tf.keras.models.load_model("mnist_model.h5")
print("Model ready ✅")


class ImageRequest(BaseModel):
    image: str


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    scores: list[float]


def preprocess(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    raw_bytes = base64.b64decode(encoded)
    img_rgba = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    background = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
    background.paste(img_rgba, mask=img_rgba.split()[3])
    img_gray = background.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img_gray, dtype="float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: ImageRequest):
    try:
        img_array = preprocess(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    raw_scores = model.predict(img_array, verbose=0)[0]
    predicted_digit = int(np.argmax(raw_scores))
    confidence = float(raw_scores[predicted_digit])

    return PredictionResponse(
        digit=predicted_digit,
        confidence=confidence,
        scores=raw_scores.tolist(),
    )