"""
main.py  —  FastAPI server that exposes an ML prediction endpoint.

Start with:
    uvicorn main:app --reload

Endpoints:
    GET  /health      → quick check that the server is alive
    POST /predict     → accepts a base64 PNG, returns predicted digit + scores
"""

import io
import base64
import numpy as np
from PIL import Image                             # Pillow — image manipulation
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="MNIST Digit Recogniser", version="1.0")

# CORS (Cross-Origin Resource Sharing)
# By default browsers block JS from calling a different port.
# This tells FastAPI to allow requests from the React dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ───────────────────────────────────────────────
# Loading a model is slow (~1 sec). We do it here so every request is fast.
print("Loading model …")
model = tf.keras.models.load_model("mnist_model.h5")
print("Model ready ✅")


# ── Request schema ────────────────────────────────────────────────────────────
# Pydantic validates incoming JSON automatically.
# The frontend will send:  { "image": "data:image/png;base64,iVBORw0KGgo..." }
class ImageRequest(BaseModel):
    image: str   # data-URL string (base64 encoded PNG from the canvas)


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    digit: int              # most likely digit   e.g. 7
    confidence: float       # probability 0-1     e.g. 0.9983
    scores: list[float]     # all 10 class probs  e.g. [0.0, 0.0, …, 0.99, …]


# ── Helper: preprocess the canvas image ──────────────────────────────────────
def preprocess(data_url: str) -> np.ndarray:
    """
    The HTML canvas gives us a colour PNG data-URL.
    MNIST was trained on 28×28 white digits on a BLACK background.
    We need to match that exactly.

    Steps:
      1. Strip the  "data:image/png;base64,"  header.
      2. Decode base64 → raw bytes.
      3. Open with Pillow → RGBA image (canvas background is transparent).
      4. Paste onto a BLACK background (matches MNIST convention).
      5. Convert to grayscale (L).
      6. Resize to 28×28.
      7. Invert if the user drew dark on light  (canvas is white bg by default).
      8. Normalise to [0, 1] and reshape to (1, 28, 28, 1).
    """
    # 1 & 2: decode base64
    header, encoded = data_url.split(",", 1)
    raw_bytes = base64.b64decode(encoded)

    # 3: open image
    img_rgba = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

    # 4: composite onto black
    background = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
    background.paste(img_rgba, mask=img_rgba.split()[3])   # use alpha as mask

    # 5 & 6: grayscale + resize
    img_gray = background.convert("L").resize((28, 28), Image.LANCZOS)

    # 7: array and check if we need to invert
    arr = np.array(img_gray, dtype="float32")
    # If the mean pixel is bright, the image has light bg / dark strokes.
    # MNIST is the opposite — invert so digit pixels are bright.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 8: normalise and add batch + channel dims  →  shape (1, 28, 28, 1)
    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Simple liveness probe — useful for debugging."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: ImageRequest):
    """
    Accepts a base64 canvas image, runs it through the CNN, returns a prediction.

    Flow:
      Request JSON  →  preprocess image  →  model.predict()  →  JSON response
    """
    try:
        img_array = preprocess(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    # model.predict returns shape (1, 10) — one probability per digit class.
    raw_scores: np.ndarray = model.predict(img_array, verbose=0)[0]

    predicted_digit = int(np.argmax(raw_scores))
    confidence = float(raw_scores[predicted_digit])

    return PredictionResponse(
        digit=predicted_digit,
        confidence=confidence,
        scores=raw_scores.tolist(),
    )