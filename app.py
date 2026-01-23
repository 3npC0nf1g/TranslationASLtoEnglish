# app.py - Railway compatible (NO OpenCV)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch
import numpy as np
import base64
import logging

from src.inference.hand_landmarks import HandLandmarkExtractor
from src.inference.landmarks_inferencer import LandmarksInferencer

# Configuration
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHECKPOINT_PATH = "checkpoints/landmarks_mlp.pt"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="ASL2English API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.2.28:3000",
        "https://deal-me-n64n.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize models
logger.info("=" * 60)
logger.info("Initializing ASL2English models...")
logger.info(f"Device: {DEVICE}")
logger.info("=" * 60)

extractor = None
inferencer = None

try:
    extractor = HandLandmarkExtractor()
    logger.info("✓ HandLandmarkExtractor loaded")

    inferencer = LandmarksInferencer(
        checkpoint_path=CHECKPOINT_PATH,
        class_names=CLASS_NAMES,
        device=DEVICE,
    )
    logger.info("✓ LandmarksInferencer loaded")
    logger.info("✓ Models loaded successfully!")

except Exception as e:
    logger.error(f"✗ Failed to load models: {e}")


# Models
class FrameRequest(BaseModel):
    frame: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    landmarks: list = None
    hand_box: dict = None


# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if extractor else "degraded",
        "device": DEVICE,
        "models_loaded": extractor is not None,
    }


@app.get("/")
def root():
    return {"name": "ASL2English API", "version": "1.0.0"}


# Inference
@app.post("/predict", response_model=PredictionResponse)
def predict(request: FrameRequest):
    """Receive base64 frame and return prediction."""
    if extractor is None or inferencer is None:
        return PredictionResponse(
            success=False,
            prediction="Error",
            confidence=0.0
        )

    try:
        # Decode frame using Pillow (NO OpenCV!)
        img_data = base64.b64decode(request.frame)
        image = Image.open(io.BytesIO(img_data))
        frame = np.array(image)

        # Flip horizontally (mirror effect)
        frame = np.fliplr(frame)

        # Convert RGB to BGR if needed (for hand landmark extractor)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]  # RGB to BGR

        # Extract landmarks
        result = extractor.extract(frame)

        if result is None:
            return PredictionResponse(
                success=False,
                prediction="No hand",
                confidence=0.0
            )

        landmarks, hand = result

        # Get prediction
        prediction, confidence = inferencer.predict(landmarks)

        # Extract bounding box
        h, w = frame.shape[:2]
        xs = [int(p.x * w) for p in hand.landmark]
        ys = [int(p.y * h) for p in hand.landmark]

        hand_box = {
            "x_min": int(min(xs)),
            "x_max": int(max(xs)),
            "y_min": int(min(ys)),
            "y_max": int(max(ys)),
        }

        # Prepare landmarks
        landmarks_list = [
            {
                "x": int(p.x * w),
                "y": int(p.y * h),
                "z": float(p.z) if hasattr(p, 'z') else 0.0
            }
            for p in hand.landmark
        ]

        return PredictionResponse(
            success=True,
            prediction=prediction,
            confidence=float(confidence),
            landmarks=landmarks_list,
            hand_box=hand_box
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return PredictionResponse(
            success=False,
            prediction="Error",
            confidence=0.0
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)