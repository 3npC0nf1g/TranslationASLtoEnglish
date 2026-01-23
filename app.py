# FastAPI Backend Server
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
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

# Enable CORS (Critical for browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://192.168.2.28:3000",
                   "https://deal-me-n64n.vercel.app/"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["POST, GET"],
    allow_headers=["*"],
)

# Initialize models ONCE at startup (not per request)
logger.info("Initializing models...")
try:
    extractor = HandLandmarkExtractor()
    inferencer = LandmarksInferencer(
        checkpoint_path=CHECKPOINT_PATH,
        class_names=CLASS_NAMES,
        device=DEVICE,
    )
    logger.info(f"Models loaded successfully on device: {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise


# Request/Response Models
class FrameRequest(BaseModel):
    frame: str  # Base64 encoded image


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    landmarks: list = None
    hand_box: dict = None
    success: bool


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": True
    }


# Main inference endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: FrameRequest):
    """
    Receive a base64-encoded frame and return hand gesture prediction.

    Args:
        request: JSON with 'frame' key containing base64 image

    Returns:
        PredictionResponse with prediction, confidence, and landmarks
    """
    try:
        # Decode base64 frame
        img_data = base64.b64decode(request.frame)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image")

        # Flip frame (was cv2.flip(frame, 1) in original)
        frame = cv2.flip(frame, 1)

        # Extract hand landmarks
        result = extractor.extract(frame)

        if result is None:
            return PredictionResponse(
                prediction="No hand",
                confidence=0.0,
                success=False
            )

        landmarks, hand = result

        # Get prediction
        prediction, confidence = inferencer.predict(landmarks)

        # Extract hand bounding box
        h, w, _ = frame.shape
        xs = [int(p.x * w) for p in hand.landmark]
        ys = [int(p.y * h) for p in hand.landmark]

        hand_box = {
            "x_min": int(min(xs)),
            "x_max": int(max(xs)),
            "y_min": int(min(ys)),
            "y_max": int(max(ys)),
        }

        # Prepare landmarks for frontend visualization
        landmarks_list = [
            {
                "x": int(p.x * w),
                "y": int(p.y * h),
                "z": float(p.z) if hasattr(p, 'z') else 0.0
            }
            for p in hand.landmark
        ]

        return PredictionResponse(
            prediction=prediction,
            confidence=float(confidence),
            landmarks=landmarks_list,
            hand_box=hand_box,
            success=True
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Batch prediction endpoint
@app.post("/predict-batch")
def predict_batch(requests: list[FrameRequest]):
    """
    Process multiple frames in one request (useful for optimization).
    """
    results = []
    for req in requests:
        result = predict(req)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn

    # Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)