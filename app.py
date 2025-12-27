import os
import sys
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

from src.inference.load_model import load_model
from src.inference.hand_detector import HandDetector
from src.inference.preprocess import preprocess_frame

# --------------------
# CONFIG
# --------------------
CHECKPOINT_PATH = "checkpoints/asl-vit-epoch=16-val_acc=0.96.ckpt"
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
INFER_EVERY_N_FRAMES = 1

# --------------------
# INIT
# --------------------
model = load_model(CHECKPOINT_PATH).to(DEVICE).eval()
hand_detector = HandDetector()
cap = cv2.VideoCapture(0)

class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

frame_count = 0
last_prediction = "?"

# --------------------
# LOOP
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # miroir
    roi, bbox, landmarks_points = hand_detector.detect(frame)

    if roi is not None and bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        # 🔲 Draw detection box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 🔴 Draw landmarks
        for cx, cy in landmarks_points:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 🧠 Inference every N frames
        if frame_count % INFER_EVERY_N_FRAMES == 0:
            try:
                x_img = preprocess_frame(roi).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(x_img)
                    pred_idx = logits.argmax(dim=1).item()
                    last_prediction = class_names[pred_idx]
            except Exception as e:
                print(f"Inference error: {e}")
                last_prediction = "?"

    # Display prediction
    cv2.putText(frame, f"Pred: {last_prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL2English - Realtime", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
