import cv2
import torch

from src.inference.hand_landmarks import HandLandmarkExtractor
from src.inference.landmarks_inferencer import LandmarksInferencer

CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHECKPOINT_PATH = "checkpoints/landmarks_mlp.pt"
INFER_EVERY_N_FRAMES = 5

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

extractor = HandLandmarkExtractor()
inferencer = LandmarksInferencer(
    checkpoint_path=CHECKPOINT_PATH,
    class_names=CLASS_NAMES,
    device=DEVICE,
)

cap = cv2.VideoCapture(0)

frame_count = 0
last_pred = ""
last_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    result = extractor.extract(frame)
    if result is not None:
        landmarks, hand = result

        if frame_count % INFER_EVERY_N_FRAMES == 0:
            last_pred, last_conf = inferencer.predict(landmarks)

        # DRAW
        h, w, _ = frame.shape
        xs, ys = [], []
        for p in hand.landmark:
            cx, cy = int(p.x * w), int(p.y * h)
            xs.append(cx)
            ys.append(cy)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"{last_pred} ({last_conf:.2f})",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )

    cv2.imshow("ASL2English", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
