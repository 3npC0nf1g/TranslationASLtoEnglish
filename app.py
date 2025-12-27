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
TEST_IMAGE_PATH = "assets/hand1_q_bot_seg_2_cropped.jpeg"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# --------------------
# LOAD MODEL
# --------------------
model = load_model(CHECKPOINT_PATH).to(DEVICE)
model.eval()

hand_detector = HandDetector()

# --------------------
# LOAD IMAGE
# --------------------
"""
if image is None:
    raise ValueError(f"Could not load image at {TEST_IMAGE_PATH}")

roi, bbox = hand_detector.detect(image)

if roi is None:
    print("No hand detected in test image")
    exit()

# --------------------
# DRAW BOX + LANDMARKS
# --------------------
x_min, y_min, x_max, y_max = bbox
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
"""
# --------------------
# PREPROCESS & INFER
# --------------------
image = cv2.imread(TEST_IMAGE_PATH)
x = preprocess_frame(image).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    confidence = probs.max().item()

print(f"Prediction: {class_names[pred_idx]} | confidence={confidence:.2f}")

"""cv2.putText(
    image,
    f"{class_names[pred_idx]} ({confidence:.2f})",
    (x_min, y_min - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 255, 0),
    3,
)"""

cv2.imshow("Test Image Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
