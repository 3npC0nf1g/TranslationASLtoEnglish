import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

RAW_DATA_DIR = Path("data/raw")
OUT_DATA_DIR = Path("data/landmarks")
TRAIN_RATIO = 0.8
IMAGE_EXTS = {".jpg", ".png", ".jpeg"}

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(img)

    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0]

    points = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)


    points -= points[0]
    norm = np.linalg.norm(points[9])
    if norm > 0:
        points /= norm

    return points.flatten()  # (63,)


def main():
    labels = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])

    print(f"Found labels: {labels}")

    for label in labels:
        img_dir = RAW_DATA_DIR / label
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

        if len(images) == 0:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        splits = {
            "train": images[:split_idx],
            "val": images[split_idx:]
        }

        for split, imgs in splits.items():
            out_dir = OUT_DATA_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{label}] {split}: {len(imgs)} samples")

            for i, img_path in enumerate(tqdm(imgs)):
                vec = extract_landmarks(img_path)
                if vec is None:
                    continue

                out_path = out_dir / f"{label}_{i:05d}.npy"
                np.save(out_path, vec)

    print("\n Dataset landmarks-only created!")

# --------------------
if __name__ == "__main__":
    main()
