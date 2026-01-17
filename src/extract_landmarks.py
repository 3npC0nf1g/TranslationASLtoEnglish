import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)


def extract_landmarks(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(img)

    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0]
    points = np.array([[p.x, p.y, p.z] for p in lm.landmark])

    # normalisation
    points -= points[0]
    points /= np.linalg.norm(points[9])

    return points.flatten().astype("float32")
