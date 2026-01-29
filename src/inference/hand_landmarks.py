import mediapipe as mp
import numpy as np
import cv2


class HandLandmarkExtractor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def extract(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)

        # NORMALISATION (same as the training)
        pts -= pts[0]
        norm = np.linalg.norm(pts[9])
        if norm > 0:
            pts /= norm

        return pts.flatten(), hand
