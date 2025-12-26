import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def detect(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

        # 🔥 ZONE PLUS LARGE
        margin = 60
        x_min = max(min(xs) - margin, 0)
        y_min = max(min(ys) - margin, 0)
        x_max = min(max(xs) + margin, w)
        y_max = min(max(ys) + margin, h)

        roi = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)

        # ✋ DRAW LANDMARKS
        if draw:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        return roi, bbox
