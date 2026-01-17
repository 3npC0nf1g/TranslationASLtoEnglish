import extract_landmarks from src.extract_landmarks
import numpy as np

images_of_A = ["path/to/image1.jpg", "path/to/image2.jpg"]  # Example image paths

for image_path in images_of_A:
    vec = extract_landmarks(image_path)
    if vec is not None:
        np.save("dataset/train/A/A_001.npy", vec)
