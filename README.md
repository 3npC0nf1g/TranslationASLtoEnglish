# ASL2English: Hand Gesture Recognition (Landmarks‑Only)

This project implements **real‑time American Sign Language (ASL) alphanumeric recognition** using **hand landmarks only** (no raw images at inference time).

The system relies on:

* **MediaPipe Hands** for 3D hand landmark extraction
* A lightweight **PyTorch MLP** trained on normalized landmarks
* **OpenCV** for real‑time webcam inference

The design follows **single‑responsibility principles** and is structured for clean experimentation, training, and production inference.

---

## Key Features

* Landmarks‑only inference (robust to lighting & background)
* Fast real‑time webcam prediction
* Lightweight MLP (63‑dim input)
* Clean, modular architecture
* Dataset preprocessing pipeline included

---

## Dataset

The model is trained using the **ASL Alphanumeric Dataset** from Kaggle:

[https://www.kaggle.com/datasets/ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

### Dataset description

* Classes: `0–9` and `A–Z`
* Format: RGB images of single hand gestures
* Usage in this project:

  * Images are **not used directly** by the model
  * Each image is processed once to extract **21 hand landmarks (x, y, z)**
  * Resulting feature vector size: **63 floats per sample**

---

## Model

**LandmarksMLP** (PyTorch)

* Input: `(63,)` flattened landmarks
* Output: `36` classes (`0–9`, `A–Z`)
* Loss: Cross‑Entropy
* Optimizer: Adam

The model was **trained on Google Colab** using the notebook:

```
notebooks/00_landmarks_MLP.ipynb
```

The final checkpoint is stored locally:

```
checkpoints/landmarks_mlp.pt
```
---
## Data Processing Pipeline

1. **Raw images** (`data/raw/<class>/*.jpg`)
2. MediaPipe extracts 21 hand landmarks
3. Landmarks are normalized:

   * Translation: wrist at origin
   * Scale: normalized by middle‑finger MCP distance
4. Saved as `.npy` vectors of shape `(63,)`
5. Used for training the MLP

---

## Running Webcam Inference

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the webcam app

For MaxOS/Linux:
```bash
python3 -m src.app.webcam
```
For Windows:
```bash
python -m src.app.webcam
```
Controls:

* `q` → quit

The app will:

* Detect one hand
* Extract landmarks
* Run inference every N frames
* Display predicted class and confidence

---

## Configuration

Key parameters can be adjusted in `webcam.py` or inference modules:

* `INFER_EVERY_N_FRAMES`
* MediaPipe confidence thresholds
* Model checkpoint path

---

## Why Landmarks‑Only?

| Aspect     | Image‑based | Landmarks‑only  |
| ---------- | ----------- | --------------- |
| Speed      | ❌ Slower    | ✅ Fast          |
| Robustness | ❌ Sensitive | ✅ Stable        |
| Model size | ❌ Large     | ✅ Small         |
| Deployment | ❌ Heavy     | ✅ Edge‑friendly |

---

# Future Improvements

* Temporal smoothing (gesture stability)
* ONNX / TensorRT export
* REST or WebSocket API
* Sequence‑based recognition (words)
* Dockerized deployment

---

## License

This project is for educational and research purposes.
Dataset license applies as defined on Kaggle.

---

## Acknowledgements

* MediaPipe Hands
* PyTorch
* Kaggle ASL Dataset contributors
