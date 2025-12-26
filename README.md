
# ASL2English — Real-Time ASL Fingerspelling to English App

This project is an end-to-end machine learning application that translates **American Sign Language (ASL) fingerspelling letters** into **English words** in real time using a webcam.

It includes:

- A deep learning model trained on the ASL alphabet (A–Z)
- Real-time hand tracking using MediaPipe
- Letter prediction and smoothing
- Debounce logic to avoid repeated letters
- Word construction + dictionary autocorrect
- Backend API + frontend interface

---

## Features

- **Real-time hand detection** (MediaPipe)
- **Live ASL letter classification** (CNN or MobileNetV3)
- **Automatic word building**
- **English autocorrect**
- **Clean modular architecture**

---

## Model

The classifier is trained on the ASL Alphabet Dataset from Kaggle:

- 26 classes (A–Z)
- Thousands of labeled images
- Images resized and augmented for robustness

You can easily swap architectures (CNN, MobileNet, EfficientNet).

---

## Folder Structure (Simplified)

```

asl2english/
│
├── data/               # Raw + processed dataset
├── notebooks/          # Jupyter notebooks (training, exploration)
├── src/
│   ├── model/          # Training, evaluation, export
│   ├── inference/      # Real-time detection + prediction
│   ├── api/            # Optional FastAPI backend
│   └── frontend/       # React UI
│
├── models/             # Saved models
└── README.md

```

---

## Real-Time Pipeline

1. Detect hand landmarks  
2. Extract Region of Interest (ROI)  
3. Preprocess ROI  
4. Predict ASL letter  
5. Smooth predictions  
6. Build words  
7. Autocorrect to English  

---

## Technologies

- Python, PyTorch Lightning
- MediaPipe Hands
- OpenCV
- FastAPI
- React
- ONNX / TFLite for deployment

---

## Contact

Feel free to reach out with questions or suggestions!

```