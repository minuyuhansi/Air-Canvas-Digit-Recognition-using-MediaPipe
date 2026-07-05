# ✍️ Air Canvas Digit Recognition using MediaPipe

An AI-powered Air Canvas application that allows users to draw digits (0–9) in the air using their index finger. The project uses **MediaPipe Gesture Recognizer** for hand tracking and gesture recognition, while an **MNIST ONNX model** classifies the drawn digit in real time.

---

## ✨ Features

- Real-time hand tracking
- Gesture recognition using MediaPipe
- Air drawing with index finger
- Draws hand landmarks and skeleton
- Recognizes digits (0–9)
- Open Palm gesture clears the canvas
- Live webcam support
- ONNX model inference using OpenCV DNN

---

## 🛠 Technologies Used

- Python
- OpenCV
- MediaPipe Tasks API
- NumPy
- ONNX
- MNIST

---


## 📦 Installation

Install the required libraries.

```bash
pip install mediapipe opencv-python numpy
```

---

## ▶️ Run

```bash
python index.py
```

Press **Q** to exit.

---

## 🎮 Controls

| Gesture | Action |
|----------|--------|
| ☝️ Pointing Up | Draw on the canvas |
| 🤞 Index + Middle Fingers Together | Move without drawing |
| 🖐 Open Palm | Clear the canvas |
| Q Key | Exit application |

---

### Drawing Digits

## 🎥 Demo Video

Watch the complete demo below.

[▶ Watch Demo](demo.mp4)

---

## 🧠 How It Works

1. Detects the user's hand using MediaPipe Gesture Recognizer.
2. Tracks the index finger position.
3. Draws the finger path on a virtual canvas.
4. Converts the drawing into a 28×28 grayscale image.
5. Sends the image to the trained MNIST ONNX model.
6. Displays the predicted digit on the screen.

---

## 👨‍💻 Author

Developed using **Python**, **MediaPipe**, **OpenCV**, and **ONNX**.