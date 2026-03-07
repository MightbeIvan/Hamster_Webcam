# 🐹 Hamster Gesture AI

A Python project that uses a webcam, hand tracking, and a neural network to detect hand gestures and trigger hamster reactions.

This project uses:

- MediaPipe for hand tracking
- PyTorch for gesture classification
- OpenCV for webcam input and display

When a gesture is detected, the program displays a hamster reaction image.

---

# ✨ Features

- Real-time hand tracking
- Custom gesture training
- Neural network gesture recognition
- Hamster reaction images
- Supports two hands
- Easy dataset collection
- Automatic training option

---

# ✋ Supported Gestures

The model currently supports **5 gestures**.

| Label | Gesture |
|------|--------|
| 0 | Heart |
| 1 | Nerd |
| 2 | ThumbsUp |
| 3 | ThumbsDown |
| 4 | Freak |

Each gesture triggers a hamster reaction image.

Required images in the project folder:

```
heart.png
nerd.png
ThumbsUp.png
ThumbsDown.png
Freak.png
```

---

# 📁 Project Structure

```
gesture_hamster/

collect.py
auto_collect.py
train.py
auto_train.py
app.py

dataset.csv
model.pth

heart.png
nerd.png
ThumbsUp.png
ThumbsDown.png
Freak.png

README.md
```

---

# ⚙️ Installation

Make sure you have **Python 3 installed**.

Install the required libraries:

```bash
pip3 install opencv-python mediapipe torch pandas
```

---

# 📷 Step 1 — Collect Gesture Data

Run the data collection script:

```bash
python3 collect.py
```

Enter a gesture label when prompted.

```
0
1
2
3
4
```

Example:

```
Enter gesture label: 0
```

Perform the **Heart gesture** in front of the webcam.

Try to collect **200–400 samples per gesture**.

Repeat the process for each gesture.

---

# 🧠 Step 2 — Train the AI

Run the training script:

```bash
python3 train.py
```

This will:

1. Load `dataset.csv`
2. Train the neural network
3. Save the trained model as:

```
model.pth
```

---

# ▶️ Step 3 — Run Gesture Recognition

Start the gesture recognition system:

```bash
python3 app.py
```

Your webcam will open and detect gestures in real time.

When a gesture is detected:

- The gesture name appears on screen
- A hamster reaction image appears

Press **ESC** to close the program.

---

# 🤖 Automatic Training (Optional)

You can automatically collect data and train the model using:

```bash
python3 auto_train.py
```

This script will automatically:

1. Collect gesture samples
2. Save the dataset
3. Train the neural network
4. Save the model

---

# ⚙️ How It Works

1. MediaPipe detects **21 hand landmarks**
2. The landmarks are converted into **42 numerical features**
3. The neural network predicts the gesture
4. The predicted gesture triggers a hamster reaction

Neural network structure:

```
Input: 42 features
Hidden Layer: 64 neurons
Hidden Layer: 32 neurons
Output: 5 gesture classes
```

---

# 🎯 Tips for Better Accuracy

- Use good lighting
- Keep your hand clearly visible
- Collect at least **300 samples per gesture**
- Record gestures from multiple angles
- Avoid background clutter

---

# ⌨️ Controls

| Key | Action |
|----|------|
| ESC | Close program |

---

# 🚀 Future Improvements

Possible upgrades include:

- Gesture smoothing
- Two-hand gesture combinations
- Animated hamster reactions
- Additional gesture classes
- Real-time training

---

# 📜 License

This project is for educational purposes and learning computer vision and machine learning.
