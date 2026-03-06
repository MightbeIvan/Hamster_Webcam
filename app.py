import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# Load hamster images
gestures = {
    0: cv2.imread("heart.png"),
    1: cv2.imread("nerd.png"),
    2: cv2.imread("ThumbsUp.png"),
    3: cv2.imread("ThumbsDown.png"),
    4: cv2.imread("Freak.png")
}

labels = [
    "Heart",
    "Nerd",
    "ThumbsUp",
    "ThumbsDown",
    "Freak"
]

# Model
model = nn.Sequential(
    nn.Linear(42, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5)
)

model.load_state_dict(torch.load("model.pth"))
model.eval()

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            data = []

            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            if len(data) == 42:

                with torch.no_grad():

                    output = model(torch.tensor(data).float().unsqueeze(0))

                    predicted_class = torch.argmax(output).item()

                if predicted_class < len(labels):

                    gesture = labels[predicted_class]

                    cv2.putText(
                        frame,
                        gesture,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    if predicted_class in gestures and gestures[predicted_class] is not None:
                        cv2.imshow("Hamster", gestures[predicted_class])

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()