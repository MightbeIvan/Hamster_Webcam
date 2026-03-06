import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

label = input("Enter gesture label (0-4): ")

with open("dataset.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

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
                    data.append(label)
                    writer.writerow(data)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()