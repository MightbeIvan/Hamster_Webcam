import cv2
import mediapipe as mp
import csv
import time

gestures = ["Heart","Nerd","ThumbsUp","ThumbsDown","Freak"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

with open("dataset.csv","a",newline="") as f:

    writer = csv.writer(f)

    for label,gesture in enumerate(gestures):

        print("Prepare gesture:",gesture)

        time.sleep(3)

        samples = 0

        while samples < 300:

            success,frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame,1)

            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    data=[]

                    for lm in hand_landmarks.landmark:
                        data.append(lm.x)
                        data.append(lm.y)

                    if len(data)==42:

                        data.append(label)
                        writer.writerow(data)

                        samples += 1

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(frame,f"Gesture: {gesture}",
                        (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

            cv2.imshow("Auto Collect",frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()