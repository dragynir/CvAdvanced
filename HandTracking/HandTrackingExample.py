import cv2
import mediapipe as mp
import time


def calculate_fps(current_time, previous_time):
    return 1 / (current_time - previous_time)


camera_id = 0
cap = cv2.VideoCapture(camera_id)

mpHands = mp.solutions.hands

hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.5,
                      )
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w, img_channels = img.shape

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        # iterate threw hands detected
        for handLandmarks in results.multi_hand_landmarks:

            for id, lm in enumerate(handLandmarks.landmark):

                # find landmark position on image (scale to image)
                cx, cy = int(lm.x * img_w), int(lm.y * img_h)

                if id == mpHands.HandLandmark.INDEX_FINGER_TIP:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                elif id == mpHands.HandLandmark.WRIST:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                elif id == mpHands.HandLandmark.THUMB_TIP:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = calculate_fps(current_time, previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
