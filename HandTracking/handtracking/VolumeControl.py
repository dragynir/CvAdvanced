import cv2
import time
import numpy as np
import math
from HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

camera_width, camera_height = 1080, 640
cap = cv2.VideoCapture(0)
cap.set(propId=3, value=camera_width)
cap.set(propId=4, value=camera_height)


hand_detector = HandDetector(min_detection_confidence=0.7,
                             min_tracking_confidence=0.5,
                             max_num_hands=1)
previous_time = 0
TIP_INTERSECTION = 40

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
min_volume, max_volume, _ = volume_interface.GetVolumeRange()


while True:

    status, img = cap.read()

    img, results, process_fps = hand_detector.process(img)
    hand_detector.draw_landmark(img, HandDetector.HandLandmark.INDEX_FINGER_TIP)

    if results:
        hand_id = 0
        hand_landmarks = results[hand_id]['parsed']
        thumb_tip = hand_landmarks[HandDetector.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[HandDetector.HandLandmark.INDEX_FINGER_TIP]

        line_cx = (thumb_tip['x'] + index_tip['x']) // 2
        line_cy = (thumb_tip['y'] + index_tip['y']) // 2
        line_length = math.hypot(thumb_tip['x'] - index_tip['x'],
                                 thumb_tip['y'] - index_tip['y'])

        cv2.circle(img, (thumb_tip['x'], thumb_tip['y']), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (index_tip['x'], index_tip['y']), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (thumb_tip['x'], thumb_tip['y']), (index_tip['x'], index_tip['y']), (255, 0, 255), 5)
        cv2.circle(img, (line_cx, line_cy), 10, (255, 0, 255), cv2.FILLED)

        if line_length < TIP_INTERSECTION:
            cv2.circle(img, (line_cx, line_cy), 10, (0, 255, 0), cv2.FILLED)

        volume = np.interp(line_length, [20, 200], [min_volume, max_volume])
        volume_interface.SetMasterVolumeLevel(volume, None)

        fill_level = abs(volume - min_volume)/(max_volume - min_volume)

        print(fill_level, volume)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), thickness=3)
        cv2.rectangle(img, (50, 400 - int(fill_level * 250)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'volume: {int(round(100 * fill_level, 2))} %',
                    (50, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


    current_time = time.time()
    fps = hand_detector.calculate_fps(current_time, previous_time)
    previous_time = current_time

    cv2.putText(img, f'fps: {str(int(fps))}',
                (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    if status:
        cv2.imshow('Image', img)
        cv2.waitKey(1)