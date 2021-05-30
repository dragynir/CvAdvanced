import cv2
import mediapipe as mp
import time


class HandDetector:

    HandLandmark = mp.solutions.hands.HandLandmark

    def __init__(self, static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5
                 ):
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.HandLandmark = self.mpHands.HandLandmark

        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                              max_num_hands=self.max_hands,
                              min_detection_confidence=self.detection_confidence,
                              min_tracking_confidence=self.tracking_confidence,
                              )
        self.mpDraw = mp.solutions.drawing_utils
        self.detection_results = {}

    def calculate_fps(self, current_time, previous_time):
        return 1 / (current_time - previous_time)

    def process(self, img, draw=False):

        start_time = time.time()
        img_rgp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, img_channels = img_rgp.shape
        results = self.hands.process(img_rgp)

        current_results = {}
        if results.multi_hand_landmarks:
            # iterate threw hands detected
            for hand_id, handLandmarks in enumerate(results.multi_hand_landmarks):

                hand_landmarks = []
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

                for landmark_id, lm in enumerate(handLandmarks.landmark):

                    # find landmark position on image (scale to image)
                    cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                    hand_landmarks.append({'x': cx, 'y': cy, 'id': landmark_id})

                current_results[hand_id] = {'parsed': hand_landmarks, 'raw': handLandmarks}

        fps = self.calculate_fps(time.time(), start_time)
        self.detection_results = current_results

        return img, current_results, fps

    def draw_hand(self, img, hand_id):

        if hand_id >= len(self.detection_results):
            return img

        hand_landmarks = self.detection_results[hand_id]['raw']
        self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def draw_landmark(self, img, landmark_number, color=(255, 0, 255), size=5):

        for ind, hand in self.detection_results.items():
            landmark = hand['parsed'][landmark_number]
            cx = landmark['x']
            cy = landmark['y']
            cv2.circle(img, (cx, cy), size, color, cv2.FILLED)

        return img

def main():
    
    hand_detector = HandDetector()

    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    previous_time = 0

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img, results, process_fps = hand_detector.process(img)

        hand_detector.draw_hand(img, hand_id=0)
        hand_detector.draw_hand(img, hand_id=1)
        hand_detector.draw_landmark(img, HandDetector.HandLandmark.INDEX_FINGER_TIP)
        current_time = time.time()
        fps = hand_detector.calculate_fps(current_time, previous_time)
        previous_time = current_time

        cv2.putText(img, f'fps: {str(int(fps))}, process_fps: {str(int(process_fps))}',
                    (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()