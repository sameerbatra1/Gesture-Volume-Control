import cv2
import mediapipe as mp
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volPer = 0

def detect_hands():
    global vol, volBar, volPer
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        frame.flags.writeable = False

        results = hands.process(frame)

        frame.flags.writeable = True

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw only the landmarks for the index finger and thumb
                for landmark in [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]]:
                    index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    pinky_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    index_x, index_y = int(index_finger_landmark.x * frame.shape[1]), int(index_finger_landmark.y * frame.shape[0])
                    thumb_x, thumb_y = int(thumb_landmark.x * frame.shape[1]), int(thumb_landmark.y * frame.shape[0])
                    pinky_x, pinky_y = int(pinky_landmark.x * frame.shape[1]), int(pinky_landmark.y * frame.shape[0])

                    # Get the center of the line
                    cX, cY = (index_x + thumb_x) // 2, (index_y + thumb_y) // 2

                    # Draw circles at the landmarks
                    cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)
                    cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 0, 0), -1)
                    cv2.circle(frame, (pinky_x, pinky_y), 5, (255, 0, 0), -1)
                    cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

                    # Draw line between index finger and thumb
                    cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)

                    # Get the length between the Index Finger and Thumb
                    length = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    # print(length)

                    # Hand Range 58 to 300
                    # Volume Range -63.5 to 0
                    # we need to convert lowest and highest value of hand range into low, high volume range
                    vol = np.interp(length, [50, 250], [minVol, maxVol])
                    volBar = np.interp(length, [50, 250], [400, 150])
                    volPer = np.interp(length, [50, 250], [0, 100])
                    print(vol)
                    volume.SetMasterVolumeLevel(vol, None)

                    if length<58:
                        cv2.circle(frame, (cX, cY), 10, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f'{int(volPer)}%', (48, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 3)
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        detect_hands()
