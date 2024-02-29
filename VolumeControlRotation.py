import cv2
import mediapipe as mp
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import c_float, c_int

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Function to calculate the quadrant based on finger position
def calculate_quadrant(x, y, width, height):
    if x < width / 2:
        if y < height / 2:
            return 1
        else:
            return 4
    else:
        if y < height / 2:
            return 2
        else:
            return 3

# Function to adjust volume based on quadrant change
def adjust_volume(current_quadrant, previous_quadrant, volume):
    # if volume == -63.5 or volume == 0:  # Ensure volume remains within range
    #     return volume

    ###### INCREASE VOLUME ######
    if current_quadrant == 1 and previous_quadrant == 4:
        return min(0, volume + 5)  # Increase volume
    elif current_quadrant == 4 and previous_quadrant == 3:
        return min(0, volume + 5)
    elif current_quadrant == 3 and previous_quadrant == 2:
        return min(0, volume + 5)
    elif current_quadrant == 2 and previous_quadrant == 1:
        return min(0, volume + 5)
    ###### DECREASE VOLUME ######
    elif current_quadrant == 4 and previous_quadrant == 1:
        return max(-63.5, volume - 5)
    elif current_quadrant == 3 and previous_quadrant == 4:
        return max(-63.5, volume - 5)
    elif current_quadrant == 2 and previous_quadrant == 3:
        return max(-63.5, volume - 5)
    elif current_quadrant == 1 and previous_quadrant == 2:
        return max(-63.5, volume - 5)
    else:
        return volume  # No change

# Main function for hand tracking and volume control
def hand_volume_control():
    global volume
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    previous_quadrant = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_x = int(hand_landmarks.landmark[8].x * width)
                index_y = int(hand_landmarks.landmark[8].y * height)
                cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)
                current_quadrant = calculate_quadrant(index_x, index_y, width, height)

                if previous_quadrant is not None:
                    adjusted_volume = adjust_volume(current_quadrant, previous_quadrant, volume.GetMasterVolumeLevel())
                    volume.SetMasterVolumeLevel(adjusted_volume, None)
                    print(volume)

                previous_quadrant = current_quadrant

                cv2.putText(frame, f"Quadrant: {current_quadrant}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Volume: {volume}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Hand Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the hand volume control function
# volume.SetMasterVolumeLevel(0, None)
hand_volume_control()
