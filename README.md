# Gesture-Volume-Control

Gesture Volume Control is a Python project that utilizes computer vision techniques to control the system volume based on hand gestures. The project uses the MediaPipe library for hand tracking and OpenCV for video processing. It allows users to adjust the volume by moving their hand in predefined quadrants, or by using distance between the thumb and index finger.

## Features

- Hand tracking using MediaPipe Hands.
- Volume adjustment based on hand position.
- Quadrant detection to determine volume control direction.
- Real-time visualization of hand position and volume level.

## Dependencies

- Python 3.x
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- PyCaw (`pip install pycaw`)
- comtypes (`pip install comtypes`)

## Setup

1. Clone the repository:

```bash
[git clone https://github.com/your_username/Gesture-Volume-Control.git]
```
2. Navigate to the project directory:
```Command Prompt
cd Gesture-Volume-Control
```
3. Install the dependencies:
``` Command Prompt
pip install -r requirements.txt
```
4. python VolumeControlRotation.py
``` Command Prompt
python VolumeControlRotation.py
OR
python VolumeControlDistance.py
```
