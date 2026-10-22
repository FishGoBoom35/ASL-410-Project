
# ASL Hand Landmark Extraction
#
# This script uses MediaPipe and OpenCV to extract hand landmarks
# from recorded ASL videos and save the coordinates to a CSV file.
# (created in an environment ".venv" in vs-code as that is how I got mine to work)
# Requirements:
# 1. Install Python
# 2. Install required libraries:
#       "python -m pip install mediapipe opencv-python" (can just put in vs-code terminal)
# 3. Download the MediaPipe Hand Landmarker model:
#       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task (will download a file that you can put in the models folder)
#    Save it in models as hand_landmarker.task
# 4. Place your ASL video inside the videos/ folder ("ASL_How" is what I named mine)
#
# Run extract_landmarks.py
# Output in data csv file

import csv
from pathlib import Path
import cv2
import mediapipe as mp

# Set up MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Open your ASL video
video = cv2.VideoCapture("videos/ASL_How.mp4")
fps = video.get(cv2.CAP_PROP_FPS)

if not video.isOpened():
    raise RuntimeError("Could not open video")

frame_number = 0
Path("data").mkdir(exist_ok=True)

csv_file = open("data/ASL_How_landmarks.csv", "w", newline="")

writer = csv.writer(csv_file)

writer.writerow(["frame", "hand", "landmark", "x", "y", "z"])
with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = video.read()

        if not success:
            break

        # Convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(frame_number * 1000 / fps)

        # Detect hands
        result = detector.detect_for_video(mp_image, timestamp)

        # Draw green dots and save landmarks
        for hand_index, hand in enumerate(result.hand_landmarks):
            for landmark_id, point in enumerate(hand):

                writer.writerow([
                    frame_number,
                    hand_index,
                    landmark_id,
                    point.x,
                    point.y,
                    point.z
                ])
            
            height, width = frame.shape[:2]
            x = int(point.x * width)
            y = int(point.y * height)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Press Q to exit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        frame_number += 1

video.release()
csv_file.close()
cv2.destroyAllWindows()

print("Video processing complete!")