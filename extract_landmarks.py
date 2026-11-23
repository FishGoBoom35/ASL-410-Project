
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
import pandas as pd

# Set up MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

columns = ["frame", "hand", "landmark", "x", "y", "z"]

def extract(video_path, model_path="models/hand_landmarker.task", show_video=True):
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

    fps = video.get(cv2.CAP_PROP_FPS) or 30  # fall back if fps is missing
    rows = []
    frame_number = 0


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

            height, width = frame.shape[:2]

            for hand_index, hand in enumerate(result.hand_landmarks):
                for landmark_id, point in enumerate(hand):
                    rows.append((frame_number, hand_index, landmark_id, point.x, point.y, point.z))

                    if show_video:
                        cv2.circle(
                            frame,
                            (int(point.x * width), int(point.y * height)),
                            4,
                            (0, 255, 0),
                            -1,
                        )
 
            if show_video:
                cv2.imshow("Hand landmarks", frame)
                # Press Q to exit early
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
            frame_number += 1
 
    video.release()
    cv2.destroyAllWindows()
 
    return pd.DataFrame(rows, columns=columns)
 
 
df = extract("videos/ASL_How.mp4", show_video=False)
print(df.head())
print(f"{df['frame'].nunique()} frames with a detected hand, {len(df)} rows total")
print("Video processing complete!")