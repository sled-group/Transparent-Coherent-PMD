import cv2
import numpy as np
from typing import Optional

def get_video(video_path: str) -> cv2.VideoCapture:
    """
    Loads a video using cv2.

    :param video_path: Path to video file (only tested with mp4).
    :return: Loaded cv2 VideoCapture for video. Be sure to call .release() on returned VideoCapture later.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Cannot open video file")
    
    return cap
    # NOTE: remember to call cap.release() later

def extract_frames(cap: cv2.VideoCapture, times: list[Optional[float]]) -> list[np.ndarray]:
    """
    Samples frames from a given video at the given times.

    :param cap: Opened cv2 video capture.
    :param times: List of times in seconds to sample frames for.
    :return: List of frames as NumPy arrays.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    frames = []

    for t in times:
        try:
            if type(t) == float:
                frame_number = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(min(frame_number, last_frame), 0.0)) # Avoid seeking before first frame and past last frame
                ret, frame = cap.read()

                if ret:
                    # Convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
                    frames.append(frame)
                else:
                    print(f"Warning: Frame at time {t} seconds could not be read.")
                    frames.append(None)
            else:
                frames.append(None)
        except:
            print(f"Warning: Failed while trying to read frame at time {t} seconds.")
            frames.append(None)
            continue

    return frames