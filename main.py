import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageSequence
import time
import os
import urllib.request

# ---------------------------------------------------------------------------
# MediaPipe removed the old `mp.solutions.hands` / `mp.solutions.face_mesh`
# API in 0.10.31+. The replacement is the "Tasks" API, which needs small
# model bundle files downloaded once. We fetch them automatically on first
# run and cache them next to this script.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
FACE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"Downloading model: {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print("Done.")


ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)

VisionRunningMode = vision.RunningMode

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

# Standard 21-point hand skeleton connections (same pairs the old
# mp.solutions.hands.HAND_CONNECTIONS used), kept here manually since the
# `solutions` module that used to expose this constant no longer exists.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)


def load_image_frames(path):
    try:
        img = Image.open(path)
        frames = []
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGBA")
            cv_img = np.array(frame)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
            frames.append(cv_img)
        return frames
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


images = {
    'default': load_image_frames('cats/default.webp'),
    'chup': load_image_frames('cats/chup.gif'),
    'lick': load_image_frames('cats/lick.gif'),
    'rocked': load_image_frames('cats/rocked.gif')
}


def get_current_frame(frames, start_time, fps=10):
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    elapsed = time.time() - start_time
    frame_index = int(elapsed * fps) % len(frames)
    return frames[frame_index]


# NOTE: gesture functions below are unchanged in logic from the original
# script. Only the landmark access changed: the old API used
# `hand_landmarks.landmark[i]`; the new Tasks API gives a plain list, so it's
# `hand_landmarks[i]` instead. Same indices (0-20), same meaning.

def is_shaka_gesture(hand_landmarks):
    index_folded = hand_landmarks[8].y > hand_landmarks[6].y
    middle_folded = hand_landmarks[12].y > hand_landmarks[10].y
    ring_folded = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_extended = hand_landmarks[20].y < hand_landmarks[18].y
    thumb_extended = abs(hand_landmarks[4].x - hand_landmarks[17].x) > abs(hand_landmarks[3].x - hand_landmarks[17].x)

    return index_folded and middle_folded and ring_folded and pinky_extended and thumb_extended


def is_shh_gesture(hand_landmarks, face_landmarks):
    index_extended = hand_landmarks[8].y < hand_landmarks[6].y
    middle_folded = hand_landmarks[12].y > hand_landmarks[10].y
    ring_folded = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_folded = hand_landmarks[20].y > hand_landmarks[18].y

    if index_extended and middle_folded and ring_folded and pinky_folded:
        if face_landmarks:
            index_tip = hand_landmarks[8]
            mouth_top = face_landmarks[13]
            mouth_bottom = face_landmarks[14]

            dist_to_mouth_y = abs(index_tip.y - (mouth_top.y + mouth_bottom.y) / 2)
            dist_to_mouth_x = abs(index_tip.x - (mouth_top.x + mouth_bottom.x) / 2)

            if dist_to_mouth_x < 0.2 and dist_to_mouth_y < 0.2:
                return True
        else:
            return True
    return False


def is_tongue_out(face_landmarks):
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]
    mouth_openness = bottom_lip.y - top_lip.y
    return mouth_openness > 0.05


def open_camera():
    """Try a few camera indices since it varies by machine which one is the
    real webcam. Uses the DirectShow backend on Windows, which opens more
    reliably there than the default backend."""
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            print(f"Opened webcam at index {idx}.")
            return cap
        cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open any webcam (tried indices 0-3).")
        print("Check that no other app (Zoom, Teams, Camera app) is using it,")
        print("and that Windows Settings > Privacy > Camera allows desktop apps access.")
        return

    current_gesture = 'default'
    gesture_start_time = time.time()
    start_ts = time.time()

    print("Press 'q' or 'ESC' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int((time.time() - start_ts) * 1000)

        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        detected_gesture = 'default'

        face_landmarks = face_results.face_landmarks[0] if face_results.face_landmarks else None

        if hand_results.hand_landmarks:
            hand_landmarks = hand_results.hand_landmarks[0]
            draw_hand_landmarks(image, hand_landmarks)

            if is_shaka_gesture(hand_landmarks):
                detected_gesture = 'rocked'
            elif is_shh_gesture(hand_landmarks, face_landmarks):
                detected_gesture = 'chup'

        if detected_gesture == 'default' and face_landmarks:
            if is_tongue_out(face_landmarks):
                detected_gesture = 'lick'

        if detected_gesture != current_gesture:
            current_gesture = detected_gesture
            gesture_start_time = time.time()

        cat_img_bgra = get_current_frame(images[current_gesture], gesture_start_time)
        if cat_img_bgra is not None:
            cv2.imshow('Cat Reaction', cat_img_bgra)

        cv2.imshow('Webcam Feed', image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()