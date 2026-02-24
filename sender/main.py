import cv2
import mediapipe as mp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

# osc settings
OSC_IP = "127.0.0.1"
OSC_PORT = 8000


# mediapipe settings
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def pinch_strength(thumb_xy, index_xy) -> float:
    d = np.linalg.norm(np.array(thumb_xy) - np.array(index_xy))
    d_open = 0.12
    d_pinched = 0.03
    s = (d_open - d) / (d_open - d_pinched)
    return clamp01(s)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # if loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    
    # flip the image horizontally
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()