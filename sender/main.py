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

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        SHOW_NEAR_POINTS = True
        if SHOW_NEAR_POINTS:
            h, w = image.shape[:2]
            for j in range(21):
                px, py = int(lm[j].x * w), int(lm[j].y * h)
                cv2.putText(image, f"{j}", (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
        print(hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()