import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

# Create base options with model path
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

# Configure hand landmarker
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Create landmarker
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# video using cv2
cap = cv2.VideoCapture(0)


# You detect cross sign if:
# Exactly 2 hands detected
# One["vertical"], one["horizontal"]
# Both have index & middle extended
# Their finger direction vectors are roughly perpendicular
# Their centers are spatially close
""" 
Step 3 — Check They Intersect (Very Important)

Even if one hand is horizontal and one vertical,
they might be far apart.

So now check spatial overlap.

A simple way:

Compute midpoint between index & middle tips for each hand.

Call them center_A and center_B.

If:

distance(center_A, center_B) < threshold


→ They’re close enough to be crossing.

Threshold should be normalized by hand size.
"""
def isJutsuActive(result):
    # if there is only ine hand
    if len(result.handedness) < 2:
        return False
    hand_landmarks = result.hand_landmarks
    orientation = {"VERTICAL": 0, "HORIZONTAL": 0}

    for landmark in hand_landmarks:
        dx = abs(landmark[9].x - landmark[0].x)
        dy = abs(landmark[9].y - landmark[0].y)
        if dy > dx:
            orientation["VERTICAL"] = landmark
        elif dx > dy:
            orientation["HORIZONTAL"] = landmark

    if not (orientation["HORIZONTAL"] and orientation["VERTICAL"]):
        return False

    if (
        (orientation["VERTICAL"][8].y < orientation["VERTICAL"][6].y)
        and (orientation["VERTICAL"][12].y < orientation["VERTICAL"][10].y)
        and (orientation["VERTICAL"][16].y > orientation["VERTICAL"][14].y)
        and (orientation["VERTICAL"][20].y > orientation["VERTICAL"][18].y)
    ):
        if (
            (orientation["HORIZONTAL"][8].x < orientation["HORIZONTAL"][6].x)
            and (orientation["HORIZONTAL"][12].x < orientation["HORIZONTAL"][10].x)
            and (orientation["HORIZONTAL"][16].x > orientation["HORIZONTAL"][14].x)
            and (orientation["HORIZONTAL"][20].x > orientation["HORIZONTAL"][18].x)
        ):
            
            print("Jutsu active")
        else:
            return False

    else:
        return False


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # flip frame to get mirror effect
    frame = cv2.flip(frame, 1)

    # convert to rgb from bgr
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Process frame (VIDEO mode requires timestamp)
    timestamp_ms = timestamp_ms = int(time.time() * 1000)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks manually
    if result.hand_landmarks:
        isJutsuActive(result)
        time.sleep(0.001)
        # print(len(result.hand_landmarks))
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
