import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

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


BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white
# Create the options that will be used for ImageSegmenter
img_base_options = python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
img_options = vision.ImageSegmenterOptions(
    base_options=img_base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True,
)
segmenter = vision.ImageSegmenter.create_from_options(img_options)

# You detect cross sign if:
# Exactly 2 hands detected
# One["vertical"], one["horizontal"]
# Both have index & middle extended
# Their finger direction vectors are roughly perpendicular
# Their centers are spatially close


def get_hand_orientation(hand):
    # Compare wrist (0) and middle MCP (9)
    dx = abs(hand[9].x - hand[0].x)
    dy = abs(hand[9].y - hand[0].y)
    return "VERTICAL" if dy > dx else "HORIZONTAL"


def index_middle_extended_vertical(hand):
    return (
        hand[8].y < hand[6].y  # index tip above PIP
        and hand[12].y < hand[10].y  # middle tip above PIP
        and hand[16].y > hand[14].y  # ring folded
        and hand[20].y > hand[18].y  # pinky folded
    )


def index_middle_extended_horizontal(hand):
    return (
        hand[8].x < hand[6].x
        and hand[12].x < hand[10].x
        and hand[16].x > hand[14].x
        and hand[20].x > hand[18].x
    )


def get_intersection_point(vertical_tip, horizontal_tip):
    return (vertical_tip.x, horizontal_tip.y)


def fingers_cross(v_hand, h_hand, threshold=0.1):  # Increased threshold
    # 1. Get the Center of the Horizontal Hand (The "Projectile")
    # Average of Index Tip (8) and Middle Tip (12)
    h_center_y = (h_hand[8].y + h_hand[12].y) / 2
    h_center_x = (h_hand[8].x + h_hand[12].x) / 2

    # 2. Define the Target Zone on the Vertical Hand (The "Target")
    # Top is Tip (8), Bottom is Knuckle (5)
    v_top_y = v_hand[8].y
    v_bottom_y = v_hand[5].y

    # We need to handle min/max because 'y' is 0 at top of screen
    v_min_y = min(v_top_y, v_bottom_y)
    v_max_y = max(v_top_y, v_bottom_y)

    # 3. The Check: Is the Horizontal Hand's Y inside the Vertical Hand's finger length?
    vertical_alignment = v_min_y <= h_center_y <= v_max_y

    # 4. Also check X alignment (Are they touching horizontally?)
    # Vertical hand X position
    v_x = v_hand[8].x
    # Check if Horizontal center X is close to Vertical X
    horizontal_touching = abs(h_center_x - v_x) < threshold

    return vertical_alignment and horizontal_touching


def is_jutsu_active(result):
    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return False

    hands = result.hand_landmarks

    vertical_hand = None
    horizontal_hand = None

    # Separate hands by orientation
    for hand in hands:
        orientation = get_hand_orientation(hand)
        if orientation == "VERTICAL":
            vertical_hand = hand
        elif orientation == "HORIZONTAL":
            horizontal_hand = hand

    if not vertical_hand or not horizontal_hand:
        return False

    # Check finger configuration
    if not index_middle_extended_vertical(vertical_hand):
        return False

    if not index_middle_extended_horizontal(horizontal_hand):
        return False

    # Check if both crosses align closely
    if fingers_cross(vertical_hand, horizontal_hand):
        return True

    return False


timestamp_ms = 0
jutsu_active = False
jutsu_start_time = 0
JUTSU_DURATION = 4000  # milliseconds (2 seconds)

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
    timestamp_ms += 33  # ~30 FPS

    # Run MediaPipe segmentation on the current video frame
    segmentation_result = segmenter.segment_for_video(mp_image, timestamp_ms)

    # Get the first confidence mask (there is usually only one for person segmentation)
    # This mask contains float values between 0.0 and 1.0
    # 0.0 = definitely background
    # 1.0 = definitely person
    confidence_mask = segmentation_result.confidence_masks[0]

    # Convert MediaPipe mask to NumPy array so OpenCV can use it
    # Shape is (height, width)
    # Values are float32 in range 0–1
    mask_np = confidence_mask.numpy_view()

    # Resize mask to match the original video frame size
    mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

    # Apply Gaussian blur to smooth edges and reduce flickering noise
    # This makes edges around fingers/hair softer and more stable
    mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)

    # Convert single-channel mask (H, W) into 3 channels (H, W, 3)
    # This allows us to multiply it with the color frame
    mask_3d = np.dstack([mask_np] * 3)

    # Create a black background with same shape as the frame
    background = np.zeros_like(frame)

    # Blend person and background using the soft mask
    # Where mask ≈ 1 → keep frame (person)
    # Where mask ≈ 0 → use background (black)
    # Values in between create smooth edges
    isolated_me = (frame * mask_3d + background * (1 - mask_3d)).astype("uint8")

    h, w = frame.shape[:2]

    final_output = frame.copy()
    if jutsu_active:
        black_canvas = np.zeros_like(frame)

        offset_x = int(w * 0.35)
        offset_y = int(h * 0.2)
        scale = 0.5

        clone_small = cv2.resize(isolated_me, (0, 0), fx=scale, fy=scale)
        ch, cw = clone_small.shape[:2]

        x_left = int(w / 2 - offset_x - cw / 2)
        x_right = int(w / 2 + offset_x - cw / 2)
        y_pos = int(h / 2 - ch / 2 - offset_y)

        # Clamp positions instead of blocking them
        x_left = max(0, min(x_left, w - cw))
        x_right = max(0, min(x_right, w - cw))
        y_pos = max(0, min(y_pos, h - ch))

        black_canvas[y_pos : y_pos + ch, x_left : x_left + cw] = clone_small
        black_canvas[y_pos : y_pos + ch, x_right : x_right + cw] = clone_small

        # black_canvas = cv2.GaussianBlur(black_canvas, (7, 7), 0)

        # Paste clones directly (no fading)
        mask_clone = black_canvas > 0
        final_output[mask_clone] = black_canvas[mask_clone]

        # Real you on top
        final_output = (final_output * (1 - mask_3d) + isolated_me).astype("uint8")

    # ALWAYS show window
    # cv2.imshow("Segmentation", final_output)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks manually
    if result.hand_landmarks:
        if is_jutsu_active(result):
            jutsu_active = True
            jutsu_start_time = timestamp_ms
            print("Jutsu active")
            print("\n")
        if jutsu_active and (timestamp_ms - jutsu_start_time > JUTSU_DURATION):
            jutsu_active = False
        # print(len(result.hand_landmarks))
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                color = (200, 0, 0) if jutsu_active else (0, 255, 0)
                cv2.circle(final_output, (x, y), 6, color, -1)

    cv2.imshow("Naruto Shadow Clone", final_output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
