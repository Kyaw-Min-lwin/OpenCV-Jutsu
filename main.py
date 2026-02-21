import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# We define the skeleton connections manually to bypass MediaPipe's messy legacy API
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm base
]

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


def get_hand_orientation(hand):
    dx = abs(hand[9].x - hand[0].x)
    dy = abs(hand[9].y - hand[0].y)
    return "VERTICAL" if dy > dx else "HORIZONTAL"


def index_middle_extended_vertical(hand):
    return (
        hand[8].y < hand[6].y
        and hand[12].y < hand[10].y
        and hand[16].y > hand[14].y
        and hand[20].y > hand[18].y
    )


def index_middle_extended_horizontal(hand):
    return (
        hand[8].x < hand[6].x
        and hand[12].x < hand[10].x
        and hand[16].x > hand[14].x
        and hand[20].x > hand[18].x
    )


def fingers_cross(v_hand, h_hand, threshold=0.1):
    h_center_y = (h_hand[8].y + h_hand[12].y) / 2
    h_center_x = (h_hand[8].x + h_hand[12].x) / 2

    v_top_y = v_hand[8].y
    v_bottom_y = v_hand[5].y

    v_min_y = min(v_top_y, v_bottom_y)
    v_max_y = max(v_top_y, v_bottom_y)

    vertical_alignment = v_min_y <= h_center_y <= v_max_y

    v_x = v_hand[8].x
    horizontal_touching = abs(h_center_x - v_x) < threshold

    return vertical_alignment and horizontal_touching


def is_jutsu_active(result):
    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return False

    hands = result.hand_landmarks

    vertical_hand = None
    horizontal_hand = None

    for hand in hands:
        orientation = get_hand_orientation(hand)
        if orientation == "VERTICAL":
            vertical_hand = hand
        elif orientation == "HORIZONTAL":
            horizontal_hand = hand

    if not vertical_hand or not horizontal_hand:
        return False

    if not index_middle_extended_vertical(vertical_hand):
        return False

    if not index_middle_extended_horizontal(horizontal_hand):
        return False

    if fingers_cross(vertical_hand, horizontal_hand):
        return True

    return False


# ==========================================
# PROCEDURAL SMOKE GENERATOR (No Assets Needed)
# ==========================================
def draw_smoke_poof(frame, cx, cy, age, max_age, base_size):
    progress = age / max_age  # 0.0 to 1.0

    # Radius expands as it ages
    r = int(base_size * 0.5 * (1.0 + progress))

    # Opacity fades exponentially (stays thick at first, then vanishes)
    opacity = 0.85 * (1.0 - (progress**1.5))

    # Empty mask for the smoke cloud
    smoke_mask = np.zeros(frame.shape[:2], dtype=np.float32)

    # Draw a cluster of 5 overlapping circles to make a "cloud"
    offsets = [
        (0, 0),
        (int(-0.4 * r), int(0.2 * r)),
        (int(0.4 * r), int(0.1 * r)),
        (int(-0.2 * r), int(-0.4 * r)),
        (int(0.3 * r), int(-0.3 * r)),
    ]
    for ox, oy in offsets:
        cv2.circle(smoke_mask, (cx + ox, cy + oy), int(r * 0.6), 1.0, -1)

    # Blur heavily to make it soft like vapor
    smoke_mask = cv2.GaussianBlur(smoke_mask, (51, 51), 0)
    smoke_mask = smoke_mask * opacity
    smoke_3d = np.dstack([smoke_mask] * 3)

    # Off-white classic anime smoke color
    smoke_color = np.full_like(frame, (220, 220, 220), dtype=np.float32)

    return frame * (1 - smoke_3d) + smoke_color * smoke_3d


timestamp_ms = 0
jutsu_active = False
jutsu_start_time = 0
JUTSU_DURATION = 40000  # milliseconds (40 seconds)
SMOKE_DURATION = 300

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms += 33

    segmentation_result = segmenter.segment_for_video(mp_image, timestamp_ms)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    confidence_mask = segmentation_result.confidence_masks[0]
    mask_np = confidence_mask.numpy_view()
    mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

    # --- 1. MASK SKELETON (Hidden logic to fix webbed fingers) ---
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append((x, y))

            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                cv2.line(mask_np, points[start_idx], points[end_idx], 1.0, thickness=25)

            for point in points:
                cv2.circle(mask_np, point, 15, 1.0, -1)

    sharp_mask = cv2.GaussianBlur(mask_np.copy(), (3, 3), 0)
    sharp_mask_3d = np.dstack([sharp_mask] * 3)

    mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
    mask_3d = np.dstack([mask_np] * 3)

    background = np.zeros_like(frame)

    isolated_me = (frame * mask_3d + background * (1 - mask_3d)).astype("uint8")

    h, w = frame.shape[:2]
    final_output = frame.copy()

    if jutsu_active:
        elapsed_ms = timestamp_ms - jutsu_start_time
        frame = frame.astype(np.float32)
        final_output = final_output.astype(np.float32)
        isolated_me_float = isolated_me.astype(np.float32)

        # Helper to process a clone to keep the loop clean
        def process_clone(output, spawn_time, scale, x_offset_func, y_offset_func):
            if elapsed_ms > spawn_time:
                c_small = cv2.resize(isolated_me_float, (0, 0), fx=scale, fy=scale)
                c_mask = cv2.resize(mask_np, (c_small.shape[1], c_small.shape[0]))
                c_mask = cv2.GaussianBlur(c_mask, (11, 11), 0)
                c_mask_3d = np.dstack([c_mask] * 3)
                ch, cw = c_small.shape[:2]

                x_pos = x_offset_func(cw, ch)
                y_pos = y_offset_func(cw, ch)
                x_pos = max(0, min(x_pos, w - cw))
                y_pos = max(0, min(y_pos, h - ch))

                roi = output[y_pos : y_pos + ch, x_pos : x_pos + cw]
                output[y_pos : y_pos + ch, x_pos : x_pos + cw] = (
                    roi * (1 - c_mask_3d) + c_small * c_mask_3d
                )

                # DRAW SMOKE IF IT JUST SPAWNED
                age = elapsed_ms - spawn_time
                if 0 <= age < SMOKE_DURATION:
                    # Center the smoke on the clone's body
                    cx, cy = x_pos + cw // 2, y_pos + ch // 2
                    output = draw_smoke_poof(output, cx, cy, age, SMOKE_DURATION, cw)
            return output

        # Z-INDEXING: Back to Front
        # 4: Back Left (500ms)
        final_output = process_clone(
            final_output,
            500,
            0.45,
            lambda cw, ch: int(w * 0.2 - cw),
            lambda cw, ch: int(h - ch),
        )
        # 3: Back Right (700ms)
        final_output = process_clone(
            final_output,
            700,
            0.45,
            lambda cw, ch: int(w * 0.2 + cw),
            lambda cw, ch: int(h - ch),
        )
        # 2: Front Left (100ms)
        final_output = process_clone(
            final_output, 100, 0.65, lambda cw, ch: 0, lambda cw, ch: h - ch
        )
        # 1: Front Right (300ms)
        final_output = process_clone(
            final_output, 300, 0.65, lambda cw, ch: w - cw, lambda cw, ch: h - ch
        )

        # =======================================================
        # --- NEW: BLUE & BLACK CURSED ENERGY AURA ---
        # =======================================================

        # A. Expand the sharp mask to create an aura boundary (keeps it close to the body)
        kernel = np.ones((5, 5), np.uint8)
        dilated_aura = cv2.dilate(sharp_mask, kernel, iterations=5)
        aura_area = np.clip(dilated_aura - sharp_mask, 0, 1)

        # B. Add random noise to make it jagged and unstable
        noise = np.random.uniform(0.0, 1.0, (h, w)).astype(np.float32)
        jagged_aura = aura_area * noise

        # C. Create the "Black" base (Darkens the background heavily behind the blue)
        black_alpha = cv2.GaussianBlur(aura_area, (15, 15), 0)
        black_alpha_3d = np.dstack([black_alpha] * 3)
        final_output = final_output * (1.0 - black_alpha_3d * 0.8)

        # D. Create the "Blue" flames
        # Squaring the noise (jagged_aura ** 2) isolates the sharpest spikes
        blue_alpha = cv2.GaussianBlur(jagged_aura**2, (7, 7), 0)
        blue_alpha_3d = np.dstack([blue_alpha] * 3)
        blue_color = np.full_like(
            frame, (118, 124, 80), dtype=np.float32
        )  # BGR Blue/Cyan

        # Add the bright blue flames on top of the darkened background
        final_output = np.clip(final_output + blue_color * blue_alpha_3d * 2.5, 0, 255)

        # =======================================================

        # ----- REAL YOU ON TOP -----
        # Because we paste the Real You back over the aura,
        # the energy perfectly wraps behind your silhouette!
        final_output = (
            final_output * (1 - sharp_mask_3d) + frame * sharp_mask_3d
        ).astype("uint8")

    # --- 2. JUTSU LOGIC & CHAKRA VISUALIZATION ---
    if result.hand_landmarks:
        if is_jutsu_active(result):
            # Only trigger a fresh start if the jutsu wasn't already active
            if not jutsu_active:
                jutsu_active = True
                jutsu_start_time = timestamp_ms
                print("Jutsu active\n")

        if jutsu_active and (timestamp_ms - jutsu_start_time > JUTSU_DURATION):
            jutsu_active = False

        # Draw the Chakra Lines!
        for hand_landmarks in result.hand_landmarks:
            pixel_points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                pixel_points.append((x, y))

            # BGR format: Normal Chakra is Cyan/Blue, Active Jutsu is Kyuubi Red
            chakra_line_color = (0, 0, 255) if jutsu_active else (255, 200, 0)
            chakra_node_color = (0, 0, 200) if jutsu_active else (255, 255, 0)

            # Draw the glowing connections
            for connection in HAND_CONNECTIONS:
                start_pt = pixel_points[connection[0]]
                end_pt = pixel_points[connection[1]]
                cv2.line(final_output, start_pt, end_pt, chakra_line_color, 2)

            # Draw the chakra nodes (joints)
            for point in pixel_points:
                cv2.circle(final_output, point, 5, chakra_node_color, -1)

    cv2.imshow("Naruto Shadow Clone", final_output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
