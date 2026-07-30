"""
Hand-gesture control for the myCobot ultraArmP340 — full XYZ + suction.

Control scheme (relative/incremental):
  - Wrist horizontal movement (frame-to-frame)  -> nudges Y (left/right)
  - Wrist vertical movement (frame-to-frame)     -> nudges Z (up/down)
  - Hand size change (grows/shrinks)             -> nudges X (forward/back reach),
                                                     used as a depth proxy since a
                                                     single RGB camera can't measure
                                                     true distance-from-camera.
  - Closed fist / open hand                      -> suction ON / OFF (GPIO)

Because a single camera has no real depth sensing, this uses *relative* control:
each frame nudges an internally-tracked target position rather than mapping hand
position directly to an absolute coordinate. This is far more forgiving of
detection jitter than absolute mapping.
"""

import time
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymycobot import ultraArmP340

# --- Hardware / model configuration ---------------------------------------

SERIAL_PORT = "COM8"
BAUD_RATE = 115200
MODEL_PATH = "hand_landmarker.task"

# --- Starting position (arm starts here after go_zero, then moves relatively) --

START_X = 235.55
START_Y = 0.0
START_Z = 100.0

# --- Safe travel limits (mm) — clamp the target position to stay in range ---

X_MIN, X_MAX = 150.0, 300.0
Y_MIN, Y_MAX = -100.0, 100.0
Z_MIN, Z_MAX = 60.0, 130.0

# --- Gesture / geometry thresholds -----------------------------------------

# Hand open/closed threshold: ratio of (fingertip-to-wrist) / (knuckle-to-wrist).
# Below this ratio = closed fist.
CLOSED_HAND_RATIO_THRESHOLD = 1.1

# --- Incremental-motion tuning ----------------------------------------------

# How many mm the target position shifts per pixel of frame-to-frame wrist
# movement. Increase for faster response, decrease for finer control.
# Bumped up from 0.30 -> 0.60 so a hand can traverse the full Y/Z travel
# range without needing to leave the camera's field of view.
SENSITIVITY_Y_MM_PER_PX = 0.60   # horizontal wrist movement -> Y
SENSITIVITY_Z_MM_PER_PX = 0.60   # vertical wrist movement -> Z

# How many mm the target X shifts per pixel of change in hand-size
# (wrist-to-middle-knuckle distance). Positive HAND_SIZE_SIGN means "hand
# looks bigger (closer to camera) -> arm moves toward X_MAX (extends out)".
# Flip HAND_SIZE_SIGN to -1 if that direction feels backwards for you.
# Bumped up from 0.20 -> 0.40 for the same reason as Y/Z above.
SENSITIVITY_X_MM_PER_PX = 0.40
HAND_SIZE_SIGN = 1

# Dead zones: ignore movement smaller than this (in pixels) to avoid drift
# from camera/detection noise when the hand is basically still.
# Kept the same as before — dead zone filters noise, sensitivity controls
# speed, so these two are independent tuning knobs.
DEAD_ZONE_WRIST_PX = 4
DEAD_ZONE_SIZE_PX = 3

# Minimum change (mm) in the target position required before we bother
# sending a new set_coords command.
MIN_MOVE_DELTA_MM = 1.5

# Minimum time (s) between move commands sent over serial.
MOVE_INTERVAL = 0.1

# Arm movement speed passed to set_coords.
MOVE_SPEED = 50

# --- Landmark indices used for gesture/geometry detection ------------------

WRIST_IDX = 0
INDEX_TIP_IDX = 8
INDEX_MCP_IDX = 5
MIDDLE_MCP_IDX = 9  # used as a stable "hand size" reference point

# Hand skeleton connections for drawing (MediaPipe hand landmark topology).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm base
]


def create_hand_detector(model_path: str):
    """Build a single-hand MediaPipe HandLandmarker for static-image mode."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def is_hand_closed(points, wrist_idx=WRIST_IDX, tip_idx=INDEX_TIP_IDX, mcp_idx=INDEX_MCP_IDX):
    """
    Determine whether the hand is closed (fist) based on index-finger geometry.

    Uses the ratio of (fingertip-to-wrist distance) to (knuckle-to-wrist
    distance) so the check is roughly scale-invariant (works regardless of
    hand size or distance from camera).
    """
    wrist = points[wrist_idx]
    index_tip = points[tip_idx]
    index_mcp = points[mcp_idx]

    tip_to_wrist = math.hypot(index_tip[0] - wrist[0], index_tip[1] - wrist[1])
    mcp_to_wrist = math.hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])

    hand_open_ratio = tip_to_wrist / (mcp_to_wrist + 1e-6)
    return hand_open_ratio < CLOSED_HAND_RATIO_THRESHOLD


def hand_size_metric(points, wrist_idx=WRIST_IDX, ref_idx=MIDDLE_MCP_IDX):
    """
    Distance (px) between wrist and middle-finger knuckle.

    Used as a rough proxy for how close the hand is to the camera: this
    distance grows as the hand approaches the camera and shrinks as it
    moves away, regardless of whether fingers are curled (unlike using a
    fingertip, which changes with the open/closed gesture).
    """
    wrist = points[wrist_idx]
    ref = points[ref_idx]
    return math.hypot(ref[0] - wrist[0], ref[1] - wrist[1])


def clamp(value, low, high):
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def apply_dead_zone(delta, dead_zone):
    """Return 0 if |delta| is within the dead zone, else return delta unchanged."""
    return delta if abs(delta) > dead_zone else 0.0


def draw_hand_overlay(img, points):
    """Draw hand skeleton connections and landmark points onto the frame."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)


def main():
    # --- Initialize robot arm ---
    mc = ultraArmP340(SERIAL_PORT, BAUD_RATE)
    mc.go_zero()

    # --- Initialize hand detector ---
    detector = create_hand_detector(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    # Internally-tracked target position (since set_coords needs absolute
    # coordinates, but we're driving it with relative hand movement).
    target_x, target_y, target_z = START_X, START_Y, START_Z
    # Move the arm to its starting position before tracking begins.
    mc.set_coords([target_x, target_y, target_z], MOVE_SPEED)

    # Previous-frame values used to compute frame-to-frame deltas.
    prev_wrist_px = None   # (x, y) in pixels
    prev_hand_size = None  # px

    # State tracked to avoid redundant/noisy serial commands.
    current_gpio_state = None  # last suction state sent (0=off, 1=on)
    last_sent_pos = (target_x, target_y, target_z)
    last_move_time = time.time()

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            h, w, _ = img.shape

            # MediaPipe expects RGB input.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            results = detector.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                # Convert normalized landmarks to pixel coordinates.
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                wrist_px = points[WRIST_IDX]
                size_px = hand_size_metric(points)

                # --- 1. Suction control (open/closed hand -> GPIO) ---
                hand_closed = is_hand_closed(points)
                new_gpio_state = 1 if hand_closed else 0

                if new_gpio_state != current_gpio_state:
                    current_gpio_state = new_gpio_state
                    mc.set_gpio_state(current_gpio_state)
                    print(f"Suction: {'ON (1)' if current_gpio_state == 1 else 'OFF (0)'}")

                # --- 2. Incremental XYZ control ---
                # Only compute deltas once we have a previous frame to compare against.
                if prev_wrist_px is not None and prev_hand_size is not None:
                    dx_px = wrist_px[0] - prev_wrist_px[0]   # horizontal wrist movement
                    dy_px = wrist_px[1] - prev_wrist_px[1]   # vertical wrist movement
                    dsize_px = size_px - prev_hand_size      # hand size change

                    # Apply dead zones so small jitter doesn't cause drift.
                    dx_px = apply_dead_zone(dx_px, DEAD_ZONE_WRIST_PX)
                    dy_px = apply_dead_zone(dy_px, DEAD_ZONE_WRIST_PX)
                    dsize_px = apply_dead_zone(dsize_px, DEAD_ZONE_SIZE_PX)

                    # Convert pixel deltas into mm nudges on each axis.
                    target_y += dx_px * SENSITIVITY_Y_MM_PER_PX
                    target_z += -dy_px * SENSITIVITY_Z_MM_PER_PX  # up (smaller y) -> +Z
                    target_x += HAND_SIZE_SIGN * dsize_px * SENSITIVITY_X_MM_PER_PX

                    # Clamp to safe travel limits.
                    target_x = clamp(target_x, X_MIN, X_MAX)
                    target_y = clamp(target_y, Y_MIN, Y_MAX)
                    target_z = clamp(target_z, Z_MIN, Z_MAX)

                prev_wrist_px = wrist_px
                prev_hand_size = size_px

                # Throttle serial writes: only send if enough time has passed
                # AND the position changed enough to matter.
                now = time.time()
                if now - last_move_time > MOVE_INTERVAL:
                    moved_enough = math.dist(
                        (target_x, target_y, target_z), last_sent_pos
                    ) >= MIN_MOVE_DELTA_MM
                    if moved_enough:
                        mc.set_coords([target_x, target_y, target_z], MOVE_SPEED)
                        last_sent_pos = (target_x, target_y, target_z)
                        last_move_time = now

                # --- Visualization ---
                draw_hand_overlay(img, points)

                state_str = "CLOSED" if hand_closed else "OPEN"
                cv2.putText(img, f"Hand: {state_str} | Suction: {new_gpio_state}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(img, f"Target X:{target_x:.1f} Y:{target_y:.1f} Z:{target_z:.1f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No hand detected this frame: reset deltas so we don't jump
                # on the next frame the hand reappears.
                prev_wrist_px = None
                prev_hand_size = None

            cv2.imshow("ultraArm Hand Control", img)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        # Ensure camera and windows are always released, even on error/exit.
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()