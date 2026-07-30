"""
Two-hand gesture control for the myCobot ultraArmP340.

Control scheme:
  RIGHT HAND (discrete / directional switch, neutral = closed fist):
    - Thumb extended, other fingers curled       -> hold to move RIGHT (+Y)
    - Pinky extended, other fingers curled       -> hold to move LEFT (-Y)
    - Closed fist (neutral)                      -> no Y movement

  LEFT HAND (continuous / relative):
    - Wrist vertical movement (frame-to-frame)  -> nudges Z (up/down)
    - Hand size change (grows/shrinks)          -> nudges X (forward/back reach),
                                                    used as a depth proxy since a
                                                    single RGB camera can't measure
                                                    true distance-from-camera.
    - Closed fist / open hand                   -> suction ON / OFF (GPIO)

IMPORTANT — camera mirroring:
  MediaPipe's left/right hand classification assumes the input image is
  mirrored (like a selfie camera). This script flips the captured frame
  horizontally before detection/display so that:
    (a) what you see on screen behaves like a normal mirror, and
    (b) MediaPipe's "Left"/"Right" labels correctly match YOUR left/right
        hand rather than being swapped.
  If your labels still seem swapped for your camera setup, set
  MIRROR_CAMERA = False below and re-test.
"""

import time
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymycobot import ultraArmP340

# --- Hardware / model configuration ---------------------------------------

SERIAL_PORT = "COM10"
BAUD_RATE = 115200
MODEL_PATH = "hand_landmarker.task"

# Flip the camera frame horizontally (selfie/mirror view). See module
# docstring above for why this matters for hand-label accuracy.
MIRROR_CAMERA = True

# --- Starting position (arm starts here after go_zero, then moves relatively) --

START_X = 235.55
START_Y = 0.0
START_Z = 130.0

# --- Safe travel limits (mm) — clamp the target position to stay in range ---

X_MIN, X_MAX = 150.0, 300.0
Y_MIN, Y_MAX = -250.0, 250.0
Z_MIN, Z_MAX = 60.0, 130.0

# --- Gesture / geometry thresholds -----------------------------------------

# Hand open/closed threshold: ratio of (fingertip-to-wrist) / (knuckle-to-wrist).
# Below this ratio = closed fist. Used for the right hand's suction gesture.
CLOSED_HAND_RATIO_THRESHOLD = 1.1

# Finger-extended threshold: a finger counts as "extended" when its tip is
# this many times farther from the wrist than its middle knuckle (pip/ip).
# Used for the left hand's thumb/pinky directional gesture.
FINGER_EXTENDED_RATIO = 1.2

# If True, "thumb extended" / "pinky extended" also require the OTHER
# fingers to be curled (a clean, exclusive gesture). If False, it only
# checks whether that one finger is extended, ignoring the rest of the hand.
REQUIRE_OTHER_FINGERS_CURLED = True

# --- Right-hand incremental-motion tuning -----------------------------------

# How many mm the target position shifts per pixel of frame-to-frame wrist
# movement / hand-size change.
# Bumped up again: 1.00 stays for Z, X pushed from 0.70 -> 1.10 for a much
# more responsive forward/backward (X) reach control.
SENSITIVITY_Z_MM_PER_PX = 1.00   # left wrist vertical movement -> Z
SENSITIVITY_X_MM_PER_PX = 1.10   # left hand size change -> X

# Positive HAND_SIZE_SIGN means "hand looks bigger (closer to camera) ->
# arm moves toward X_MAX (extends out)". Flip to -1 if that feels backwards.
HAND_SIZE_SIGN = 1

# Dead zones: ignore movement smaller than this (in pixels) to avoid drift
# from camera/detection noise when the hand is basically still.
DEAD_ZONE_WRIST_PX = 4
DEAD_ZONE_SIZE_PX = 3

# --- Left-hand directional-switch tuning ------------------------------------

# How fast Y moves (mm/sec) while the thumb-only or pinky-only gesture is
# held. Speed-based (not per-frame) so it's independent of camera frame rate.
LEFT_HAND_Y_SPEED_MM_PER_SEC = 60.0

# --- Shared motion / serial tuning ------------------------------------------

# Minimum change (mm) in the target position required before we bother
# sending a new set_coords command.
MIN_MOVE_DELTA_MM = 1.5

# Minimum time (s) between move commands sent over serial.
MOVE_INTERVAL = 0.1

# Arm movement speed passed to set_coords.
MOVE_SPEED = 50

# --- Calibration -------------------------------------------------------

# How long (s) both hands must be held open, simultaneously, before
# calibration locks in and the arm moves to its starting position. A short
# hold (rather than a single frame) avoids accidentally triggering
# calibration while the user is still getting into position.
CALIBRATION_HOLD_SECONDS = 1.5

# --- Landmark indices --------------------------------------------------

WRIST_IDX = 0

THUMB_TIP_IDX, THUMB_IP_IDX = 4, 3
INDEX_TIP_IDX, INDEX_PIP_IDX, INDEX_MCP_IDX = 8, 6, 5
MIDDLE_TIP_IDX, MIDDLE_PIP_IDX, MIDDLE_MCP_IDX = 12, 10, 9
RING_TIP_IDX, RING_PIP_IDX = 16, 14
PINKY_TIP_IDX, PINKY_PIP_IDX = 20, 18

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
    """Build a two-hand MediaPipe HandLandmarker for static-image mode."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def is_hand_closed(points, wrist_idx=WRIST_IDX, tip_idx=INDEX_TIP_IDX, mcp_idx=INDEX_MCP_IDX):
    """
    Determine whether the hand is closed (fist) based on index-finger geometry.
    Used for the right hand's suction on/off gesture.
    """
    wrist = points[wrist_idx]
    index_tip = points[tip_idx]
    index_mcp = points[mcp_idx]

    tip_to_wrist = math.hypot(index_tip[0] - wrist[0], index_tip[1] - wrist[1])
    mcp_to_wrist = math.hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])

    hand_open_ratio = tip_to_wrist / (mcp_to_wrist + 1e-6)
    return hand_open_ratio < CLOSED_HAND_RATIO_THRESHOLD


def is_finger_extended(points, tip_idx, pip_idx, wrist_idx=WRIST_IDX, ratio=FINGER_EXTENDED_RATIO):
    """
    Generic finger-extended check: true when the fingertip is meaningfully
    farther from the wrist than that finger's middle knuckle is. Works for
    any finger (including the thumb, using its IP joint in place of a PIP)
    and is roughly orientation-independent since it's based on radial
    distance from the wrist rather than absolute screen direction.
    """
    wrist = points[wrist_idx]
    tip = points[tip_idx]
    pip = points[pip_idx]

    tip_to_wrist = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
    pip_to_wrist = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])

    return tip_to_wrist > pip_to_wrist * ratio


def classify_left_hand_gesture(points):
    """
    Classify the left hand's gesture into one of: "thumb", "pinky", "neutral".

    - "thumb": thumb extended (and, if REQUIRE_OTHER_FINGERS_CURLED, all
      other fingers curled) -> move arm right (+Y).
    - "pinky": pinky extended (and, if REQUIRE_OTHER_FINGERS_CURLED, all
      other fingers curled) -> move arm left (-Y).
    - "neutral": closed fist / anything else -> no Y movement.
    """
    thumb_ext = is_finger_extended(points, THUMB_TIP_IDX, THUMB_IP_IDX)
    index_ext = is_finger_extended(points, INDEX_TIP_IDX, INDEX_PIP_IDX)
    middle_ext = is_finger_extended(points, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX)
    ring_ext = is_finger_extended(points, RING_TIP_IDX, RING_PIP_IDX)
    pinky_ext = is_finger_extended(points, PINKY_TIP_IDX, PINKY_PIP_IDX)

    if REQUIRE_OTHER_FINGERS_CURLED:
        others_curled_for_thumb = not (index_ext or middle_ext or ring_ext or pinky_ext)
        others_curled_for_pinky = not (index_ext or middle_ext or ring_ext or thumb_ext)
        if thumb_ext and others_curled_for_thumb:
            return "thumb"
        if pinky_ext and others_curled_for_pinky:
            return "pinky"
        return "neutral"
    else:
        if thumb_ext:
            return "thumb"
        if pinky_ext:
            return "pinky"
        return "neutral"


def hand_size_metric(points, wrist_idx=WRIST_IDX, ref_idx=MIDDLE_MCP_IDX):
    """
    Distance (px) between wrist and middle-finger knuckle.
    Used as a rough depth proxy for the right hand's X (reach) control.
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


def draw_hand_overlay(img, points, label=None):
    """Draw hand skeleton connections, landmark points, and an optional label."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)
    if label:
        wrist = points[WRIST_IDX]
        cv2.putText(img, label, (wrist[0] - 20, wrist[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)


def wait_for_calibration(cap, detector):
    """
    Block until the user shows both hands open simultaneously, held for
    CALIBRATION_HOLD_SECONDS, to confirm they're ready to start.

    This is the "lock in the base position" step: nothing moves until the
    user deliberately signals readiness, so the arm doesn't jump to its
    starting coordinates the instant the script launches.

    Returns True once calibration succeeds, or False if the user pressed
    'q' to quit before calibrating.
    """
    hold_start = None

    while True:
        success, img = cap.read()
        if not success:
            continue

        if MIRROR_CAMERA:
            img = cv2.flip(img, 1)

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        hands_detected = 0
        both_open = False

        if results.hand_landmarks:
            hands_detected = len(results.hand_landmarks)
            all_open = True
            for hand_landmarks in results.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                draw_hand_overlay(img, points)
                if is_hand_closed(points):
                    all_open = False
            both_open = hands_detected >= 2 and all_open

        now = time.time()
        if both_open:
            if hold_start is None:
                hold_start = now
            held_for = now - hold_start
            remaining = max(0.0, CALIBRATION_HOLD_SECONDS - held_for)
            cv2.putText(img, f"Hold both hands open... {remaining:.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if held_for >= CALIBRATION_HOLD_SECONDS:
                cv2.putText(img, "Calibrated!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("ultraArm Two-Hand Control", img)
                cv2.waitKey(400)  # brief pause so the confirmation is visible
                return True
        else:
            hold_start = None
            cv2.putText(img, "Show BOTH hands open to calibrate", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if 0 < hands_detected < 2:
                cv2.putText(img, f"({hands_detected}/2 hands detected)", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ultraArm Two-Hand Control", img)
        if cv2.waitKey(1) == ord("q"):
            return False


def main():
    # --- Initialize robot arm ---
    mc = ultraArmP340(SERIAL_PORT, BAUD_RATE)
    mc.go_zero()

    # --- Initialize hand detector ---
    detector = create_hand_detector(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    # --- Calibration: wait for the user to show both hands open before ---
    # --- moving the arm to its starting position. -------------------------
    calibrated = wait_for_calibration(cap, detector)
    if not calibrated:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        return

    # Internally-tracked target position (since set_coords needs absolute
    # coordinates, but we're driving it with relative/discrete hand control).
    # This is only moved to now, after calibration locks it in.
    target_x, target_y, target_z = START_X, START_Y, START_Z
    mc.set_coords([target_x, target_y, target_z], MOVE_SPEED)

    # Ensure suction starts OFF, matching current_gpio_state below.
    mc.set_gpio_state(0)

    # Previous-frame values for the LEFT hand's relative X/Z control.
    prev_depth_wrist_px = None
    prev_depth_hand_size = None

    # State tracked to avoid redundant/noisy serial commands.
    current_gpio_state = 0  # last suction state sent (0=off, 1=on); starts OFF post-calibration
    last_sent_pos = (target_x, target_y, target_z)
    last_move_time = time.time()
    last_frame_time = time.time()  # used for speed-based right-hand Y control

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            if MIRROR_CAMERA:
                img = cv2.flip(img, 1)

            h, w, _ = img.shape

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            # MediaPipe expects RGB input.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            results = detector.detect(mp_image)

            depth_hand_seen = False
            direction_hand_seen = False

            if results.hand_landmarks:
                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    # Top handedness classification for this detected hand.
                    handedness_label = results.handedness[i][0].category_name  # "Left" or "Right"

                    if handedness_label == "Left":
                        depth_hand_seen = True
                        wrist_px = points[WRIST_IDX]
                        size_px = hand_size_metric(points)

                        # --- Suction control (open/closed hand -> GPIO) ---
                        hand_closed = is_hand_closed(points)
                        new_gpio_state = 1 if hand_closed else 0
                        if new_gpio_state != current_gpio_state:
                            current_gpio_state = new_gpio_state
                            mc.set_gpio_state(current_gpio_state)
                            print(f"Suction: {'ON (1)' if current_gpio_state == 1 else 'OFF (0)'}")

                        # --- Relative Z (vertical wrist movement) and X (hand size) ---
                        if prev_depth_wrist_px is not None and prev_depth_hand_size is not None:
                            dy_px = wrist_px[1] - prev_depth_wrist_px[1]
                            dsize_px = size_px - prev_depth_hand_size

                            dy_px = apply_dead_zone(dy_px, DEAD_ZONE_WRIST_PX)
                            dsize_px = apply_dead_zone(dsize_px, DEAD_ZONE_SIZE_PX)

                            target_z += -dy_px * SENSITIVITY_Z_MM_PER_PX  # up (smaller y) -> +Z
                            target_x += HAND_SIZE_SIGN * dsize_px * SENSITIVITY_X_MM_PER_PX

                            target_x = clamp(target_x, X_MIN, X_MAX)
                            target_z = clamp(target_z, Z_MIN, Z_MAX)

                        prev_depth_wrist_px = wrist_px
                        prev_depth_hand_size = size_px

                        draw_hand_overlay(img, points, label="L")
                        state_str = "CLOSED" if hand_closed else "OPEN"
                        cv2.putText(img, f"Left: {state_str} | Suction: {new_gpio_state}",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    elif handedness_label == "Right":
                        direction_hand_seen = True
                        gesture = classify_left_hand_gesture(points)

                        # Speed-based Y nudge: framerate-independent movement
                        # while the gesture is held.
                        if gesture == "thumb":
                            target_y += LEFT_HAND_Y_SPEED_MM_PER_SEC * dt
                        elif gesture == "pinky":
                            target_y -= LEFT_HAND_Y_SPEED_MM_PER_SEC * dt

                        target_y = clamp(target_y, Y_MIN, Y_MAX)

                        draw_hand_overlay(img, points, label="R")
                        cv2.putText(img, f"Right gesture: {gesture.upper()}",
                                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            if not depth_hand_seen:
                # Right hand left the frame: reset its relative-tracking
                # baseline so we don't jump when it reappears.
                prev_depth_wrist_px = None
                prev_depth_hand_size = None

            # Throttle serial writes: only send if enough time has passed
            # AND the position changed enough to matter.
            if now - last_move_time > MOVE_INTERVAL:
                moved_enough = math.dist(
                    (target_x, target_y, target_z), last_sent_pos
                ) >= MIN_MOVE_DELTA_MM
                if moved_enough:
                    mc.set_coords([target_x, target_y, target_z], MOVE_SPEED)
                    last_sent_pos = (target_x, target_y, target_z)
                    last_move_time = now

            cv2.putText(img, f"Target X:{target_x:.1f} Y:{target_y:.1f} Z:{target_z:.1f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("ultraArm Two-Hand Control", img)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()