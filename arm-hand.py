import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymycobot import ultraArmP340

# --- 1. Initialize MyCobot ultraArmP340 ---
mc = ultraArmP340("COM11", 115200)
mc.go_zero()

# Fixed X and Y coordinates (from your script parameters)
FIXED_X = 235.55
FIXED_Y = 0.0

# Minimum and maximum Z height limits for ultraArm
MIN_Z = 60.0
MAX_Z = 130.0

# --- 2. Initialize MediaPipe Hand Detector ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Track single hand controlling the arm
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]

cap = cv2.VideoCapture(0)

# Track states to minimize unnecessary serial commands
current_gpio_state = None
last_target_z = None

# Time threshold to prevent flooding serial port with coordinate commands
last_move_time = time.time()
MOVE_INTERVAL = 0.1  # Send position update every 100ms max

while True:
    success, img = cap.read()
    if not success:
        break

    h, w, _ = img.shape

    # Convert image format for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect hand landmarks
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Wrist and finger keypoints
        wrist = points[0]
        index_tip = points[8]
        index_mcp = points[5]

        # -------------------------------------------------------------
        # 1. Open / Closed Hand Detection (GPIO Control)
        # -------------------------------------------------------------
        tip_to_wrist = math.hypot(index_tip[0] - wrist[0], index_tip[1] - wrist[1])
        mcp_to_wrist = math.hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])

        # Ratio normalized against hand size (works regardless of hand gesture/distance)
        hand_open_ratio = tip_to_wrist / (mcp_to_wrist + 1e-6)
        is_closed = hand_open_ratio < 1.1

        new_gpio_state = 1 if is_closed else 0
        if new_gpio_state != current_gpio_state:
            current_gpio_state = new_gpio_state
            mc.set_gpio_state(current_gpio_state)
            print(f"Gripper State: {'CLOSED (1)' if current_gpio_state == 1 else 'OPEN (0)'}")

        # -------------------------------------------------------------
        # 2. Hand Vertical Movement Mapping (Z-Height Control)
        # -------------------------------------------------------------
        # Wrist Y normalized coordinate in frame (0.0 at top, 1.0 at bottom)
        wrist_y_norm = hand_landmarks[0].y

        # Camera frame Y ranges from top (0.2) to bottom (0.8) for comfortable hand height
        # Higher hand position (lower Y value) -> High Z (130)
        # Lower hand position (higher Y value) -> Low Z (60)
        norm_clamped = max(0.4, min(0.8, wrist_y_norm))
        normalized_height = (0.8 - norm_clamped) / (0.8 - 0.4)  # Maps 0.2->1.0 (top) and 0.8->0.0 (bottom)

        target_z = round(MIN_Z + normalized_height * (MAX_Z - MIN_Z), 1)

        # Send movement commands at fixed intervals if position changed significantly
        if time.time() - last_move_time > MOVE_INTERVAL:
            if last_target_z is None or abs(target_z - last_target_z) >= 3.0:
                mc.set_coords([FIXED_X, FIXED_Y, target_z], 50)
                last_target_z = target_z
                last_move_time = time.time()

        # -------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

        # On-screen display
        state_str = "CLOSED" if is_closed else "OPEN"
        cv2.putText(img, f"Hand: {state_str} | GPIO: {new_gpio_state}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f"Target Z Height: {target_z} mm", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ultraArm Hand Control", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()