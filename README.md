# ultraarm-vision

Control a **myCobot ultraArm P340** with your hands, using only a webcam.

The script tracks both hands with MediaPipe and turns gestures into joint-angle commands (J1/J2/J3) and suction on/off. No gloves, markers, or depth camera are needed.

## How it works

1. A webcam frame is captured and mirrored (selfie view).
2. MediaPipe's HandLandmarker detects up to two hands and labels each as Left or Right.
3. Each hand's landmarks are turned into a gesture.
4. Gestures update internally tracked target joint angles, which are clamped to safe limits.
5. Updated angles are sent to the arm over serial with `set_angles`, throttled to avoid flooding the port.

## Controls

### Right hand: joint switches (neutral = closed fist)

| Gesture | Action |
|---|---|
| Thumb extended, other fingers curled | Rotate **J1** (base) in the positive direction |
| Pinky extended, other fingers curled | Rotate **J1** (base) in the negative direction |
| Index finger extended, others curled | Increase **J3** (elbow) |
| Index + middle extended, others curled | Decrease **J3** (elbow) |
| Closed fist | No movement |

Movement is speed-based (degrees per second while the gesture is held), so it doesn't depend on camera frame rate.

### Left hand: continuous control

| Gesture | Action |
|---|---|
| Move wrist up / down | Nudge **J2** (shoulder) up / down |
| Closed fist | Suction **ON** |
| Open hand | Suction **OFF** |

**J4** (accessory) has an unlimited range and is not mapped to any gesture. It stays at its starting angle.

### Calibration

On launch, nothing moves until you hold **both hands open** in view for 1.5 seconds. This prevents the arm from jumping the moment the script starts. Press `q` at any time to quit.

## Joint limits

Targets are clamped to these ranges:

| Joint | Range |
|---|---|
| J1 | -150° to +170° |
| J2 | -20° to +90° |
| J3 | -5° to +70° |
| J4 | Unlimited (not driven) |

## Requirements

- Python 3.9+
- A myCobot ultraArm P340 connected over USB serial
- A webcam
- Python packages:

```bash
pip install opencv-python mediapipe pymycobot
```

- The MediaPipe hand landmarker model, saved as `hand_landmarker.task` next to the script:

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Setup and usage

1. Connect the arm and find its serial port (for example `COM11` on Windows, `/dev/ttyUSB0` on Linux).
2. Edit the configuration at the top of `two_hand_gesture_control_joints.py`:

   ```python
   SERIAL_PORT = "COM11"
   BAUD_RATE = 115200
   MODEL_PATH = "hand_landmarker.task"
   ```

3. Clear the workspace around the arm.
4. Run:

   ```bash
   python two_hand_gesture_control_joints.py
   ```

5. The arm homes with `go_zero()`. Show both hands open until calibration completes, then start gesturing.

## Configuration

All tuning constants are at the top of the script.

| Constant | Purpose |
|---|---|
| `MIRROR_CAMERA` | Flip the frame horizontally. If Left/Right hands seem swapped, set to `False`. |
| `START_J1`–`START_J4` | Joint angles the arm moves to after calibration |
| `J1_MIN/MAX`, `J2_MIN/MAX`, `J3_MIN/MAX` | Safe joint limits |
| `RIGHT_HAND_J1_SPEED_DEG_PER_SEC` | J1 rotation speed while a gesture is held |
| `RIGHT_HAND_J3_SPEED_DEG_PER_SEC` | J3 speed while a gesture is held |
| `SENSITIVITY_J2_DEG_PER_PX` | Degrees of J2 per pixel of wrist movement |
| `DEAD_ZONE_WRIST_PX` | Ignore wrist jitter smaller than this |
| `FINGER_EXTENDED_RATIO` | How far a fingertip must be from the wrist to count as extended |
| `CLOSED_HAND_RATIO_THRESHOLD` | Threshold for fist vs. open hand |
| `REQUIRE_OTHER_FINGERS_CURLED` | Require a clean, exclusive gesture (recommended) |
| `MOVE_INTERVAL` / `MIN_MOVE_DELTA_DEG` | Serial throttling |
| `MOVE_SPEED` | Arm speed passed to `set_angles` |
| `CALIBRATION_HOLD_SECONDS` | How long both hands must be held open to calibrate |

## Tips and troubleshooting

- **Left and right are swapped:** toggle `MIRROR_CAMERA`.
- **A joint moves the wrong way:** swap the `+=` and `-=` for that gesture in the main loop, or negate the relevant speed/sensitivity constant. The direction of positive rotation depends on the arm's mounting, so check each joint at low speed first.
- **Gestures misfire:** use even lighting, keep your hand fully in frame, and tune `FINGER_EXTENDED_RATIO`. Keeping `REQUIRE_OTHER_FINGERS_CURLED = True` helps avoid accidental triggers.
- **J2 drifts or jitters:** increase `DEAD_ZONE_WRIST_PX` or lower `SENSITIVITY_J2_DEG_PER_PX`.
- **Arm movement is choppy:** lower `MOVE_INTERVAL` or `MIN_MOVE_DELTA_DEG`, but expect more serial traffic.

## Safety

- Keep the workspace clear and your hand near the power switch the first time you run this.
- Start with a low `MOVE_SPEED` and low joint speeds, then increase gradually.
- Joint limits in the script are software clamps only. Confirm them against your own setup and attached accessories before relying on them.
- If hand tracking is lost, the arm holds its last target rather than returning home.

## Project structure

```
ultraarm-vision/
├── two_hand_gesture_control_joints.py   # main script
├── hand_landmarker.task                 # MediaPipe model (download separately)
└── README.md
```

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for hand landmark detection
- [pymycobot](https://github.com/elephantrobotics/pymycobot) for arm control
- [Elephant Robotics](https://www.elephantrobotics.com/) for the myCobot ultraArm P340
- [Red Gutz Trading](https://www.redgutz.com/) for letting us use the myCobot UltraArm P340
