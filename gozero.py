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
