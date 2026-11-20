import cv2
import mediapipe as mp
from pathlib import Path
import threading
import time
import numpy as np
import math

# =====================================
# MODEL LOCATION
# =====================================

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# =====================================
# SHARED RESULT STORAGE
# =====================================

latest_result = {"result": None}
result_lock = threading.Lock()


def result_callback(result, output_image, timestamp_ms):
    with result_lock:
        latest_result["result"] = result


# =====================================
# MEDIAPIPE SETUP
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

# =====================================
# HAND CONNECTIONS
# =====================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# =====================================
# CAMERA
# =====================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Could not open webcam")
    raise SystemExit

print("AirDraw Started! Press Q to Quit")

# =====================================
# DRAWING VARIABLES
# =====================================

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

prev_x = None
prev_y = None

current_color = (0, 0, 255)  # Red

brush_thickness = 5
colors = [
    (0,0,255),      # Red
    (0,255,0),      # Green
    (255,0,0),      # Blue
    (0,255,255),    # Yellow
    (255,255,255)   # White
]

color_names = [
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "WHITE"
]

color_index = 0

last_pinch_time = 0
open_hand_start = None

cv2.namedWindow("AirDraw", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "AirDraw",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# =====================================
# MAIN LOOP
# =====================================

with HandLandmarker.create_from_options(options) as landmarker:

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))

            # -----------------------------
            # Detection Frame
            # -----------------------------

            small_frame = cv2.resize(frame, (320, 240))

            rgb_frame = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = time.perf_counter_ns() // 1_000_000

            landmarker.detect_async(
                mp_image,
                timestamp_ms
            )

            # -----------------------------
            # Fetch Latest Result
            # -----------------------------

            with result_lock:
                result = latest_result["result"]

            # -----------------------------
            # Hand Detected
            # -----------------------------

            if result is not None and result.hand_landmarks:

                for hand in result.hand_landmarks:

                    points = []

                    for landmark in hand:

                        x = int(
                            landmark.x * frame.shape[1]
                        )

                        y = int(
                            landmark.y * frame.shape[0]
                        )

                        points.append((x, y))

                    if len(points) < 21:
                        continue

                    # ==========================
                    # Finger State Detection
                    # ==========================

                    index_up = points[8][1] < points[6][1]
                    middle_up = points[12][1] < points[10][1]
                    ring_up = points[16][1] < points[14][1]
                    pinky_up=points[20][1] < points[18][1]
                    

                    # ==========================
                    # FIST = CLEAR CANVAS
                    # ==========================

                    fist = (
                        points[8][1] > points[6][1] and
                        points[12][1] > points[10][1] and
                        points[16][1] > points[14][1] and
                        points[20][1] > points[18][1]
                    )

                    if fist:

                       eraser_x = points[0][0]
                       eraser_y = points[0][1]

                       cv2.circle(
                       canvas,
                       (eraser_x, eraser_y),
                        40,
                       (0, 0, 0),
                       -1
                       )

                       cv2.circle(
                         frame,
                         (eraser_x, eraser_y),
                          40,
                         (255,255,255),
                          2
                         )

                    # ==========================
                    # DRAW MODE
                    # ==========================

                    draw_mode = (
                        index_up and
                        not middle_up and
                        not ring_up and
                        not pinky_up
                    )

                    x, y = points[8]
                    thumb_x, thumb_y = points[4]

                    distance = math.hypot(
                       x - thumb_x,
                       y - thumb_y
                    )

                    current_time = time.time()
                    peace_sign = (
                    index_up and
                    middle_up and
                    not ring_up and
                    not pinky_up
                    )

                    if peace_sign:

                       if current_time - last_pinch_time > 0.7:

                          color_index += 1

                          if color_index >= len(colors):
                            color_index = 0

                          current_color = colors[color_index]

                          last_pinch_time = current_time

                    open_hand = (
                      index_up and
                      middle_up and
                      ring_up and
                      pinky_up and
                      not fist
                    )
                    print(open_hand)
                    current_time=time.time()
                    if open_hand:

                      if open_hand_start is None:
                        open_hand_start = current_time

                      elapsed = current_time - open_hand_start

                      cv2.putText(
                          frame,
                          f"HOLD OPEN: {elapsed:.1f}s",
                         (20,120),
                         cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                        (0,0,255),
                          2
                         )

                      if elapsed >= 2:

                        canvas[:] = 0
                        open_hand_start = None

                    else:
                       open_hand_start = None


                

                    # ==========================
                    # DRAWING
                    # ==========================
                if draw_mode:

                        if prev_x is not None:

                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (x, y),
                                current_color,
                                brush_thickness
                            )

                        prev_x, prev_y = x, y

                else:

                        prev_x = None
                        prev_y = None

                    # ==========================
                    # HAND CONNECTIONS
                    # ==========================

                for start, end in connections:

                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # ==========================
                    # HAND POINTS
                    # ==========================

                for point in points:

                        cv2.circle(
                            frame,
                            point,
                            5,
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Show Your Hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # =================================
            # MERGE CANVAS
            # =================================

            frame = cv2.add(frame, canvas)

            # =================================
            # COLOR MENU
            # =================================

            cv2.rectangle(frame, (0, 0), (128, 60), (0, 0, 255), -1)
            cv2.rectangle(frame, (128, 0), (256, 60), (0, 255, 0), -1)
            cv2.rectangle(frame, (256, 0), (384, 60), (255, 0, 0), -1)
            cv2.rectangle(frame, (384, 0), (512, 60), (0, 255, 255), -1)
            cv2.rectangle(frame, (512, 0), (640, 60), (255, 255, 255), -1)
            button_width = 128

            start_x = color_index * button_width

            cv2.rectangle(
               frame,
               (start_x,0),
               (start_x + button_width,60),
               (255,255,255),
                4
            )

            cv2.putText(
                frame,
                "RED",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "GREEN",
                (140, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "BLUE",
                (280, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "YELLOW",
                (390, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "WHITE",
                (525, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            # =================================
            # DISPLAY
            # =================================
            cv2.circle(
               frame,
               (600,100),
                20,
              current_color,
              -1
            )

            cv2.putText(
               frame,
               color_names[color_index],
               (500,145),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               current_color,
               2
             )
            cv2.imshow(
                "AirDraw",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

print("AirDraw Closed")
