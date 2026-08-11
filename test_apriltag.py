"""Standalone AprilTag detection test -- isolated from grid_nav.py/ROS2.

Opens the USB camera directly, runs pupil_apriltags on every frame, and
prints what it finds (ALL tags, not just id 4, so you can tell "detector
finds nothing at all" apart from "detector finds tags but not id 4").
Also saves an annotated snapshot to apriltag_debug.jpg every frame so you
can inspect it (scp it off the Pi, or open it in a file browser) even
with no display attached.

Usage:
    python test_apriltag.py [device_index]

Defaults to device index 0. Press Ctrl+C to stop.
"""
import sys
import time

import cv2

try:
    from pupil_apriltags import Detector
except ImportError:
    print("ERROR: pupil_apriltags is not importable in this Python environment.")
    print("Run:  pip install pupil-apriltags")
    sys.exit(1)

DEVICE_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WIDTH = 640
HEIGHT = 480
TAG_FAMILY = 'tag36h11'
TAG_ID = 4
TAG_SIZE_CM = 16.0
FX = 600.0
FY = 600.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0

print(f"Opening camera device index {DEVICE_INDEX} at {WIDTH}x{HEIGHT}...")
cap = cv2.VideoCapture(DEVICE_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
if not cap.isOpened():
    print(f"ERROR: could not open camera at device index {DEVICE_INDEX}")
    sys.exit(1)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera opened. Actual resolution reported: {actual_w:.0f}x{actual_h:.0f}")
if int(actual_w) != WIDTH or int(actual_h) != HEIGHT:
    print("WARNING: actual resolution differs from the WIDTH/HEIGHT this script assumes -- "
          "CX/CY (image center) will be wrong, which can hurt pose estimation (not raw detection).")

detector = Detector(families=TAG_FAMILY, quad_decimate=1.0)
print(f"Detector ready: family={TAG_FAMILY}, watching for id={TAG_ID}, tag_size={TAG_SIZE_CM}cm")
print("Press Ctrl+C to stop.\n")

frame_count = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: frame grab failed")
            time.sleep(0.2)
            continue
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(FX, FY, CX, CY), tag_size=TAG_SIZE_CM / 100.0,
        )

        if not detections:
            print(f"[frame {frame_count}] no tags detected at all")
        else:
            ids_found = [d.tag_id for d in detections]
            print(f"[frame {frame_count}] found {len(detections)} tag(s), ids={ids_found}")

        annotated = frame.copy()
        for det in detections:
            corners = det.corners.astype(int)
            color = (0, 255, 0) if det.tag_id == TAG_ID else (0, 165, 255)
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(annotated, p1, p2, color, 2)
            center = tuple(det.center.astype(int))
            cv2.circle(annotated, center, 4, color, -1)

            if det.tag_id == TAG_ID:
                tx, ty, tz = (float(v) for v in det.pose_t.flatten())
                distance_cm = (tx * tx + ty * ty + tz * tz) ** 0.5 * 100.0
                label = f"id={det.tag_id} dist={distance_cm:.1f}cm"
                print(f"    -> MATCH id={TAG_ID}: distance={distance_cm:.1f}cm, "
                      f"pose_t=({tx:.3f}, {ty:.3f}, {tz:.3f})m")
            else:
                label = f"id={det.tag_id}"
            cv2.putText(annotated, label, (center[0] - 20, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite('apriltag_debug.jpg', annotated)

        time.sleep(0.2)  # ~5Hz, matches APRILTAG_FPS in grid_nav.py
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()
