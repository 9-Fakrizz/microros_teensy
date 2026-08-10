"""
Grid navigation with live GUI -- runs on the Pi5 as a ROS2 node.

Robot model this script assumes:
  - Position (x, y) is tracked at the FRONT-LEFT wheel's contact point --
    which is also treated as one CORNER (not the center) of the robot's
    square ROBOT_SIZE_CM x ROBOT_SIZE_CM footprint for A* clearance
    purposes (see ROBOT_FOOTPRINT_RADIUS_CM / inflate_obstacles()).
  - Straight-line distance is measured from the LEFT wheel encoder
    (wheel_encoder data[0] -- see IMPORTANT note below).
  - Turning is done by a PIVOT about the left wheel: left wheel stays
    stopped, only the right wheel drives, so the tracked (x, y) point
    does not move during a turn -- only heading changes.
  - Movement is planned on a grid of GRID_SPACING_CM cells using A*
    (8-directional: N/S/E/W plus the 4 diagonals), so the robot CAN
    drive diagonally and will route around any cells marked as
    obstacles in the GUI. Obstacles are inflated by the robot's
    footprint radius before each A* search (see set_goal()), so the
    plan already accounts for the real 60x60cm body clearing everything,
    not just the single tracked reference point. The planned path is
    compressed into a sequence of straight-line legs (each a single
    rotate + drive), one per run of consecutive same-direction grid
    steps -- not one tiny hop per cell. If no path exists (goal fully
    blocked, even before inflation), the goal is rejected and logged.

IMPORTANT -- firmware encoder wiring:
  wheel_encoder data[0] must be the LEFT wheel's encoder for the
  position math here to be valid (see src/main.cpp setup(): whichever
  attachInterrupt() is active feeds data[0]). If you've swapped it to
  the right wheel for testing (per notebook_debug.txt), swap it back
  to the left wheel before running this script.

GUI:
  Served as a local web page (no display/X11 needed on the Pi) --
  a grid view shows the robot's tracked position (arrow = heading),
  its path so far, and the current goal. A form + "Go" button let you
  send a new goal (in cm) at any time, including while the robot is
  mid-move. A status readout shows raw IMU yaw and the current target
  heading in degrees, for debugging the IMU. A live MJPEG feed from a
  USB webcam (see CAMERA_DEVICE_INDEX) is also shown, if available, with
  a backup-camera-style distance HUD (see CameraRangefinder) and live
  obstacle detection boxes (see ObstacleDetector) drawn over it -- a
  top-anchored vertical edge (a leg or pillar) whose floor-contact point
  falls between the 50cm and 100cm VFOV lines is flagged as a detection.
  NOTE: detection is NOT currently wired into avoidance -- ObstacleWatcher
  (the safety supervisor that would hold the robot and pin a confirmed
  detection into the A* obstacle map) is deliberately PAUSED right now
  (see its docstring) while detection itself is still being tuned. The
  GUI's "Obstacle Avoidance Enabled" toggle and detection boxes are live,
  but nothing gets pinned/avoided automatically yet -- obstacles still
  need to be added manually (click a grid cell) for A* to route around
  them.

  Open it from any browser on the same network:
      http://<pi5-ip-address>:8080

Requires Flask (pip install flask) and OpenCV (sudo apt install
python3-opencv, or pip install opencv-python) in addition to your
ROS2 env.

Run (after sourcing your ROS2 setup):
    python3 grid_nav.py
"""

import heapq
import math
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

from flask import Flask, jsonify, request, Response

SCRIPT_VERSION = "v2.0 - web GUI"

# ---------------- Configuration ----------------
IMU_TOPIC = "/imu_data"
CMD_VEL_TOPIC = "/cmd_vel"
WHEEL_ENCODER_TOPIC = "/wheel_encoder"
ENCODER_INDEX_PRIMARY_PULSES = 0  # must be the LEFT wheel -- see module docstring

# Final calibrated value (see notebook_debug.txt for the full derivation
# history: 183.5 -> 118.4 -> 131.6 -> 122.3 -> 110 -> 105 -> this).
# Derived from 14 runs at 250cm/box (26250 commanded pulses each) with
# measured error averaging -5.79cm (undershoot), i.e. ~244.2cm real
# distance -> 26250 / 244.2 = ~107.5 pulses/cm. Accepted tolerance going
# forward is +-10cm at 250cm range (~4%); the raw run-to-run spread
# (-13cm to 0cm) is wider than the systematic bias this value corrects
# for, so don't expect this to zero out every individual run.
# heading_hold.py is NOT updated to this value -- only validated
# separately, at a different test distance.
PULSES_PER_CM = 112

FORWARD_SPEED = 0.20           # m/s, straight-line drive speed
ROTATE_SPEED = 0.20            # max commanded speed magnitude while pivoting

# If the robot pivots the WRONG way (heading error grows instead of
# shrinking) during testing, flip this to -1. Left wheel stays at 0
# regardless of this value -- it only affects rotation direction.
PIVOT_SIGN = 1

# During pivot, we send linear.x = cmd and angular.z = PIVOT_ANGULAR_SIGN * cmd.
# The firmware's differential mix is:
#   left  = (lin_x - ang_z) * INVERT
#   right = (lin_x + ang_z) * INVERT
# angular.z = +cmd zeroes "left" (lin_x - ang_z = 0); angular.z = -cmd
# zeroes "right" (lin_x + ang_z = 0). Testing showed +cmd actually stops
# the physical RIGHT wheel and drives the physical LEFT wheel -- backwards
# from what we want -- so this is set to -1 to zero the other term instead.
# Flip back to +1 if it turns out backwards again.
PIVOT_ANGULAR_SIGN = -1

# Flips which physical turn direction counts as "+Y". Was -1, but testing
# at the 5m-scale/0.25-speed setup showed Y now goes the wrong way again --
# flipped back to +1. Flip back to -1 if it turns out reversed again.
Y_AXIS_SIGN = 1

HEADING_TOLERANCE_DEG = 3.0    # stop pivoting once within this of target

# Rotate-phase PID: scales pivot speed down as heading error shrinks
# (instead of a constant speed followed by a hard stop at tolerance).
# Output is clamped to +/-ROTATE_SPEED and applied to BOTH linear.x and
# angular.z (see control_loop) -- that's what keeps the left wheel at
# exactly 0 regardless of the PID output's sign/magnitude.
# Gains lowered (was KP=0.8, KD=0.05) -- response was too aggressive/jerky.
ROTATE_KP = 0.5
ROTATE_KI = 0.0
ROTATE_KD = 0.03
ROTATE_MAX_INTEGRAL = 0.3
# PWM floor so the pivot doesn't stall out approaching zero error before
# actually reaching HEADING_TOLERANCE_DEG.
ROTATE_MIN_OUTPUT = 0.05

# Drive-phase PID: keeps the robot on its cardinal heading while driving
# straight. Kept small/clamped -- the ROTATE phase does the real turning,
# not this. Gains lowered (was KP=1.0, KD=0.1) -- same reason as ROTATE.
DRIVE_KP = 0.6
DRIVE_KI = 0.0
DRIVE_KD = 0.05
DRIVE_MAX_INTEGRAL = 0.3
MAX_ANGULAR_Z_HOLD = 0.20

# Testing toggle: while False, DRIVE phase sends angular.z = 0 (pure
# open-loop straight driving, no heading correction). Distance calibration
# is confirmed good now, and open-loop driving was letting the robot
# slowly curve off its heading -- re-enabled to correct that.
DRIVE_HEADING_HOLD_ENABLED = True

# Same left/right term-swap issue as PIVOT_ANGULAR_SIGN above was suspected
# to apply here too, so this was flipped to +1 from the naive -correction
# convention (borrowed from heading_hold.py). Testing showed +1 makes it
# WORSE -- aggressive correction that never settles at the setpoint, the
# exact "wrong sign fights itself" failure case -- so this is reverted to
# -1. The pivot-phase wiring swap does NOT seem to carry over to this
# phase's correction sign.
DRIVE_CORRECTION_SIGN = -1

LOOP_HZ = 20.0                 # control loop rate
GUI_HZ = 12.0                  # GUI poll/redraw rate

GRID_SPACING_CM = 50            # gridline spacing (box size), cosmetic + step-mode size
GRID_HALF_EXTENT_CM = 500       # initial view: +/- this many cm (10m x 10m total)

# Step-mode distance per box, for isolating the distance calibration by
# measuring one grid box at a time instead of a whole multi-box leg in one
# go. Defaults to matching the visual grid spacing.
STEP_SIZE_CM = GRID_SPACING_CM

# A* plans over the same GRID_SPACING_CM cells shown on the GUI grid.
# Search is bounded to the visible grid so an unreachable goal (e.g. fully
# walled off) fails fast instead of scanning an unbounded plane.
PLANNING_HALF_EXTENT_CELLS = GRID_HALF_EXTENT_CM // GRID_SPACING_CM
SQRT2 = math.sqrt(2.0)

# Robot footprint, for A* obstacle clearance: (x, y) is tracked at the
# FRONT-LEFT wheel, which is one CORNER of the robot's square body, not
# its center. Because that reference corner's position relative to the
# footprint's other three corners rotates with heading (and A* here plans
# over a static cell graph without per-cell heading awareness), obstacles
# are inflated by the worst-case distance from that corner to the
# footprint's farthest (diagonally opposite) corner -- the full diagonal,
# NOT half the side length -- so the real box clears every obstacle
# regardless of which of the 8 discrete headings the robot ends up facing
# while passing through a given area. Inflation is applied fresh at
# set_goal() time (see inflate_obstacles()); the raw self.obstacles set
# used for GUI display/toggling is never itself modified.
ROBOT_SIZE_CM = 60.0
ROBOT_FOOTPRINT_RADIUS_CM = ROBOT_SIZE_CM * SQRT2
ROBOT_INFLATION_CELLS = math.ceil(ROBOT_FOOTPRINT_RADIUS_CM / GRID_SPACING_CM)

# How far (cm) a confirmed obstacle's blocked footprint extends AWAY from
# the robot, starting at its measured near surface -- NOT centered on the
# single measured point. distance_cm is the obstacle's nearest edge (from
# the detection box's bottom edge), so blocking a box centered on that
# point would incorrectly treat clear floor between the robot and the
# obstacle as blocked. Instead the box's near edge sits at distance_cm and
# extends this far past it.
OBSTACLE_BOX_DEPTH_CM = 2 * GRID_SPACING_CM

# USB webcam streamed to the GUI as MJPEG over /video_feed. Device index
# matches OpenCV/V4L2 numbering (0 = /dev/video0). If you have more than
# one video device (e.g. a webcam plus some other UVC device), check
# `ls /dev/video*` and `v4l2-ctl --list-devices` on the Pi to find the
# right index.
CAMERA_DEVICE_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
CAMERA_JPEG_QUALITY = 80        # 0-100, higher = better quality/more bandwidth

# Real-world floor distances (cm) the backup-camera-style HUD draws a
# horizontal guide line for, once CameraRangefinder is calibrated. Each is
# projected to whatever pixel row it actually falls at (near = low in
# frame, far = near the horizon line) -- lines outside the visible frame
# for the current calibration are simply skipped.
GUIDE_DISTANCES_CM = [50, 100, 150, 200, 250, 300]

# Lightweight obstacle detector (ObstacleDetector) -- no trained model
# file, no HOG. VERTICAL-LINE pattern matching: testing with
# black_value_max=0 (black-line rejection effectively off) showed that a
# pillar or a human leg produces a strong, mostly-unbroken vertical Canny
# edge running from the TOP of the frame straight down to wherever it
# touches the floor -- unlike floor clutter/texture, which doesn't form
# a long continuous vertical line. So instead of generic "any big blob"
# contour detection, ObstacleDetector specifically looks for these tall,
# thin, top-anchored vertical edge segments, and estimates each one's
# distance from where its LOWEST point (floor-contact point) falls on
# CameraRangefinder's distance-per-row mapping.
DETECTION_FPS = 5.0                     # detection is heavier than streaming; runs at its own slower rate

# Ignore black/white straight line markings (floor tape, tile grout
# seams, thresholds) via HSV color segmentation, applied to the EDGE MAP
# before the vertical-line search runs: any Canny edge pixel that falls
# on a "black" (V below black_value_max) or "white" (S below
# white_sat_max AND V above white_value_min) region of the frame is
# erased first. All three are GUI/API-adjustable -- tune them live
# against the debug feed, since the right cutoff depends on actual
# ambient lighting. (NOTE: setting black_value_max too high can eat a
# real leg/pillar's own edge if it's dark -- that's what testing at 0
# revealed the vertical-line pattern in the first place.)
DETECTION_BLACK_VALUE_MAX = 90      # 0-255 HSV V; below this = "black"
DETECTION_WHITE_SAT_MAX = 40        # 0-255 HSV S; below this (AND V above WHITE_VALUE_MIN) = "white"
DETECTION_WHITE_VALUE_MIN = 200     # 0-255 HSV V; above this (AND S below WHITE_SAT_MAX) = "white"

# VERTICAL-LINE obstacle condition:
#   1. The edge must extend from near the top of the frame downward --
#      it starts within the top VERTICAL_TOP_MARGIN_FRACTION of the
#      downscaled frame's own height (fraction, so it scales with
#      DETECTION_DOWNSCALE automatically).
#   2. Its bottom endpoint (foot) must not be farther than the 100cm
#      VFOV line -- see SAFETY DISTANCE FILTER below, which reuses the
#      existing calibrated CameraRangefinder Y-to-distance mapping
#      unchanged. Nearer than the 50cm line is fine too -- it just gets
#      clamped to 50cm rather than rejected (see below).
# A small vertical morphological CLOSE (VERTICAL_CLOSE_KSIZE tall, in
# downscaled pixels) bridges small gaps first, so a real leg/pillar's
# otherwise-continuous edge isn't split into several short fragments by
# minor noise/blur breaks. Internal only, not GUI-exposed.
DETECTION_VERTICAL_TOP_MARGIN_FRACTION = 0.05
DETECTION_VERTICAL_CLOSE_KSIZE = 15

# SAFETY DISTANCE FILTER -- THE core obstacle condition: a vertical
# line's FOOT (its lowest point, full-frame pixel row) is converted to
# a real-world floor distance via CameraRangefinder.distance_for_row().
#   - Farther than DETECTION_OBJECT_MAX_DISTANCE_CM (behind the 100cm
#     line): the estimate is considered unreliable and the whole
#     detection is dropped -- not clamped, not reported.
#   - Nearer than DETECTION_OBJECT_MIN_DISTANCE_CM: NOT rejected. When an
#     obstacle is very close, its primary vertical edge tends to merge
#     with noise near the bottom of the frame, making the exact
#     close-range distance unreliable -- but "there's something very
#     close" is still real signal, so instead of dropping it the
#     reported distance is CLAMPED to DETECTION_OBJECT_MIN_DISTANCE_CM,
#     the closest fixed value this detector will ever report.
# Distinct from the old frame-crop band (removed) -- this filters
# individual detections by their OWN measured distance, not by cropping
# the source frame.
DETECTION_OBJECT_MIN_DISTANCE_CM = 50.0    # closest reported distance -- nearer detections are clamped to this, not dropped
DETECTION_OBJECT_MAX_DISTANCE_CM = 100.0   # farther than this (behind the 100cm line) is dropped as unreliable

# Performance: this downscale factor cuts the pixel count the Canny
# pipeline has to churn through -- keeps CPU/bandwidth low even with the
# /debug_feed stream running. Not exposed as GUI-adjustable.
DETECTION_DOWNSCALE = 0.5   # (0, 1.0]; e.g. 0.5 = quarter the pixels (half width x half height) before segmentation

# Locked-in default calibration so the HUD/obstacle detector work
# immediately at startup without re-calibrating through the GUI every
# run -- height=26cm, tilt derived from calibrating at 100cm (see
# notebook_debug.txt, "Camera HUD / distance rangefinder" section).
# Recalibrate via the GUI form any time the physical camera mount changes.
CAMERA_DEFAULT_HEIGHT_CM = 26.0
CAMERA_DEFAULT_TILT_DEG = 14.6

# Obstacle-confirmation safety supervisor (ObstacleWatcher): a single
# detection tick could be a glitch (lighting flicker, motion blur, a
# passing shadow), so a detection alone doesn't get pinned into the A*
# obstacle map immediately. Instead, the robot HOLDS (stops) for
# OBSTACLE_HOLD_DURATION_S seconds and counts how many separate ticks
# during that hold still see something in the 100-150cm band. Only if
# that count reaches OBSTACLE_CONFIRM_COUNT does it get pinned + trigger
# an A* replan; otherwise it's treated as a glitch and the interrupted
# leg just resumes. Both are live-adjustable from the GUI.
OBSTACLE_HOLD_DURATION_S = 5.0
OBSTACLE_CONFIRM_COUNT = 5
# -------------------------------------------------


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, current):
    d = target - current
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def normalize_angle(a):
    return angle_diff(a, 0.0)


# 8-connected neighbor offsets: (di, dj, step_cost). Orthogonal steps cost 1
# cell, diagonal steps cost sqrt(2) cells (true Euclidean distance between
# diagonally-adjacent cell centers).
_ASTAR_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
]


def _octile_heuristic(a, b):
    # Admissible heuristic for 8-directional movement with unit/sqrt(2)
    # costs. Plain Manhattan distance (|dx|+|dy|) overestimates the true
    # cost once diagonal moves are allowed (a diagonal step covers 2 cells
    # of Manhattan distance for sqrt(2) ~= 1.414 cost, not 2), which would
    # make A* not guaranteed to return the shortest path. This "octile"
    # distance is the exact cost of the optimal path on an obstacle-free
    # grid, so it's both admissible and consistent here.
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def astar_search(start_cell, goal_cell, obstacles):
    """8-directional A* over grid cells. obstacles is a set of (i, j)
    blocked cells. Returns a list of cells from start_cell to goal_cell
    (inclusive), or None if no path exists."""
    if start_cell == goal_cell:
        return [start_cell]

    limit = PLANNING_HALF_EXTENT_CELLS

    def in_bounds(c):
        return -limit <= c[0] <= limit and -limit <= c[1] <= limit

    open_heap = [(0.0, start_cell)]
    g_cost = {start_cell: 0.0}
    came_from = {}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for di, dj, step_cost in _ASTAR_NEIGHBORS:
            neighbor = (current[0] + di, current[1] + dj)
            if neighbor in closed or not in_bounds(neighbor) or neighbor in obstacles:
                continue
            if di != 0 and dj != 0:
                # Don't let a diagonal step cut through the corner formed by
                # two blocked orthogonal cells -- standard grid-A* rule so
                # the path never squeezes between two "walls" that aren't
                # actually passable.
                if (current[0] + di, current[1]) in obstacles and (current[0], current[1] + dj) in obstacles:
                    continue
            tentative = g_cost[current] + step_cost
            if tentative < g_cost.get(neighbor, float('inf')):
                g_cost[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + _octile_heuristic(neighbor, goal_cell), neighbor))

    return None


def path_to_legs(cell_path, cell_size_cm):
    """Compress a list of adjacent grid cells into ('move', ux, uy,
    distance_cm) legs -- one per run of consecutive same-direction steps,
    so a long straight or diagonal stretch becomes a single rotate+drive
    leg instead of one tiny hop per cell."""
    legs = []
    i = 1
    n = len(cell_path)
    while i < n:
        di = cell_path[i][0] - cell_path[i - 1][0]
        dj = cell_path[i][1] - cell_path[i - 1][1]
        steps = 1
        j = i + 1
        while j < n and (cell_path[j][0] - cell_path[j - 1][0], cell_path[j][1] - cell_path[j - 1][1]) == (di, dj):
            steps += 1
            j += 1
        step_cm = cell_size_cm * (SQRT2 if (di != 0 and dj != 0) else 1.0)
        norm = math.hypot(di, dj)
        legs.append(('move', di / norm, dj / norm, steps * step_cm))
        i = j
    return legs


def inflate_obstacles(obstacles, radius_cells):
    """Expand a set of blocked (i, j) cells outward by radius_cells in a
    circular stamp. Planning-time-only -- returns a NEW set, never mutates
    the original -- so A* can treat the robot as a single point while
    still guaranteeing its real ROBOT_SIZE_CM x ROBOT_SIZE_CM footprint
    clears every obstacle (see ROBOT_INFLATION_CELLS derivation above)."""
    if radius_cells <= 0:
        return set(obstacles)
    inflated = set()
    r2 = radius_cells * radius_cells
    for (bi, bj) in obstacles:
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if di * di + dj * dj <= r2:
                    inflated.add((bi + di, bj + dj))
    return inflated


class CameraStreamer:
    """Grabs frames from a USB webcam in a background thread and keeps the
    latest one JPEG-encoded and ready to serve. Decoupling capture from
    the Flask request handler means multiple browser tabs (or just slow
    HTTP writes) don't stall frame grabbing, and a client that hasn't
    polled recently always gets the freshest frame instead of a queued
    stale one."""

    def __init__(self, device_index, width, height, fps, jpeg_quality):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._cap = None
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_frame = None   # raw BGR ndarray, for ObstacleDetector -- avoids a JPEG decode round-trip
        self._running = False
        self._thread = None

    def start(self):
        """Returns True if the camera opened successfully."""
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        interval = 1.0 / self.fps
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        while self._running:
            start = time.monotonic()
            ok, frame = self._cap.read()
            if ok:
                ok2, buf = cv2.imencode('.jpg', frame, encode_params)
                if ok2:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._latest_frame = frame
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


class CameraRangefinder:
    """'Aim and measure' floor-distance calibration + a backup-camera-style
    perspective HUD (horizontal distance guide lines) for the camera feed.

    The crosshair drawn on the video feed sits dead-center of the frame,
    i.e. exactly on the camera's optical axis. To calibrate: physically
    point the robot/camera so the crosshair lines up with a floor mark at
    a KNOWN real-world distance, enter that distance plus the camera's
    mounted height above the floor, and calibrate. Because the crosshair
    point is on the optical axis by definition, no lens/FOV data is needed
    for THIS step -- simple right-triangle trig solves the camera's
    downward tilt angle directly:

        tan(tilt) = height / distance  =>  tilt = atan(height / distance)

    Pick a calibration distance you can position precisely and that lands
    somewhere in the middle of the frame, not right at the far/blurry edge
    -- a close, easy-to-align mark (e.g. 100cm) calibrates the tilt angle
    more precisely than a far one, since the same pixel-alignment error
    corresponds to a smaller angular error up close.

    The floor is flat but the camera's projection of it is NOT linear --
    equal steps of real-world distance get squeezed into shrinking bands
    of image row as you look toward the horizon (this is why a distant
    30cm gap looks like a sliver of pixels while the same 30cm right in
    front of the camera spans a big chunk of the frame). To map that
    correctly for the OTHER rows of the image (not just the crosshair
    row), we additionally need the camera's vertical field of view
    (vfov_deg) to get a pixel focal length:

        f_px = (frame_height_px / 2) / tan(vfov_deg / 2)

    Then for any real-world floor distance d, the row it projects to is:

        phi(d) = atan(height / d)              -- look-down angle to d
        y = frame_height_px/2 + f_px * tan(phi(d) - tilt)

    This is exactly what draws the perspective guide lines -- like a car's
    backup camera overlay -- showing where fixed real-world distances
    actually fall in the live image, which shrink together near the
    horizon just like the real floor does.
    """

    DEFAULT_VFOV_DEG = 60.0  # typical-ish USB webcam vertical FOV; tune via the GUI for your camera

    def __init__(self):
        self._lock = threading.Lock()
        # Pre-populated with the locked-in defaults (see CAMERA_DEFAULT_*
        # above) so the HUD/obstacle detector are usable immediately at
        # startup -- recalibrate via the GUI form to override these.
        self.height_cm = CAMERA_DEFAULT_HEIGHT_CM
        self.tilt_deg = CAMERA_DEFAULT_TILT_DEG
        self.vfov_deg = self.DEFAULT_VFOV_DEG

    def calibrate(self, height_cm, known_distance_cm, vfov_deg=None):
        if height_cm <= 0 or known_distance_cm <= 0:
            return False
        if vfov_deg is not None and vfov_deg <= 0:
            return False
        tilt_deg = math.degrees(math.atan(height_cm / known_distance_cm))
        with self._lock:
            self.height_cm = height_cm
            self.tilt_deg = tilt_deg
            if vfov_deg is not None:
                self.vfov_deg = vfov_deg
        return True

    @staticmethod
    def _row_for_distance(height_cm, tilt_deg, vfov_deg, distance_cm, frame_height_px):
        """Pixel row (0 = top) that real-world floor distance distance_cm
        projects to, or None if it falls outside the visible frame (too
        close/behind the camera, or beyond the horizon)."""
        if distance_cm <= 0:
            return None
        phi = math.atan(height_cm / distance_cm)
        theta = math.radians(tilt_deg)
        angle_offset = phi - theta
        if abs(angle_offset) >= math.radians(89.0):
            return None  # numerically unstable this close to the horizon/behind-camera limit
        f_px = (frame_height_px / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        y = frame_height_px / 2.0 + f_px * math.tan(angle_offset)
        if y < 0.0 or y > frame_height_px:
            return None
        return y

    def row_for_distance(self, distance_cm, frame_height_px):
        """Public entry point used by ObstacleDetector to crop the frame to
        only the region farther than a given real-world distance."""
        with self._lock:
            height_cm = self.height_cm
            tilt_deg = self.tilt_deg
            vfov_deg = self.vfov_deg
        if height_cm is None or tilt_deg is None:
            return None
        return self._row_for_distance(height_cm, tilt_deg, vfov_deg, distance_cm, frame_height_px)

    @staticmethod
    def _distance_for_row(height_cm, tilt_deg, vfov_deg, y_px, frame_height_px):
        """Inverse of _row_for_distance: real-world floor distance for a
        given pixel row, or None if that row looks above the horizon (never
        hits the floor) or calibration is missing."""
        theta = math.radians(tilt_deg)
        f_px = (frame_height_px / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        angle_offset = math.atan((y_px - frame_height_px / 2.0) / f_px)
        phi = theta + angle_offset
        if phi <= 0.0 or phi >= math.radians(89.5):
            return None
        return height_cm / math.tan(phi)

    def distance_for_row(self, y_px, frame_height_px):
        """Public entry point used by ObstacleDetector to estimate distance
        to a detected box from the pixel row of its floor-contact point
        (bottom edge of the bounding box)."""
        with self._lock:
            height_cm = self.height_cm
            tilt_deg = self.tilt_deg
            vfov_deg = self.vfov_deg
        if height_cm is None or tilt_deg is None:
            return None
        return self._distance_for_row(height_cm, tilt_deg, vfov_deg, y_px, frame_height_px)

    def bearing_deg_for_column(self, x_px, frame_width_px, frame_height_px):
        """Horizontal angle (deg) of a pixel column from the camera's
        forward boresight -- positive = to the right. Reuses the SAME
        pixel focal length as the vertical projection (square-pixel
        assumption: one focal length in pixels serves both axes; the
        horizontal and vertical FOVs only differ because frame width !=
        frame height), so no separate horizontal-FOV calibration is
        needed. Used to convert a detected box's horizontal position into
        a bearing for placing it on the world map."""
        with self._lock:
            tilt_deg = self.tilt_deg
            vfov_deg = self.vfov_deg
        if tilt_deg is None:
            return None
        f_px = (frame_height_px / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        angle = math.atan((x_px - frame_width_px / 2.0) / f_px)
        return math.degrees(angle)

    def get_snapshot(self, frame_height_px):
        with self._lock:
            height_cm = self.height_cm
            tilt_deg = self.tilt_deg
            vfov_deg = self.vfov_deg

        crosshair_distance_cm = None
        guide_lines = []
        if height_cm is not None and tilt_deg is not None and tilt_deg > 0:
            crosshair_distance_cm = height_cm / math.tan(math.radians(tilt_deg))
            for d in GUIDE_DISTANCES_CM:
                y = self._row_for_distance(height_cm, tilt_deg, vfov_deg, d, frame_height_px)
                if y is not None:
                    guide_lines.append({'distance_cm': d, 'y_frac': y / frame_height_px})

        return {
            'calibrated': tilt_deg is not None,
            'height_cm': height_cm,
            'tilt_deg': tilt_deg,
            'vfov_deg': vfov_deg,
            'crosshair_distance_cm': crosshair_distance_cm,
            'guide_lines': guide_lines,
        }


class ObstacleDetector:
    """No trained model file, no HOG. VERTICAL-EDGE obstacle detection
    built on the existing calibrated VFOV distance mapping
    (CameraRangefinder.distance_for_row() -- unchanged, reused as-is):

    1. Compute plain Canny edges on the full frame (the distance-band
       crop was tried and cancelled -- see notebook_debug.txt),
       downscaled by `downscale`. A small vertical morphological CLOSE
       bridges minor gaps first, so a real leg/pillar's otherwise-
       continuous edge isn't split into several short fragments by
       noise/blur breaks -- "strong/continuous edge lines."
    2. Find each edge line that starts near the TOP of the frame (within
       DETECTION_VERTICAL_TOP_MARGIN_FRACTION) and extends downward --
       a pillar or a human leg produces exactly this pattern, unlike
       floor clutter/texture.
    3. For each such line, take its LOWEST point (the "foot") and look
       up the real-world distance it represents via
       CameraRangefinder.distance_for_row() -- the existing calibrated
       Y-to-distance mapping, not touched here.
    4. Obstacle condition: accept unless that foot distance is farther
       than DETECTION_OBJECT_MAX_DISTANCE_CM (behind the 100cm VFOV
       line) -- then the detection is dropped as unreliable, not
       reported. Nearer than DETECTION_OBJECT_MIN_DISTANCE_CM (the 50cm
       line) is accepted, not dropped -- close range is where the
       primary edge tends to merge with noise at the very bottom of the
       frame, so rather than trust that noisy exact pixel row, the
       reported distance is CLAMPED to DETECTION_OBJECT_MIN_DISTANCE_CM,
       the closest fixed value ever reported. E.g. a leg's edge running
       from the top of the frame down past the 50cm line is still
       detected, just reported as "~50cm" rather than whatever
       (unreliable) closer number the raw pixel row implied.

    Black/white LINE color segmentation (floor tape, tile grout seams)
    is still computed as line_coverage_pct, a tuning/debug stat, but is
    NOT applied to erase edges -- a dark leg/pillar is exactly the kind
    of thing black_value_max would otherwise wipe out before the
    vertical search ever ran (confirmed with a real cv2 test; see
    notebook_debug.txt). The top-anchor condition above already rejects
    ordinary (short, wide) floor markings on shape alone.

    The frame served over the live debug stream (/debug_feed, multipart
    PNG -- not JPEG, see notebook_debug.txt: JPEG's lossy block
    compression was found to bleed rejected regions back into visibility
    on decode) is the vertical-closed edge map with a GREEN box drawn
    around each ACCEPTED candidate (foot at or nearer than the 100cm
    line, including near-clamped ones) -- candidates rejected for being
    behind the 100cm line get no box at all, so what's boxed on the
    debug feed always matches what's actually detected. All three HSV
    line thresholds are GUI/API-adjustable -- tune them against the live
    debug feed.

    Runs in its own background thread at DETECTION_FPS, reading the
    latest raw frame from a CameraStreamer.
    """

    def __init__(self, camera: 'CameraStreamer', rangefinder: CameraRangefinder,
                 fps=DETECTION_FPS,
                 black_value_max=DETECTION_BLACK_VALUE_MAX,
                 white_sat_max=DETECTION_WHITE_SAT_MAX,
                 white_value_min=DETECTION_WHITE_VALUE_MIN,
                 downscale=DETECTION_DOWNSCALE):
        self.camera = camera
        self.rangefinder = rangefinder
        self.fps = fps

        self._settings_lock = threading.Lock()
        self._black_value_max = black_value_max
        self._white_sat_max = white_sat_max
        self._white_value_min = white_value_min
        self._downscale = downscale

        self._lock = threading.Lock()
        self._detections = []
        self._line_coverage_pct = None    # % of ROI pixels classified as a black/white line
        self._object_coverage_pct = None  # % of ROI (contour) area classified as obstacle
        self._debug_png = None            # latest processed (line-masked) edge map, PNG-encoded
        self._running = False
        self._thread = None

    def set_black_value_max(self, value):
        """Live-adjustable -- 0-255 HSV V (brightness) cutoff. A pixel
        is rejected as "black" if its V is BELOW this."""
        value = int(value)
        if not (0 <= value <= 255):
            return False
        with self._settings_lock:
            self._black_value_max = value
        return True

    def get_black_value_max(self):
        with self._settings_lock:
            return self._black_value_max

    def set_white_sat_max(self, saturation):
        """Live-adjustable -- 0-255 HSV S (saturation) cutoff, paired with
        white_value_min. A pixel is rejected as "white" if its S
        is BELOW this AND its V is above white_value_min."""
        saturation = int(saturation)
        if not (0 <= saturation <= 255):
            return False
        with self._settings_lock:
            self._white_sat_max = saturation
        return True

    def get_white_sat_max(self):
        with self._settings_lock:
            return self._white_sat_max

    def set_white_value_min(self, value):
        """Live-adjustable -- 0-255 HSV V (brightness) cutoff, paired with
        white_sat_max. See set_white_sat_max()."""
        value = int(value)
        if not (0 <= value <= 255):
            return False
        with self._settings_lock:
            self._white_value_min = value
        return True

    def get_white_value_min(self):
        with self._settings_lock:
            return self._white_value_min

    def get_line_coverage(self):
        """Latest frame's % of ROI pixels classified as a black/white
        line -- or None before the first detection cycle has run. Purely
        a tuning aid right now (watch it drop to ~0 over a clean floor
        patch as the HSV cutoffs are dialed in)."""
        with self._lock:
            return self._line_coverage_pct

    def get_object_coverage(self):
        """Latest frame's % of ROI pixels classified as obstacle (post
        floor+line rejection, filled) -- or None before the first
        detection cycle has run."""
        with self._lock:
            return self._object_coverage_pct

    def set_downscale(self, factor):
        """Internal-only performance knob -- fraction (0, 1.0] the ROI is
        resized by before the HSV mask. Smaller = cheaper but less
        precise (thin lines more likely to get missed)."""
        if not (0.0 < factor <= 1.0):
            return False
        with self._settings_lock:
            self._downscale = factor
        return True

    def get_downscale(self):
        with self._settings_lock:
            return self._downscale

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self):
        interval = 1.0 / self.fps
        while self._running:
            start = time.monotonic()
            frame = self.camera.get_frame()
            if frame is not None:
                detections, line_coverage_pct, object_coverage_pct, debug_png = self._detect(frame)
                with self._lock:
                    self._detections = detections
                    self._line_coverage_pct = line_coverage_pct
                    self._object_coverage_pct = object_coverage_pct
                    self._debug_png = debug_png
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _detect(self, frame):
        frame_height_px, frame_width_px = frame.shape[:2]

        # Crop CANCELLED -- process the full frame instead of the
        # min_distance_cm-max_distance_cm band. roi_top stays 0 so the
        # rest of the pipeline (which still adds roi_top back to get
        # full-frame Y coordinates) needs no other changes.
        roi_top = 0
        roi = frame

        # Downscale before the HSV mask/Canny -- cheap, and the debug
        # frame is served at this resolution too (no need to upscale it
        # back).
        scale = self.get_downscale()
        small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else roi
        inv_scale = 1.0 / scale

        # Black/white LINE color segmentation -- still computed (as
        # line_coverage_pct, a tuning/debug stat), but NO LONGER applied
        # to erase edges before the vertical-line search. Confirmed with
        # a real cv2 test: a dark leg/pillar (which is exactly what
        # black_value_max is tuned to reject as floor tape) got wiped
        # from the edge map entirely before the vertical search ever
        # ran -- this is precisely the bug testing at black_value_max=0
        # exposed. Floor tape doesn't need a color-based veto here
        # anyway: a real painted line is short and wide, while the
        # vertical shape filter below (top-anchored, tall, thin) already
        # rejects that shape on its own, so color-based erasure was only
        # ever hurting real vertical obstacles.
        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat = hsv_small[..., 1]
        val = hsv_small[..., 2]
        black_value_max = self.get_black_value_max()
        white_sat_max = self.get_white_sat_max()
        white_value_min = self.get_white_value_min()
        line_mask = (val < black_value_max) | ((sat < white_sat_max) & (val > white_value_min))
        line_pixel_count = int(line_mask.sum())
        total_pixel_count = line_mask.shape[0] * line_mask.shape[1]
        line_coverage_pct = (line_pixel_count / total_pixel_count) * 100.0 if total_pixel_count > 0 else 0.0

        # Plain Canny edges -- full, unfiltered by color. The
        # vertical-line SHAPE filter below is what separates real
        # pillars/legs from floor clutter, not a color veto.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # A lighter blur + lower Canny thresholds than a first pass here
        # -- GaussianBlur(5,5) + Canny(50,150) smoothed real obstacle
        # edges (moderate-contrast, e.g. a ~40-gray-level step) below
        # the low threshold entirely on the already-downscaled frame,
        # confirmed with a real cv2 test (0 edges found for an obstacle
        # that should produce hundreds). (3,3) + Canny(30,90) still
        # rejects sensor noise while actually detecting real edges.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)

        # VERTICAL-LINE pattern match: a small vertical morphological
        # CLOSE bridges minor gaps in an otherwise-continuous vertical
        # edge first (a real leg/pillar edge can have small breaks from
        # noise/blur that would otherwise split it into several short
        # fragments).
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, DETECTION_VERTICAL_CLOSE_KSIZE))
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug frame: the vertical-closed edge map, in color so a green
        # box can be drawn around each ACCEPTED candidate (foot at or
        # nearer than the 100cm line, including clamped near-range ones,
        # see below) -- rejected candidates (behind the 100cm line) get
        # no box at all. Drawn in this same downscaled pixel space the
        # edges/contours are already in, so no extra scaling is needed
        # here.
        debug_img = cv2.cvtColor(vertical_edges, cv2.COLOR_GRAY2BGR)

        small_height_px = vertical_edges.shape[0]
        top_margin_px = DETECTION_VERTICAL_TOP_MARGIN_FRACTION * small_height_px
        min_distance_cm = DETECTION_OBJECT_MIN_DISTANCE_CM
        max_distance_cm = DETECTION_OBJECT_MAX_DISTANCE_CM

        detections = []
        total_object_area = 0.0
        for c in contours:
            x_roi, y_roi, w, h = cv2.boundingRect(c)
            # Obstacle condition, exactly two parts:
            #   1. The edge must extend from near the top of the frame
            #      downward (y_roi within the top margin).
            #   2. Its bottom endpoint (the foot -- checked below via
            #      distance_for_row) must not be behind the 100cm line.
            if y_roi > top_margin_px:
                continue

            x_full = x_roi * inv_scale
            y_full = y_roi * inv_scale
            w_full = w * inv_scale
            h_full = h * inv_scale
            x, y = x_full, y_full + roi_top  # full-FRAME coordinates from here on

            # The line's LOWEST point is its floor-contact point --
            # convert to a real-world distance. Behind max_distance_cm
            # (the 100cm line) the estimate is untrusted and the whole
            # detection is dropped. Nearer than min_distance_cm (the
            # 50cm line) is NOT dropped -- up close the primary edge
            # tends to merge with noise right at the bottom of the
            # frame, so instead of trusting that noisy exact row, the
            # reported distance is CLAMPED to min_distance_cm, the
            # closest fixed value this detector ever reports.
            distance_cm = self.rangefinder.distance_for_row(y + h_full, frame_height_px)
            accepted = distance_cm is not None and distance_cm <= max_distance_cm
            if accepted and distance_cm < min_distance_cm:
                distance_cm = min_distance_cm

            if not accepted:
                # Rejected for being behind the 100cm line -- no box
                # drawn on the debug frame for these, only accepted
                # (incl. near-clamped) candidates get one.
                continue

            # Box drawn in the debug image's own (downscaled) pixel
            # space -- green, since only accepted candidates reach here.
            cv2.rectangle(debug_img, (x_roi, y_roi), (x_roi + w, y_roi + h), (0, 255, 0), 2)

            total_object_area += w_full * h_full
            bearing_deg = self.rangefinder.bearing_deg_for_column(x + w_full / 2.0, frame_width_px, frame_height_px)
            left_bearing_deg = self.rangefinder.bearing_deg_for_column(x, frame_width_px, frame_height_px)
            right_bearing_deg = self.rangefinder.bearing_deg_for_column(x + w_full, frame_width_px, frame_height_px)
            detections.append({'label': 'obstacle', 'x': int(x), 'y': int(y),
                                'w': int(w_full), 'h': int(h_full), 'distance_cm': distance_cm,
                                'bearing_deg': bearing_deg,
                                'left_bearing_deg': left_bearing_deg,
                                'right_bearing_deg': right_bearing_deg})

        frame_area = float(frame_height_px * frame_width_px)
        object_coverage_pct = (total_object_area / frame_area) * 100.0 if frame_area > 0 else 0.0

        # Debug frame for /debug_feed: the vertical-closed edge map (post
        # line-erasure) with a green box drawn around each ACCEPTED
        # candidate -- so you should see continuous vertical lines for
        # real pillars/legs (boxed if within 100cm), but no black/white
        # line edges and no box for anything rejected as too far.
        # PNG (lossless), NOT JPEG -- see notebook_debug.txt: JPEG's
        # block-based DCT compression bleeds erased regions back into
        # visibility on decode. The frame here is tiny (downscaled) and
        # served at DETECTION_FPS, not CAMERA_FPS, so PNG's extra size
        # doesn't matter.
        debug_png = None
        ok, buf = cv2.imencode('.png', debug_img)
        if ok:
            debug_png = buf.tobytes()

        return detections, line_coverage_pct, object_coverage_pct, debug_png

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def get_debug_frame(self):
        """Latest processed frame, PNG-encoded (lossless -- see _detect),
        for the /debug_feed stream -- or None before the first detection
        cycle. This is the vertical-closed Canny edge map with
        black/white line edges already erased, PLUS a green box drawn
        around each ACCEPTED candidate line (foot within 100cm) --
        candidates rejected as too far get no box (see class
        docstring)."""
        with self._lock:
            return self._debug_png


class ObstacleWatcher:
    """Safety supervisor bridging ObstacleDetector's obstacle-candidate
    signal and GridNavNode's map-aware cell-confirmation flow.

    PAUSED right now: ObstacleDetector is being rebuilt from scratch
    (currently only does step 1 -- black/white line rejection, see its
    docstring) and doesn't produce an obstacle-candidate signal yet, so
    _is_candidate() always returns False here -- no hold ever starts,
    no cell ever gets pinned. The rest of this class (hold/confirm-count
    timing, per-cell state machine, enabled toggle) is left intact and
    wired up, ready for a real candidate signal once detection is built
    back on top of the cleaned/line-free frame.

    Logic each tick (once a real candidate signal exists again):
      Identify the FRONT CELL (the world grid cell directly ahead of the
      robot right now -- node.get_front_cell()) and check its map status
      (node.get_cell_status()) BEFORE doing anything else:
        - DONE + CLEAR  -> nothing to do, keep driving normally.
        - DONE + OBSTACLE -> already known/pinned into the A* map
          (avoidance already handled by planning), nothing to do here.
        - NOT_DONE -> this is the only case that does any work:
          evaluate _is_candidate(). A single candidate frame is NOT
          enough (could be a glitch), so:
            1. The first candidate frame while driving toward a
               NOT_DONE front cell HOLDS the robot (stops, freezes the
               current leg) for hold_duration_s seconds.
            2. Every tick during the hold where _is_candidate() is still
               true increments a hit counter.
            3. When the hold window ends: if hit_count reached
               confirm_count, the cell is confirmed OBSTACLE -- pinned
               into node.obstacles (so A* avoids it) and the goal is
               replanned. Otherwise the cell is confirmed CLEAR. Either
               way the cell becomes DONE and is never re-processed.

    hold_duration_s and confirm_count are live-adjustable from the GUI
    (unchanged from before), and the whole watcher can be toggled on/off
    (enabled) -- when disabled, detection/streaming keep running for
    display, but no holds are ever started (and any in-progress hold is
    released immediately). Runs its own background thread polling the
    detector at DETECTION_FPS.
    """

    def __init__(self, node: 'GridNavNode', detector: ObstacleDetector,
                 hold_duration_s=OBSTACLE_HOLD_DURATION_S,
                 confirm_count=OBSTACLE_CONFIRM_COUNT):
        self.node = node
        self.detector = detector

        self._lock = threading.Lock()
        self.hold_duration_s = hold_duration_s
        self.confirm_count = confirm_count
        self.enabled = True
        self._holding = False
        self._hold_start = 0.0
        self._hit_count = 0
        self._hold_cell = None   # the front cell this hold is evaluating

        self._running = False
        self._thread = None

    def set_hold_duration(self, seconds):
        if seconds <= 0:
            return False
        with self._lock:
            self.hold_duration_s = seconds
        return True

    def set_confirm_count(self, count):
        if count <= 0:
            return False
        with self._lock:
            self.confirm_count = int(count)
        return True

    def set_enabled(self, enabled: bool):
        """Master on/off toggle from the GUI. Turning it off immediately
        releases any in-progress hold so the robot isn't left stuck."""
        with self._lock:
            self.enabled = enabled
            was_holding = self._holding
            if not enabled:
                self._holding = False
        if not enabled and was_holding:
            self.node.set_obstacle_hold(False)
        return True

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self):
        interval = 1.0 / DETECTION_FPS
        while self._running:
            self._tick()
            time.sleep(interval)

    # PAUSED -- see class docstring. ObstacleDetector's vertical-edge
    # detection isn't wired into a candidate signal yet, so no hold ever
    # starts until this is flipped and _is_candidate() below does real
    # work. Surfaced via get_status()['paused'] so the GUI can show this
    # honestly instead of a normal-looking "clear"/enabled status that
    # implies avoidance is active when it structurally cannot fire.
    PAUSED = True

    def _is_candidate(self):
        # No real candidate signal wired up yet -- see PAUSED above.
        return False

    def _tick(self):
        with self._lock:
            enabled = self.enabled
            holding = self._holding
        if not enabled:
            return

        if not holding:
            front_cell = self.node.get_front_cell()
            if front_cell is None or not self.node.is_driving():
                return
            # DONE (either CLEAR or OBSTACLE) -- already resolved, nothing
            # to do: CLEAR needs no action, OBSTACLE is already pinned and
            # handled by A* planning.
            if self.node.get_cell_status(front_cell) != 'NOT_DONE':
                return
            if self._is_candidate():
                with self._lock:
                    self._holding = True
                    self._hold_start = time.monotonic()
                    self._hit_count = 1
                    self._hold_cell = front_cell
                self.node.set_obstacle_hold(True)
            return

        if self._is_candidate():
            with self._lock:
                self._hit_count += 1

        with self._lock:
            elapsed = time.monotonic() - self._hold_start
            hold_duration_s = self.hold_duration_s
            hit_count = self._hit_count
            confirm_count = self.confirm_count
            hold_cell = self._hold_cell

        if elapsed >= hold_duration_s:
            if hit_count >= confirm_count:
                self.node.set_cell_status(hold_cell, 'OBSTACLE')
                if self.node.pin_cells([hold_cell]):
                    self.node.replan_current_goal()
            else:
                self.node.set_cell_status(hold_cell, 'CLEAR')
            with self._lock:
                self._holding = False
                self._hold_cell = None
            self.node.set_obstacle_hold(False)

    def get_candidate_cells(self):
        """The single front cell currently being evaluated (mid-hold), as
        a list for GUI compatibility -- lets the GUI show a stable
        'being confirmed' box for the whole hold window."""
        with self._lock:
            return [self._hold_cell] if self._hold_cell is not None else []

    def get_status(self):
        with self._lock:
            holding = self._holding
            hit_count = self._hit_count
            hold_start = self._hold_start
            hold_duration_s = self.hold_duration_s
            confirm_count = self.confirm_count
            enabled = self.enabled
        remaining_s = max(0.0, hold_duration_s - (time.monotonic() - hold_start)) if holding else 0.0
        return {
            'enabled': enabled,
            'paused': self.PAUSED,
            'holding': holding,
            'hit_count': hit_count,
            'confirm_count': confirm_count,
            'hold_duration_s': hold_duration_s,
            'remaining_s': remaining_s,
        }


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_max, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self, initial_error=0.0):
        # Seed prev_error with the actual current error (not 0) so the
        # first compute() after a reset doesn't see a fake error jump from
        # 0 -> real_error and fire a derivative-kick spike (dominates the
        # output on a large setpoint change, e.g. rotating to a new target
        # heading -- this was causing the "very fast angular" burst at the
        # start of each leg).
        self.integral = 0.0
        self.prev_error = initial_error

    def compute(self, error):
        self.integral += error * self.dt
        self.integral = max(-self.i_max, min(self.i_max, self.integral))
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, output))


class GridNavNode(Node):
    def __init__(self):
        super().__init__('grid_nav_node')

        self.dt = 1.0 / LOOP_HZ

        # Symmetric output range: PID sign follows error sign, PIVOT_SIGN is
        # applied separately in control_loop.
        self.rotate_pid = PID(ROTATE_KP, ROTATE_KI, ROTATE_KD,
                               -ROTATE_SPEED, ROTATE_SPEED, ROTATE_MAX_INTEGRAL, self.dt)
        self.drive_pid = PID(DRIVE_KP, DRIVE_KI, DRIVE_KD,
                              -MAX_ANGULAR_Z_HOLD, MAX_ANGULAR_Z_HOLD, DRIVE_MAX_INTEGRAL, self.dt)

        # Live-adjustable via the web GUI (see set_speed()) -- start from the
        # module defaults above, but can be changed at runtime without
        # restarting the node, to test how speed affects distance accuracy.
        self.forward_speed = FORWARD_SPEED
        self.rotate_speed = ROTATE_SPEED

        self._lock = threading.Lock()

        # Pose tracked at the left wheel, in cm. (0, 0) at node start.
        self.x = 0.0
        self.y = 0.0

        self.current_yaw = None     # raw IMU yaw, radians
        self.heading_ref = None     # yaw captured at startup == "+X" (east)

        self.last_pulses = None     # most recent raw pulse count from encoder

        # Navigation state: 'IDLE' | 'RUNNING'
        self.state = 'IDLE'
        self.legs = []               # list of ('move', ux, uy, distance_cm) | ('heading', end_dir_deg)
        self.leg_idx = 0
        self.phase = None            # 'ROTATE' | 'DRIVE'
        self.leg_target_heading = 0.0
        self.leg_target_distance_cm = 0.0
        self.leg_baseline_pulses = 0
        self.leg_progress_cm = 0.0   # unsigned live distance traveled this leg (cm)
                                      # (not yet committed to x/y -- see control_loop DRIVE)
        self.leg_start_x = 0.0       # x/y at the start of the current 'move' leg, plus the
        self.leg_start_y = 0.0       # unit direction vector -- together with leg_progress_cm
        self.leg_unit_dx = 0.0       # these give the live in-progress display position
        self.leg_unit_dy = 0.0
        self.leg_boxes_crossed = 0   # how many GRID_SPACING_CM boxes crossed so far this leg,
                                      # for pushing live trail points as each box is reached
        self.goal = None             # (gx, gy) for display
        self.end_dir_deg = None      # requested final heading, degrees (or None)

        # Obstacle map for A* planning: set of blocked (i, j) grid cells,
        # cell (i, j) centered at (i * GRID_SPACING_CM, j * GRID_SPACING_CM).
        # Edited live from the GUI (click a cell to toggle it).
        self.obstacles = set()
        # Subset of self.obstacles that came from a CONFIRMED camera
        # detection (ObstacleWatcher), not a manual GUI click -- tracked
        # separately purely so the GUI can draw them solid red/distinctly
        # from manually-toggled cells.
        self.pinned_cells = set()

        # Map-aware cell validation (ObstacleWatcher's area-density
        # confirmation flow): (i, j) -> 'CLEAR' | 'OBSTACLE'. A cell
        # missing from this dict is implicitly NOT_DONE (never checked).
        # Once a cell is DONE (either value present), it's never
        # re-processed -- see get_front_cell()/get_cell_status().
        self.cell_status = {}

        self.planned_path = []       # [(x_cm, y_cm), ...] cell centers of the last A* route, for GUI overlay

        # Step mode: pause fully after each STEP_SIZE_CM of DRIVE travel
        # and wait for continue_step() before resuming, so you can measure
        # one grid box at a time instead of a whole leg in one go.
        self.step_mode = False
        self.awaiting_continue = False
        self.step_baseline_pulses = 0

        # Set True by ObstacleWatcher while confirming a camera-detected
        # obstacle (stop-and-recheck window) -- control_loop holds all-zero
        # cmd_vel and does nothing else while this is set, then resumes the
        # SAME leg from where it left off once cleared (nothing about the
        # leg's phase/baseline pulses is touched during a hold).
        self.obstacle_hold = False

        self.path = [(0.0, 0.0)]     # visited points, for GUI trail

        self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.imu_callback, 10)
        self.encoder_sub = self.create_subscription(
            Int32MultiArray, WHEEL_ENCODER_TOPIC, self.encoder_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f'=== grid_nav.py {SCRIPT_VERSION} ===')
        self.get_logger().info('Waiting for first IMU message to lock heading reference...')

    # ---------------- ROS callbacks ----------------

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        with self._lock:
            self.current_yaw = yaw
            if self.heading_ref is None:
                self.heading_ref = yaw
                self.get_logger().info(
                    f'Heading reference locked (this is "+X"): {math.degrees(yaw):.1f} deg'
                )

    def encoder_callback(self, msg: Int32MultiArray):
        data = msg.data
        if len(data) <= ENCODER_INDEX_PRIMARY_PULSES:
            self.get_logger().warn('wheel_encoder message shorter than expected indices')
            return
        with self._lock:
            self.last_pulses = data[ENCODER_INDEX_PRIMARY_PULSES]

    # ---------------- Goal handling ----------------

    @staticmethod
    def _to_cell(x_cm, y_cm):
        return (round(x_cm / GRID_SPACING_CM), round(y_cm / GRID_SPACING_CM))

    def toggle_obstacle(self, i, j):
        with self._lock:
            cell = (i, j)
            if cell in self.obstacles:
                self.obstacles.discard(cell)
                self.pinned_cells.discard(cell)
            else:
                self.obstacles.add(cell)

    def clear_obstacles(self):
        with self._lock:
            self.obstacles.clear()
            self.pinned_cells.clear()

    def set_goal(self, gx, gy, end_dir_deg=None, step_mode=False):
        with self._lock:
            start_cell = self._to_cell(self.x, self.y)
            goal_cell = self._to_cell(gx, gy)

            legs = []
            planned_path = []
            if start_cell != goal_cell:
                # Inflate obstacles by the robot's footprint radius before
                # searching -- A* itself still treats the robot as a
                # single point, but against a map that already accounts
                # for the real ROBOT_SIZE_CM box's clearance needs.
                inflated_obstacles = inflate_obstacles(self.obstacles, ROBOT_INFLATION_CELLS)
                cell_path = astar_search(start_cell, goal_cell, inflated_obstacles)
                if cell_path is None:
                    self.goal = (gx, gy)
                    self.get_logger().warn(
                        f'No path to ({gx:.1f}, {gy:.1f}) cm -- blocked by obstacles (incl. robot '
                        f'clearance margin) or out of range.'
                    )
                    return
                legs = path_to_legs(cell_path, GRID_SPACING_CM)
                planned_path = [(c[0] * GRID_SPACING_CM, c[1] * GRID_SPACING_CM) for c in cell_path]

            if end_dir_deg is not None:
                # Rotate-only leg: no drive phase, just turn to face this
                # heading (degrees, relative to heading_ref where 0 = +X)
                # once position legs are done.
                legs.append(('heading', end_dir_deg))

            self.goal = (gx, gy)
            self.end_dir_deg = end_dir_deg
            self.legs = legs
            self.leg_idx = 0
            self.planned_path = planned_path
            self.step_mode = step_mode
            self.awaiting_continue = False

            if not legs:
                self.state = 'IDLE'
                self.get_logger().info('Goal is at (or within tolerance of) current position.')
                return

            self.state = 'RUNNING'
            self._start_leg_locked()
            self.get_logger().info(
                f'New goal: ({gx:.1f}, {gy:.1f}) cm -- {len(legs)} leg(s)'
                f'{" [step mode]" if step_mode else ""}'
            )

    def continue_step(self):
        """Resume DRIVE after a step-mode pause. No-op if not currently
        paused (e.g. button clicked twice, or clicked while not running)."""
        with self._lock:
            if not self.awaiting_continue or self.last_pulses is None:
                return
            self.awaiting_continue = False
            self.step_baseline_pulses = self.last_pulses

    def _start_leg_locked(self):
        """Caller must hold self._lock."""
        leg = self.legs[self.leg_idx]
        if self.heading_ref is None:
            # No IMU data yet -- can't compute a target heading. Bail to
            # IDLE; control_loop will simply do nothing until IMU arrives
            # and the user re-sends the goal.
            self.state = 'IDLE'
            self.get_logger().warn('No IMU data yet -- cannot start leg. Try the goal again shortly.')
            return

        if leg[0] == 'move':
            _, ux, uy, distance_cm = leg
            # heading_ref + 0deg == "+X" (ux=1,uy=0); Y_AXIS_SIGN flips which
            # physical turn direction counts as "+Y" -- same convention as
            # before, just generalized to any of the 8 grid directions via
            # atan2 instead of only handling the 4 cardinal cases.
            target = self.heading_ref + math.atan2(Y_AXIS_SIGN * uy, ux)
            self.leg_target_heading = normalize_angle(target)
            self.leg_target_distance_cm = distance_cm
            self.leg_start_x = self.x
            self.leg_start_y = self.y
            self.leg_unit_dx = ux
            self.leg_unit_dy = uy
        else:  # 'heading' -- leg[1] is an absolute end direction in degrees
            target = self.heading_ref + math.radians(leg[1])
            self.leg_target_heading = normalize_angle(target)
            self.leg_target_distance_cm = 0.0

        self.leg_progress_cm = 0.0
        self.leg_boxes_crossed = 0
        self.phase = 'ROTATE'
        self.rotate_pid.reset(angle_diff(self.leg_target_heading, self.current_yaw))

    def set_speed(self, forward=None, rotate=None):
        """Adjust drive/rotate speed live, without restarting the node."""
        with self._lock:
            if forward is not None and forward > 0:
                self.forward_speed = forward
            if rotate is not None and rotate > 0:
                self.rotate_speed = rotate
                self.rotate_pid.out_min = -rotate
                self.rotate_pid.out_max = rotate
        self.get_logger().info(
            f'Speed updated: forward={self.forward_speed:.3f} rotate={self.rotate_speed:.3f}'
        )

    # ---------------- Control loop ----------------

    def control_loop(self):
        twist = Twist()

        with self._lock:
            if self.obstacle_hold:
                self.cmd_pub.publish(twist)  # all-zero -- held for obstacle confirmation
                return

            if self.state != 'RUNNING' or self.current_yaw is None or self.last_pulses is None:
                self.cmd_pub.publish(twist)  # all-zero
                return

            if self.awaiting_continue:
                self.cmd_pub.publish(twist)  # all-zero -- paused between step-mode boxes
                return

            if self.phase == 'ROTATE':
                error = angle_diff(self.leg_target_heading, self.current_yaw)
                if abs(math.degrees(error)) <= HEADING_TOLERANCE_DEG:
                    if self.legs[self.leg_idx][0] == 'heading':
                        # Rotate-only leg (final end direction) -- no DRIVE
                        # phase, no position change. Leg is done as soon as
                        # heading is reached.
                        self.leg_idx += 1
                        self.cmd_pub.publish(twist)
                        if self.leg_idx >= len(self.legs):
                            self.state = 'IDLE'
                            self.get_logger().info(
                                f'Goal reached: ({self.x:.1f}, {self.y:.1f}) cm, '
                                f'facing {math.degrees(self.leg_target_heading - self.heading_ref):.1f} deg'
                            )
                        else:
                            self._start_leg_locked()
                        return

                    self.leg_baseline_pulses = self.last_pulses
                    self.step_baseline_pulses = self.last_pulses
                    self.phase = 'DRIVE'
                    self.drive_pid.reset(error)
                    self.cmd_pub.publish(twist)  # brief all-zero pause between phases
                    return

                cmd = self.rotate_pid.compute(error) * PIVOT_SIGN
                # Floor the magnitude so the pivot doesn't stall out as the
                # PID output shrinks near zero error, before actually
                # reaching HEADING_TOLERANCE_DEG. (rotate_speed is live-
                # adjustable, but the floor stays a fixed fraction so it
                # keeps working across the adjustable range.)
                if abs(cmd) < ROTATE_MIN_OUTPUT:
                    cmd = math.copysign(ROTATE_MIN_OUTPUT, cmd if cmd != 0 else error)
                # See PIVOT_ANGULAR_SIGN comment -- this zeroes the firmware
                # term that corresponds to the physical LEFT wheel.
                twist.linear.x = cmd
                twist.angular.z = PIVOT_ANGULAR_SIGN * cmd
                self.cmd_pub.publish(twist)
                return

            # phase == 'DRIVE'
            traveled_pulses = self.last_pulses - self.leg_baseline_pulses
            traveled_cm = abs(traveled_pulses) / PULSES_PER_CM
            # Live progress along this leg's direction, updated every tick
            # (not yet committed to x/y) -- lets the GUI show real-time
            # position and distance-so-far while driving, like
            # heading_hold.py's periodic distance print, instead of only
            # jumping at leg end. Unsigned: direction is carried by
            # leg_unit_dx/dy (the robot only ever drives forward, having
            # already turned to face the right way in ROTATE).
            self.leg_progress_cm = traveled_cm

            # Push a live trail point each time a full grid box is crossed
            # (not just when the whole leg finishes), so the GUI path/trail
            # updates progressively as the robot passes each box instead of
            # only jumping once at leg completion.
            boxes_crossed = int(traveled_cm // GRID_SPACING_CM)
            if boxes_crossed > self.leg_boxes_crossed:
                self.leg_boxes_crossed = boxes_crossed
                box_x = self.leg_start_x + self.leg_unit_dx * traveled_cm
                box_y = self.leg_start_y + self.leg_unit_dy * traveled_cm
                self.path.append((box_x, box_y))

            if traveled_cm >= self.leg_target_distance_cm:
                self.x = self.leg_start_x + self.leg_unit_dx * traveled_cm
                self.y = self.leg_start_y + self.leg_unit_dy * traveled_cm
                self.path.append((self.x, self.y))
                self.leg_progress_cm = 0.0

                self.leg_idx += 1
                self.cmd_pub.publish(twist)  # all-zero between legs
                if self.leg_idx >= len(self.legs):
                    self.state = 'IDLE'
                    self.get_logger().info(
                        f'Goal reached: ({self.x:.1f}, {self.y:.1f}) cm'
                    )
                else:
                    self._start_leg_locked()
                return

            if self.step_mode:
                step_traveled_cm = abs(self.last_pulses - self.step_baseline_pulses) / PULSES_PER_CM
                if step_traveled_cm >= STEP_SIZE_CM:
                    self.awaiting_continue = True
                    self.cmd_pub.publish(twist)  # all-zero -- full stop for measuring
                    self.get_logger().info(
                        f'Step complete ({step_traveled_cm:.1f}cm this box, '
                        f'{traveled_cm:.1f}/{self.leg_target_distance_cm:.1f}cm total). '
                        f'Waiting for continue.'
                    )
                    return

            twist.linear.x = self.forward_speed
            if DRIVE_HEADING_HOLD_ENABLED:
                herr = angle_diff(self.leg_target_heading, self.current_yaw)
                correction = self.drive_pid.compute(herr)
                twist.angular.z = DRIVE_CORRECTION_SIGN * correction
            # else: angular.z stays 0 -- pure open-loop straight driving
            self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())  # all zeros

    # ---------------- Obstacle confirmation hooks (used by ObstacleWatcher) ----------------

    def _live_position_locked(self):
        """Caller must hold self._lock. self.x/self.y are only updated
        when a 'move' leg fully COMPLETES (see control_loop's DRIVE
        phase) -- while mid-leg they still hold the position from when
        the leg STARTED. This returns the true current position,
        committed x/y plus in-progress DRIVE movement along the current
        leg's direction. Used by get_snapshot() (so the GUI marker moves
        in real time instead of jumping only at leg completion) and by
        get_front_cell() (so a multi-cell leg's "cell ahead" advances as
        the robot actually drives through it, not just once at leg
        start -- see notebook_debug.txt for the bug this fixed: a leg
        spanning several grid cells was only ever getting its FIRST cell
        checked by ObstacleWatcher, since get_front_cell() used to read
        the stale leg-start self.x/self.y directly)."""
        x, y = self.x, self.y
        if self.state == 'RUNNING' and self.phase == 'DRIVE' and self.legs \
                and self.legs[self.leg_idx][0] == 'move':
            x = self.leg_start_x + self.leg_unit_dx * self.leg_progress_cm
            y = self.leg_start_y + self.leg_unit_dy * self.leg_progress_cm
        return x, y

    def is_driving(self):
        """True if control_loop is actively working a leg right now --
        i.e. there's real motion an obstacle hold would actually interrupt."""
        with self._lock:
            return self.state == 'RUNNING' and not self.awaiting_continue and not self.obstacle_hold

    def get_front_cell(self):
        """The single world grid cell directly ahead of the robot's
        CURRENT (live, not stale leg-start) position and heading, one
        GRID_SPACING_CM step forward. Used by ObstacleWatcher's
        map-aware confirmation flow to decide which cell a NOT_DONE
        observation should be attributed to. Returns None if heading
        isn't known yet."""
        with self._lock:
            if self.heading_ref is None or self.current_yaw is None:
                return None
            heading_deg = math.degrees(angle_diff(self.current_yaw, self.heading_ref))
            rad = math.radians(heading_deg)
            live_x, live_y = self._live_position_locked()
            front_x = live_x + GRID_SPACING_CM * math.cos(rad)
            front_y = live_y + GRID_SPACING_CM * math.sin(rad)
            return self._to_cell(front_x, front_y)

    def get_cell_status(self, cell):
        """'NOT_DONE' (never checked), 'CLEAR' (checked, no obstacle), or
        'OBSTACLE' (checked, confirmed obstacle) -- see cell_status."""
        with self._lock:
            return self.cell_status.get(cell, 'NOT_DONE')

    def set_cell_status(self, cell, status):
        """Mark a cell DONE with the given result ('CLEAR' or
        'OBSTACLE'). Once set, get_front_cell()/get_cell_status() callers
        should not re-process this cell."""
        with self._lock:
            self.cell_status[cell] = status

    def set_obstacle_hold(self, hold: bool):
        with self._lock:
            self.obstacle_hold = hold
            if hold:
                self._commit_live_position_locked()
        if hold:
            self.stop_robot()

    def _commit_live_position_locked(self):
        """Caller must hold self._lock. self.x/self.y are normally only
        updated when a 'move' leg fully COMPLETES (see control_loop's
        DRIVE phase) -- while mid-leg they still hold the position from
        when the leg STARTED, even though the robot has actually
        traveled leg_progress_cm further by now (see
        _live_position_locked(), which computes but doesn't mutate this).
        Called when an obstacle hold begins so that pin_cells()/
        replan_current_goal() reason about the robot's TRUE current
        position -- without this, a hold triggered mid-drive would pin
        the obstacle and replan from a stale, pre-leg position,
        potentially routing the "avoidance" path right past the
        obstacle's actual real-world location."""
        if self.state != 'RUNNING' or self.phase != 'DRIVE' or not self.legs:
            return
        if self.legs[self.leg_idx][0] != 'move':
            return
        self.x, self.y = self._live_position_locked()
        # Shrink the remaining distance on this leg to match, so if this
        # turns out to be a glitch (no replan) and the SAME leg just
        # resumes, it still ends at the original target point instead of
        # overshooting by however much was already covered before the
        # hold.
        self.leg_target_distance_cm -= self.leg_progress_cm
        self.leg_start_x = self.x
        self.leg_start_y = self.y
        self.leg_progress_cm = 0.0
        self.leg_boxes_crossed = 0
        self.leg_baseline_pulses = self.last_pulses

    def pin_cells(self, cells):
        """Permanently add an arbitrary collection of (i, j) grid cells to
        the A* obstacle map, tracked in pinned_cells (a subset of
        obstacles) so the GUI can draw confirmed/camera-pinned cells
        distinctly (solid) from manually-toggled ones (light). Returns the
        list of cells actually pinned (may be empty)."""
        cells = list(cells)
        if not cells:
            return []
        with self._lock:
            self.obstacles.update(cells)
            self.pinned_cells.update(cells)
        self.get_logger().warn(f'Obstacle confirmed -- pinned {len(cells)} grid cell(s): {sorted(cells)}')
        return cells

    def replan_current_goal(self):
        """Re-submit the current goal so set_goal()'s A* search picks up
        newly pinned obstacle cells and routes around them."""
        with self._lock:
            goal = self.goal
            end_dir_deg = self.end_dir_deg
            step_mode = self.step_mode
        if goal is None:
            return
        self.get_logger().warn(f'Replanning path to {goal} around newly pinned obstacle(s)')
        self.set_goal(goal[0], goal[1], end_dir_deg, step_mode)

    def reset_position(self):
        """Zero the tracked (x, y) without restarting the node. Goals are
        relative to this tracked position, not the physical start point --
        if it drifts from reality (e.g. a prior run stopped early), later
        goals will be off by exactly that drift. Call this right before a
        fresh test run to realign tracked (0, 0) with wherever the robot
        physically is right now."""
        with self._lock:
            self.state = 'IDLE'
            self.x = 0.0
            self.y = 0.0
            self.path = [(0.0, 0.0)]
            self.goal = None
            self.legs = []
            self.leg_idx = 0
            self.planned_path = []
        self.stop_robot()
        self.get_logger().info('Position reset to (0, 0). Obstacles left as-is.')

    # ---------------- Snapshot for the GUI thread ----------------

    def get_snapshot(self):
        with self._lock:
            yaw_deg = math.degrees(self.current_yaw) if self.current_yaw is not None else None
            ref_deg = math.degrees(self.heading_ref) if self.heading_ref is not None else None
            heading_deg = None
            if self.current_yaw is not None and self.heading_ref is not None:
                heading_deg = math.degrees(angle_diff(self.current_yaw, self.heading_ref))
            target_deg = None
            if self.state == 'RUNNING':
                target_deg = math.degrees(angle_diff(self.leg_target_heading, self.heading_ref))

            display_x, display_y = self._live_position_locked()

            return {
                'x': display_x,
                'y': display_y,
                'path': list(self.path),
                'planned_path': list(self.planned_path),
                'obstacles': [list(c) for c in self.obstacles],
                'pinned_cells': [list(c) for c in self.pinned_cells],
                'goal': self.goal,
                'state': self.state,
                'phase': self.phase,
                'yaw_deg': yaw_deg,
                'ref_deg': ref_deg,
                'heading_deg': heading_deg,   # yaw relative to startup reference; 0 = "+X"
                'target_heading_deg': target_deg,
                'leg_idx': self.leg_idx,
                'leg_count': len(self.legs),
                'end_dir_deg': self.end_dir_deg,
                'step_mode': self.step_mode,
                'awaiting_continue': self.awaiting_continue,
                'leg_progress_cm': abs(self.leg_progress_cm) if self.phase == 'DRIVE' else None,
                'leg_target_distance_cm': self.leg_target_distance_cm if self.phase == 'DRIVE' else None,
                'forward_speed': self.forward_speed,
                'rotate_speed': self.rotate_speed,
            }


HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>grid_nav.py</title>
<style>
  body { font-family: sans-serif; background: #1e1e1e; color: #eee; margin: 0; padding: 8px; font-size: 13px; }
  h1 { font-size: 13px; font-weight: normal; color: #aaa; margin: 0 0 6px 0; }
  .layout { display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap; }
  .left { display: flex; flex-direction: column; gap: 6px; min-width: 220px; }
  .camera { display: flex; flex-direction: column; gap: 4px; min-width: 260px; max-width: 400px; }
  .camera-wrap { position: relative; }
  .camera img { width: 100%; background: #111; border: 1px solid #444; border-radius: 4px; display: block; }
  .camera .k { font-size: 10px; color: #999; text-transform: uppercase; }
  /* Small "+" fixed at the optical-axis center -- what you physically aim
     at a known-distance floor mark to calibrate. */
  .crosshair { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .crosshair::before, .crosshair::after { content: ''; position: absolute; top: 50%; left: 50%; background: #fff; box-shadow: 0 0 2px #000; }
  .crosshair::before { width: 18px; height: 2px; transform: translate(-50%, -50%); }
  .crosshair::after { width: 2px; height: 18px; transform: translate(-50%, -50%); }
  /* Dashed "straight ahead" reference line, like a backup camera's path guide. */
  .center-line { position: absolute; top: 0; left: 50%; width: 0; height: 100%;
                 border-left: 2px dashed rgba(255,255,255,0.35); pointer-events: none; }
  /* Perspective distance guide lines, positioned/colored dynamically by JS. */
  .guide-lines { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .guide-line { position: absolute; left: 0; width: 100%; height: 2px; box-shadow: 0 0 3px #000; }
  .guide-line-label { position: absolute; right: 4px; font-size: 11px; font-family: monospace;
                       text-shadow: 0 0 3px #000, 0 0 3px #000; transform: translateY(-100%); }
  /* Detected obstacle boxes, positioned/sized dynamically by JS as % of frame. */
  .detection-boxes { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .detection-box { position: absolute; border: 2px solid; box-sizing: border-box; }
  .detection-box-label { position: absolute; top: 0; left: 0; transform: translateY(-100%);
                          font-size: 11px; font-family: monospace; white-space: nowrap;
                          text-shadow: 0 0 3px #000, 0 0 3px #000; }
  .right { flex: 1; }
  #canvas { background: #111; border: 1px solid #444; display: block; max-width: 100%; height: auto; cursor: crosshair; }
  form { background: #262626; border: 1px solid #444; padding: 6px 8px; border-radius: 5px; }
  form .row { margin-bottom: 4px; }
  input { width: 80px; font-size: 12px; padding: 2px 3px; }
  button { font-size: 12px; padding: 4px 10px; margin-top: 2px; width: 100%; }
  label { display: block; font-size: 10px; color: #aaa; margin-bottom: 1px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .stat-box { background: #262626; border: 1px solid #444; border-radius: 5px; padding: 4px 6px; }
  .stat-box .k { font-size: 9px; color: #999; text-transform: uppercase; }
  .stat-box .v { font-family: monospace; font-size: 12px; color: #eee; margin-top: 1px; }
  .stat-box.wide { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>grid_nav.py -- live position (poll __POLL_MS__ms)</h1>
<div class="layout">
  <div class="left">
    <form id="goalForm">
      <div class="row"><label>Goal X (cm)</label><input id="goalX" type="number" value="0" step="1"></div>
      <div class="row"><label>Goal Y (cm)</label><input id="goalY" type="number" value="0" step="1"></div>
      <div class="row"><label>End Direction (deg, 0=+X)</label><input id="goalDir" type="number" value="0" step="1"></div>
      <div class="row"><label style="display:inline"><input id="goalStep" type="checkbox" style="width:auto"> Step mode (pause every __STEP_SIZE__m box)</label></div>
      <button type="submit">Go</button>
    </form>
    <button id="resetBtn" style="background:#5a2a2a;">Reset Position to (0,0)</button>
    <button id="continueBtn" style="background:#2a5a2a; display:none;">Continue to Next Box</button>
    <button id="clearObstaclesBtn" style="background:#5a4a1a;">Clear Obstacles</button>
    <div style="font-size:11px; color:#999;">Click a grid cell to toggle it as an obstacle (A* routes around it, diagonals allowed).</div>
    <form id="speedForm">
      <div class="row"><label>Drive Speed (0-1)</label><input id="speedFwd" type="number" value="__FORWARD_SPEED__" step="0.01" min="0.01" max="1"></div>
      <div class="row"><label>Rotate Speed (0-1)</label><input id="speedRot" type="number" value="__ROTATE_SPEED__" step="0.01" min="0.01" max="1"></div>
      <button type="submit">Set Speed</button>
    </form>
    <div class="stats" id="stats">
      <div class="stat-box wide"><div class="k">Position</div><div class="v" id="s-pos">--</div></div>
      <div class="stat-box"><div class="k">State</div><div class="v" id="s-state">--</div></div>
      <div class="stat-box"><div class="k">Phase</div><div class="v" id="s-phase">--</div></div>
      <div class="stat-box"><div class="k">Leg</div><div class="v" id="s-leg">--</div></div>
      <div class="stat-box wide"><div class="k">Leg Progress</div><div class="v" id="s-progress">--</div></div>
      <div class="stat-box"><div class="k">End Dir</div><div class="v" id="s-enddir">--</div></div>
      <div class="stat-box wide"><div class="k">Speed (drive / rotate)</div><div class="v" id="s-speed">--</div></div>
      <div class="stat-box wide"><div class="k">IMU Yaw (raw)</div><div class="v" id="s-yaw">--</div></div>
      <div class="stat-box wide"><div class="k">Heading (ref=0)</div><div class="v" id="s-heading">--</div></div>
      <div class="stat-box wide"><div class="k">Target Heading</div><div class="v" id="s-target">--</div></div>
      <div class="stat-box wide"><div class="k">Obstacle Watch</div><div class="v" id="s-obwatch">--</div></div>
    </div>
    <form id="obWatchForm">
      <div class="row"><label style="display:inline"><input id="obEnabled" type="checkbox" style="width:auto" checked> Obstacle Avoidance Enabled</label></div>
      <div class="row"><label>Obstacle Hold Duration (sec)</label><input id="obHoldSec" type="number" value="5" step="0.5" min="0.5"></div>
      <div class="row"><label>Confirm Count (hits during hold)</label><input id="obConfirmCount" type="number" value="5" step="1" min="1"></div>
      <button type="submit">Set Obstacle Watch</button>
    </form>
  </div>
  <div class="camera">
    <div class="k">Camera</div>
    <div class="camera-wrap">
      <img id="cameraFeed" src="/video_feed" alt="camera feed"
           onerror="this.replaceWith(Object.assign(document.createElement('div'), {textContent: 'Camera unavailable', style: 'color:#999; padding:12px; border:1px solid #444; border-radius:4px;'}))">
      <div class="center-line"></div>
      <div class="guide-lines" id="guideLines"></div>
      <div class="detection-boxes" id="detectionBoxes"></div>
      <div class="crosshair"></div>
    </div>
    <form id="camCalibForm">
      <div class="row"><label>Camera Height Above Floor (cm)</label><input id="camHeight" type="number" value="26" step="0.5"></div>
      <div class="row"><label>Known Distance at Crosshair (cm)</label><input id="camDist" type="number" value="100" step="1"></div>
      <div class="row"><label>Vertical FOV (deg, tune for accuracy)</label><input id="camVfov" type="number" value="60" step="1"></div>
      <button type="submit">Calibrate (crosshair on floor mark)</button>
    </form>
    <div class="stat-box wide"><div class="k">Tilt / Crosshair Distance</div><div class="v" id="s-cam">not calibrated</div></div>
    <form id="detectSizeForm">
      <div class="row"><label>Black V Max (0-255)</label><input id="detectBlackVMax" type="number" value="90" step="1" min="0" max="255"></div>
      <div class="row"><label>White S Max (0-255)</label><input id="detectWhiteSMax" type="number" value="40" step="1" min="0" max="255"></div>
      <div class="row"><label>White V Min (0-255)</label><input id="detectWhiteVMin" type="number" value="200" step="1" min="0" max="255"></div>
      <button type="submit">Set Detection Params</button>
    </form>
    <div class="stat-box wide"><div class="k">Line / Object Coverage (%)</div><div class="v" id="s-detect-size">--</div></div>
    <div class="stat-box wide"><div class="k">Front Cell Status</div><div class="v" id="s-front-cell">--</div></div>
    <div class="k">Detection Debug (edge map, line edges erased)</div>
    <img id="debugFeed" src="/debug_feed" alt="detection debug feed"
         onerror="this.replaceWith(Object.assign(document.createElement('div'), {textContent: 'Debug feed unavailable', style: 'color:#999; padding:12px; border:1px solid #444; border-radius:4px;'}))">
  </div>
  <div class="right">
    <canvas id="canvas" width="__CANVAS_PX__" height="__CANVAS_PX__"></canvas>
  </div>
</div>

<script>
const HALF_EXTENT = __HALF_EXTENT__;
const SPACING = __SPACING__;
const CANVAS_PX = __CANVAS_PX__;
const ROBOT_SIZE = __ROBOT_SIZE__;
const OBSTACLE_BOX_DEPTH = __OBSTACLE_BOX_DEPTH__;
const SCALE = CANVAS_PX / (2 * HALF_EXTENT); // px per cm

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let path = [];
let lastGoal = null;
let liveDetections = [];  // latest camera detections (distance_cm/bearing_deg), updated by pollDetections()
let candidateCells = [];  // cells accumulated so far during an in-progress obstacle hold, updated by pollObstacleWatch()

// Draw a single SPACING x SPACING world-cm grid cell as a filled+stroked
// box on the canvas -- shared by obstacles/pinned-cells/candidate-cells/
// live-detection rendering so they all look like consistent grid boxes.
function drawCellBox(i, j, fillStyle, strokeStyle) {
  const cx = i * SPACING;
  const cy = j * SPACING;
  const [px, py] = toPx(cx - SPACING / 2, cy + SPACING / 2);
  const size = SPACING * SCALE;
  ctx.fillStyle = fillStyle;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 1;
  ctx.fillRect(px, py, size, size);
  ctx.strokeRect(px, py, size, size);
}

function toPx(xcm, ycm) {
  return [CANVAS_PX / 2 + xcm * SCALE, CANVAS_PX / 2 - ycm * SCALE];
}

function draw(state) {
  ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);

  // grid
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  for (let c = -HALF_EXTENT; c <= HALF_EXTENT; c += SPACING) {
    let [px, ] = toPx(c, 0);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, CANVAS_PX); ctx.stroke();
    let [, py] = toPx(0, c);
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(CANVAS_PX, py); ctx.stroke();
  }
  // axes
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 1.5;
  let [ox, oy] = toPx(0, 0);
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, CANVAS_PX); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(CANVAS_PX, oy); ctx.stroke();

  // axis tick labels, in meters (1 decimal) -- internal math stays in cm
  // throughout, this only affects the displayed text. Skips 0 on each axis
  // to avoid overlap at the origin. View is a fixed +/-HALF_EXTENT square
  // (no panning), so the origin (ox, oy) is always the canvas center --
  // labels always go below/right of the axes.
  ctx.fillStyle = '#999';
  ctx.font = '11px monospace';
  for (let c = -HALF_EXTENT; c <= HALF_EXTENT; c += SPACING) {
    if (c === 0) continue;
    const m = (c / 100).toFixed(1);
    let [px, ] = toPx(c, 0);
    ctx.textAlign = 'center';
    ctx.fillText(m, px, oy + 14);
    let [, py] = toPx(0, c);
    ctx.textAlign = 'left';
    ctx.fillText(m, ox + 4, py + 4);
  }
  ctx.textAlign = 'left';
  ctx.fillStyle = '#ccc';
  ctx.fillText('X (m)', CANVAS_PX - 40, oy - 6);
  ctx.fillText('Y (m)', ox + 6, 12);

  // obstacles (blocked A* cells) -- manually-toggled cells drawn light/
  // semi-transparent; CONFIRMED camera-pinned cells (a subset) drawn
  // solid full red so they visually stand out as "this one is real,
  // camera-confirmed, not just a manual test block." These are permanent
  // (stay until manually cleared/toggled) -- if a cell you expect to see
  // pinned isn't here, it was never actually confirmed (see candidate/live
  // boxes below for what's still being evaluated).
  if (state.obstacles) {
    const pinnedKeys = new Set((state.pinned_cells || []).map(c => `${c[0]},${c[1]}`));
    for (const cell of state.obstacles) {
      const isPinned = pinnedKeys.has(`${cell[0]},${cell[1]}`);
      drawCellBox(cell[0], cell[1], isPinned ? '#ff0000' : 'rgba(255,60,60,0.35)', '#ff3c3c');
    }
  }

  // planned A* route (cell-to-cell, includes diagonals) -- dashed, distinct
  // from the solid blue trail of where the robot has actually been.
  if (state.planned_path && state.planned_path.length > 1) {
    ctx.strokeStyle = '#7CFC00';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    let [psx, psy] = toPx(state.planned_path[0][0], state.planned_path[0][1]);
    ctx.moveTo(psx, psy);
    for (const p of state.planned_path.slice(1)) {
      let [ppx, ppy] = toPx(p[0], p[1]);
      ctx.lineTo(ppx, ppy);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // path
  if (path.length > 1) {
    ctx.strokeStyle = '#4da3ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let [sx, sy] = toPx(path[0][0], path[0][1]);
    ctx.moveTo(sx, sy);
    for (const p of path.slice(1)) {
      let [px, py] = toPx(p[0], p[1]);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // candidate cells -- accumulated across the WHOLE current hold window
  // (not just the latest tick), so this box stays stable/visible for the
  // full hold instead of flickering with every noisy per-frame detection.
  // Drawn yellow: "being evaluated, not confirmed yet."
  for (const cell of candidateCells) {
    drawCellBox(cell[0], cell[1], 'rgba(255,220,0,0.45)', '#ffdc00');
  }

  // live camera-detected obstacles (this instant's raw detections, not yet
  // even part of a hold, and not currently pinned into the obstacle map --
  // ObstacleWatcher is paused, see its docstring) -- box projection: near
  // edge at the measured distance, extending OBSTACLE_BOX_DEPTH further
  // away (not centered on the point), spanning left-edge to right-edge
  // bearing. Expected to flicker frame to frame -- it's the raw/
  // unconfirmed signal, not the persisted obstacle map (see the solid red
  // boxes above for what's actually pinned).
  if (state.heading_deg !== null && state.heading_deg !== undefined) {
    for (const det of liveDetections) {
      if (det.distance_cm === null || det.distance_cm === undefined) continue;
      if (det.bearing_deg === null || det.bearing_deg === undefined) continue;
      const bearings = [det.left_bearing_deg, det.bearing_deg, det.right_bearing_deg]
            .filter(b => b !== null && b !== undefined);
      const distances = [det.distance_cm, det.distance_cm + OBSTACLE_BOX_DEPTH];
      const cellIs = [], cellJs = [];
      for (const bearingDeg of bearings) {
        const worldBearingRad = (state.heading_deg + bearingDeg) * Math.PI / 180;
        const cosB = Math.cos(worldBearingRad), sinB = Math.sin(worldBearingRad);
        for (const d of distances) {
          const ox = state.x + d * cosB;
          const oy = state.y + d * sinB;
          cellIs.push(Math.round(ox / SPACING));
          cellJs.push(Math.round(oy / SPACING));
        }
      }
      if (!cellIs.length) continue;
      const iMin = Math.min(...cellIs), iMax = Math.max(...cellIs);
      const jMin = Math.min(...cellJs), jMax = Math.max(...cellJs);
      for (let ci = iMin; ci <= iMax; ci++) {
        for (let cj = jMin; cj <= jMax; cj++) {
          drawCellBox(ci, cj, 'rgba(255,165,0,0.35)', '#ffa500');
        }
      }
    }
  }

  // goal marker
  if (state.goal) {
    let [gx, gy] = toPx(state.goal[0], state.goal[1]);
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(gx - 6, gy - 6); ctx.lineTo(gx + 6, gy + 6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 6, gy + 6); ctx.lineTo(gx + 6, gy - 6); ctx.stroke();
  }

  // robot footprint box (ROBOT_SIZE x ROBOT_SIZE cm) -- state.x/y is the
  // FRONT-LEFT wheel, one CORNER of the box (not its center), so in the
  // robot's own local frame (+x = forward, +y = left) the box spans
  // local x in [-ROBOT_SIZE, 0] (body is BEHIND the front edge) and
  // local y in [-ROBOT_SIZE, 0] (body is to the RIGHT of the left edge).
  // Rotated into world space by the current heading, since the box's
  // orientation (not just position) changes as the robot turns.
  const headingDeg = state.heading_deg ?? 0;
  const rad = headingDeg * Math.PI / 180;
  const cosH = Math.cos(rad), sinH = Math.sin(rad);
  const localCorners = [[0, 0], [0, -ROBOT_SIZE], [-ROBOT_SIZE, -ROBOT_SIZE], [-ROBOT_SIZE, 0]];
  const boxPx = localCorners.map(([lx, ly]) => toPx(
    state.x + lx * cosH - ly * sinH,
    state.y + lx * sinH + ly * cosH
  ));
  ctx.fillStyle = 'rgba(77,163,255,0.15)';
  ctx.strokeStyle = '#4da3ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(boxPx[0][0], boxPx[0][1]);
  for (let k = 1; k < boxPx.length; k++) ctx.lineTo(boxPx[k][0], boxPx[k][1]);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // robot arrow (heading: 0deg = +X, math convention; canvas Y is flipped)
  const [rx, ry] = toPx(state.x, state.y);
  const len = SPACING * 0.8 * SCALE;
  const tipX = rx + len * Math.cos(rad);
  const tipY = ry - len * Math.sin(rad);
  ctx.strokeStyle = '#ffa500';
  ctx.fillStyle = '#ffa500';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.beginPath();
  ctx.arc(tipX, tipY, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(rx, ry, 3, 0, 2 * Math.PI);
  ctx.fillStyle = '#4da3ff';
  ctx.fill();

  // direction label next to the arrow tip: degrees + nearest axis direction
  // (relative to heading_ref captured at startup, where 0deg = +X -- this
  // is NOT true compass north, just this session's local reference frame)
  ctx.fillStyle = '#ffa500';
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`${headingDeg.toFixed(0)}° (${axisLabel(headingDeg)})`, tipX + 8, tipY);
}

function axisLabel(deg) {
  const dirs = ['+X', '+X/+Y', '+Y', '-X/+Y', '-X', '-X/-Y', '-Y', '+X/-Y'];
  const idx = Math.round(((deg % 360) + 360) % 360 / 45) % 8;
  return dirs[idx];
}

function fmt(v) { return (v === null || v === undefined) ? 'n/a' : v.toFixed(1) + 'deg'; }

function set(id, text) { document.getElementById(id).textContent = text; }

function updateStatus(state) {
  const legInfo = (state.state === 'RUNNING') ? `${state.leg_idx}/${state.leg_count}` : '-';
  set('s-pos', `(${(state.x / 100).toFixed(1)}, ${(state.y / 100).toFixed(1)}) m`);
  set('s-state', state.awaiting_continue ? 'PAUSED (measure now)' : state.state);
  set('s-phase', state.phase ?? '-');
  set('s-leg', legInfo);
  set('s-progress', (state.leg_progress_cm === null || state.leg_progress_cm === undefined)
        ? '-' : `${(state.leg_progress_cm / 100).toFixed(1)} / ${(state.leg_target_distance_cm / 100).toFixed(1)} m`);
  set('s-enddir', state.end_dir_deg === null || state.end_dir_deg === undefined
        ? 'n/a' : `${state.end_dir_deg.toFixed(0)}deg`);
  set('s-speed', `${state.forward_speed.toFixed(2)} / ${state.rotate_speed.toFixed(2)}`);
  set('s-yaw', fmt(state.yaw_deg));
  set('s-heading', fmt(state.heading_deg));
  set('s-target', fmt(state.target_heading_deg));
  document.getElementById('continueBtn').style.display = state.awaiting_continue ? 'block' : 'none';
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const state = await res.json();
    path = state.path;
    lastGoal = state.goal;
    draw(state);
    updateStatus(state);
  } catch (e) {
    set('s-state', 'connection lost');
  }
}

document.getElementById('goalForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const x = parseFloat(document.getElementById('goalX').value);
  const y = parseFloat(document.getElementById('goalY').value);
  const dir = parseFloat(document.getElementById('goalDir').value);
  const step = document.getElementById('goalStep').checked;
  await fetch('/api/goal', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: x, y: y, dir: dir, step: step})
  });
});

document.getElementById('continueBtn').addEventListener('click', async () => {
  await fetch('/api/continue_step', {method: 'POST'});
});

document.getElementById('speedForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const forward = parseFloat(document.getElementById('speedFwd').value);
  const rotate = parseFloat(document.getElementById('speedRot').value);
  await fetch('/api/set_speed', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({forward: forward, rotate: rotate})
  });
});

document.getElementById('resetBtn').addEventListener('click', async () => {
  await fetch('/api/reset_position', {method: 'POST'});
});

document.getElementById('clearObstaclesBtn').addEventListener('click', async () => {
  await fetch('/api/clear_obstacles', {method: 'POST'});
});

canvas.addEventListener('click', async (ev) => {
  const rect = canvas.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * (CANVAS_PX / rect.width);
  const py = (ev.clientY - rect.top) * (CANVAS_PX / rect.height);
  const xcm = (px - CANVAS_PX / 2) / SCALE;
  const ycm = -(py - CANVAS_PX / 2) / SCALE;
  const i = Math.round(xcm / SPACING);
  const j = Math.round(ycm / SPACING);
  await fetch('/api/toggle_obstacle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({i: i, j: j})
  });
});

document.getElementById('camCalibForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const height = parseFloat(document.getElementById('camHeight').value);
  const dist = parseFloat(document.getElementById('camDist').value);
  const vfov = parseFloat(document.getElementById('camVfov').value);
  await fetch('/api/camera_calibrate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({height_cm: height, distance_cm: dist, vfov_deg: vfov})
  });
});

function guideLineColor(distanceCm) {
  if (distanceCm <= 100) return '#ff4d4d';   // near -- red
  if (distanceCm <= 200) return '#ffd24d';   // mid -- yellow
  return '#4dff88';                          // far -- green
}

function renderGuideLines(cs) {
  const container = document.getElementById('guideLines');
  container.innerHTML = '';
  if (!cs.calibrated) return;
  for (const g of cs.guide_lines) {
    const color = guideLineColor(g.distance_cm);
    const topPct = `${(g.y_frac * 100).toFixed(2)}%`;

    const line = document.createElement('div');
    line.className = 'guide-line';
    line.style.top = topPct;
    line.style.background = color;
    container.appendChild(line);

    const label = document.createElement('div');
    label.className = 'guide-line-label';
    label.style.top = topPct;
    label.style.color = color;
    label.textContent = `${g.distance_cm}cm`;
    container.appendChild(label);
  }
}

async function pollCamera() {
  try {
    const res = await fetch('/api/camera_state');
    const cs = await res.json();
    if (cs.calibrated) {
      set('s-cam', `${cs.tilt_deg.toFixed(1)}° tilt, ${cs.vfov_deg.toFixed(0)}° vfov -- `
            + `${(cs.crosshair_distance_cm / 100).toFixed(2)} m at crosshair`);
    } else {
      set('s-cam', 'not calibrated');
    }
    renderGuideLines(cs);
  } catch (e) {
    set('s-cam', 'connection lost');
  }
}

function detectionColor() {
  return '#ffa500';
}

async function pollDetections() {
  try {
    const res = await fetch('/api/detections');
    const ds = await res.json();
    liveDetections = ds.detections;  // cached for draw() to project onto the grid map
    const container = document.getElementById('detectionBoxes');
    container.innerHTML = '';
    for (const det of ds.detections) {
      const color = detectionColor(det.label);
      const box = document.createElement('div');
      box.className = 'detection-box';
      box.style.left = `${(det.x / ds.frame_width * 100).toFixed(2)}%`;
      box.style.top = `${(det.y / ds.frame_height * 100).toFixed(2)}%`;
      box.style.width = `${(det.w / ds.frame_width * 100).toFixed(2)}%`;
      box.style.height = `${(det.h / ds.frame_height * 100).toFixed(2)}%`;
      box.style.borderColor = color;

      const label = document.createElement('div');
      label.className = 'detection-box-label';
      label.style.color = color;
      const distText = (det.distance_cm === null || det.distance_cm === undefined)
            ? '?m' : `${(det.distance_cm / 100).toFixed(2)}m`;
      label.textContent = `${det.label} ~${distText}`;
      box.appendChild(label);

      container.appendChild(box);
    }
    if (ds.line_coverage_pct !== null && ds.line_coverage_pct !== undefined) {
      const obj = (ds.object_coverage_pct === null || ds.object_coverage_pct === undefined)
            ? '?' : ds.object_coverage_pct.toFixed(1);
      set('s-detect-size', `${ds.line_coverage_pct.toFixed(1)}% / ${obj}%`);
    } else {
      set('s-detect-size', 'n/a (no camera)');
    }
  } catch (e) {
    // detection endpoint unavailable -- leave last-drawn boxes as-is
  }
}

document.getElementById('detectSizeForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const blackVMax = parseFloat(document.getElementById('detectBlackVMax').value);
  const whiteSMax = parseFloat(document.getElementById('detectWhiteSMax').value);
  const whiteVMin = parseFloat(document.getElementById('detectWhiteVMin').value);
  await fetch('/api/detection_settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      black_value_max: blackVMax,
      white_sat_max: whiteSMax,
      white_value_min: whiteVMin
    })
  });
});

async function pollObstacleWatch() {
  try {
    const res = await fetch('/api/obstacle_watch');
    const ow = await res.json();
    candidateCells = ow.candidate_cells || [];
    if (!ow.active) {
      set('s-obwatch', 'n/a (no camera)');
    } else if (ow.paused) {
      set('s-obwatch', 'PAUSED -- detection not wired to avoidance yet (add obstacles manually)');
    } else if (!ow.enabled) {
      set('s-obwatch', 'DISABLED');
    } else if (ow.holding) {
      set('s-obwatch', `HOLDING -- ${ow.hit_count}/${ow.confirm_count} confirmations, ${ow.remaining_s.toFixed(1)}s left`);
    } else {
      set('s-obwatch', `clear (hold=${ow.hold_duration_s}s, confirm=${ow.confirm_count})`);
    }
    if (ow.front_cell) {
      set('s-front-cell', `(${ow.front_cell[0]}, ${ow.front_cell[1]}) -- ${ow.front_cell_status}`);
    } else {
      set('s-front-cell', 'n/a');
    }
  } catch (e) {
    set('s-obwatch', 'connection lost');
  }
}

document.getElementById('obWatchForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const enabled = document.getElementById('obEnabled').checked;
  const holdSec = parseFloat(document.getElementById('obHoldSec').value);
  const confirmCount = parseInt(document.getElementById('obConfirmCount').value, 10);
  await fetch('/api/obstacle_watch_settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({enabled: enabled, hold_duration_s: holdSec, confirm_count: confirmCount})
  });
});

setInterval(poll, __POLL_MS__);
setInterval(pollCamera, __POLL_MS__);
setInterval(pollDetections, __POLL_MS__);
setInterval(pollObstacleWatch, __POLL_MS__);
poll();
pollCamera();
pollDetections();
pollObstacleWatch();
</script>
</body>
</html>
"""

WEB_PORT = 8080
GUI_POLL_MS = int(1000 / GUI_HZ)
CANVAS_PX = 900  # bumped up from 700 for a bigger view of the 5m x 5m grid


def render_page():
    return (HTML_PAGE
            .replace('__HALF_EXTENT__', str(GRID_HALF_EXTENT_CM))
            .replace('__SPACING__', str(GRID_SPACING_CM))
            .replace('__CANVAS_PX__', str(CANVAS_PX))
            .replace('__POLL_MS__', str(GUI_POLL_MS))
            .replace('__STEP_SIZE__', f'{STEP_SIZE_CM / 100:.1f}')
            .replace('__FORWARD_SPEED__', f'{FORWARD_SPEED:.2f}')
            .replace('__ROTATE_SPEED__', f'{ROTATE_SPEED:.2f}')
            .replace('__ROBOT_SIZE__', str(ROBOT_SIZE_CM))
            .replace('__OBSTACLE_BOX_DEPTH__', str(OBSTACLE_BOX_DEPTH_CM)))


def create_app(node: GridNavNode, camera: 'CameraStreamer | None',
               rangefinder: CameraRangefinder,
               detector: 'ObstacleDetector | None',
               watcher: 'ObstacleWatcher | None') -> Flask:
    app = Flask(__name__)
    # Werkzeug's request logging is noisy at GUI_HZ polling rates (and would
    # be far worse for the continuous /video_feed stream).
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    @app.route('/')
    def index():
        return Response(render_page(), mimetype='text/html')

    @app.route('/video_feed')
    def video_feed():
        if camera is None:
            return Response('Camera not available', status=503)

        def gen():
            interval = 1.0 / CAMERA_FPS
            while True:
                jpeg = camera.get_jpeg()
                if jpeg is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                time.sleep(interval)

        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/debug_feed')
    def debug_feed():
        if detector is None:
            return Response('Detector not available', status=503)

        def gen():
            interval = 1.0 / DETECTION_FPS
            while True:
                png = detector.get_debug_frame()
                if png is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/png\r\n\r\n' + png + b'\r\n')
                time.sleep(interval)

        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/camera_calibrate', methods=['POST'])
    def api_camera_calibrate():
        data = request.get_json(force=True)
        try:
            height_cm = float(data['height_cm'])
            distance_cm = float(data['distance_cm'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'invalid height/distance'}), 400
        vfov_deg = data.get('vfov_deg', None)
        try:
            vfov_deg = float(vfov_deg) if vfov_deg not in (None, '') else None
        except (TypeError, ValueError):
            vfov_deg = None
        if not rangefinder.calibrate(height_cm, distance_cm, vfov_deg):
            return jsonify({'ok': False, 'error': 'height/distance/vfov must be positive'}), 400
        return jsonify({'ok': True})

    @app.route('/api/camera_state')
    def api_camera_state():
        return jsonify(rangefinder.get_snapshot(CAMERA_HEIGHT))

    @app.route('/api/detections')
    def api_detections():
        return jsonify({
            'frame_width': CAMERA_WIDTH,
            'frame_height': CAMERA_HEIGHT,
            'detections': detector.get_detections() if detector is not None else [],
            'line_coverage_pct': detector.get_line_coverage() if detector is not None else None,
            'object_coverage_pct': detector.get_object_coverage() if detector is not None else None,
        })

    @app.route('/api/detection_settings', methods=['POST'])
    def api_detection_settings():
        if detector is None:
            return jsonify({'ok': False, 'error': 'no detector running (camera unavailable)'}), 400
        data = request.get_json(force=True)
        ok = True
        if 'black_value_max' in data:
            try:
                ok = detector.set_black_value_max(float(data['black_value_max'])) and ok
            except (TypeError, ValueError):
                ok = False
        if 'white_sat_max' in data:
            try:
                ok = detector.set_white_sat_max(float(data['white_sat_max'])) and ok
            except (TypeError, ValueError):
                ok = False
        if 'white_value_min' in data:
            try:
                ok = detector.set_white_value_min(float(data['white_value_min'])) and ok
            except (TypeError, ValueError):
                ok = False
        if not ok:
            return jsonify({'ok': False, 'error': 'invalid detection settings'}), 400
        return jsonify({'ok': True})

    @app.route('/api/obstacle_watch')
    def api_obstacle_watch():
        if watcher is None:
            return jsonify({'active': False})
        status = watcher.get_status()
        status['active'] = True
        status['candidate_cells'] = [list(c) for c in watcher.get_candidate_cells()]
        front_cell = node.get_front_cell()
        status['front_cell'] = list(front_cell) if front_cell is not None else None
        status['front_cell_status'] = node.get_cell_status(front_cell) if front_cell is not None else None
        return jsonify(status)

    @app.route('/api/obstacle_watch_settings', methods=['POST'])
    def api_obstacle_watch_settings():
        if watcher is None:
            return jsonify({'ok': False, 'error': 'no obstacle watcher running (camera unavailable)'}), 400
        data = request.get_json(force=True)
        ok = True
        if 'hold_duration_s' in data:
            try:
                ok = watcher.set_hold_duration(float(data['hold_duration_s'])) and ok
            except (TypeError, ValueError):
                ok = False
        if 'confirm_count' in data:
            try:
                ok = watcher.set_confirm_count(int(data['confirm_count'])) and ok
            except (TypeError, ValueError):
                ok = False
        if 'enabled' in data:
            watcher.set_enabled(bool(data['enabled']))
        if not ok:
            return jsonify({'ok': False, 'error': 'invalid hold_duration_s/confirm_count'}), 400
        return jsonify({'ok': True})

    @app.route('/api/state')
    def api_state():
        return jsonify(node.get_snapshot())

    @app.route('/api/goal', methods=['POST'])
    def api_goal():
        data = request.get_json(force=True)
        try:
            gx = float(data['x'])
            gy = float(data['y'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'invalid x/y'}), 400
        end_dir = data.get('dir', None)
        try:
            end_dir = float(end_dir) if end_dir not in (None, '') else None
        except (TypeError, ValueError):
            end_dir = None
        step_mode = bool(data.get('step', False))
        node.set_goal(gx, gy, end_dir, step_mode)
        return jsonify({'ok': True})

    @app.route('/api/set_speed', methods=['POST'])
    def api_set_speed():
        data = request.get_json(force=True)
        forward = data.get('forward', None)
        rotate = data.get('rotate', None)
        try:
            forward = float(forward) if forward not in (None, '') else None
            rotate = float(rotate) if rotate not in (None, '') else None
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'invalid speed value'}), 400
        node.set_speed(forward=forward, rotate=rotate)
        return jsonify({'ok': True})

    @app.route('/api/toggle_obstacle', methods=['POST'])
    def api_toggle_obstacle():
        data = request.get_json(force=True)
        try:
            i = int(data['i'])
            j = int(data['j'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'invalid cell'}), 400
        node.toggle_obstacle(i, j)
        return jsonify({'ok': True})

    @app.route('/api/clear_obstacles', methods=['POST'])
    def api_clear_obstacles():
        node.clear_obstacles()
        return jsonify({'ok': True})

    @app.route('/api/reset_position', methods=['POST'])
    def api_reset_position():
        node.reset_position()
        return jsonify({'ok': True})

    @app.route('/api/continue_step', methods=['POST'])
    def api_continue_step():
        node.continue_step()
        return jsonify({'ok': True})

    return app


def main(args=None):
    print(f'grid_nav.py {SCRIPT_VERSION}')
    rclpy.init(args=args)
    node = GridNavNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    camera = CameraStreamer(CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
                             CAMERA_FPS, CAMERA_JPEG_QUALITY)
    if camera.start():
        node.get_logger().info(f'Camera streaming from device index {CAMERA_DEVICE_INDEX}')
    else:
        node.get_logger().warn(
            f'Could not open camera at device index {CAMERA_DEVICE_INDEX} -- '
            f'/video_feed will report unavailable. Check `ls /dev/video*`.'
        )
        camera = None

    rangefinder = CameraRangefinder()

    detector = None
    watcher = None
    if camera is not None:
        detector = ObstacleDetector(camera, rangefinder)
        detector.start()
        node.get_logger().info(f'Obstacle detector running at {DETECTION_FPS} Hz')

        watcher = ObstacleWatcher(node, detector)
        watcher.start()
        node.get_logger().info('Obstacle confirmation watcher running')

    app = create_app(node, camera, rangefinder, detector, watcher)
    node.get_logger().info(f'Web GUI at http://<this-device-ip>:{WEB_PORT}')
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        if watcher is not None:
            watcher.stop()
        if detector is not None:
            detector.stop()
        if camera is not None:
            camera.stop()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
