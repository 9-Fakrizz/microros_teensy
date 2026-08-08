"""
Grid navigation with live GUI -- runs on the Pi5 as a ROS2 node.

Robot model this script assumes:
  - Position (x, y) is tracked at the LEFT wheel's contact point.
  - Straight-line distance is measured from the LEFT wheel encoder
    (wheel_encoder data[0] -- see IMPORTANT note below).
  - Turning is done by a PIVOT about the left wheel: left wheel stays
    stopped, only the right wheel drives, so the tracked (x, y) point
    does not move during a turn -- only heading changes.
  - Movement is planned on a grid of GRID_SPACING_CM cells using A*
    (8-directional: N/S/E/W plus the 4 diagonals), so the robot CAN
    drive diagonally and will route around any cells marked as
    obstacles in the GUI. The planned path is compressed into a
    sequence of straight-line legs (each a single rotate + drive),
    one per run of consecutive same-direction grid steps -- not one
    tiny hop per cell. If no path exists (goal fully blocked), the
    goal is rejected and logged.

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
  USB webcam (see CAMERA_DEVICE_INDEX) is also shown, if available.

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
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

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

    DEFAULT_VFOV_DEG = 45.0  # typical-ish USB webcam vertical FOV; tune via the GUI for your camera

    def __init__(self):
        self._lock = threading.Lock()
        self.height_cm = None
        self.tilt_deg = None
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
        self.planned_path = []       # [(x_cm, y_cm), ...] cell centers of the last A* route, for GUI overlay

        # Step mode: pause fully after each STEP_SIZE_CM of DRIVE travel
        # and wait for continue_step() before resuming, so you can measure
        # one grid box at a time instead of a whole leg in one go.
        self.step_mode = False
        self.awaiting_continue = False
        self.step_baseline_pulses = 0

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
            else:
                self.obstacles.add(cell)

    def clear_obstacles(self):
        with self._lock:
            self.obstacles.clear()

    def set_goal(self, gx, gy, end_dir_deg=None, step_mode=False):
        with self._lock:
            start_cell = self._to_cell(self.x, self.y)
            goal_cell = self._to_cell(gx, gy)

            legs = []
            planned_path = []
            if start_cell != goal_cell:
                cell_path = astar_search(start_cell, goal_cell, self.obstacles)
                if cell_path is None:
                    self.goal = (gx, gy)
                    self.get_logger().warn(
                        f'No path to ({gx:.1f}, {gy:.1f}) cm -- blocked by obstacles or out of range.'
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

            # Live display position: committed x/y plus in-progress DRIVE
            # movement along the current leg's direction, so the GUI marker
            # and position readout move in real time instead of jumping
            # only when a leg completes.
            display_x, display_y = self.x, self.y
            if self.state == 'RUNNING' and self.phase == 'DRIVE' and self.legs \
                    and self.legs[self.leg_idx][0] == 'move':
                display_x = self.leg_start_x + self.leg_unit_dx * self.leg_progress_cm
                display_y = self.leg_start_y + self.leg_unit_dy * self.leg_progress_cm

            return {
                'x': display_x,
                'y': display_y,
                'path': list(self.path),
                'planned_path': list(self.planned_path),
                'obstacles': [list(c) for c in self.obstacles],
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
  body { font-family: sans-serif; background: #1e1e1e; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 16px; font-weight: normal; color: #aaa; margin: 0 0 12px 0; }
  .layout { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .left { display: flex; flex-direction: column; gap: 12px; min-width: 260px; }
  .camera { display: flex; flex-direction: column; gap: 6px; min-width: 320px; max-width: 480px; }
  .camera-wrap { position: relative; }
  .camera img { width: 100%; background: #111; border: 1px solid #444; border-radius: 4px; display: block; }
  .camera .k { font-size: 11px; color: #999; text-transform: uppercase; }
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
  .right { flex: 1; }
  #canvas { background: #111; border: 1px solid #444; display: block; max-width: 100%; height: auto; cursor: crosshair; }
  form { background: #262626; border: 1px solid #444; padding: 10px; border-radius: 6px; }
  form .row { margin-bottom: 8px; }
  input { width: 90px; font-size: 14px; padding: 4px; }
  button { font-size: 14px; padding: 6px 16px; margin-top: 4px; width: 100%; }
  label { display: block; font-size: 12px; color: #aaa; margin-bottom: 2px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-box { background: #262626; border: 1px solid #444; border-radius: 6px; padding: 8px 10px; }
  .stat-box .k { font-size: 11px; color: #999; text-transform: uppercase; }
  .stat-box .v { font-family: monospace; font-size: 15px; color: #eee; margin-top: 2px; }
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
    </div>
  </div>
  <div class="camera">
    <div class="k">Camera</div>
    <div class="camera-wrap">
      <img id="cameraFeed" src="/video_feed" alt="camera feed"
           onerror="this.replaceWith(Object.assign(document.createElement('div'), {textContent: 'Camera unavailable', style: 'color:#999; padding:12px; border:1px solid #444; border-radius:4px;'}))">
      <div class="center-line"></div>
      <div class="guide-lines" id="guideLines"></div>
      <div class="crosshair"></div>
    </div>
    <form id="camCalibForm">
      <div class="row"><label>Camera Height Above Floor (cm)</label><input id="camHeight" type="number" value="26" step="0.5"></div>
      <div class="row"><label>Known Distance at Crosshair (cm)</label><input id="camDist" type="number" value="100" step="1"></div>
      <div class="row"><label>Vertical FOV (deg, tune for accuracy)</label><input id="camVfov" type="number" value="45" step="1"></div>
      <button type="submit">Calibrate (crosshair on floor mark)</button>
    </form>
    <div class="stat-box wide"><div class="k">Tilt / Crosshair Distance</div><div class="v" id="s-cam">not calibrated</div></div>
  </div>
  <div class="right">
    <canvas id="canvas" width="__CANVAS_PX__" height="__CANVAS_PX__"></canvas>
  </div>
</div>

<script>
const HALF_EXTENT = __HALF_EXTENT__;
const SPACING = __SPACING__;
const CANVAS_PX = __CANVAS_PX__;
const SCALE = CANVAS_PX / (2 * HALF_EXTENT); // px per cm

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let path = [];
let lastGoal = null;

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

  // obstacles (blocked A* cells)
  if (state.obstacles) {
    ctx.fillStyle = 'rgba(255,60,60,0.35)';
    ctx.strokeStyle = '#ff3c3c';
    ctx.lineWidth = 1;
    for (const cell of state.obstacles) {
      const cx = cell[0] * SPACING;
      const cy = cell[1] * SPACING;
      const [px, py] = toPx(cx - SPACING / 2, cy + SPACING / 2);
      const size = SPACING * SCALE;
      ctx.fillRect(px, py, size, size);
      ctx.strokeRect(px, py, size, size);
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

  // goal marker
  if (state.goal) {
    let [gx, gy] = toPx(state.goal[0], state.goal[1]);
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(gx - 6, gy - 6); ctx.lineTo(gx + 6, gy + 6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 6, gy + 6); ctx.lineTo(gx + 6, gy - 6); ctx.stroke();
  }

  // robot arrow (heading: 0deg = +X, math convention; canvas Y is flipped)
  const headingDeg = state.heading_deg ?? 0;
  const rad = headingDeg * Math.PI / 180;
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

setInterval(poll, __POLL_MS__);
setInterval(pollCamera, __POLL_MS__);
poll();
pollCamera();
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
            .replace('__ROTATE_SPEED__', f'{ROTATE_SPEED:.2f}'))


def create_app(node: GridNavNode, camera: 'CameraStreamer | None',
               rangefinder: CameraRangefinder) -> Flask:
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

    app = create_app(node, camera, rangefinder)
    node.get_logger().info(f'Web GUI at http://<this-device-ip>:{WEB_PORT}')
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.stop()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
