"""
mock_robot.py -- run grid_nav.py's REAL navigation / obstacle / AprilTag /
web-GUI logic against a SIMULATED robot instead of real ROS2 + hardware,
so you can test the whole app on any machine -- no Teensy, no camera, no
ROS2 install, no micro-ROS agent.

Run:
    python mock_robot.py

Then open http://localhost:8080 in a browser -- it is the EXACT same web
GUI grid_nav.py serves for real; only the "hardware" underneath is faked.
A small command console also runs in this terminal (type `help`).

------------------------------------------------------------------------
What's simulated
------------------------------------------------------------------------
- IMU + wheel encoder feedback: a background thread reads the REAL
  GridNavNode's own control_loop() state (state / phase / homing_stage /
  leg_target_heading / homing_target_heading) each tick and integrates a
  simple true (yaw, traveled-pulses) pair, then feeds it back in through
  the SAME imu_callback()/encoder_callback() entry points a real
  micro-ROS bridge would use. This deliberately does NOT try to reverse
  -engineer the published Twist/PWM mixing (that's real-hardware-wiring
  -specific, see PIVOT_ANGULAR_SIGN etc. in grid_nav.py) -- reading the
  node's own already-decided intent (which phase it's in) is simpler and
  exactly matches what that phase is SUPPOSED to do.
- The start-point AprilTag: FakeAprilTagLocalizer computes a plausible
  visible / distance / frame-offset reading each tick from the simulated
  robot's true pose vs. a configured world tag position (see
  TAG_WORLD_X_CM/TAG_WORLD_Y_CM below) -- so Setup / Confirm Start
  Position / the full return-home sequence can all be tested without a
  camera. No noise is injected -- this is for exercising the app's LOGIC,
  not for testing robustness to sensor error.

------------------------------------------------------------------------
What's NOT simulated
------------------------------------------------------------------------
- The live camera feed / debug feed -- reports "unavailable" in the GUI,
  same as grid_nav.py's own graceful behavior with no camera attached.
- Automatic obstacle detection (ObstacleWatcher) -- but A* obstacle
  AVOIDANCE is still fully testable: click grid cells on the GUI map to
  toggle them as obstacles, exactly like with a real robot.
"""

import math
import sys
import threading
import time
import types

# ---------------- Stub the ROS2 pieces grid_nav.py imports ----------------
# Only rclpy/message types are faked -- cv2, Flask, numpy stay real, so
# the actual web GUI is served for real over HTTP.
for _name in ('rclpy', 'rclpy.node', 'sensor_msgs', 'sensor_msgs.msg',
              'geometry_msgs', 'geometry_msgs.msg', 'std_msgs', 'std_msgs.msg'):
    sys.modules[_name] = types.ModuleType(_name)


class _StubLogger:
    def info(self, *a, **k):
        if a:
            print(f'[grid_nav] {a[0]}')

    def warn(self, *a, **k):
        if a:
            print(f'[grid_nav] WARN: {a[0]}')


class _StubNode:
    """Replaces rclpy.node.Node -- GridNavNode.__init__ calls
    create_subscription/create_publisher/create_timer once each; none of
    them need to do anything real here (this script drives
    control_loop()/imu_callback()/encoder_callback() directly instead of
    through rclpy's subscription/timer machinery)."""

    def __init__(self, name):
        pass

    def create_subscription(self, *a, **k):
        return None

    def create_publisher(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return None

    def get_logger(self):
        return _StubLogger()

    def destroy_node(self):
        pass


class _FakeTwist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0)
        self.angular = types.SimpleNamespace(z=0.0)


class _FakeImu:
    def __init__(self, yaw_rad):
        self.orientation = types.SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(yaw_rad / 2.0), w=math.cos(yaw_rad / 2.0)
        )


class _FakeEncoder:
    def __init__(self, pulses):
        self.data = [int(pulses)]


sys.modules['rclpy.node'].Node = _StubNode
sys.modules['sensor_msgs.msg'].Imu = object
sys.modules['geometry_msgs.msg'].Twist = _FakeTwist
sys.modules['std_msgs.msg'].Int32MultiArray = object
sys.modules['rclpy'].init = lambda args=None: None
sys.modules['rclpy'].shutdown = lambda: None
sys.modules['rclpy'].spin = lambda node: None

import grid_nav as gn  # noqa: E402 -- must come after the stubs above

# ---------------- Simulation tuning -- console/GUI only, not physical ----
SIM_ROTATE_RATE_DEG_S = 60.0   # how fast the simulated robot turns
SIM_DRIVE_RATE_CM_S = 25.0     # how fast the simulated robot drives straight
SIM_TICK_HZ = 20.0             # matches grid_nav.py's own LOOP_HZ

# Where the start-point AprilTag "physically sits", in the SAME world
# coordinate frame the robot's own (x, y) live in. (0, -200) means 200cm
# in the -Y direction from wherever (0, 0) ends up after calibration --
# i.e. the tag is 2m "ahead" of the robot's default starting pose, which
# starts facing -90deg (see boot heading-lock below) so it's looking
# straight at the tag from the very first tick.
TAG_WORLD_X_CM = 0.0
TAG_WORLD_Y_CM = -200.0
TAG_FOV_DEG = 70.0              # how wide a "field of view" the fake camera has


class FakeAprilTagLocalizer:
    """Minimal stand-in for grid_nav.AprilTagLocalizer -- implements
    exactly the subset of its interface GridNavNode/create_app() use, but
    derives each reading from the SIMULATED robot's true pose vs a fixed
    world tag position instead of a real camera + pupil_apriltags.

    Runs its own background thread (like the real class) so the
    trigger-on-close-enough side effect (correct_position_from_tag())
    fires from OUTSIDE any GridNavNode lock, exactly like the real
    class's own _tick() does -- calling it from inside get_status() would
    risk deadlocking against GridNavNode's own lock, since that's called
    from within control_loop() while the lock is already held."""

    def __init__(self, node, world_x_cm=TAG_WORLD_X_CM, world_y_cm=TAG_WORLD_Y_CM,
                 fov_deg=TAG_FOV_DEG, tag_id=6, tag_size_cm=16.0,
                 trigger_distance_cm=50.0, frame_width=640, frame_height=480):
        self.node = node
        self._world_x_cm = world_x_cm
        self._world_y_cm = world_y_cm
        self._fov_deg = fov_deg
        self._frame_width = frame_width
        self._frame_height = frame_height

        self._settings_lock = threading.Lock()
        self._tag_id = int(tag_id)
        self._start_tag_id = int(tag_id)
        self._tag_size_cm = tag_size_cm
        self._start_tag_size_cm = tag_size_cm
        self._trigger_distance_cm = trigger_distance_cm

        self._lock = threading.Lock()
        self._visible = False
        self._distance_cm = None
        self._bbox = None
        self._triggered = False
        self.forced_hidden = False  # console `tag off` -- force "not visible" for testing the odometry fallback

        self._running = False
        self._thread = None

    # -- pin-position tag config (not exercised by the sim, just satisfies the API) --
    def get_tag_id(self):
        with self._settings_lock:
            return self._tag_id

    def set_tag_id(self, v):
        with self._settings_lock:
            self._tag_id = int(v)
        return True

    def get_world_position(self):
        with self._settings_lock:
            return (self._world_x_cm, self._world_y_cm)

    def set_world_position(self, x, y):
        with self._settings_lock:
            self._world_x_cm = float(x)
            self._world_y_cm = float(y)
        return True

    def get_tag_size(self):
        with self._settings_lock:
            return self._tag_size_cm

    def set_tag_size(self, v):
        v = float(v)
        if v <= 0:
            return False
        with self._settings_lock:
            self._tag_size_cm = v
        return True

    def get_trigger_distance(self):
        with self._settings_lock:
            return self._trigger_distance_cm

    def set_trigger_distance(self, v):
        v = float(v)
        if v <= 0:
            return False
        with self._settings_lock:
            self._trigger_distance_cm = v
        return True

    # -- start-point tag config --
    def get_start_tag_id(self):
        with self._settings_lock:
            return self._start_tag_id

    def set_start_tag_id(self, v):
        with self._settings_lock:
            self._start_tag_id = int(v)
        return True

    def get_start_tag_size(self):
        with self._settings_lock:
            return self._start_tag_size_cm

    def set_start_tag_size(self, v):
        v = float(v)
        if v <= 0:
            return False
        with self._settings_lock:
            self._start_tag_size_cm = v
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
        interval = 1.0 / 10.0
        while self._running:
            start = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                print(f'[mock_robot] FakeAprilTagLocalizer tick failed: {e!r}')
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _tick(self):
        if self.forced_hidden:
            with self._lock:
                self._visible = False
                self._distance_cm = None
                self._bbox = None
                self._triggered = False
            return

        with self.node._lock:
            current_yaw = self.node.current_yaw
            heading_ref = self.node.heading_ref
            x, y = self.node._live_position_locked()
        if current_yaw is None or heading_ref is None:
            with self._lock:
                self._visible = False
                self._distance_cm = None
                self._bbox = None
                self._triggered = False
            return

        dx = self._world_x_cm - x
        dy = self._world_y_cm - y
        distance_cm = math.hypot(dx, dy)
        bearing = heading_ref + math.atan2(gn.Y_AXIS_SIGN * dy, dx)
        heading_error = gn.angle_diff(bearing, current_yaw)
        half_fov = math.radians(self._fov_deg / 2.0)

        if distance_cm < 1.0 or abs(heading_error) > half_fov:
            visible, bbox, offset_frac = False, None, None
        else:
            visible = True
            offset_frac = max(-1.0, min(1.0, heading_error / half_fov))
            box_w = max(20.0, min(self._frame_width * 0.9, 30000.0 / distance_cm))
            center_x = self._frame_width / 2.0 + offset_frac * (self._frame_width / 2.0 - box_w / 2.0)
            bbox = {'x': center_x - box_w / 2.0, 'y': self._frame_height / 2.0 - box_w / 2.0,
                    'w': box_w, 'h': box_w}

        trigger_distance_cm = self.get_trigger_distance()
        triggered = visible and distance_cm <= trigger_distance_cm

        with self._lock:
            self._visible = visible
            self._distance_cm = distance_cm
            self._bbox = bbox
            self._triggered = triggered

        if triggered:
            world_x_cm, world_y_cm = self.get_world_position()
            self.node.correct_position_from_tag(world_x_cm, world_y_cm)

    def get_status(self):
        with self._lock:
            return {
                'available': True,
                'visible': self._visible,
                'distance_cm': self._distance_cm,
                'triggered': self._triggered,
                'bbox': self._bbox,
                'frame_width': self._frame_width,
                'frame_height': self._frame_height,
            }

    def get_start_status(self):
        with self._lock:
            return {
                'available': True,
                'visible': self._visible,
                'distance_cm': self._distance_cm,
                'bbox': self._bbox,
                'frame_width': self._frame_width,
                'frame_height': self._frame_height,
            }


class RobotSimulator:
    """Drives GridNavNode's control_loop() on a fixed-rate background
    thread (nothing else will, since there's no real rclpy timer here),
    and feeds back simulated IMU/encoder data based on whichever phase
    the node itself says it's currently in -- see the module docstring
    for why this is simpler/more correct than reverse-engineering the
    published Twist's PWM-mixing semantics."""

    def __init__(self, node):
        self.node = node
        self._pose_lock = threading.Lock()
        # Boot facing -90deg (see grid_nav.py's own imu_callback comment:
        # "point the robot at the start AprilTag before/at power-on") --
        # matches TAG_WORLD_Y_CM being negative, so the fake tag is
        # visible from the very first tick without needing to drive
        # anywhere first.
        self._true_yaw = math.radians(-90.0)
        self._true_pulses = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def teleport(self, x_cm, y_cm, heading_deg=None):
        """Console `teleport` command -- jump the simulated robot
        somewhere else instantly, e.g. to test homing from far away
        without waiting for a real drive there.

        Aborts any in-progress goal/leg/mode-path/homing FIRST -- setting
        x/y directly while a 'move' leg is mid-DRIVE doesn't stick,
        because the node's own live-position math (used for
        get_snapshot() and by the DRIVE phase itself) recomputes position
        from leg_start_x/y + progress every tick, silently overwriting a
        bare x/y assignment on the very next control_loop() tick -- the
        same "rigid shift" reasoning correct_position_from_tag() and
        commit_live_position() exist for in grid_nav.py itself."""
        self.node.abort_to_idle()
        with self.node._lock:
            self.node.x = float(x_cm)
            self.node.y = float(y_cm)
        if heading_deg is not None:
            with self._pose_lock:
                self._true_yaw = self.node.heading_ref + math.radians(heading_deg) \
                    if self.node.heading_ref is not None else math.radians(heading_deg)

    def _loop(self):
        interval = 1.0 / SIM_TICK_HZ
        node = self.node
        # Seed the first IMU reading immediately -- real hardware doesn't
        # wait either; control_loop just does nothing until this arrives.
        node.imu_callback(_FakeImu(self._true_yaw))
        while self._running:
            start = time.monotonic()
            try:
                node.control_loop()
                self._simulate_physics(interval)
                node.imu_callback(_FakeImu(self._true_yaw))
                node.encoder_callback(_FakeEncoder(self._true_pulses))
            except Exception as e:
                print(f'[mock_robot] simulation tick failed: {e!r}')
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _simulate_physics(self, dt):
        with node_lock_snapshot(self.node) as snap:
            state, phase, stopped, homing_stage = (
                snap['state'], snap['phase'], snap['stopped_for_obstacle'], snap['homing_stage']
            )
            leg_target_heading = snap['leg_target_heading']
            homing_target_heading = snap['homing_target_heading']

        if stopped:
            return

        if state == 'RUNNING' and phase == 'ROTATE':
            self._rotate_toward(leg_target_heading, dt)
        elif state == 'RUNNING' and phase == 'DRIVE':
            self._drive_forward(dt)
        elif state == 'HOMING':
            # Only the final STRAIGHT stage sets state='HOMING' directly
            # (RETURN_X/ROTATE are ordinary RUNNING legs, already covered
            # above) -- drive forward while holding the target heading.
            if homing_target_heading is not None:
                self._rotate_toward(homing_target_heading, dt, rate_scale=0.3)
            self._drive_forward(dt)
        # else: IDLE / awaiting_continue / etc. -- no motion.

    def _rotate_toward(self, target_heading, dt, rate_scale=1.0):
        with self._pose_lock:
            err = gn.angle_diff(target_heading, self._true_yaw)
            step = math.radians(SIM_ROTATE_RATE_DEG_S) * dt * rate_scale
            if abs(err) <= step:
                self._true_yaw = target_heading
            else:
                self._true_yaw += step if err > 0 else -step
            self._true_yaw = gn.normalize_angle(self._true_yaw)

    def _drive_forward(self, dt):
        with self._pose_lock:
            distance_cm = SIM_DRIVE_RATE_CM_S * dt
            self._true_pulses += distance_cm * self.node.pulses_per_cm

    def status_line(self):
        with self.node._lock:
            x, y = self.node.x, self.node.y
            state, phase = self.node.state, self.node.phase
            homing_stage = self.node.homing_stage
        heading_deg = math.degrees(self._true_yaw)
        return (f'x={x:7.1f}cm  y={y:7.1f}cm  heading={heading_deg:6.1f}deg  '
                f'state={state:<8} phase={str(phase):<7} homing_stage={homing_stage}')


class node_lock_snapshot:
    """Tiny context manager: grab everything _simulate_physics() needs
    from the node in ONE lock acquisition, as a plain dict, so the actual
    physics math below runs without holding the node's lock at all."""

    def __init__(self, node):
        self.node = node

    def __enter__(self):
        self.node._lock.acquire()
        n = self.node
        self._snap = {
            'state': n.state, 'phase': n.phase, 'stopped_for_obstacle': n.stopped_for_obstacle,
            'homing_stage': n.homing_stage, 'leg_target_heading': n.leg_target_heading,
            'homing_target_heading': getattr(n, 'homing_target_heading', None),
        }
        self.node._lock.release()
        return self._snap

    def __exit__(self, *exc):
        return False


def run_console(node, sim, apriltag):
    print()
    print('mock_robot.py -- simulated grid_nav.py console')
    print('Open the GUI:  http://localhost:8080')
    print("Type 'help' for console commands, 'quit' to stop.")
    print()
    while True:
        try:
            line = input('mock> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ('quit', 'exit'):
            break
        elif cmd == 'help':
            print('  pos                       -- print simulated position/heading/state')
            print('  teleport <x> <y> [heading] -- jump the simulated robot (cm, deg)')
            print('  tag on|off                -- force the fake AprilTag visible/hidden')
            print('  tag show                  -- print the fake AprilTag\'s current reading')
            print('  quit                      -- stop the simulator')
        elif cmd == 'pos':
            print('  ' + sim.status_line())
        elif cmd == 'teleport':
            if len(parts) < 3:
                print('  usage: teleport <x_cm> <y_cm> [heading_deg]')
                continue
            try:
                x, y = float(parts[1]), float(parts[2])
                heading = float(parts[3]) if len(parts) > 3 else None
            except ValueError:
                print('  x/y/heading must be numbers')
                continue
            sim.teleport(x, y, heading)
            print(f'  teleported to ({x}, {y})' + (f' @ {heading}deg' if heading is not None else ''))
        elif cmd == 'tag':
            sub = parts[1].lower() if len(parts) > 1 else ''
            if sub == 'off':
                apriltag.forced_hidden = True
                print('  fake AprilTag forced HIDDEN (testing the odometry-only fallback)')
            elif sub == 'on':
                apriltag.forced_hidden = False
                print('  fake AprilTag visibility restored to the simulated geometry')
            elif sub == 'show':
                print('  ' + str(apriltag.get_start_status()))
            else:
                print('  usage: tag on|off|show')
        else:
            print(f"  unknown command '{cmd}' -- type 'help'")


def build():
    """Constructs the simulated node/AprilTag/physics-sim/Flask app, all
    started, but does NOT bind a real port or block -- used both by
    main() (which then serves it for real) and by anything that wants to
    exercise the app programmatically (e.g. via app.test_client())."""
    node = gn.GridNavNode()
    node.cmd_pub = types.SimpleNamespace(publish=lambda t: None)  # no real motor to command

    rangefinder = gn.CameraRangefinder()
    apriltag = FakeAprilTagLocalizer(node)
    apriltag.start()
    node.apriltag_localizer = apriltag

    sim = RobotSimulator(node)
    sim.start()

    mode_settings = gn.ModePathSettings()
    app = gn.create_app(node, None, rangefinder, None, None, apriltag, mode_settings)
    return node, sim, apriltag, app


def main():
    print(f'mock_robot.py -- simulating grid_nav.py {gn.SCRIPT_VERSION}')
    node, sim, apriltag, app = build()

    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=gn.WEB_PORT, threaded=True, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    time.sleep(0.5)  # let Flask actually bind before printing the URL

    try:
        run_console(node, sim, apriltag)
    finally:
        sim.stop()
        apriltag.stop()
        print('Stopped.')


if __name__ == '__main__':
    main()
