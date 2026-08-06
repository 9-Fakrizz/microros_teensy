"""
Grid navigation with live GUI -- runs on the Pi5 as a ROS2 node.

Robot model this script assumes:
  - Position (x, y) is tracked at the LEFT wheel's contact point.
  - Straight-line distance is measured from the LEFT wheel encoder
    (wheel_encoder data[0] -- see IMPORTANT note below).
  - Turning is done by a PIVOT about the left wheel: left wheel stays
    stopped, only the right wheel drives, so the tracked (x, y) point
    does not move during a turn -- only heading changes.
  - The robot never drives diagonally. To reach goal (gx, gy) it
    resolves the X-axis distance first (turn to face +X/-X, drive),
    then the Y-axis distance (turn to face +Y/-Y, drive).

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
  heading in degrees, for debugging the IMU.

  Open it from any browser on the same network:
      http://<pi5-ip-address>:8080

Requires Flask (pip install flask) in addition to your ROS2 env.

Run (after sourcing your ROS2 setup):
    python3 grid_nav.py
"""

import math
import threading

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

# From the calibration data in notebook_debug.txt (~183.5 pulses/cm,
# consistent across the -35000 and -70000 pulse test runs). Re-derive
# if wheel/tire/encoder changes.
PULSES_PER_CM = 183.5

FORWARD_SPEED = 0.15           # m/s, straight-line drive speed
ROTATE_SPEED = 0.12            # commanded speed magnitude while pivoting

# If the robot pivots the WRONG way (heading error grows instead of
# shrinking) during testing, flip this to -1. Left wheel stays at 0
# regardless of this value -- it only affects rotation direction.
PIVOT_SIGN = 1

HEADING_TOLERANCE_DEG = 3.0    # stop pivoting once within this of target
POSITION_EPSILON_CM = 1.0      # skip an axis leg smaller than this

# Light heading-hold correction while driving straight (keeps the
# robot from wandering off its cardinal heading mid-leg). Kept small
# and clamped -- the ROTATE phase does the real turning, not this.
HEADING_HOLD_KP = 1.0
MAX_ANGULAR_Z_HOLD = 0.3

LOOP_HZ = 20.0                 # control loop rate
GUI_HZ = 5.0                   # GUI redraw rate

GRID_SPACING_CM = 20            # gridline spacing, cosmetic only
GRID_HALF_EXTENT_CM = 200       # initial view: +/- this many cm
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


class GridNavNode(Node):
    def __init__(self):
        super().__init__('grid_nav_node')

        self.dt = 1.0 / LOOP_HZ

        self._lock = threading.Lock()

        # Pose tracked at the left wheel, in cm. (0, 0) at node start.
        self.x = 0.0
        self.y = 0.0

        self.current_yaw = None     # raw IMU yaw, radians
        self.heading_ref = None     # yaw captured at startup == "+X" (east)

        self.last_pulses = None     # most recent raw pulse count from encoder

        # Navigation state: 'IDLE' | 'RUNNING'
        self.state = 'IDLE'
        self.legs = []               # list of ('x'|'y', delta_cm)
        self.leg_idx = 0
        self.phase = None            # 'ROTATE' | 'DRIVE'
        self.leg_target_heading = 0.0
        self.leg_target_distance_cm = 0.0
        self.leg_baseline_pulses = 0
        self.goal = None             # (gx, gy) for display

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

    def set_goal(self, gx, gy):
        with self._lock:
            dx = gx - self.x
            dy = gy - self.y
            legs = []
            if abs(dx) >= POSITION_EPSILON_CM:
                legs.append(('x', dx))
            if abs(dy) >= POSITION_EPSILON_CM:
                legs.append(('y', dy))

            self.goal = (gx, gy)
            self.legs = legs
            self.leg_idx = 0

            if not legs:
                self.state = 'IDLE'
                self.get_logger().info('Goal is at (or within tolerance of) current position.')
                return

            self.state = 'RUNNING'
            self._start_leg_locked()
            self.get_logger().info(f'New goal: ({gx:.1f}, {gy:.1f}) cm -- {len(legs)} leg(s)')

    def _start_leg_locked(self):
        """Caller must hold self._lock."""
        axis, delta = self.legs[self.leg_idx]
        if self.heading_ref is None:
            # No IMU data yet -- can't compute a target heading. Bail to
            # IDLE; control_loop will simply do nothing until IMU arrives
            # and the user re-sends the goal.
            self.state = 'IDLE'
            self.get_logger().warn('No IMU data yet -- cannot start leg. Try the goal again shortly.')
            return

        if axis == 'x':
            target = self.heading_ref if delta > 0 else self.heading_ref + math.pi
        else:
            target = self.heading_ref + (math.pi / 2.0) if delta > 0 else self.heading_ref - (math.pi / 2.0)

        self.leg_target_heading = normalize_angle(target)
        self.leg_target_distance_cm = abs(delta)
        self.phase = 'ROTATE'

    # ---------------- Control loop ----------------

    def control_loop(self):
        twist = Twist()

        with self._lock:
            if self.state != 'RUNNING' or self.current_yaw is None or self.last_pulses is None:
                self.cmd_pub.publish(twist)  # all-zero
                return

            if self.phase == 'ROTATE':
                error = angle_diff(self.leg_target_heading, self.current_yaw)
                if abs(math.degrees(error)) <= HEADING_TOLERANCE_DEG:
                    self.leg_baseline_pulses = self.last_pulses
                    self.phase = 'DRIVE'
                    self.cmd_pub.publish(twist)  # brief all-zero pause between phases
                    return

                turn_dir = 1.0 if error > 0 else -1.0
                cmd = ROTATE_SPEED * turn_dir * PIVOT_SIGN
                # linear.x == angular.z always zeroes the left-wheel term
                # in the firmware's differential mix, regardless of sign.
                twist.linear.x = cmd
                twist.angular.z = cmd
                self.cmd_pub.publish(twist)
                return

            # phase == 'DRIVE'
            traveled_pulses = self.last_pulses - self.leg_baseline_pulses
            traveled_cm = abs(traveled_pulses) / PULSES_PER_CM

            if traveled_cm >= self.leg_target_distance_cm:
                axis, delta = self.legs[self.leg_idx]
                signed_cm = math.copysign(traveled_cm, delta)
                if axis == 'x':
                    self.x += signed_cm
                else:
                    self.y += signed_cm
                self.path.append((self.x, self.y))

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

            herr = angle_diff(self.leg_target_heading, self.current_yaw)
            correction = max(-MAX_ANGULAR_Z_HOLD, min(MAX_ANGULAR_Z_HOLD, HEADING_HOLD_KP * herr))
            twist.linear.x = FORWARD_SPEED
            twist.angular.z = -correction  # sign convention matches heading_hold.py
            self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())  # all zeros

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
            return {
                'x': self.x,
                'y': self.y,
                'path': list(self.path),
                'goal': self.goal,
                'state': self.state,
                'phase': self.phase,
                'yaw_deg': yaw_deg,
                'ref_deg': ref_deg,
                'heading_deg': heading_deg,   # yaw relative to startup reference; 0 = "+X"
                'target_heading_deg': target_deg,
                'leg_idx': self.leg_idx,
                'leg_count': len(self.legs),
            }


HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>grid_nav.py</title>
<style>
  body { font-family: sans-serif; background: #1e1e1e; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 16px; font-weight: normal; color: #aaa; }
  #canvas { background: #111; border: 1px solid #444; display: block; margin-bottom: 12px; }
  #status { white-space: pre; font-family: monospace; font-size: 13px; background: #262626;
            border: 1px solid #444; padding: 10px; display: inline-block; min-width: 320px; }
  form { margin-top: 12px; }
  input { width: 70px; font-size: 14px; padding: 4px; }
  button { font-size: 14px; padding: 5px 14px; margin-left: 6px; }
  label { margin-right: 4px; }
</style>
</head>
<body>
<h1>grid_nav.py -- live position (poll __POLL_MS__ms)</h1>
<canvas id="canvas" width="__CANVAS_PX__" height="__CANVAS_PX__"></canvas>
<div id="status">connecting...</div>
<form id="goalForm">
  <label>Goal X (cm)</label><input id="goalX" type="number" value="0" step="1">
  <label>Goal Y (cm)</label><input id="goalY" type="number" value="0" step="1">
  <button type="submit">Go</button>
</form>

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
}

function fmt(v) { return (v === null || v === undefined) ? 'n/a' : v.toFixed(1) + 'deg'; }

function updateStatus(state) {
  const legInfo = (state.state === 'RUNNING') ? `${state.leg_idx}/${state.leg_count}` : '-';
  document.getElementById('status').textContent =
    `pos:    (${state.x.toFixed(1)}, ${state.y.toFixed(1)}) cm\\n` +
    `state:  ${state.state}  phase: ${state.phase}\\n` +
    `leg:    ${legInfo}\\n` +
    `IMU yaw (raw):     ${fmt(state.yaw_deg)}\\n` +
    `heading (ref=0):   ${fmt(state.heading_deg)}\\n` +
    `target heading:    ${fmt(state.target_heading_deg)}`;
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
    document.getElementById('status').textContent = 'connection lost: ' + e;
  }
}

document.getElementById('goalForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const x = parseFloat(document.getElementById('goalX').value);
  const y = parseFloat(document.getElementById('goalY').value);
  await fetch('/api/goal', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: x, y: y})
  });
});

setInterval(poll, __POLL_MS__);
poll();
</script>
</body>
</html>
"""

WEB_PORT = 8080
GUI_POLL_MS = int(1000 / GUI_HZ)
CANVAS_PX = 700


def render_page():
    return (HTML_PAGE
            .replace('__HALF_EXTENT__', str(GRID_HALF_EXTENT_CM))
            .replace('__SPACING__', str(GRID_SPACING_CM))
            .replace('__CANVAS_PX__', str(CANVAS_PX))
            .replace('__POLL_MS__', str(GUI_POLL_MS)))


def create_app(node: GridNavNode) -> Flask:
    app = Flask(__name__)
    # Werkzeug's request logging is noisy at GUI_HZ polling rates.
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    @app.route('/')
    def index():
        return Response(render_page(), mimetype='text/html')

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
        node.set_goal(gx, gy)
        return jsonify({'ok': True})

    return app


def main(args=None):
    print(f'grid_nav.py {SCRIPT_VERSION}')
    rclpy.init(args=args)
    node = GridNavNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = create_app(node)
    node.get_logger().info(f'Web GUI at http://<this-device-ip>:{WEB_PORT}')
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
