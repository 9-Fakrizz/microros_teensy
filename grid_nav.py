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
  A grid view shows the robot's tracked position (arrow = heading),
  its path so far, and the current goal. Text boxes + a "Go" button
  let you send a new goal (in cm) at any time, including while the
  robot is mid-move. A status readout shows raw IMU yaw and the
  current target heading in degrees, for debugging the IMU.

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

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.patches as patches

SCRIPT_VERSION = "v1.0"

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


class GridNavGUI:
    def __init__(self, node: GridNavNode):
        self.node = node

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.22)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_title('grid_nav.py -- live position')

        e = GRID_HALF_EXTENT_CM
        self.ax.set_xlim(-e, e)
        self.ax.set_ylim(-e, e)
        self.ax.set_xticks(range(-e, e + 1, GRID_SPACING_CM))
        self.ax.set_yticks(range(-e, e + 1, GRID_SPACING_CM))
        self.ax.grid(True, linewidth=0.5, alpha=0.5)
        self.ax.axhline(0, color='gray', linewidth=0.8)
        self.ax.axvline(0, color='gray', linewidth=0.8)

        (self.path_line,) = self.ax.plot([], [], '-', color='tab:blue', linewidth=1.5)
        (self.goal_marker,) = self.ax.plot([], [], 'x', color='red', markersize=12, markeredgewidth=2)
        self.robot_arrow = None

        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax_x = plt.axes([0.15, 0.05, 0.15, 0.06])
        ax_y = plt.axes([0.35, 0.05, 0.15, 0.06])
        ax_go = plt.axes([0.55, 0.05, 0.15, 0.06])
        self.tb_x = TextBox(ax_x, 'Goal X ', initial='0')
        self.tb_y = TextBox(ax_y, 'Goal Y ', initial='0')
        self.btn_go = Button(ax_go, 'Go')
        self.btn_go.on_clicked(self._on_go)

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / GUI_HZ))
        self.timer.add_callback(self._redraw)
        self.timer.start()

    def _on_go(self, event):
        try:
            gx = float(self.tb_x.text)
            gy = float(self.tb_y.text)
        except ValueError:
            self.node.get_logger().warn(f'Invalid goal input: x={self.tb_x.text!r} y={self.tb_y.text!r}')
            return
        self.node.set_goal(gx, gy)

    def _redraw(self):
        snap = self.node.get_snapshot()

        xs = [p[0] for p in snap['path']]
        ys = [p[1] for p in snap['path']]
        self.path_line.set_data(xs, ys)

        if snap['goal'] is not None:
            self.goal_marker.set_data([snap['goal'][0]], [snap['goal'][1]])

        if self.robot_arrow is not None:
            self.robot_arrow.remove()
            self.robot_arrow = None

        heading_deg = snap['heading_deg'] if snap['heading_deg'] is not None else 0.0
        heading_rad = math.radians(heading_deg)
        arrow_len = GRID_SPACING_CM * 0.8
        dx = arrow_len * math.cos(heading_rad)
        dy = arrow_len * math.sin(heading_rad)
        self.robot_arrow = self.ax.add_patch(patches.FancyArrow(
            snap['x'], snap['y'], dx, dy,
            width=2.5, head_width=8, head_length=8,
            color='tab:orange', length_includes_head=True
        ))

        def fmt(v, suffix='deg'):
            return f'{v:.1f}{suffix}' if v is not None else 'n/a'

        leg_info = f"{snap['leg_idx']}/{snap['leg_count']}" if snap['state'] == 'RUNNING' else '-'
        self.status_text.set_text(
            f"pos:    ({snap['x']:.1f}, {snap['y']:.1f}) cm\n"
            f"state:  {snap['state']}  phase: {snap['phase']}\n"
            f"leg:    {leg_info}\n"
            f"IMU yaw (raw):     {fmt(snap['yaw_deg'])}\n"
            f"heading (ref=0):   {fmt(snap['heading_deg'])}\n"
            f"target heading:    {fmt(snap['target_heading_deg'])}"
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main(args=None):
    print(f'grid_nav.py {SCRIPT_VERSION}')
    rclpy.init(args=args)
    node = GridNavNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    gui = GridNavGUI(node)
    try:
        gui.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
