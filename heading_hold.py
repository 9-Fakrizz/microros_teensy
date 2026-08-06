"""
PID heading-hold controller -- runs directly on the Pi5 as a ROS2 node.

Subscribes to /imu (heading feedback) and /wheel_encoder (distance feedback),
publishes /cmd_vel to drive straight, and prints distance traveled.

Run (after sourcing your ROS2 setup):
    python3 heading_hold.py
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

SCRIPT_VERSION = "v1.6 - single-encoder raw-pulse conversion"

# ---------------- Configuration ----------------
IMU_TOPIC = "/imu_data"
CMD_VEL_TOPIC = "/cmd_vel"
WHEEL_ENCODER_TOPIC = "/wheel_encoder"

# Teensy firmware layout (see encoder_data[] in timer_callback, src/main.cpp):
#   data[0] = primary encoder RAW pulse count (currently the right wheel;
#             only one physical encoder is enabled at a time)
#   data[1] = secondary encoder raw pulse count (disabled -- always 0)
#   data[2] = primary encoder glitch count (pulses rejected as noise)
#   data[3] = secondary encoder glitch count (always 0 while disabled)
# Meter conversion is NOT done on the firmware side anymore -- it happens
# here, using WHEEL_DIAMETER_MM / PULSES_PER_REV below.
ENCODER_INDEX_PRIMARY_PULSES = 0
ENCODER_INDEX_PRIMARY_GLITCH = 2

# Wheel geometry for pulses -> meters conversion. Update these if you
# change wheels/tires or the encoder's pulses-per-revolution.
WHEEL_DIAMETER_MM = 125.0
PULSES_PER_REV = 1000.0
WHEEL_CIRCUMFERENCE_M = math.pi * (WHEEL_DIAMETER_MM / 1000.0)
METERS_PER_PULSE = WHEEL_CIRCUMFERENCE_M / PULSES_PER_REV

# Multiplicative calibration: measured_distance * DISTANCE_SCALE_FACTOR = actual distance.
# NOTE: this was originally derived when the firmware did the pulses->meters
# conversion itself; now that conversion happens here instead (see
# METERS_PER_PULSE above), re-derive this against real measured distances
# again before trusting it -- the old derivation no longer applies as-is.
DISTANCE_SCALE_FACTOR = 1.05

FORWARD_SPEED = 0.15         # m/s, constant forward speed
LOOP_HZ = 20.0                 # control loop rate

KP = 1.5
KI = 0.0
KD = 0.15

MAX_ANGULAR_Z = 1.0            # rad/s clamp on correction output
MAX_INTEGRAL = 1.0             # anti-windup clamp

PRINT_EVERY_N_LOOPS = 10       # print distance every N control loop ticks (~0.5s at 20Hz)
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


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_max, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        self.integral = max(-self.i_max, min(self.i_max, self.integral))
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, output))


class HeadingHoldNode(Node):
    def __init__(self):
        super().__init__('heading_hold_node')

        self.dt = 1.0 / LOOP_HZ
        self.pid = PID(KP, KI, KD, -MAX_ANGULAR_Z, MAX_ANGULAR_Z, MAX_INTEGRAL, self.dt)

        self.current_yaw = None
        self.target_yaw = None

        # Encoder / distance tracking (single active encoder)
        self.baseline_pulses = None
        self.distance_cm = 0.0
        self._loop_count = 0

        self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.imu_callback, 10)
        self.encoder_sub = self.create_subscription(
            Int32MultiArray, WHEEL_ENCODER_TOPIC, self.encoder_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f'=== heading_hold.py {SCRIPT_VERSION} ===')
        self.get_logger().info(f'Subscribed to {IMU_TOPIC} and {WHEEL_ENCODER_TOPIC}')
        self.get_logger().info(f'Publishing to {CMD_VEL_TOPIC}')
        self.get_logger().info('Waiting for first IMU message to lock heading...')

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if self.target_yaw is None:
            self.target_yaw = self.current_yaw
            self.get_logger().info(
                f'Locked target heading: {math.degrees(self.target_yaw):.1f} deg'
            )

    def encoder_callback(self, msg: Int32MultiArray):
        data = msg.data
        if len(data) <= ENCODER_INDEX_PRIMARY_PULSES:
            self.get_logger().warn('wheel_encoder message shorter than expected indices')
            return

        pulses = data[ENCODER_INDEX_PRIMARY_PULSES]

        # Teensy accumulates pulses from boot (never resets), so capture a
        # baseline on the first message -> distance starts at 0 for this run.
        if self.baseline_pulses is None:
            self.baseline_pulses = pulses
            self.get_logger().info(f'Encoder baseline set: pulses={pulses}')
            return

        raw_m = (pulses - self.baseline_pulses) * METERS_PER_PULSE
        self.distance_cm = raw_m * 100.0 * DISTANCE_SCALE_FACTOR

    def control_loop(self):
        if self.target_yaw is None or self.current_yaw is None:
            return  # haven't received IMU data yet

        error = angle_diff(self.target_yaw, self.current_yaw)
        correction = self.pid.compute(error)

        twist = Twist()
        twist.linear.x = FORWARD_SPEED
        twist.angular.z = -correction  # sign flip: steer back toward target
        self.cmd_pub.publish(twist)

        self._loop_count += 1
        if self._loop_count % PRINT_EVERY_N_LOOPS == 0:
            self.get_logger().info(f'Distance: {self.distance_cm:.1f}cm')

    def stop_robot(self):
        twist = Twist()  # all zeros
        self.cmd_pub.publish(twist)


def main(args=None):
    print(f'heading_hold.py {SCRIPT_VERSION}')
    rclpy.init(args=args)
    node = HeadingHoldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping robot...')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
