#include <Arduino.h>
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h>
#include <sensor_msgs/msg/imu.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <Wire.h>
#include "SparkFun_BNO08x_Arduino_Library.h"

// =====================================================
// ---------------- Hardware Pins ----------------------
// =====================================================
const int DIR1_PIN = 21;
const int DIR2_PIN = 20;
const int PWM1_PIN = 23;
const int PWM2_PIN = 22;

const int led_relay_pin = 38;

#define BNO08X_RST 40
#define BNO08X_I2C_ADDR 0x4B
BNO08x myIMU;

// ---------------- Encoder Pins -----------------------
const int L_A_MINUS = 14;
const int L_A_PLUS  = 15;
const int R_A_MINUS = 16;
const int R_A_PLUS  = 17;

// =====================================================
// ---------------- Robot Parameters -------------------
// =====================================================
const float WHEEL_DIAMETER_MM = 125.0;
const float CIRCUMFERENCE_M =
  (3.14159265 * WHEEL_DIAMETER_MM) / 1000.0;

const float PULSES_PER_REV = 1000.0;
const float MAX_SPEED = 1.0;     // m/s max speed

// ---------------- Timing -----------------------------
const uint32_t IMU_REPORT_INTERVAL_MS = 50;   // must match TIMER_PERIOD_MS
const uint32_t TIMER_PERIOD_MS        = 50;   // 20 Hz publish rate
const unsigned long CMD_TIMEOUT_MS    = 500;  // stop motors if no cmd_vel

// ---------------- Encoder debounce --------------------
// Minimum time between accepted pulses, in microseconds. Reject anything
// faster than this as noise/glitch rather than a real quadrature edge.
// Tune based on max expected pulse rate:
//   max_wheel_rev_per_sec = MAX_SPEED / CIRCUMFERENCE_M
//   max_pulses_per_sec    = max_wheel_rev_per_sec * PULSES_PER_REV
//   min_pulse_interval_us = 1e6 / max_pulses_per_sec
// Pick something a bit below that; default here is conservative.
const unsigned long ENCODER_DEBOUNCE_US = 200;

volatile unsigned long lastLeftPulseMicros = 0;
volatile unsigned long lastRightPulseMicros = 0;

// ใช้ volatile สำหรับตัวแปรที่ถูกดัดแปลงในฟังก์ชัน Interrupt
volatile int64_t leftPulseCount  = 0;
volatile int64_t rightPulseCount = 0;

// ---------------- Encoder noise filter -----------------
// Rate-based filter: real movement produces a BURST of pulses within
// each TIMER_PERIOD_MS window; isolated electrical noise produces only
// 1-2 stray pulses per window. Each timer tick, we look at how many NEW
// pulses arrived since the last tick — if that delta is below
// ENCODER_NOISE_THRESHOLD_PULSES, we treat it as noise and drop it
// (don't add it to distance). Otherwise we keep it.
//
// Raw ISR counts (leftPulseCount/rightPulseCount) are left untouched for
// diagnostics; these filtered accumulators are what get converted to
// meters and published.
//
// Tune this threshold based on testing: it must be higher than your
// observed static-noise pulse count per window, but low enough that it
// doesn't eat real slow-speed motion. At PULSES_PER_REV=1000 and
// CIRCUMFERENCE_M computed above, one pulse per 50ms window corresponds
// to roughly (CIRCUMFERENCE_M / PULSES_PER_REV) / (TIMER_PERIOD_MS/1000)
// m/s of "invisible" minimum speed — keep that in mind if you raise it.
const int64_t ENCODER_NOISE_THRESHOLD_PULSES = 3;

int64_t leftFilteredPulses  = 0;
int64_t rightFilteredPulses = 0;
int64_t lastRawLeftPulse    = 0;
int64_t lastRawRightPulse   = 0;

// =====================================================
// ---------------- micro-ROS Objects ------------------
// =====================================================
rcl_subscription_t subscriber;
rcl_publisher_t imu_pub;
rcl_publisher_t encoder_pub;

geometry_msgs__msg__Twist msg_cmd;
sensor_msgs__msg__Imu msg_imu;
std_msgs__msg__Float32MultiArray msg_encoder;

float encoder_data[4]; // [0]=left_m, [1]=right_m, [2]=left_raw_pulses, [3]=right_raw_pulses

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

volatile unsigned long last_cmd_time = 0;
bool imu_ready = false;
bool motors_stopped_by_watchdog = false;

// ---------------- IMU data cache ---------------------
// The SparkFun BNO08x library stores ALL report types (rotation vector,
// gyro, accel, ...) in one shared internal struct with a union. Since we
// enable three separate reports, each getSensorEvent() call only updates
// the fields for whichever report just arrived. Reading getQuatI() /
// getGyroX() / getAccelX() at an arbitrary later time (e.g. from the
// timer callback) can silently return stale or wrong-report data.
// To avoid this, we cache each report type into our own variables the
// moment it arrives, gated by getSensorEventID().
struct ImuCache {
  float quat_i = 0, quat_j = 0, quat_k = 0, quat_real = 1; // identity quat
  float gyro_x = 0, gyro_y = 0, gyro_z = 0;
  float accel_x = 0, accel_y = 0, accel_z = 0;
  bool has_quat = false;
  bool has_gyro = false;
  bool has_accel = false;
};
ImuCache imu_cache;

// =====================================================
#define RCCHECK(fn) { rcl_ret_t rc = fn; if(rc != RCL_RET_OK) error_loop(1); }
#define RCSOFTCHECK(fn) { rcl_ret_t rc = fn; (void) rc; }

// blink_count differentiates failure sites: 1 = RCCHECK/ROS init,
// 2 = IMU init failure. Fast-blinks blink_count times, pauses, repeats.
void error_loop(int blink_count) {
  pinMode(LED_BUILTIN, OUTPUT);
  while (1) {
    for (int i = 0; i < blink_count; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(150);
      digitalWrite(LED_BUILTIN, LOW);
      delay(150);
    }
    delay(800);
  }
}

// =====================================================
// ---------------- Motor Control ----------------------
// =====================================================
// ---------------- Motor direction inversion -----------
// Flip either of these to -1 if that wheel spins backward relative to
// a positive commanded velocity. If the WHOLE robot drives backward on
// a forward command, set BOTH to -1. If only turning direction is
// reversed, set only ONE to -1.
const int LEFT_MOTOR_INVERT  = -1;
const int RIGHT_MOTOR_INVERT = -1;

void set_motors(float lin_x, float ang_z) {
    float left  = (lin_x - ang_z) * LEFT_MOTOR_INVERT;
    float right = (lin_x + ang_z) * RIGHT_MOTOR_INVERT;

    auto drive = [](int d_pin, int p_pin, float val) {
        digitalWrite(d_pin, val >= 0 ? HIGH : LOW);
        analogWrite(p_pin, (int)constrain(fabs(val) * 255.0f, 0, 255));
    };

    drive(DIR1_PIN, PWM1_PIN, left);
    drive(DIR2_PIN, PWM2_PIN, right);
}

void cmd_vel_callback(const void * msgin) {
    const geometry_msgs__msg__Twist * msg = (const geometry_msgs__msg__Twist *)msgin;
    last_cmd_time = millis();
    motors_stopped_by_watchdog = false;
    set_motors(msg->linear.x, msg->angular.z);
}

// =====================================================
// ---------------- Encoder Interrupts -----------------
// =====================================================
void leftEncoderISR() {
  unsigned long now = micros();
  if (now - lastLeftPulseMicros < ENCODER_DEBOUNCE_US) return; // reject glitch
  lastLeftPulseMicros = now;

  // อ่านสถานะของอีกขาเพื่อตัดสินทิศทาง (หากหมุนสลับทางให้สลับ HIGH/LOW)
  if (digitalRead(L_A_MINUS) == HIGH) {
    leftPulseCount++;
  } else {
    leftPulseCount--;
  }
}

void rightEncoderISR() {
  unsigned long now = micros();
  if (now - lastRightPulseMicros < ENCODER_DEBOUNCE_US) return; // reject glitch
  lastRightPulseMicros = now;

  if (digitalRead(R_A_MINUS) == HIGH) {
    rightPulseCount++;
  } else {
    rightPulseCount--;
  }
}

// =====================================================
// ---------------- Timer Callback ---------------------
// =====================================================
void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  (void) last_call_time;
  if (timer == NULL) return;

  // ---------- IMU ----------
  // Read from our own imu_cache (updated in loop() per report type) —
  // NOT from myIMU.getQuatI()/getGyroX()/etc. directly, since those all
  // read a shared internal struct that gets overwritten by whichever
  // report type arrived most recently, regardless of which value you ask for.
  if (imu_ready) {
    msg_imu.header.stamp.sec = rmw_uros_epoch_millis() / 1000;
    msg_imu.header.stamp.nanosec = (rmw_uros_epoch_millis() % 1000) * 1000000;

    // frame_id string is set once in setup(); not touched here.

    msg_imu.orientation.x = imu_cache.quat_i;
    msg_imu.orientation.y = imu_cache.quat_j;
    msg_imu.orientation.z = imu_cache.quat_k;
    msg_imu.orientation.w = imu_cache.quat_real;

    msg_imu.angular_velocity.x = imu_cache.gyro_x;
    msg_imu.angular_velocity.y = imu_cache.gyro_y;
    msg_imu.angular_velocity.z = imu_cache.gyro_z;

    msg_imu.linear_acceleration.x = imu_cache.accel_x;
    msg_imu.linear_acceleration.y = imu_cache.accel_y;
    msg_imu.linear_acceleration.z = imu_cache.accel_z;

    msg_imu.orientation_covariance[0] = -1;
    msg_imu.angular_velocity_covariance[0] = -1;
    msg_imu.linear_acceleration_covariance[0] = -1;

    RCSOFTCHECK(rcl_publish(&imu_pub, &msg_imu, NULL));
  }

  // ---------- Encoder ----------
  noInterrupts();
  int64_t currentLeftPulse = leftPulseCount;
  int64_t currentRightPulse = rightPulseCount;
  interrupts();

  // Rate-based noise filter: only count this window's pulses as real
  // motion if the burst size clears the noise threshold. See comment
  // at ENCODER_NOISE_THRESHOLD_PULSES definition for tuning notes.
  int64_t deltaLeft  = currentLeftPulse  - lastRawLeftPulse;
  int64_t deltaRight = currentRightPulse - lastRawRightPulse;
  lastRawLeftPulse  = currentLeftPulse;
  lastRawRightPulse = currentRightPulse;

  if (abs(deltaLeft) < ENCODER_NOISE_THRESHOLD_PULSES)  deltaLeft  = 0;
  if (abs(deltaRight) < ENCODER_NOISE_THRESHOLD_PULSES) deltaRight = 0;

  leftFilteredPulses  += deltaLeft;
  rightFilteredPulses += deltaRight;

  encoder_data[0] = (leftFilteredPulses / PULSES_PER_REV) * CIRCUMFERENCE_M;
  encoder_data[1] = (rightFilteredPulses / PULSES_PER_REV) * CIRCUMFERENCE_M;
  // Raw (unfiltered) pulse counts included for calibration/diagnostics —
  // compare these against the filtered distance above to see the filter
  // working, and against a known real-world distance to verify PULSES_PER_REV.
  encoder_data[2] = (float)currentLeftPulse;
  encoder_data[3] = (float)currentRightPulse;

  msg_encoder.data.data = encoder_data;
  msg_encoder.data.size = 4;
  msg_encoder.data.capacity = 4;

  RCSOFTCHECK(rcl_publish(&encoder_pub, &msg_encoder, NULL));
}


// =====================================================
// ---------------- Setup ------------------------------
// =====================================================
void setup() {

  set_microros_transports();
  pinMode(LED_BUILTIN, OUTPUT);

  while (rmw_uros_ping_agent(100, 120) != RCL_RET_OK) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(300);
  }

  Wire.begin();

  if (!myIMU.begin(BNO08X_I2C_ADDR, Wire, -1, BNO08X_RST)) {
    error_loop(2); // IMU init failure — distinct blink pattern from ROS failures
  }

  // Report interval matches TIMER_PERIOD_MS so fresh data lines up with each publish.
  myIMU.enableGameRotationVector(IMU_REPORT_INTERVAL_MS);
  myIMU.enableGyro(IMU_REPORT_INTERVAL_MS);
  myIMU.enableAccelerometer(IMU_REPORT_INTERVAL_MS);

  pinMode(DIR1_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);
  pinMode(PWM1_PIN, OUTPUT);
  pinMode(PWM2_PIN, OUTPUT);
  pinMode(led_relay_pin, OUTPUT);
  digitalWrite(led_relay_pin, LOW);

  // Encoder pins as INPUT_PULLUP (digital quadrature signals)
  pinMode(L_A_PLUS, INPUT_PULLUP);
  pinMode(L_A_MINUS, INPUT_PULLUP);
  pinMode(R_A_PLUS, INPUT_PULLUP);
  pinMode(R_A_MINUS, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(L_A_PLUS), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(R_A_PLUS), rightEncoderISR, RISING);

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "teensy_bot", "", &support));

  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "cmd_vel"));

  RCCHECK(rclc_publisher_init_default(
    &imu_pub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
    "imu_data"));

  RCCHECK(rclc_publisher_init_default(
    &encoder_pub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "wheel_encoder"));

  RCCHECK(rclc_timer_init_default(
    &timer,
    &support,
    RCL_MS_TO_NS(TIMER_PERIOD_MS),
    timer_callback));

  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_subscription(
    &executor, &subscriber, &msg_cmd,
    &cmd_vel_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  // Set static frame_id once — no need to recompute strlen()/reassign every tick.
  static const char FRAME_ID[] = "imu_link";
  msg_imu.header.frame_id.data = (char*)FRAME_ID;
  msg_imu.header.frame_id.size = strlen(FRAME_ID);
  msg_imu.header.frame_id.capacity = sizeof(FRAME_ID);

  digitalWrite(led_relay_pin, HIGH);
  last_cmd_time = millis();
}


// =====================================================
// ---------------- Loop -------------------------------
// =====================================================
void loop() {
  // 1. อ่านข้อมูลจาก I2C ตลอดเวลาเพื่อเคลียร์บัฟเฟอร์ (ป้องกันคอขวด)
  // getSensorEvent() returns true whenever ANY enabled report arrives
  // (game rotation vector, gyro, OR accel — they interleave). We must
  // check getSensorEventID() and cache each type into our own variables
  // immediately, or later reads risk mixing data from different reports
  // (see ImuCache comment above) — this was the root cause of orientation
  // values not matching the real world.
  if (myIMU.getSensorEvent() == true) {
    switch (myIMU.getSensorEventID()) {
      case SH2_GAME_ROTATION_VECTOR:
        imu_cache.quat_i    = myIMU.getQuatI();
        imu_cache.quat_j    = myIMU.getQuatJ();
        imu_cache.quat_k    = myIMU.getQuatK();
        imu_cache.quat_real = myIMU.getQuatReal();
        imu_cache.has_quat  = true;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        imu_cache.gyro_x    = myIMU.getGyroX();
        imu_cache.gyro_y    = myIMU.getGyroY();
        imu_cache.gyro_z    = myIMU.getGyroZ();
        imu_cache.has_gyro  = true;
        break;
      case SH2_ACCELEROMETER:
        imu_cache.accel_x   = myIMU.getAccelX();
        imu_cache.accel_y   = myIMU.getAccelY();
        imu_cache.accel_z   = myIMU.getAccelZ();
        imu_cache.has_accel = true;
        break;
      default:
        break; // some other report we didn't ask for; ignore
    }
    imu_ready = imu_cache.has_quat && imu_cache.has_gyro && imu_cache.has_accel;
  }

  // 2. Command watchdog: stop motors if no cmd_vel received recently.
  // Prevents the robot from running away if the agent/link drops.
  if (!motors_stopped_by_watchdog &&
      (millis() - last_cmd_time > CMD_TIMEOUT_MS)) {
    set_motors(0.0f, 0.0f);
    motors_stopped_by_watchdog = true;
  }

  // 3. รัน micro-ROS Agent
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
}