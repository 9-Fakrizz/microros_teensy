# microros_teensy

Firmware + navigation stack for an autonomous tennis-ball-collecting
robot: a Teensy 4.1 running micro-ROS talks to a Raspberry Pi 5, which
runs the path-planning / vision / web-GUI brain (`grid_nav.py`).

The whole Pi-side stack can also run **without any hardware at all**
via `mock_robot.py` — the exact same `grid_nav.py` code, driven by a
simulated robot, on any Windows/Mac/Linux machine.

```
┌─────────────────────┐        USB (serial + power)        ┌──────────────────────┐
│   Teensy 4.1         │◄───────────────────────────────────►│   Raspberry Pi 5      │
│   (src/main.cpp)     │        micro-ROS / ROS 2 topics      │   grid_nav.py         │
│  - motor PWM/DIR      │                                      │  - A* path planning   │
│  - BNO08x IMU         │                                      │  - obstacle detection │
│  - wheel encoder      │                                      │  - AprilTag homing    │
│  - watchdog stop      │                                      │  - Flask web GUI      │
└─────────────────────┘                                      └──────────────────────┘
                                                                        ▲
                                                                        │ HTTP
                                                                 operator's browser
                                                              http://<pi-ip>:8080
```

---

## Repository layout

| Path | What it is |
|---|---|
| `grid_nav.py` | The Pi-side brain: a single `rclpy` node (`GridNavNode`) that does A* grid path planning, PID rotate/drive control, camera-based obstacle detection, AprilTag-assisted homing, **and** hosts the Flask web GUI (`/api/*` routes). Identical whether it's talking to real hardware or `mock_robot.py`. |
| `mock_robot.py` | Drop-in stand-in for the ROS 2 + hardware layer. Stubs `rclpy`/`sensor_msgs`/`geometry_msgs`/`std_msgs` in `sys.modules` **before** importing `grid_nav`, so `GridNavNode` runs completely unmodified against a simulated IMU/encoder/AprilTag. Also gives you an interactive console (`pos`, `teleport`, `tag on/off`, ...). |
| `heading_hold.py` | Older/simpler standalone PID heading-hold script (drive straight, hold heading, print distance). Independent `PULSES_PER_CM` calibration from `grid_nav.py` — see [Known limitations](#known-limitations). |
| `test_apriltag.py` | Standalone AprilTag detector test, isolated from ROS 2 — opens the webcam, prints every tag it sees, saves an annotated `apriltag_debug.jpg` each frame. Useful for debugging camera/tag issues without the full stack. |
| `src/main.cpp` | Teensy 4.1 firmware (PlatformIO/Arduino): motor driver, quadrature-ish encoder ISRs with noise filtering, BNO08x IMU, micro-ROS node, command-timeout watchdog. |
| `platformio.ini` | PlatformIO project config for the Teensy 4.1 build (`teensy41` board, `micro_ros_arduino` + SparkFun BNO08x libraries). |
| `Dockerfile` | Builds the `grid-nav` image (`ros:jazzy-ros-base` + OpenCV + Flask + optional `pupil-apriltags`) for real-robot deployment. |
| `docker-compose.yml` | Runs the micro-ROS agent + `grid_nav.py` together as two containers on the Pi. |
| `DEPLOY.md` | Step-by-step guide for deploying to a (new) Raspberry Pi 5 with Docker. |
| `GUIDE.txt` | Full user + developer guide: every web GUI field explained, all `/api/*` routes, `mock_robot.py` console commands, and the testing workflow. |
| `notebook_debug.txt` | Running debug log — root causes and fixes for every bug hit so far (encoder drift, PID derivative-kick, PULSES_PER_CM calibration saga, obstacle-detection edge cases, etc.), plus a summary of open items. |

---

## Quick start

### Option A — Mock robot (no hardware needed)

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install flask opencv-python numpy
python mock_robot.py
```

Open **http://localhost:8080**. The same terminal is also an
interactive console (type `help`) — see `GUIDE.txt` §2.2 for the full
command list (`pos`, `teleport <x> <y> [heading]`, `tag on/off/show`,
`quit`).

### Option B — Real robot (Raspberry Pi 5 + Teensy 4.1)

1. Flash the Teensy with `src/main.cpp` (PlatformIO, `pio run -t upload`,
   environment `teensy41`).
2. On the Pi 5, copy at least `grid_nav.py`, `Dockerfile`, and
   `docker-compose.yml` into one folder, then:

   ```bash
   ls /dev/ttyACM*   # confirm the Teensy's serial device
   ls /dev/video*    # confirm the USB webcam's device
   # edit docker-compose.yml if either isn't ttyACM0 / video0
   docker compose up -d --build
   ```

3. Open **http://\<pi's-ip\>:8080** from any device on the same
   network.

   Full details, troubleshooting, and day-to-day commands are in
   [`DEPLOY.md`](./DEPLOY.md).

### First-time startup (both modes, identical)

1. Power on — the robot won't move yet; it's waiting to lock a heading
   reference.
2. Point it at the "start" AprilTag, ~2 m away, roughly centered in
   frame.
3. Wait for **Setup Status** to turn ready, then press **Confirm Start
   Position**. This locks `(0, 0)` / 0° heading to the current spot and
   records the measured distance as the return-home target.
4. Movement controls unlock.

See `GUIDE.txt` §1 for a full section-by-section walkthrough of the
web control panel (Manual Goal, Preset Coverage Paths A/B/C,
Return-to-Home, Obstacle Avoidance, AprilTag pin-position, camera
calibration, etc.).

---

## System overview

- **Path planning** — 8-directional A* over a 50 cm grid (octile
  heuristic), with obstacles inflated by the robot's full 60 cm × 60 cm
  footprint diagonal before every search (the tracked position is a
  *corner*, the front-left wheel, not the center).
- **Motion control** — each leg is a ROTATE phase (pivot about the left
  wheel, PID on heading error) followed by a DRIVE phase (PID
  heading-hold + encoder-counted distance).
- **Obstacle detection** — no ML model: Canny edges → contour/shape
  filtering on the camera feed, confirmed over several consecutive
  frames before a cell is pinned into the A* map and the path is
  replanned.
- **Homing** — a three-stage return sequence (drive X to 0 → rotate to
  face the start tag → drive straight, refining the stop distance from
  live AprilTag readings, falling back to odometry if the tag isn't
  visible).
- **Communication** — Teensy ↔ Pi over USB (micro-ROS agent, serial,
  115200 baud), Pi ↔ browser over Flask/HTTP.

### ROS 2 topics

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Pi → Teensy | `linear.x` / `angular.z`; firmware mixes `left=(lin_x-ang_z)`, `right=(lin_x+ang_z)`. |
| `/imu_data` | `sensor_msgs/Imu` | Teensy → Pi | From the BNO08x game rotation vector + gyro + accelerometer, published at 20 Hz. |
| `/wheel_encoder` | `std_msgs/Int32MultiArray` | Teensy → Pi | `[primary_pulses, secondary_pulses(=0), primary_glitches, secondary_glitches(=0)]`. Only one physical encoder is active at a time (currently the right wheel) — see [Known limitations](#known-limitations). |
| `/reset_encoder` | `std_msgs/Empty` | Pi → Teensy | Zeroes the pulse/glitch counters without a USB replug. |

### Key firmware safety behavior

- **Command watchdog** — if no `/cmd_vel` arrives for `CMD_TIMEOUT_MS`
  (500 ms), motors are stopped.
- **IMU init failure** and **general ROS init failure** blink the
  Teensy's LED in two distinct patterns (`error_loop(2)` vs
  `error_loop(1)`) so a stuck boot is diagnosable without a serial
  console.
- **Encoder noise filtering** is two-layer: per-pulse width/gap
  validation in the ISR, plus a "stationary gate" that only counts
  pulses while motion is actually commanded.

---

## Development & testing without hardware

`mock_robot.py` stubs the ROS 2 stack so `grid_nav.py`'s real
`GridNavNode` class runs untouched. For scripted tests:

```python
import mock_robot
node, sim, apriltag, app = mock_robot.build()   # no Flask server, no console loop
client = app.test_client()
client.post("/api/confirm_start")
node.run_path([(100, 0), (100, 100)])
# poll client.get("/api/state") until state == "IDLE"
```

If you change **how position is tracked live** during a leg/homing
stage in `grid_nav.py` (`leg_progress_cm`, `homing_*` fields,
`_live_position_locked()`), double-check
`FakeAprilTagLocalizer._tick()` in `mock_robot.py` still reads position
the same way — it deliberately mirrors `grid_nav.py`'s own internal
pattern instead of reading `node.x`/`node.y` raw, and a future change to
one side without the other reintroduces a frozen-position bug that has
already been hit and fixed once.

Full API route reference (`/api/goal`, `/api/run_mode`,
`/api/toggle_obstacle`, `/api/camera_calibrate`, ...) is in
`GUIDE.txt` §2.3.

---

## Known limitations

See `notebook_debug.txt` for full root-cause writeups. In short:

- **`PULSES_PER_CM` (112) is only valid at `FORWARD_SPEED = ROTATE_SPEED
  = 0.20`.** Raising speed measurably undercounts distance (leading
  theory: the ISR's noise filters start clipping real pulses at higher
  wheel speed). Re-run the calibration procedure before trusting
  distance at any other speed.
- **Only one wheel encoder is active** (currently right; left is wired
  but disabled in firmware) — the two encoders don't agree reliably.
- **`heading_hold.py` and `grid_nav.py` use different `PULSES_PER_CM`**
  (110 vs 112) — calibrated independently, never reconciled.
- **Camera `vfov_deg` is a guess** (60°/45° depending on script), not
  measured — affects distance accuracy away from the single calibrated
  crosshair point.
- **Robot footprint inflation radius and its rotation direction on the
  GUI map are not yet verified against the physical robot.**

## Planned improvements

Ranked roughly by impact — see `notebook_debug.txt` and the project's
slide deck appendix for the full reasoning:

1. Switch wheel-distance tracking to the Teensy 4.1's hardware
   quadrature decoder instead of interrupt+debounce heuristics (removes
   the speed-dependent calibration problem entirely).
2. Fix the encoder signal at the hardware level (shielded/twisted wiring,
   common ground, decoupling caps) rather than filtering in software.
3. Two-point camera calibration to solve for the real `vfov_deg`.
4. Verify the footprint-inflation direction/conservativeness against the
   real robot.
5. Reconcile `PULSES_PER_CM` between `heading_hold.py` and `grid_nav.py`.
6. Re-enable and reconcile the second wheel encoder.

---

## Requirements

**Pi side:** Python 3, `flask`, `opencv-python` (or `python3-opencv`),
`numpy`, a ROS 2 install (Jazzy) for real-robot mode. Optional:
`pupil-apriltags` for AprilTag position correction / homing (silently
disabled if not installed).

**Teensy side:** [PlatformIO](https://platformio.org/), board
`teensy41`, libraries `micro-ROS/micro_ros_arduino` and SparkFun BNO08x
Arduino Library (pulled automatically via `platformio.ini`).

## License

No license file is currently included in this repository — add one
(e.g. MIT) before treating this as open for external reuse.
