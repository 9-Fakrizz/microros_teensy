# Deploying to a new Raspberry Pi 5

Runs the micro-ROS agent and `grid_nav.py` as two Docker containers,
started together with one command.

## 1. One-time setup on the new Pi5

Install Docker (Raspberry Pi OS / Ubuntu):

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
```

Copy this project onto the Pi (via `git clone`, `scp`, or a USB drive) --
you need at least `grid_nav.py`, `Dockerfile`, and `docker-compose.yml` in
the same folder.

## 2. Check your device paths

USB enumeration order isn't guaranteed to match your old Pi5. Plug in the
Teensy and the USB webcam, then check:

```bash
ls /dev/ttyACM*   # the Teensy (micro-ROS agent's serial device)
ls /dev/video*    # the USB webcam (grid_nav.py's camera)
```

If either isn't `/dev/ttyACM0` / `/dev/video0`, edit `docker-compose.yml`'s
`devices:` and `command:` lines for the matching service to the actual
path.

## 3. Build and start

```bash
docker compose up -d --build
```

This builds the `grid-nav` image (first run only takes a few minutes --
`pupil-apriltags` may need to compile from source if no prebuilt aarch64
wheel is available) and starts both containers in the background.

## 4. Check it's working

```bash
docker compose logs -f microros-agent   # should show the Teensy connecting
docker compose logs -f grid-nav         # should show "grid_nav.py vX.X" and the web GUI line
```

Open the GUI from any browser on the same network:

```
http://<this-pi's-ip>:8080
```

## Day-to-day use

```bash
docker compose up -d       # start (after the first build, no --build needed unless grid_nav.py changed)
docker compose down        # stop both containers
docker compose restart grid-nav   # restart just grid_nav.py, e.g. after editing it
```

Both containers are set to `restart: unless-stopped`, so they come back
automatically after a Pi reboot or power loss -- no need to manually
start the agent then run `grid_nav.py` anymore.

## If something doesn't come up

- **`docker compose logs -f grid-nav` shows an rclpy/DDS error, or
  grid_nav.py never sees IMU/encoder data**: the two containers likely
  aren't on the same ROS2 domain, or the agent isn't actually connected to
  the Teensy -- check `docker compose logs -f microros-agent` first.
- **Camera shows "unavailable" in the GUI**: confirm `/dev/video0` in
  `docker-compose.yml` actually matches `ls /dev/video*` on this Pi, and
  that the container was rebuilt/restarted after any change
  (`docker compose up -d --build`).
- **`pupil-apriltags` fails to build during `docker compose build`**: it's
  optional (see `grid_nav.py`'s own docstring) -- remove the
  `pip3 install ... pupil-apriltags` line from `Dockerfile` and rebuild if
  you don't need AprilTag-based position correction/homing.
