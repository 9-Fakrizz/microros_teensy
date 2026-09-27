# Runs grid_nav.py as a ROS2 node + Flask GUI. Must match the ROS2 distro
# the micro-ROS agent container uses (see docker-compose.yml) -- DDS
# message types are distro-specific, so a mismatch here means grid_nav.py
# and the agent silently can't talk to each other.
FROM ros:jazzy-ros-base

# python3-opencv pulls in a real apt-built OpenCV (with numpy) for arm64 --
# far more reliable on a Raspberry Pi than `pip install opencv-python`,
# which often has no prebuilt aarch64 wheel and fails to build from source
# without a long list of extra apt packages.
#
# build-essential + cmake are only here so `pip install pupil-apriltags`
# (optional AprilTag support -- see grid_nav.py's own docstring) can build
# from source if PyPI has no prebuilt aarch64 wheel for it. If you don't
# need AprilTag features, delete the pupil-apriltags line below and these
# two packages to make the image build noticeably faster.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-opencv \
        python3-pip \
        python3-flask \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# --break-system-packages: this image's Python is only used to run
# grid_nav.py, so there's no concern about clobbering a system Python
# Ubuntu itself depends on.
RUN pip3 install --no-cache-dir --break-system-packages pupil-apriltags

WORKDIR /app
COPY grid_nav.py .

# Matches the ROS2 setup grid_nav.py's own docstring says to source
# before running it natively -- here that happens once, at container
# start, via this image's ros_entrypoint.sh (baked into the ros:jazzy
# base image) sourcing /opt/ros/jazzy/setup.bash automatically.
CMD ["python3", "grid_nav.py"]
