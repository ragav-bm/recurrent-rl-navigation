# ROS2 Humble base image
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install development tools and ROS packages
RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-dev-tools \
    git \
    curl \
    wget \
    vim \
    nano \
    tmux \
    openssh-client \
    build-essential \
    cmake \
    gdb \
    gazebo \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-rviz2 \
    ros-humble-tf-transformations \
    ros-humble-turtlebot3-msgs \
    ros-humble-xacro \
    ros-humble-rmw-zenoh-cpp \
    iproute2 \
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libglu1-mesa \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN useradd -ms /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Give vscode ownership of workspace area
RUN mkdir -p /workspaces && chown -R vscode:vscode /workspaces

# Configure ROS environment globally
RUN cat >> /etc/bash.bashrc << 'EOF'

# Source ROS2 automatically
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Auto-detect workspace
_ws() {
    echo "/workspaces/$(ls /workspaces/ | head -1)"
}

# Gazebo model
export TURTLEBOT3_MODEL=burger

# TensorBoard shortcut
alias tens='tensorboard --logdir ./ --port 6007'

# Activate Python virtual environment
alias venv='source $(_ws)/.venv/bin/activate'

# TurtleBot3 workspace setup
tb3setup() {
    cd $(_ws) || return
    source /opt/ros/humble/setup.bash
    source /opt/turtlebot3_ws/install/setup.bash
    export TURTLEBOT3_MODEL=burger
    echo "TurtleBot3 workspace ready ($(basename $(_ws)))"
}

# Full RL development environment
tb3venv() {
    cd $(_ws) || return
    source .venv/bin/activate
    source /opt/ros/humble/setup.bash
    source /opt/turtlebot3_ws/install/setup.bash
    export TURTLEBOT3_MODEL=burger
    echo "ROS2 + TurtleBot3 + Python virtualenv ready ($(basename $(_ws)))"
}

# === AUTOMATIC ZENOH ISOLATION ===
_setup_zenoh() {
    local WS_NAME="$(ls /workspaces/ 2>/dev/null | head -1)"
    if [ -z "$WS_NAME" ]; then return; fi

    local PORT=$(( 7448 + $(echo "$WS_NAME" | cksum | cut -d' ' -f1) % 100 ))

    local ROUTER_CFG="/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5"
    local SESSION_CFG="/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5"

    sudo tee "$ROUTER_CFG" > /dev/null 2>&1 << EOCFG
{
  mode: "router",
  listen: {
    endpoints: ["tcp/127.0.0.1:${PORT}"]
  },
  scouting: {
    multicast: {
      enabled: false
    }
  },
  transport: {
    shared_memory: {
      enabled: false
    }
  }
}
EOCFG

    sudo tee "$SESSION_CFG" > /dev/null 2>&1 << EOCFG
{
  mode: "client",
  connect: {
    endpoints: ["tcp/127.0.0.1:${PORT}"]
  },
  scouting: {
    multicast: {
      enabled: false
    }
  },
  transport: {
    shared_memory: {
      enabled: false
    }
  }
}
EOCFG

    if ! ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        ros2 run rmw_zenoh_cpp rmw_zenohd &>/dev/null &
        disown
        sleep 1
        echo "✅ Zenoh router started | port=${PORT} | workspace=${WS_NAME}"
    fi
}
_setup_zenoh

# === AUTOMATIC GAZEBO ISOLATION ===
_setup_gazebo() {
    local WS_NAME="$(ls /workspaces/ 2>/dev/null | head -1)"
    if [ -z "$WS_NAME" ]; then return; fi
    local GZ_PORT=$(( 11345 + $(echo "$WS_NAME" | cksum | cut -d' ' -f1) % 100 ))
    export GAZEBO_MASTER_URI="http://localhost:${GZ_PORT}"
    [[ $- == *i* ]] && echo "🎮 Gazebo port=${GZ_PORT} | workspace=${WS_NAME}"
}
_setup_gazebo

EOF

# Tmux config
RUN cat > /home/vscode/.tmux.conf << 'EOF'
set -g mouse on
set -g status-style 'bg=blue fg=white'
set -g window-status-current-style 'bg=white fg=black bold'
set -g base-index 1
set -g history-limit 10000
EOF
RUN chown vscode:vscode /home/vscode/.tmux.conf

# Suppress MOTD
RUN touch /home/vscode/.hushlogin

# Set working directory
WORKDIR /workspaces

# Switch to vscode user
USER vscode

# Default shell
SHELL ["/bin/bash", "-c"]