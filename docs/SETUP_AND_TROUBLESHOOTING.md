# Environment Setup, Parallel Worktrees & Troubleshooting Guide

This document provides detailed instructions for setting up the **DevContainer environment**, running **parallel Git Worktree experiments**, verifying **automatic container isolation**, and troubleshooting common **ROS 2 / Gazebo / X11** issues.

---

## Table of Contents

1. [Prerequisites & DevContainer CLI Installation](#1-prerequisites--devcontainer-cli-installation)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Automatic Container Isolation Details](#3-automatic-container-isolation-details)
4. [Automated Benchmark CLI Reference](#4-automated-benchmark-cli-reference)
5. [Running Multiple Experiments in Parallel (Git Worktrees)](#5-running-multiple-experiments-in-parallel-git-worktrees)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites & DevContainer CLI Installation

### Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- [DevContainer CLI](https://github.com/devcontainers/cli): `npm install -g @devcontainers/cli`
- VS Code with Dev Containers extension (optional, for GUI editing/debugging)

### Installing DevContainer CLI

**Method 1: Using NVM (Recommended — no `sudo` required)**

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install 20
npm install -g @devcontainers/cli
devcontainer --version
```

**Method 2: Using NodeSource APT Repository**

```bash
sudo apt remove -y nodejs npm
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g @devcontainers/cli
devcontainer --version
```

---

## 2. Development Environment Setup

### Build and Start the Container

First time or after modifying `.devcontainer/Dockerfile`:

```bash
cd ~/workspaces/Mapless-RL-Navigation
devcontainer up --workspace-folder . --remove-existing-container --build-no-cache
```

Normal start (when image is already built):

```bash
devcontainer up --workspace-folder .
```

This builds the Docker image (ROS 2 Humble, Gazebo, Python tools, tmux), starts the container with GPU passthrough and X11 forwarding, and runs `.scripts/post_create.sh` which:
1. Builds the TurtleBot3 colcon workspace (`/opt/turtlebot3_ws`)
2. Creates the Python virtual environment
3. Auto-detects the host GPU architecture and installs the matching PyTorch CUDA wheel
4. Installs `requirements.txt`

### Launch or Reattach the tmux Session

```bash
# Start preconfigured tmux session
devcontainer exec --workspace-folder . bash -lc '/workspaces/Mapless-RL-Navigation/.scripts/start_tmux.sh'

# Reattach later
devcontainer exec --workspace-folder . tmux attach -t ros2
```

### Attach VS Code (Optional)

```bash
devcontainer open --workspace-folder .
```

Or in VS Code: `Ctrl+Shift+P` → **"Dev Containers: Attach to Running Container..."**. Closing VS Code does **not** stop running processes inside the container.

---

## 3. Automatic Container Isolation Details

Each container is isolated automatically via `/etc/bash.bashrc` on shell startup:

1. **Zenoh Router (`rmw_zenohd`)**: Computes a deterministic port from the workspace folder name hash, disables shared memory in the Zenoh config, and starts `rmw_zenohd` in the background.
2. **Gazebo (`GAZEBO_MASTER_URI`)**: Assigns a unique Gazebo master port per workspace so multiple `gzserver` instances run simultaneously without colliding on port `11345`.
3. **PyTorch CUDA Auto-Detection (`.scripts/install_torch.sh`)**:

| GPU Series | Compute Capability | PyTorch CUDA Wheel |
| :--- | :---: | :---: |
| NVIDIA RTX 5070 / 5080 / 5090 | `sm_120` | `cu132` |
| NVIDIA RTX 4060 – 4090 | `sm_89` | `cu124` |
| NVIDIA RTX 3060 – 3090 | `sm_86` | `cu121` |

**Deterministic Port Formula:**

```bash
ZENOH_PORT=$(( 7448 + $(echo "$WORKSPACE_NAME" | cksum | cut -d' ' -f1) % 100 ))
GAZEBO_PORT=$(( 11345 + $(echo "$WORKSPACE_NAME" | cksum | cut -d' ' -f1) % 100 ))
```

**Verify Active Ports on Host:**

```bash
ss -tlnp | grep rmw_zenohd
ss -tlnp | grep 113
```

---

## 4. Automated Benchmark CLI Reference

```bash
# Dry run — preview planned containers and seeds
python -m benchmark.runner --mode parallel --dry-run

# Run all configured experiments
python -m benchmark.runner --mode parallel

# Run specific algorithm variants and seeds
python -m benchmark.runner --mode parallel --experiments sac_lstm_per sac_mlp --seeds 42 123 456

# Skip Gazebo GUI warmup (if containers already have Gazebo running)
python -m benchmark.runner --mode parallel --skip-gazebo-warmup

# Override Gazebo retries and warmup duration
python -m benchmark.runner --mode parallel --gazebo-retries 3 --gazebo-warmup-time 30

# Sequential debug mode inside a single container
python -m benchmark.runner --mode debug --experiments sac_lstm_per

# Stop all benchmark containers and clean up worktrees
python -m benchmark.runner --mode cleanup
```

---

## 5. Running Multiple Experiments in Parallel (Git Worktrees)

You can run multiple custom experiments simultaneously in separate containers using Git Worktrees while sharing the ROS 2 TurtleBot3 workspace (`/opt/turtlebot3_ws`).

### Create Experiment Branches & Worktrees

```bash
cd ~/workspaces/Mapless-RL-Navigation

git branch exp/baseline
git branch exp/reward-shaping
git branch exp/new-obs-space

mkdir -p ~/workspaces/git_worktree/mapless_rl_navigation/experiments
git worktree add ~/workspaces/git_worktree/mapless_rl_navigation/experiments/baseline exp/baseline
git worktree add ~/workspaces/git_worktree/mapless_rl_navigation/experiments/reward_shaping exp/reward-shaping
git worktree add ~/workspaces/git_worktree/mapless_rl_navigation/experiments/new_obs_space exp/new-obs-space
```

### Build & Launch Experiment Containers

```bash
devcontainer up --workspace-folder ~/workspaces/git_worktree/mapless_rl_navigation/experiments/baseline --build-no-cache --remove-existing-container
devcontainer up --workspace-folder ~/workspaces/git_worktree/mapless_rl_navigation/experiments/reward_shaping --build-no-cache --remove-existing-container

devcontainer exec --workspace-folder ~/workspaces/git_worktree/mapless_rl_navigation/experiments/baseline bash -lc '/workspaces/baseline/.scripts/start_tmux.sh'
devcontainer exec --workspace-folder ~/workspaces/git_worktree/mapless_rl_navigation/experiments/reward_shaping bash -lc '/workspaces/reward_shaping/.scripts/start_tmux.sh'
```

### Cleanup Worktrees

```bash
docker stop $(docker ps -q --filter "label=devcontainer.local_folder=$HOME/workspaces/git_worktree/mapless_rl_navigation/experiments/new_obs_space")
git worktree remove ~/workspaces/git_worktree/mapless_rl_navigation/experiments/new_obs_space
git branch -d exp/new-obs-space
```

---

## 6. Troubleshooting

### Gazebo GUI Not Showing (`gzclient` / `rviz2`)

```bash
ls /tmp/.X11-unix/     # Check available X11 displays
export DISPLAY=:1
gzclient
```

Or grant Docker X11 access from the **host**:

```bash
xhost +local:docker
```

### Gazebo Fails in Multiple Containers (Exit Code 255)

Ensure `GAZEBO_MASTER_URI` uses a distinct port per container:

```bash
echo $GAZEBO_MASTER_URI
export GAZEBO_MASTER_URI=http://localhost:11400
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4_moving_obs.launch.py gui:=false
```

### ROS 2 Daemon Crash (`!rclpy.ok()`) or `/scan` Not Detected

When many nodes launch rapidly, the ROS 2 CLI daemon can stall. Restart it with:

```bash
ros2 daemon stop
ros2 daemon start
sleep 2
ros2 topic list
ros2 topic echo /scan sensor_msgs/msg/LaserScan --once
```

### ROS 2 Nodes Cannot Communicate

Verify that the Zenoh router daemon is running inside the container:

```bash
pgrep -f rmw_zenohd
source /etc/bash.bashrc
ros2 topic list
```

### Rebuilding Colcon Workspace Inside Container

```bash
devcontainer exec --workspace-folder . bash -lc "source /opt/ros/humble/setup.bash && cd /opt/turtlebot3_ws && rm -rf build install log && colcon build --symlink-install"
```
