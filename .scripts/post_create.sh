#!/bin/bash
# .scripts/post_create.sh
set -e

WS_NAME="${LOCAL_WORKSPACE_FOLDER_BASENAME:-$(basename $(pwd))}"

# --- Build TurtleBot3 workspace ---
source /opt/ros/humble/setup.bash
cd /opt/turtlebot3_ws
rm -rf build install log
colcon build --symlink-install

# --- Setup Python venv ---
cd /workspaces/${WS_NAME}
if [ -d .venv ]; then
    sudo chown -R vscode:vscode .venv || true
    rm -rf .venv || true
fi
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Install PyTorch (GPU-aware) ---
bash .scripts/install_torch.sh

# --- Install remaining requirements ---
if [ -f requirements.txt ]; then
    pip install --no-cache-dir --no-build-isolation -r requirements.txt
fi