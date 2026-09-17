#!/bin/bash

# Auto-detect workspace
WORKSPACE=$(find /workspaces -maxdepth 1 -mindepth 1 -type d | head -1)

# Kill existing session if any
tmux kill-session -t work 2>/dev/null

source /etc/bash.bashrc

# Window 0: Plain terminal
tmux new-session -d -s work -n "plain"
tmux send-keys -t work:plain "cd $WORKSPACE && source /etc/bash.bashrc" Enter

# Window 1: tb3setup
tmux new-window -t work -n "tb3setup"
tmux send-keys -t work:tb3setup "source /etc/bash.bashrc && tb3setup" Enter

# Window 2: tb3venv 1
tmux new-window -t work -n "tb3venv1"
tmux send-keys -t work:tb3venv1 "source /etc/bash.bashrc && tb3venv" Enter

# Window 3: tb3venv 2
tmux new-window -t work -n "tb3venv2"
tmux send-keys -t work:tb3venv2 "source /etc/bash.bashrc && tb3venv" Enter

# Start at window 0
tmux select-window -t work:plain

# Attach to session
tmux attach -t work