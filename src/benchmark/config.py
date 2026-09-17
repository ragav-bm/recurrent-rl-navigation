"""
Benchmark Configuration for Mapless RL Navigation Experiments.
Defines experiment configurations, benchmark seeds, and runner parameters.
"""
import os
import pathlib

BENCHMARK_ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parent

# Default evaluation seeds for statistical significance
DEFAULT_SEEDS = [42, 123, 456]

# Results and log storage directories
RESULTS_DIR = os.path.join(PROJECT_ROOT.parent, "runs", "benchmarks")

# Benchmark Experiments Suite
EXPERIMENTS = {
    "sac_lstm_per": {
        "description": "SAC + LSTM + PER (Recurrent Continuous Control with R2D2 Burn-In)",
        "script": "train/train_lstm.py",
        "args": ["--train", "--buffer-type", "per"],
        "tag": "lstm_per",
    },
    "sac_lstm_uniform": {
        "description": "SAC + LSTM + Uniform Replay (Ablation: No Priority)",
        "script": "train/train_lstm.py",
        "args": ["--train", "--buffer-type", "uniform"],
        "tag": "lstm_uniform",
    },
    "sac_mlp": {
        "description": "SAC + MLP Feedforward Baseline (No Recurrence)",
        "script": "train/train_mlp.py",
        "args": ["--train"],
        "tag": "mlp",
    },
    "sb3_recurrent_ppo": {
        "description": "Stable-Baselines3 Recurrent PPO Baseline",
        "script": "train/train_sb3.py",
        "args": ["--algorithm", "recurrent_ppo"],
        "tag": "sb3_recurrent_ppo",
    },
}

