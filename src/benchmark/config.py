"""
Benchmark Configuration — Single source of truth.
"""
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

GAZEBO_GUI_WARMUP_SECONDS = 40       # How long to keep GUI open for obstacles to start
GAZEBO_HEADLESS_SETTLE_SECONDS = 10  # Wait after headless relaunch
GAZEBO_MAX_RETRIES = 5               # Max attempts to spawn robot
SCAN_CHECK_TIMEOUT = 10              # Seconds to wait for /scan per check

# ═══════════════════════════════════════════════════
# SEEDS — Start with one, expand later
# ═══════════════════════════════════════════════════
SEEDS = [42]
# Future: SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════
EXPERIMENTS = {
    "sac_lstm_per": {
        "description": "SAC + LSTM + PER (Full Pipeline)",
        "script": "train/train_lstm.py",
        "args": ["--train", "--buffer-type", "per"],
        "tag": "lstm_per",
    },
    "sac_lstm": {
        "description": "SAC + LSTM (Uniform Buffer, No PER)",
        "script": "train/train_lstm.py",
        "args": ["--train", "--buffer-type", "uniform"],
        "tag": "lstm_uniform",
    },
    "sac_mlp": {
        "description": "SAC + MLP (Feedforward Baseline)",
        "script": "train/train_mlp.py",
        "args": ["--train"],
        "tag": "mlp",
    },
}

# ═══════════════════════════════════════════════════
# PARALLEL MODE SETTINGS
# ═══════════════════════════════════════════════════
WORKTREE_BASE = PROJECT_ROOT.parent / "benchmark_worktrees"
GIT_BRANCH = "main"


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════
def get_all_runs(experiment_filter=None, seed_filter=None):
    """Generate all (experiment, seed) combinations."""
    seeds = seed_filter if seed_filter else SEEDS
    experiments = EXPERIMENTS
    if experiment_filter:
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in experiment_filter}

    runs = []
    for exp_key, exp_cfg in experiments.items():
        for seed in seeds:
            run_id = f"{exp_cfg['tag']}_seed{seed}"
            runs.append({
                "experiment_key": exp_key,
                "run_id": run_id,
                "seed": seed,
                "script": exp_cfg["script"],
                "args": exp_cfg["args"] + ["--seed", str(seed)],
                "tag": exp_cfg["tag"],
                "description": exp_cfg["description"],
            })
    return runs


def print_plan(runs):
    print(f"\n{'═' * 70}")
    print("BENCHMARK PLAN")
    print(f"{'═' * 70}")
    print(f"{'#':<4} {'Run ID':<25} {'Seed':<6} {'Description'}")
    print(f"{'─' * 70}")
    for i, run in enumerate(runs, 1):
        print(f"{i:<4} {run['run_id']:<25} {run['seed']:<6} {run['description']}")
    print(f"{'═' * 70}")
    print(f"Total runs: {len(runs)}\n")
