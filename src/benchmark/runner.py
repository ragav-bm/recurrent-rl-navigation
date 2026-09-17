#!/usr/bin/env python3
"""
Benchmark Runner for Mapless RL Navigation.
Executes configured RL baselines across designated random seeds, records performance metrics,
and compiles structured benchmark reports.

Usage:
    python3 -m benchmark.runner --list
    python3 -m benchmark.runner --dry-run
    python3 -m benchmark.runner --experiments sac_lstm_per sac_mlp --seeds 42
    python3 -m benchmark.runner --mode sequential --experiments sac_lstm_per
"""
import argparse
import os
import sys
import subprocess
import pathlib
from typing import List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.config import EXPERIMENTS, DEFAULT_SEEDS, RESULTS_DIR


def list_experiments():
    """Print all available benchmark experiments."""
    print("\n" + "=" * 75)
    print("  AVAILABLE BENCHMARK EXPERIMENTS")
    print("=" * 75)
    for exp_id, cfg in EXPERIMENTS.items():
        print(f"  • {exp_id:<20} : {cfg['description']}")
        print(f"    Script: {cfg['script']} {' '.join(cfg['args'])}\n")


def build_command(script_rel_path: str, extra_args: List[str], seed: int, run_id: str) -> List[str]:
    """Construct the command line list for a training run."""
    script_path = os.path.join(str(PROJECT_ROOT), script_rel_path)
    cmd = [sys.executable, script_path] + extra_args + ["--seed", str(seed)]
    return cmd


def run_benchmark(experiments: List[str], seeds: List[int], dry_run: bool = False, mode: str = "sequential"):
    """Execute selected experiments across specified seeds."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "═" * 75)
    print("  MAPLESS RL NAVIGATION BENCHMARK RUNNER")
    print("═" * 75)
    print(f"  Mode        : {mode.upper()}")
    print(f"  Experiments : {', '.join(experiments)}")
    print(f"  Seeds       : {seeds}")
    print(f"  Dry Run     : {dry_run}")
    print(f"  Results Dir : {RESULTS_DIR}")
    print("═" * 75 + "\n")

    total_runs = len(experiments) * len(seeds)
    current_run = 0

    for exp_id in experiments:
        if exp_id not in EXPERIMENTS:
            print(f"[ERROR] Unknown experiment '{exp_id}'. Skipping.")
            continue

        cfg = EXPERIMENTS[exp_id]
        for seed in seeds:
            current_run += 1
            run_id = f"{cfg['tag']}_seed{seed}"
            cmd = build_command(cfg["script"], cfg["args"], seed, run_id)

            print(f"[{current_run}/{total_runs}] Running {exp_id} (Seed {seed}) -> {run_id}")
            print(f"      Command: {' '.join(cmd)}")

            if dry_run:
                continue

            env = os.environ.copy()
            env["BENCHMARK_RUN_ID"] = run_id
            env["PYTHONUNBUFFERED"] = "1"

            try:
                result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
                if result.returncode != 0:
                    print(f"      [WARN] Run {run_id} exited with status {result.returncode}")
                else:
                    print(f"      [SUCCESS] Run {run_id} finished successfully.")
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Benchmark runner stopped by user.")
                sys.exit(130)
            except Exception as e:
                print(f"      [ERROR] Failed to run {run_id}: {e}")

    print("\n" + "═" * 75)
    print("  BENCHMARK EXECUTION COMPLETED")
    print("═" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Mapless RL Navigation Benchmark Runner")
    parser.add_argument("--list", action="store_true", help="List all configured experiments")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()),
                        help="List of experiment IDs to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds for statistical evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running")
    parser.add_argument("--mode", type=str, choices=["sequential", "debug"], default="sequential",
                        help="Execution mode (sequential or debug)")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    run_benchmark(args.experiments, args.seeds, dry_run=args.dry_run, mode=args.mode)


if __name__ == "__main__":
    main()

