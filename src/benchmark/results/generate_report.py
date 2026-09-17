#!/usr/bin/env python3
"""
Report Generator — Produces formatted benchmark reports.

Supports:
  - Console (pretty tables)
  - Markdown
  - CSV
  - LaTeX

Usage:
    python -m benchmark.results.generate_report --domain gazebo
    python -m benchmark.results.generate_report --domain pomujoco
    python -m benchmark.results.generate_report --domain gazebo --format markdown --output report.md
    python -m benchmark.results.generate_report --domain pomujoco --format csv --output results.csv
"""
import argparse
import sys
import os
import pathlib
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

SRC_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from benchmark.results.collector import ResultCollector, RunResult, print_scan_summary


# ═══════════════════════════════════════════════════════════
# REPORT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

class ExperimentGroup:
    """Aggregated results for one experiment across seeds."""

    def __init__(self, experiment_key: str):
        self.experiment_key = experiment_key
        self.runs: List[RunResult] = []

    def add(self, run: RunResult):
        self.runs.append(run)

    @property
    def n_seeds(self) -> int:
        return len(self.runs)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.runs if r.status == "completed")

    @property
    def seeds(self) -> List[int]:
        return sorted(set(r.seed for r in self.runs))

    @property
    def architecture(self) -> str:
        for r in self.runs:
            if r.architecture:
                return r.architecture
        return "?"

    @property
    def buffer_type(self) -> str:
        for r in self.runs:
            if r.buffer_type:
                return r.buffer_type
        return "?"

    @property
    def best_rewards(self) -> List[float]:
        return [r.best_eval_reward for r in self.runs
                if r.status == "completed" and r.best_eval_reward > float("-inf")]

    @property
    def success_rates(self) -> List[float]:
        return [r.best_eval_success_rate for r in self.runs
                if r.status == "completed" and r.best_eval_success_rate >= 0]

    @property
    def mean_best_reward(self) -> float:
        vals = self.best_rewards
        return float(np.mean(vals)) if vals else 0.0

    @property
    def std_best_reward(self) -> float:
        vals = self.best_rewards
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    @property
    def mean_success_rate(self) -> float:
        vals = self.success_rates
        return float(np.mean(vals)) if vals else -1.0

    @property
    def std_success_rate(self) -> float:
        vals = self.success_rates
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    @property
    def mean_steps(self) -> float:
        vals = [r.total_steps for r in self.runs if r.status == "completed" and r.total_steps > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_duration(self) -> float:
        vals = [r.training_duration_sec for r in self.runs
                if r.status == "completed" and r.training_duration_sec > 0]
        return float(np.mean(vals)) if vals else 0.0


# ═══════════════════════════════════════════════════════════
# GROUPING
# ═══════════════════════════════════════════════════════════

def group_by_experiment(runs: List[RunResult]) -> Dict[str, ExperimentGroup]:
    """Group runs by experiment key."""
    groups: Dict[str, ExperimentGroup] = {}
    for run in runs:
        key = run.experiment or "unknown"
        if key not in groups:
            groups[key] = ExperimentGroup(key)
        groups[key].add(run)
    return groups


def group_by_env_and_experiment(runs: List[RunResult]) -> Dict[str, Dict[str, ExperimentGroup]]:
    """Group runs by (environment, experiment)."""
    result: Dict[str, Dict[str, ExperimentGroup]] = defaultdict(dict)
    for run in runs:
        env = run.env_name or "gazebo_nav"
        key = run.experiment or "unknown"
        if key not in result[env]:
            result[env][key] = ExperimentGroup(key)
        result[env][key].add(run)
    return dict(result)


# ═══════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════════════════

def _fmt_duration(sec: float) -> str:
    if sec <= 0:
        return "-"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


FRIENDLY_NAMES = {
    "lstm_per": "SAC-LSTM + PER",
    "lstm_uniform": "SAC-LSTM + Uniform",
    "sac_mlp": "SAC-MLP",
    "sb3_ppo": "SB3-PPO",
    "sb3_recurrent_ppo": "SB3-RecurrentPPO",
    "sb3_sac": "SB3-SAC",
    "sb3_td3": "SB3-TD3",
    "sac_lstm_per_po": "SAC-LSTM+PER (PO)",
    "sac_lstm_uniform_po": "SAC-LSTM+Uni (PO)",
    "sac_mlp_po": "SAC-MLP (PO)",
    "sac_lstm_per_fo": "SAC-LSTM+PER (FO)",
    "sac_mlp_fo": "SAC-MLP (FO)",
    "sac_lstm_per_flicker": "SAC-LSTM+PER (Flicker)",
}


def report_console(runs: List[RunResult], domain: str):
    """Print a formatted console report."""
    groups = group_by_experiment(runs)

    max_id_len = max((len(r.run_id) for r in runs), default=40)
    table_width = max(max_id_len + 60, 90)  # dynamic total width

    print(f"\n{'═' * table_width}")
    print(f"  BENCHMARK REPORT — {domain.upper()}")
    print(f"{'═' * table_width}")

    # Determine if we have success rates (Gazebo) or just rewards (MuJoCo)
    has_sr = any(g.mean_success_rate >= 0 for g in groups.values())

    if has_sr:
        # Gazebo-style table with success rates
        print(f"\n  {'Experiment':<28} {'Arch':<6} {'Buf':<8} {'Seeds':<6} "
              f"{'Best SR%':<12} {'Best Reward':<16} {'Duration':<10}")
        print(f"  {'─' * 86}")

        sorted_groups = sorted(groups.items(),
                               key=lambda x: x[1].mean_success_rate, reverse=True)

        for exp_key, group in sorted_groups:
            name = FRIENDLY_NAMES.get(exp_key, exp_key)
            sr_str = (f"{group.mean_success_rate:.1f} ± {group.std_success_rate:.1f}%"
                      if group.mean_success_rate >= 0 else "-")
            reward_str = (f"{group.mean_best_reward:.1f} ± {group.std_best_reward:.1f}"
                          if group.best_rewards else "-")
            dur_str = _fmt_duration(group.mean_duration)

            completed = f"{group.n_completed}/{group.n_seeds}"

            print(f"  {name:<28} {group.architecture:<6} {group.buffer_type:<8} "
                  f"{completed:<6} {sr_str:<12} {reward_str:<16} {dur_str:<10}")
    else:
        # MuJoCo-style table with rewards only
        print(f"\n  {'Experiment':<28} {'Arch':<6} {'Buf':<8} {'Seeds':<6} "
              f"{'Best Reward':<18} {'Final Reward':<18} {'Duration':<10}")
        print(f"  {'─' * 94}")

        sorted_groups = sorted(groups.items(),
                               key=lambda x: x[1].mean_best_reward, reverse=True)

        for exp_key, group in sorted_groups:
            name = FRIENDLY_NAMES.get(exp_key, exp_key)
            best_str = (f"{group.mean_best_reward:.1f} ± {group.std_best_reward:.1f}"
                        if group.best_rewards else "-")
            # Final reward
            final_vals = [r.final_eval_reward for r in group.runs
                          if r.status == "completed" and r.final_eval_reward > float("-inf")]
            if final_vals:
                final_str = f"{np.mean(final_vals):.1f} ± {np.std(final_vals):.1f}"
            else:
                final_str = "-"
            dur_str = _fmt_duration(group.mean_duration)
            completed = f"{group.n_completed}/{group.n_seeds}"

            print(f"  {name:<28} {group.architecture:<6} {group.buffer_type:<8} "
                  f"{completed:<6} {best_str:<18} {final_str:<18} {dur_str:<10}")

    max_id_len = max((len(r.run_id) for r in runs), default=40)
    id_width = max(max_id_len + 2, 40)  # at least 40, or wider if needed
    separator_len = id_width + 6 + 12 + 12 + 8 + 10 + 10

    print(f"\n{'─' * separator_len}")
    print(f"  PER-SEED BREAKDOWN")
    print(f"{'─' * separator_len}")
    print(f"  {'Run ID':<{id_width}} {'Seed':<6} {'Status':<12} "
          f"{'Best Reward':<12} {'SR%':<8} {'Steps':<10}")
    print(f"  {'─' * (separator_len - 2)}")

    for run in sorted(runs, key=lambda r: (r.experiment, r.seed)):
        sr_str = f"{run.best_eval_success_rate:.1f}%" if run.best_eval_success_rate >= 0 else "-"
        reward_str = f"{run.best_eval_reward:.1f}" if run.best_eval_reward > float("-inf") else "-"
        steps_str = f"{run.total_steps:,}" if run.total_steps > 0 else "-"

        print(f"  {run.run_id:<{id_width}} {run.seed:<6} {run.status:<12} "
              f"{reward_str:<12} {sr_str:<8} {steps_str:<10}")

    print(f"\n{'═' * separator_len}\n")


# ═══════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════

def report_markdown(runs: List[RunResult], domain: str) -> str:
    """Generate a Markdown report."""
    groups = group_by_experiment(runs)
    lines = []

    lines.append(f"# Benchmark Report — {domain.upper()}")
    lines.append("")
    lines.append(f"**Total runs:** {len(runs)} | "
                 f"**Completed:** {sum(1 for r in runs if r.status == 'completed')}")
    lines.append("")

    has_sr = any(g.mean_success_rate >= 0 for g in groups.values())

    if has_sr:
        lines.append("## Results Summary")
        lines.append("")
        lines.append("| Experiment | Arch | Buffer | Seeds | Best SR% | Best Reward | Duration |")
        lines.append("|---|---|---|---|---|---|---|")

        sorted_groups = sorted(groups.items(),
                               key=lambda x: x[1].mean_success_rate, reverse=True)

        for exp_key, group in sorted_groups:
            name = FRIENDLY_NAMES.get(exp_key, exp_key)
            sr_str = (f"{group.mean_success_rate:.1f} ± {group.std_success_rate:.1f}%"
                      if group.mean_success_rate >= 0 else "-")
            reward_str = (f"{group.mean_best_reward:.1f} ± {group.std_best_reward:.1f}"
                          if group.best_rewards else "-")
            dur_str = _fmt_duration(group.mean_duration)
            completed = f"{group.n_completed}/{group.n_seeds}"
            lines.append(f"| {name} | {group.architecture} | {group.buffer_type} | "
                         f"{completed} | {sr_str} | {reward_str} | {dur_str} |")
    else:
        lines.append("## Results Summary")
        lines.append("")
        lines.append("| Experiment | Arch | Buffer | Seeds | Best Reward | Final Reward | Duration |")
        lines.append("|---|---|---|---|---|---|---|")

        sorted_groups = sorted(groups.items(),
                               key=lambda x: x[1].mean_best_reward, reverse=True)

        for exp_key, group in sorted_groups:
            name = FRIENDLY_NAMES.get(exp_key, exp_key)
            best_str = (f"{group.mean_best_reward:.1f} ± {group.std_best_reward:.1f}"
                        if group.best_rewards else "-")
            final_vals = [r.final_eval_reward for r in group.runs
                          if r.status == "completed" and r.final_eval_reward > float("-inf")]
            final_str = (f"{np.mean(final_vals):.1f} ± {np.std(final_vals):.1f}"
                         if final_vals else "-")
            dur_str = _fmt_duration(group.mean_duration)
            completed = f"{group.n_completed}/{group.n_seeds}"
            lines.append(f"| {name} | {group.architecture} | {group.buffer_type} | "
                         f"{completed} | {best_str} | {final_str} | {dur_str} |")

    # Per-seed table
    lines.append("")
    lines.append("## Per-Seed Breakdown")
    lines.append("")
    lines.append("| Run ID | Seed | Status | Best Reward | SR% | Steps |")
    lines.append("|---|---|---|---|---|---|")

    for run in sorted(runs, key=lambda r: (r.experiment, r.seed)):
        sr_str = f"{run.best_eval_success_rate:.1f}%" if run.best_eval_success_rate >= 0 else "-"
        reward_str = f"{run.best_eval_reward:.1f}" if run.best_eval_reward > float("-inf") else "-"
        steps_str = f"{run.total_steps:,}" if run.total_steps > 0 else "-"
        lines.append(f"| {run.run_id} | {run.seed} | {run.status} | "
                     f"{reward_str} | {sr_str} | {steps_str} |")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# CSV REPORT
# ═══════════════════════════════════════════════════════════

def report_csv(runs: List[RunResult]) -> str:
    """Generate CSV output."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "run_id", "experiment", "architecture", "buffer", "seed", "env",
        "status", "total_episodes", "total_steps", "total_updates",
        "best_eval_reward", "best_eval_success_rate",
        "final_eval_reward", "training_duration_sec"
    ])

    for run in sorted(runs, key=lambda r: (r.experiment, r.seed)):
        writer.writerow([
            run.run_id, run.experiment, run.architecture, run.buffer_type,
            run.seed, run.env_name, run.status, run.total_episodes,
            run.total_steps, run.total_updates, run.best_eval_reward,
            run.best_eval_success_rate, run.final_eval_reward,
            run.training_duration_sec
        ])

    return output.getvalue()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from training logs."
    )
    parser.add_argument("--domain", choices=["gazebo", "pomujoco", "popgym", "all"],
                        default="gazebo",
                        help="Which benchmark domain to report on")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--format", choices=["console", "markdown", "csv"],
                        default="console")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    workspace_root = SRC_DIR.parent

    # Determine results directory
    if args.results_dir:
        results_dirs = [(args.domain, args.results_dir)]
    elif args.domain == "all":
        results_dirs = [
            ("gazebo", str(workspace_root / "results" / "gazebo" / "models")),
            ("pomujoco", str(workspace_root / "results" / "pomujoco" / "models")),
            ("popgym", str(workspace_root / "results" / "popgym" / "models")),  # ← ADD
        ]
    elif args.domain == "gazebo":
        model_dir_candidates = [
            str(workspace_root / "results" / "gazebo" / "models"),
            str(SRC_DIR / "models"),
        ]
        results_dirs = [("gazebo", d) for d in model_dir_candidates if os.path.isdir(d)]
        if not results_dirs:
            results_dirs = [("gazebo", str(SRC_DIR / "models"))]
    elif args.domain == "pomujoco":
        results_dirs = [
            ("pomujoco", str(workspace_root / "results" / "pomujoco" / "models"))
        ]
    elif args.domain == "popgym":                                              # ← ADD
        results_dirs = [                                                        # ← ADD
            ("popgym", str(workspace_root / "results" / "popgym" / "models"))   # ← ADD
        ]                                                                       # ← ADD
    else:
        results_dirs = [
            ("pomujoco", str(workspace_root / "results" / "pomujoco" / "models"))
        ]

    # Collect all runs
    all_runs: List[RunResult] = []
    for domain, results_dir in results_dirs:
        print(f"\n  Scanning {domain}: {results_dir}")
        collector = ResultCollector(results_dir, domain)
        runs = collector.scan_all()
        all_runs.extend(runs)
        print_scan_summary(runs)

    if not all_runs:
        print("\n  [ERROR] No runs found. Check --results-dir or run the benchmark first.")
        sys.exit(1)

    # Generate report
    if args.format == "console":
        report_console(all_runs, args.domain)
    elif args.format == "markdown":
        md = report_markdown(all_runs, args.domain)
        if args.output:
            pathlib.Path(args.output).write_text(md)
            print(f"\n  ✓ Markdown report saved to: {args.output}")
        else:
            print(md)
    elif args.format == "csv":
        csv_str = report_csv(all_runs)
        if args.output:
            pathlib.Path(args.output).write_text(csv_str)
            print(f"\n  ✓ CSV report saved to: {args.output}")
        else:
            print(csv_str)


if __name__ == "__main__":
    main()