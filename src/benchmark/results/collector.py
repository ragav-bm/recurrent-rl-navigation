#!/usr/bin/env python3
"""
Results Collector — Parses structured training logs from ALL train scripts.

Supports:
  - train_lstm.py   (Gazebo SAC-LSTM)
  - train_mlp.py    (Gazebo SAC-MLP)
  - train_sb3.py    (Gazebo SB3: PPO, RecurrentPPO, SAC, TD3)
  - benchmark.pomujoco.train  (PO-MuJoCo)

Parses:
  - Structured HEADER block  → hyperparameters
  - Training episode logs    → episode rewards, steps, success
  - Evaluation logs          → avg reward, success rate
  - Best model notifications → best performance tracking
  - Structured FOOTER block  → final metrics, wall-clock time, status
  - metrics.json (if exists) → machine-readable backup

Usage:
    from benchmark.results.collector import ResultCollector
    collector = ResultCollector(results_dir, domain="gazebo")
    runs = collector.scan_all()
"""
import os
import re
import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from glob import glob


@dataclass
class RunResult:
    """Complete result from a single training run."""
    # Identity
    run_id: str = ""
    run_dir: str = ""
    log_file: str = ""
    domain: str = ""          # "gazebo" or "pomujoco"

    # Status
    status: str = "unknown"   # "completed", "interrupted", "failed", "running"

    # Experiment config (from header)
    algorithm: str = ""       # "SAC", "PPO", "TD3", etc.
    architecture: str = ""    # "LSTM", "MLP", "SB3"
    buffer_type: str = ""     # "per", "uniform", "simple", "default", "replay"
    seed: int = 0
    env_name: str = ""        # "HalfCheetah-v4" or "" for Gazebo

    # Observability (PO-MuJoCo)
    observability: str = ""   # "PO", "FO", "PO_flicker0.2"
    flicker_prob: float = 0.0

    # Architecture details (from header)
    hidden_dim: int = 0
    batch_size: int = 0
    seq_len: int = 0
    burnin_len: int = 0
    gamma: float = 0.0
    tau: float = 0.0
    alpha_init: float = 0.0
    reward_scale: float = 0.0
    lr_q: float = 0.0
    lr_policy: float = 0.0
    lr_alpha: float = 0.0
    grad_clip: float = 0.0
    buffer_size: int = 0
    max_episodes: int = 0
    max_steps_per_ep: int = 0

    # Training progress (from logs + footer)
    total_episodes: int = 0
    total_steps: int = 0
    total_updates: int = 0
    training_duration_sec: float = 0.0

    # Evaluation metrics (from eval logs + footer)
    best_eval_reward: float = float("-inf")
    best_eval_episode: int = 0
    best_eval_success_rate: float = -1.0
    final_eval_reward: float = float("-inf")
    final_eval_std: float = 0.0

    # Episode history (for learning curves)
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    eval_rewards: List[Tuple[int, float]] = field(default_factory=list)  # (episode, avg_reward)
    eval_success_rates: List[Tuple[int, float]] = field(default_factory=list)  # (episode, sr%)

    # Meta
    timestamp: str = ""
    device: str = ""
    gpu: str = ""
    model_dir: str = ""

    # Experiment classification (for runner)
    experiment: str = ""      # e.g., "sac_lstm_per_po", "lstm_per", "sb3_ppo"

    def is_complete(self) -> bool:
        return self.status == "completed"

    def summary_str(self) -> str:
        sr = f" | SR: {self.best_eval_success_rate:.1f}%" if self.best_eval_success_rate >= 0 else ""
        return (f"{self.run_id}: {self.status} | Best: {self.best_eval_reward:.1f}{sr} | "
                f"Eps: {self.total_episodes} | Steps: {self.total_steps}")


class ResultCollector:
    """
    Scans result directories, parses logs, and collects RunResults.
    """

    def __init__(self, results_dir: str, domain: str = "gazebo"):
        """
        Args:
            results_dir: Path to models directory (contains run subdirectories)
            domain: "gazebo" or "pomujoco"
        """
        self.results_dir = results_dir
        self.domain = domain

    def scan_all(self) -> List[RunResult]:
        """Scan all run directories and parse logs."""
        runs = []
        results_path = pathlib.Path(self.results_dir)

        if not results_path.exists():
            print(f"[WARN] Results directory not found: {self.results_dir}")
            return runs

        # Each subdirectory = one run
        for run_dir in sorted(results_path.iterdir()):
            if not run_dir.is_dir():
                continue

            # Find log file
            log_files = sorted(run_dir.glob("train_log_*.txt"))
            if not log_files:
                continue

            # Use the most recent log file
            log_file = log_files[-1]

            result = self._parse_log(str(log_file), str(run_dir))
            if result:
                runs.append(result)

        return runs

    def _parse_log(self, log_file: str, run_dir: str) -> Optional[RunResult]:
        """Parse a single training log file."""
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"[WARN] Cannot read {log_file}: {e}")
            return None

        result = RunResult(
            run_dir=run_dir,
            log_file=log_file,
            domain=self.domain,
            run_id=os.path.basename(run_dir),
        )

        # Check for metrics.json first (machine-readable backup)
        metrics_json = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_json):
            self._parse_metrics_json(metrics_json, result)

        # Parse structured blocks
        self._parse_header(content, result)
        self._parse_training_logs(content, result)
        self._parse_eval_logs(content, result)
        self._parse_footer(content, result)

        # Classify experiment
        self._classify_experiment(result)

        return result

    # ─── METRICS JSON ──────────────────────────────────────────

    def _parse_metrics_json(self, json_path: str, result: RunResult):
        """Parse metrics.json if available (highest priority)."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            result.status = data.get("status", "unknown").lower()
            result.env_name = data.get("env", "")
            result.architecture = data.get("arch", "").upper()
            result.buffer_type = data.get("buffer", "")
            result.seed = data.get("seed", 0)
            result.observability = data.get("observability", "")
            result.flicker_prob = data.get("flicker", 0.0)
            result.total_episodes = data.get("total_episodes", 0)
            result.total_steps = data.get("total_steps", 0)
            result.total_updates = data.get("total_updates", 0)
            result.training_duration_sec = data.get("training_duration_sec", 0.0)
            result.best_eval_reward = data.get("best_eval_reward", float("-inf"))
            result.best_eval_episode = data.get("best_eval_episode", 0)
            result.final_eval_reward = data.get("final_eval_reward", float("-inf"))
            result.final_eval_std = data.get("final_eval_std", 0.0)
            result.device = data.get("device", "")
            result.gpu = data.get("gpu", "")
            result.timestamp = data.get("timestamp", "")

            hp = data.get("hyperparameters", {})
            if hp:
                result.hidden_dim = hp.get("hidden_dim", 0)
                result.batch_size = hp.get("batch_size", 0)
                result.seq_len = hp.get("seq_len", 0)
                result.burnin_len = hp.get("burnin_len", 0)
                result.gamma = hp.get("gamma", 0.0)
                result.tau = hp.get("tau", 0.0)
                result.alpha_init = hp.get("alpha_init", 0.0)
                result.reward_scale = hp.get("reward_scale", 0.0)
                result.lr_q = hp.get("lr_q", 0.0)
                result.lr_policy = hp.get("lr_policy", 0.0)
                result.lr_alpha = hp.get("lr_alpha", 0.0)
                result.grad_clip = hp.get("grad_clip", 0.0)
                result.buffer_size = hp.get("buffer_size", 0)
                result.max_episodes = hp.get("max_episodes", 0)
                result.max_steps_per_ep = hp.get("max_steps_per_ep", 0)

        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Cannot parse {json_path}: {e}")

    # ─── HEADER PARSING ────────────────────────────────────────

    def _parse_header(self, content: str, result: RunResult):
        """Parse the TRAINING CONFIGURATION header block."""
        # Match the header block
        header_match = re.search(
            r"TRAINING CONFIGURATION.*?═{40,}(.*?)═{40,}",
            content, re.DOTALL
        )
        if not header_match:
            return

        block = header_match.group(1)

        # Parse key-value pairs
        patterns = {
            "algorithm": r"Algorithm\s*:\s*(\S+)",
            "architecture": r"Architecture\s*:\s*(\S+)",
            "buffer_type": r"Buffer\s*:\s*(.+)",
            "seed": r"Seed\s*:\s*(\d+)",
            "device": r"Device\s*:\s*(.+)",
            "gpu": r"GPU\s*:\s*(.+)",
            "env_name": r"Environment\s*:\s*(\S+)",
            "observability": r"Observability\s*:\s*(\S+)",
            "hidden_dim": r"Hidden dim\s*:\s*(\d+)",
            "batch_size": r"Batch size\s*:\s*(\d+)",
            "seq_len": r"Seq len\s*:\s*(\d+)",
            "burnin_len": r"Burn-in len\s*:\s*(\d+)",
            "gamma": r"Gamma\s*:\s*([\d.]+)",
            "tau": r"Tau\s*:\s*([\d.]+)",
            "alpha_init": r"Alpha \(init\)\s*:\s*([\d.]+)",
            "reward_scale": r"Reward Scale\s*:\s*([\d.]+)",
            "lr_q": r"LR \(Q\)\s*:\s*([\d.e\-]+)",
            "lr_policy": r"LR \(Policy\)\s*:\s*([\d.e\-]+)",
            "lr_alpha": r"LR \(Alpha\)\s*:\s*([\d.e\-]+)",
            "grad_clip": r"Grad Clip\s*:\s*([\d.]+)",
            "max_episodes": r"Max Episodes\s*:\s*(\d+)",
            "max_steps_per_ep": r"Max Steps/Ep\s*:\s*(\d+)",
            "buffer_size": r"Buffer Size\s*:\s*(\d+)",
            "timestamp": r"Timestamp\s*:\s*(.+)",
            "flicker_prob": r"Flicker Prob\s*:\s*([\d.]+)",
        }

        for attr, pattern in patterns.items():
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                current = getattr(result, attr, None)

                # Don't overwrite if already set from metrics.json
                if current and current != 0 and current != "" and current != 0.0:
                    continue

                # Type conversion
                if attr in ("seed", "hidden_dim", "batch_size", "seq_len",
                            "burnin_len", "max_episodes", "max_steps_per_ep",
                            "buffer_size"):
                    try:
                        setattr(result, attr, int(val))
                    except ValueError:
                        pass
                elif attr in ("gamma", "tau", "alpha_init", "reward_scale",
                              "lr_q", "lr_policy", "lr_alpha", "grad_clip",
                              "flicker_prob"):
                    try:
                        setattr(result, attr, float(val))
                    except ValueError:
                        pass
                else:
                    setattr(result, attr, val)

    # ─── TRAINING LOG PARSING ──────────────────────────────────

    def _parse_training_logs(self, content: str, result: RunResult):
        """Parse episode training logs."""
        # Pattern: TRAINING Ep XXXX | Steps: XXX | Total: XXXXXXX | Reward: XXXXX.XX | ...
        pattern = (
            r"(?:TRAINING\s+)?Ep\s+(\d+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*"
            r"Total:\s*(\d+)\s*\|\s*Reward:\s*([-\d.]+)"
        )
        matches = re.findall(pattern, content)

        for ep_str, steps_str, total_str, reward_str in matches:
            ep = int(ep_str)
            steps = int(steps_str)
            reward = float(reward_str)
            result.episode_rewards.append(reward)
            result.episode_steps.append(steps)

        # Update total from last training log entry if footer didn't set it
        if matches and result.total_episodes == 0:
            result.total_episodes = int(matches[-1][0])
        if matches and result.total_steps == 0:
            result.total_steps = int(matches[-1][2])

    # ─── EVALUATION LOG PARSING ────────────────────────────────

    def _parse_eval_logs(self, content: str, result: RunResult):
        """Parse evaluation logs (multiple formats supported)."""

        # Format 1 (Gazebo): "Summary | Avg Reward: XXX.XX | Success Rate: XX.X%"
        pattern1 = (
            r"(?:---.*?Evaluation.*?Episode\s+(\d+).*?---\s*\n)"
            r"(?:.*?\n)*?"
            r"Summary\s*\|\s*Avg Reward:\s*([-\d.]+)\s*\|\s*Success Rate:\s*([\d.]+)%"
        )
        for match in re.finditer(pattern1, content, re.MULTILINE):
            ep = int(match.group(1))
            avg_r = float(match.group(2))
            sr = float(match.group(3))
            result.eval_rewards.append((ep, avg_r))
            result.eval_success_rates.append((ep, sr))

        # If pattern1 didn't match eval episodes, try simpler format
        if not result.eval_rewards:
            # Fallback: find all "Summary | Avg Reward: X | Success Rate: X%"
            pattern1b = r"Summary\s*\|\s*Avg Reward:\s*([-\d.]+)\s*\|\s*Success Rate:\s*([\d.]+)%"
            eval_matches = re.findall(pattern1b, content)
            # We don't know the exact episode, but we can enumerate
            from benchmark.pomujoco.config import EVAL_INTERVAL as EI
            for i, (avg_r, sr) in enumerate(eval_matches):
                ep_est = (i + 1) * 50  # estimate
                result.eval_rewards.append((ep_est, float(avg_r)))
                result.eval_success_rates.append((ep_est, float(sr)))

        # Format 2 (PO-MuJoCo): "  [EVAL] 1234.5 ± 56.7"
        pattern2 = r"\[EVAL\]\s*([-\d.]+)\s*[±]\s*([-\d.]+)"
        eval2_matches = re.findall(pattern2, content)
        if eval2_matches and not result.eval_rewards:
            for i, (avg_r, std_r) in enumerate(eval2_matches):
                ep_est = (i + 1) * 50
                result.eval_rewards.append((ep_est, float(avg_r)))

        # Best model notifications
        # Format: "** New best ... (Success: XX.X%, Reward: XX.XX) **"
        best_pattern = (
            r"\*\*\s*New best.*?"
            r"(?:Success:\s*([\d.]+)%.*?Reward:\s*([-\d.]+)"
            r"|Reward:\s*([-\d.]+))"
        )
        for match in re.finditer(best_pattern, content):
            sr = match.group(1)
            r1 = match.group(2)
            r2 = match.group(3)
            reward = float(r1) if r1 else (float(r2) if r2 else 0.0)
            success = float(sr) if sr else -1.0

            if reward > result.best_eval_reward:
                result.best_eval_reward = reward
            if success > result.best_eval_success_rate:
                result.best_eval_success_rate = success

        # "** New best: 1500.3 **" (simpler PO-MuJoCo format)
        simple_best = r"\*\*\s*New best:\s*([-\d.]+)\s*\*\*"
        for match in re.finditer(simple_best, content):
            reward = float(match.group(1))
            if reward > result.best_eval_reward:
                result.best_eval_reward = reward

        # Update final eval from last eval entry
        if result.eval_rewards:
            result.final_eval_reward = result.eval_rewards[-1][1]

    # ─── FOOTER PARSING ────────────────────────────────────────

    def _parse_footer(self, content: str, result: RunResult):
        """Parse the TRAINING COMPLETE footer block."""
        footer_match = re.search(
            r"TRAINING COMPLETE.*?═{40,}(.*?)═{40,}",
            content, re.DOTALL
        )
        if not footer_match:
            # No footer = likely still running or crashed
            if result.status == "unknown":
                if result.episode_rewards:
                    result.status = "running"
                else:
                    result.status = "failed"
            return

        block = footer_match.group(1)

        # Status
        status_match = re.search(r"Status\s*:\s*(\S+)", block)
        if status_match:
            raw = status_match.group(1).strip().lower()
            if raw == "completed":
                result.status = "completed"
            elif raw == "interrupted":
                result.status = "interrupted"
            else:
                result.status = raw

        # Total Episodes
        ep_match = re.search(r"Total Episodes\s*:\s*(\d+)", block)
        if ep_match:
            result.total_episodes = int(ep_match.group(1))

        # Total Steps
        steps_match = re.search(r"Total Steps\s*:\s*(\d+)", block)
        if steps_match:
            result.total_steps = int(steps_match.group(1))

        # Total Updates
        updates_match = re.search(r"Total Updates\s*:\s*(\d+)", block)
        if updates_match:
            result.total_updates = int(updates_match.group(1))

        # Wall-Clock Time: "Xh Ym Zs"
        time_match = re.search(r"Wall-Clock Time\s*:\s*(\d+)h\s*(\d+)m\s*(\d+)s", block)
        if time_match:
            h, m, s = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3))
            result.training_duration_sec = h * 3600 + m * 60 + s

        # Best Eval SR
        sr_match = re.search(r"Best Eval SR\s*:\s*([\d.]+)%", block)
        if sr_match:
            sr = float(sr_match.group(1))
            if sr > result.best_eval_success_rate:
                result.best_eval_success_rate = sr

        # Best Eval Reward
        reward_match = re.search(r"Best Eval Reward\s*:\s*([-\d.]+)", block)
        if reward_match:
            raw = reward_match.group(1)          
            if raw != '-':                       
                r = float(raw)
                if r > result.best_eval_reward:
                    result.best_eval_reward = r


                    
        # Best Eval Episode (PO-MuJoCo)
        best_ep_match = re.search(r"Best Eval Episode\s*:\s*(\d+)", block)
        if best_ep_match:
            result.best_eval_episode = int(best_ep_match.group(1))

        final_match = re.search(r"Final Eval Reward\s*:\s*([-\d.]+)", block)
        if final_match:
            raw = final_match.group(1)           
            if raw != '-':
                result.final_eval_reward = float(raw)

        # Final Eval Std (PO-MuJoCo)
        std_match = re.search(r"Final Eval Std\s*:\s*([-\d.]+)", block)
        if std_match:
            raw = std_match.group(1)             
            if raw != '-':
                result.final_eval_std = float(raw)

        # Algorithm (SB3)
        algo_match = re.search(r"Algorithm\s*:\s*(\S+)", block)
        if algo_match and not result.algorithm:
            result.algorithm = algo_match.group(1).strip()

    # ─── EXPERIMENT CLASSIFICATION ─────────────────────────────

    def _classify_experiment(self, result: RunResult):
        """Classify the run into an experiment key for grouping."""
        arch = result.architecture.lower()
        buf = result.buffer_type.lower()
        obs = result.observability.upper()
        algo = result.algorithm.lower() if result.algorithm else ""

        # PO-MuJoCo experiments
        if self.domain == "pomujoco":
            if arch == "lstm" and "per" in buf:
                if "FO" in obs:
                    result.experiment = "sac_lstm_per_fo"
                elif "flicker" in obs:
                    result.experiment = "sac_lstm_per_flicker"
                else:
                    result.experiment = "sac_lstm_per_po"
            elif arch == "lstm" and "uniform" in buf:
                result.experiment = "sac_lstm_uniform_po"
            elif arch == "mlp":
                if "FO" in obs:
                    result.experiment = "sac_mlp_fo"
                else:
                    result.experiment = "sac_mlp_po"
            else:
                result.experiment = f"{arch}_{buf}"

        elif self.domain == "popgym":
            if arch == "lstm" and "per" in buf:
                result.experiment = "sac_lstm_per"
            elif arch == "lstm" and "uniform" in buf:
                result.experiment = "sac_lstm_uniform"
            elif arch == "mlp":
                result.experiment = "sac_mlp"
            else:
                result.experiment = f"{arch}_{buf}"

        # Gazebo experiments
        elif self.domain == "gazebo":
            if "sb3" in arch.lower() or algo in ("ppo", "recurrent_ppo", "td3"):
                result.experiment = f"sb3_{algo}" if algo else "sb3_unknown"
            elif arch == "lstm":
                if "per" in buf:
                    result.experiment = "lstm_per"
                else:
                    result.experiment = "lstm_uniform"
            elif arch == "mlp":
                result.experiment = "sac_mlp"
            else:
                result.experiment = f"{arch}_{buf}"

        # Detect experiment from directory name as fallback
        if not result.experiment:
            dir_name = os.path.basename(result.run_dir).lower()
            if "lstm_per" in dir_name:
                result.experiment = "lstm_per"
            elif "lstm_uniform" in dir_name:
                result.experiment = "lstm_uniform"
            elif "mlp" in dir_name:
                result.experiment = "sac_mlp"
            elif "sb3" in dir_name:
                for algo_key in ("ppo", "recurrent_ppo", "sac", "td3"):
                    if algo_key in dir_name:
                        result.experiment = f"sb3_{algo_key}"
                        break
            else:
                result.experiment = "unknown"


def print_scan_summary(runs: List[RunResult]):
    """Print a quick summary of scanned results."""
    if not runs:
        print("  No runs found.")
        return

    completed = [r for r in runs if r.status == "completed"]
    interrupted = [r for r in runs if r.status == "interrupted"]
    failed = [r for r in runs if r.status in ("failed", "running")]

    print(f"\n  Scanned: {len(runs)} runs")
    print(f"  ✓ Completed:   {len(completed)}")
    print(f"  ⚠ Interrupted: {len(interrupted)}")
    print(f"  ✗ Failed/Running: {len(failed)}")

    # Group by experiment
    from collections import Counter
    exp_counts = Counter(r.experiment for r in runs)
    print(f"\n  By experiment:")
    for exp, count in sorted(exp_counts.items()):
        ok = sum(1 for r in runs if r.experiment == exp and r.status == "completed")
        print(f"    {exp:<25} {ok}/{count} completed")