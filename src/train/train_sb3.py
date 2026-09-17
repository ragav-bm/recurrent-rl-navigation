#!/usr/bin/env python3
"""
Stable Baselines3 Training Script — Handles PPO, RecurrentPPO, SAC, TD3.

Fully compatible with:
  - benchmark.results.collector (auto-parsing header/footer/logs)
  - benchmark.results.generate_report (tables)
  - benchmark.results.paper_results (LaTeX, plots, stats)
  - benchmark runner (BENCHMARK_RUN_ID env var)

Usage:
    python train_sb3.py --algorithm ppo --seed 42
    python train_sb3.py --algorithm recurrent_ppo --seed 42
    python train_sb3.py --algorithm sac --seed 42
    python train_sb3.py --algorithm td3 --seed 42
    python train_sb3.py --algorithm ppo --test --model_path path/to/model
"""
import rclpy
import argparse
import time
import os
import sys
import pathlib
import torch
import numpy as np
import platform
import json

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import *


class DualLogger(object):
    """Logs output to both the terminal and a file simultaneously."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def print_training_header(args, device, state_dim, action_dim, run_model_dir,
                          algo_config, total_timesteps, tensorboard_log):
    """Print a machine-parseable header for the results collector."""
    algo_name = args.algorithm
    hyperparams = algo_config["hyperparams"]
    policy_type = algo_config["policy"]

    # Determine buffer type for collector
    if algo_name in ("ppo", "recurrent_ppo"):
        buffer_label = "default (rollout)"
    elif algo_name in ("sac", "td3"):
        buffer_label = "replay"
    else:
        buffer_label = "unknown"

    print(f"\n{'═' * 60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'═' * 60}")
    print(f"  Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Algorithm        : {algo_name.upper()}")
    print(f"  Architecture     : SB3")
    print(f"  Policy           : {policy_type}")
    print(f"  Buffer           : {buffer_label}")
    print(f"  Seed             : {args.seed}")
    print(f"  Device           : {device}")
    print(f"  GPU              : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  State dim        : {state_dim}")
    print(f"  Action dim       : {action_dim}")
    print(f"  Hidden dim       : {HIDDEN_DIM}")
    print(f"  Gamma            : {hyperparams.get('gamma', GAMMA)}")
    print(f"  Tau              : {hyperparams.get('tau', TAU)}")
    print(f"  LR               : {hyperparams.get('learning_rate', 3e-4)}")
    print(f"  Batch size       : {hyperparams.get('batch_size', 64)}")
    print(f"  Max Episodes     : {MAX_EPISODES}")
    print(f"  Max Steps/Ep     : {MAX_STEPS_PER_EP}")
    print(f"  Total Timesteps  : {total_timesteps}")
    print(f"  Buffer Size      : {hyperparams.get('buffer_size', BUFFER_SIZE)}")
    print(f"  Eval Interval    : {EVAL_INTERVAL}")
    print(f"  Checkpoint Intv  : {CHECKPOINT_INTERVAL}")
    print(f"  Model Dir        : {run_model_dir}")
    print(f"  TensorBoard      : {tensorboard_log}")
    # Algorithm-specific hyperparameters
    if algo_name in ("ppo", "recurrent_ppo"):
        print(f"  N Steps          : {hyperparams.get('n_steps', 2048)}")
        print(f"  N Epochs         : {hyperparams.get('n_epochs', 10)}")
        print(f"  GAE Lambda       : {hyperparams.get('gae_lambda', 0.95)}")
        print(f"  Clip Range       : {hyperparams.get('clip_range', 0.2)}")
        print(f"  Ent Coef         : {hyperparams.get('ent_coef', 0.01)}")
        print(f"  Max Grad Norm    : {hyperparams.get('max_grad_norm', 0.5)}")
    elif algo_name in ("sac", "td3"):
        print(f"  Train Freq       : {hyperparams.get('train_freq', 1)}")
        print(f"  Gradient Steps   : {hyperparams.get('gradient_steps', 1)}")
        if algo_name == "sac":
            print(f"  Ent Coef         : {hyperparams.get('ent_coef', 'auto')}")
        if algo_name == "td3":
            print(f"  Policy Delay     : {hyperparams.get('policy_delay', 2)}")
    print(f"  Python           : {platform.python_version()}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA             : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════
# ALGORITHM REGISTRY
# ═══════════════════════════════════════════════════════════════
ALGORITHM_REGISTRY = {
    "ppo": {
        "class_path": "stable_baselines3.PPO",
        "policy": "MlpPolicy",
        "hyperparams": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": GAMMA,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
        },
    },
    "recurrent_ppo": {
        "class_path": "sb3_contrib.RecurrentPPO",
        "policy": "MlpLstmPolicy",
        "hyperparams": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": GAMMA,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
        },
    },
    "sac": {
        "class_path": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "hyperparams": {
            "learning_rate": 3e-4,
            "buffer_size": BUFFER_SIZE,
            "batch_size": 256,
            "gamma": GAMMA,
            "tau": TAU,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },
    "td3": {
        "class_path": "stable_baselines3.TD3",
        "policy": "MlpPolicy",
        "hyperparams": {
            "learning_rate": 3e-4,
            "buffer_size": BUFFER_SIZE,
            "batch_size": 256,
            "gamma": GAMMA,
            "tau": TAU,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
        },
    },
}


def import_algorithm(class_path: str):
    """Dynamically import an SB3 algorithm class."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ═══════════════════════════════════════════════════════════════
# SB3 ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════
import gymnasium as gym


class SB3EnvWrapper(gym.Env):
    """Wraps PGRCEnv for SB3 compatibility."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(
            low=np.array([LINEAR_MIN, ANGULAR_MIN], dtype=np.float32),
            high=np.array([LINEAR_MAX, ANGULAR_MAX], dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0

    @property
    def active_goals(self):
        return self.env.active_goals

    @property
    def is_evaluating(self):
        return self.env.is_evaluating

    @is_evaluating.setter
    def is_evaluating(self, val):
        self.env.is_evaluating = val

    @property
    def last_termination_reason(self):
        return self.env.last_termination_reason

    @last_termination_reason.setter
    def last_termination_reason(self, val):
        self.env.last_termination_reason = val

    @property
    def current_start_idx(self):
        return self.env.current_start_idx

    @property
    def current_goal_idx(self):
        return self.env.current_goal_idx

    @property
    def current_goal_type(self):
        return self.env.current_goal_type

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._step_count += 1
        obs, reward, done, truncated, info = self.env.step(action, max_steps=MAX_STEPS_PER_EP)
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """
    Log episode results in the standard format:
      TRAINING Ep XXXX | Steps: XXX | Total: XXXXXXX | Reward: XXXXX.XX | ...

    This format is parsed by collector.py _parse_training_logs().
    """

    def __init__(self, raw_env, writer=None, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.tb_writer = writer
        self.episode_count = 0
        self.ep_reward = 0.0
        self.ep_steps = 0

    def _on_step(self) -> bool:
        self.ep_steps += 1

        # Accumulate reward
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", self.locals.get("reward", [0]))
        if isinstance(rewards, (list, np.ndarray)) and len(rewards) > 0:
            self.ep_reward += rewards[0]
        else:
            self.ep_reward += float(rewards)

        # Check episode end
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        episode_ended = dones[0] if isinstance(dones, (list, np.ndarray)) else bool(dones)

        if episode_ended:
            self.episode_count += 1
            result_tag = self.raw_env.last_termination_reason.upper()
            s_idx = self.raw_env.current_start_idx
            g_idx = self.raw_env.current_goal_idx
            g_type = self.raw_env.current_goal_type

            # Standard format — parsed by collector.py
            print(f" TRAINING Ep {self.episode_count:4d} | Steps: {self.ep_steps:3d} | "
                  f"Total: {self.num_timesteps:7d} | Reward: {self.ep_reward:8.2f} | "
                  f"Path: {s_idx}->{g_idx} ({g_type}) | Result: {result_tag}")

            if self.logger:
                self.logger.record("train/episode_reward", self.ep_reward)
                self.logger.record("train/episode_steps", self.ep_steps)
                self.logger.record("train/success", 1.0 if result_tag == "GOAL_REACHED" else 0.0)

            self.ep_reward = 0.0
            self.ep_steps = 0

        return True


class CustomEvalCallback(BaseCallback):
    """
    Evaluation callback matching existing eval logic.

    Outputs in format parsed by collector.py:
      Summary | Avg Reward: XXX.XX | Success Rate: XX.X%
      ** New best evaluation model saved: ... (Success: XX.X%, Reward: XX.XX) **
    """

    def __init__(self, raw_env, sb3_env, eval_interval_episodes, run_model_dir, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.sb3_env = sb3_env
        self.eval_interval = eval_interval_episodes
        self.run_model_dir = run_model_dir
        self.episode_count = 0
        self.best_success_rate = -1.0
        self.best_reward = -float("inf")
        self.best_model_path = None
        self.best_model_episode = 0
        self._last_eval_ep = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        if (isinstance(dones, (list, np.ndarray)) and dones[0]) or bool(dones):
            self.episode_count += 1

        if self.episode_count > 0 and \
           self.episode_count % self.eval_interval == 0 and \
           self._last_eval_ep != self.episode_count:
            self._last_eval_ep = self.episode_count
            self._run_evaluation()

        return True

    def _run_evaluation(self):
        env = self.raw_env
        eval_goals = env.active_goals * 2
        n_eval = len(eval_goals)
        eval_rewards = []
        eval_successes = 0

        env.is_evaluating = True
        env.last_termination_reason = "EVAL_START"

        is_recurrent = hasattr(self.model.policy, "lstm_states")

        print(f"\n--- Starting Evaluation for Episode {self.episode_count} ({n_eval} episodes) ---")

        for i, goal_coords in enumerate(eval_goals):
            obs, _ = self.sb3_env.reset(options={"eval_goal_coords": goal_coords})
            ep_reward = 0.0

            if is_recurrent:
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)

            for step in range(MAX_STEPS_PER_EP):
                if is_recurrent:
                    action, lstm_states = self.model.predict(
                        obs, state=lstm_states, episode_start=episode_start, deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, done, truncated, _ = self.sb3_env.step(action)
                ep_reward += reward

                if done or truncated:
                    if env.last_termination_reason.upper() == "GOAL_REACHED":
                        eval_successes += 1
                    break

            eval_rewards.append(ep_reward)

        env.is_evaluating = False
        env.last_termination_reason = "TRAIN_RESUME"

        avg_reward = np.mean(eval_rewards)
        success_rate = (eval_successes / n_eval) * 100

        # Standard format — parsed by collector.py _parse_eval_logs()
        print(f"Summary | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}%")

        if self.logger:
            self.logger.record("eval/avg_reward", avg_reward)
            self.logger.record("eval/success_rate", success_rate)

        if success_rate > self.best_success_rate or \
           (success_rate == self.best_success_rate and avg_reward > self.best_reward):
            self.best_success_rate = success_rate
            self.best_reward = avg_reward
            self.best_model_episode = self.episode_count

            # Remove previous best model
            if self.best_model_path and os.path.exists(self.best_model_path + ".zip"):
                try:
                    os.remove(self.best_model_path + ".zip")
                except OSError as e:
                    print(f"  [Warning] Could not remove old best model: {e}")

            self.best_model_path = os.path.join(
                self.run_model_dir, f"best_eval_model_ep_{self.episode_count}"
            )
            self.model.save(self.best_model_path)
            # Standard format — parsed by collector.py _parse_eval_logs() best model
            print(f"** New best evaluation model saved: best_eval_model_ep_{self.episode_count} "
                  f"(Success: {success_rate:.1f}%, Reward: {avg_reward:.2f}) **")

        print()


class CheckpointCallback(BaseCallback):
    """Periodic checkpoint saving."""

    def __init__(self, save_freq_episodes, run_model_dir, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq_episodes
        self.run_model_dir = run_model_dir
        self.episode_count = 0
        self._last_ckpt = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        if (isinstance(dones, (list, np.ndarray)) and dones[0]) or bool(dones):
            self.episode_count += 1

        if self.episode_count > 0 and \
           self.episode_count % self.save_freq == 0 and \
           self._last_ckpt != self.episode_count:
            self._last_ckpt = self.episode_count
            path = os.path.join(self.run_model_dir, f"model_ep_{self.episode_count}")
            self.model.save(path)
            print(f"  [Checkpoint] Saved: {path}")

        return True


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    training_start_time = time.time()  # ← FIXED: moved inside main()

    parser = argparse.ArgumentParser(description="SB3 Training Suite")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=list(ALGORITHM_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    # --- Seed ---
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[SEED] Set to: {args.seed}")

    # --- ROS2 ---
    rclpy.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- Output Directory ---
    algo_name = args.algorithm
    benchmark_run_id = os.environ.get("BENCHMARK_RUN_ID", None)
    if benchmark_run_id:
        run_model_dir = os.path.join(MODEL_DIR, benchmark_run_id)
    elif not args.test:
        run_model_dir = os.path.join(MODEL_DIR, f"sb3_{algo_name}_{timestamp}")
    else:
        run_model_dir = MODEL_DIR

    os.makedirs(run_model_dir, exist_ok=True)

    # --- Dual Logger ---
    log_filename = f"train_log_{timestamp}.txt" if not args.test else f"test_log_{timestamp}.txt"
    log_filepath = os.path.join(run_model_dir, log_filename)
    sys.stdout = DualLogger(log_filepath)
    sys.stderr = sys.stdout
    print(f"Terminal logs saved to: {log_filepath}")
    print(f"Device: {device}")
    print(f"Algorithm: {algo_name}")

    # --- Environment ---
    from envs.pgrc_env_map_goal import PGRCEnv
    raw_env = PGRCEnv()
    env = SB3EnvWrapper(raw_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Algorithm Config ---
    algo_config = ALGORITHM_REGISTRY[algo_name]
    AlgorithmClass = import_algorithm(algo_config["class_path"])
    policy_type = algo_config["policy"]
    hyperparams = algo_config["hyperparams"].copy()

    total_timesteps = args.total_timesteps or (MAX_EPISODES * MAX_STEPS_PER_EP)

    # --- TensorBoard ---
    tb_tag = benchmark_run_id if benchmark_run_id else f"sb3_{algo_name}_{timestamp}"
    tensorboard_log = f"{LOG_BASE_DIR}{tb_tag}"

    # ═══════════════════════════════════════════════════════════
    # STRUCTURED HEADER — parsed by collector.py _parse_header()
    # ═══════════════════════════════════════════════════════════
    print_training_header(args, device, state_dim, action_dim, run_model_dir,
                          algo_config, total_timesteps, tensorboard_log)

    # --- Tracking variables ---
    interrupted = False
    total_episodes = 0
    total_steps_count = 0
    total_updates = 0
    best_eval_success_rate = -1.0
    best_eval_reward = -float('inf')
    best_eval_episode = 0
    final_model_path = ""

    # ═══════════════════════════════════════════════════════════
    # TEST MODE
    # ═══════════════════════════════════════════════════════════
    if args.test:
        model_path = args.model_path or os.path.join(run_model_dir, "best_model")
        model = AlgorithmClass.load(model_path, env=env)
        env.is_evaluating = True
        is_recurrent = hasattr(model.policy, "lstm_states")

        print(f"\n--- RUNNING IN TEST MODE ---")
        print(f"  Model: {model_path}")
        print(f"  Episodes: {TEST_EPISODES}\n")

        test_rewards = []
        test_successes = 0

        for ep in range(1, TEST_EPISODES + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            if is_recurrent:
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)

            for step in range(TESTING_MAX_STEPS_PER_EP):
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs, state=lstm_states, episode_start=episode_start, deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                if done or truncated:
                    break

            total_episodes = ep
            result_tag = raw_env.last_termination_reason.upper()
            test_rewards.append(ep_reward)
            if result_tag == "GOAL_REACHED":
                test_successes += 1

            print(f"[TEST] Ep {ep:4d} | Steps: {step+1:3d} | Reward: {ep_reward:8.2f} | Result: {result_tag}")

        total_steps_count = 0  # Not tracked in test mode
        best_eval_reward = np.mean(test_rewards)
        best_eval_success_rate = (test_successes / TEST_EPISODES) * 100

        print(f"\n  Test Summary: Avg Reward: {best_eval_reward:.2f} | "
              f"Success Rate: {best_eval_success_rate:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # TRAIN MODE
    # ═══════════════════════════════════════════════════════════
    else:
        model = AlgorithmClass(
            policy_type,
            env,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            verbose=0,
            device=device,
            **hyperparams,
        )

        # Create callbacks
        eval_cb = CustomEvalCallback(
            raw_env=raw_env,
            sb3_env=env,
            eval_interval_episodes=EVAL_INTERVAL,
            run_model_dir=run_model_dir,
        )
        episode_log_cb = EpisodeLogCallback(raw_env)
        checkpoint_cb = CheckpointCallback(
            save_freq_episodes=CHECKPOINT_INTERVAL,
            run_model_dir=run_model_dir,
        )

        callbacks = [episode_log_cb, eval_cb, checkpoint_cb]

        print(f"Starting training for {total_timesteps} timesteps...\n")

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Saving final model...")
            interrupted = True

        # Save final model
        final_model_path = os.path.join(run_model_dir, "final_model")
        model.save(final_model_path)
        print(f"Final model saved: {final_model_path}")

        # Collect metrics from callbacks
        total_episodes = episode_log_cb.episode_count
        total_steps_count = model.num_timesteps
        best_eval_success_rate = eval_cb.best_success_rate
        best_eval_reward = eval_cb.best_reward
        best_eval_episode = eval_cb.best_model_episode

        # Compute total updates (algorithm-specific)
        try:
            if algo_name in ("sac", "td3"):
                # Off-policy: gradient_steps per env step (roughly)
                gradient_steps = hyperparams.get("gradient_steps", 1)
                total_updates = total_steps_count * gradient_steps
            elif algo_name in ("ppo", "recurrent_ppo"):
                # On-policy: n_updates = timesteps / n_steps * n_epochs
                n_steps = hyperparams.get("n_steps", 2048)
                n_epochs = hyperparams.get("n_epochs", 10)
                total_updates = (total_steps_count // n_steps) * n_epochs
            else:
                total_updates = total_steps_count
        except Exception:
            total_updates = total_steps_count

    # ═══════════════════════════════════════════════════════════
    # CLEANUP & STRUCTURED FOOTER
    # ═══════════════════════════════════════════════════════════
    env.close()

    training_duration = time.time() - training_start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)

    # Determine final status
    if args.test:
        train_status = "COMPLETED"
    elif interrupted:
        train_status = "INTERRUPTED"
    else:
        train_status = "COMPLETED"

    # ═══════════════════════════════════════════════════════════
    # STRUCTURED FOOTER — parsed by collector.py _parse_footer()
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Status           : {train_status}")
    print(f"  Algorithm        : {algo_name.upper()}")
    print(f"  Architecture     : SB3")
    print(f"  Buffer           : {'default' if algo_name in ('ppo', 'recurrent_ppo') else 'replay'}")
    print(f"  Seed             : {args.seed}")
    print(f"  Total Episodes   : {total_episodes}")
    print(f"  Total Steps      : {total_steps_count}")
    print(f"  Total Updates    : {total_updates}")
    print(f"  Wall-Clock Time  : {hours}h {minutes}m {seconds}s")
    print(f"  Best Eval SR     : {best_eval_success_rate:.1f}%")
    print(f"  Best Eval Reward : {best_eval_reward:.2f}")
    print(f"  Best Eval Episode: {best_eval_episode}")
    if final_model_path:
        print(f"  Final Model      : {final_model_path}")
    print(f"  Model Dir        : {run_model_dir}")
    print(f"{'═' * 60}")

    # ═══════════════════════════════════════════════════════════
    # METRICS JSON — machine-readable backup for paper_results.py
    # ═══════════════════════════════════════════════════════════
    metrics_json = {
        "status": train_status.lower(),
        "algorithm": algo_name.upper(),
        "architecture": "SB3",
        "policy": algo_config["policy"],
        "buffer": "default" if algo_name in ("ppo", "recurrent_ppo") else "replay",
        "seed": args.seed,
        "env_name": "gazebo_nav",
        "total_episodes": total_episodes,
        "total_steps": total_steps_count,
        "total_updates": total_updates,
        "training_duration_sec": training_duration,
        "wall_clock_time": f"{hours}h {minutes}m {seconds}s",
        "best_eval_reward": best_eval_reward,
        "best_eval_success_rate": best_eval_success_rate,
        "best_eval_episode": best_eval_episode,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp": timestamp,
        "run_tag": tb_tag,
        "benchmark_run_id": benchmark_run_id or "",
        "hyperparameters": {
            "algorithm": algo_name,
            "policy": algo_config["policy"],
            "hidden_dim": HIDDEN_DIM,
            "gamma": hyperparams.get("gamma", GAMMA),
            "tau": hyperparams.get("tau", TAU),
            "learning_rate": hyperparams.get("learning_rate", 3e-4),
            "batch_size": hyperparams.get("batch_size", 64),
            "buffer_size": hyperparams.get("buffer_size", BUFFER_SIZE),
            "max_episodes": MAX_EPISODES,
            "max_steps_per_ep": MAX_STEPS_PER_EP,
            "total_timesteps": total_timesteps,
            **{k: v for k, v in hyperparams.items()
               if k not in ("gamma", "tau", "learning_rate", "batch_size", "buffer_size")},
        },
    }

    metrics_path = os.path.join(run_model_dir, "metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2, default=str)
        print(f"\n  Metrics JSON: {metrics_path}")
    except Exception as e:
        print(f"\n  [WARN] Could not save metrics.json: {e}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()