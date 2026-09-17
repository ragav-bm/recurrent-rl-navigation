#!/usr/bin/env python3
import rclpy
import argparse
import time
import os
import sys
import pathlib
import torch
import numpy as np
import threading
from queue import Queue
from torch.utils.tensorboard import SummaryWriter
import platform

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


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sac_agent.sac_v2_mlp import SAC_Trainer_MLP
from sac_agent.common.buffers import ReplayBufferSimple
from envs.pgrc_env_map_goal import PGRCEnv
from config.config import *

# Shared state
data_queue = Queue(maxsize=1000)
train_event = threading.Event()
train_event.set()
train_metrics = {"total_updates": 0}


def print_training_header(args, device, state_dim, action_dim, run_model_dir):
    """Print a machine-parseable header for the results collector."""
    print(f"\n{'═' * 60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'═' * 60}")
    print(f"  Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Algorithm        : SAC")
    print(f"  Architecture     : MLP")
    print(f"  Buffer           : Simple (Uniform)")
    print(f"  Seed             : {args.seed}")
    print(f"  Device           : {device}")
    print(f"  GPU              : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  State dim        : {state_dim}")
    print(f"  Action dim       : {action_dim}")
    print(f"  Hidden dim       : {HIDDEN_DIM}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Gamma            : {GAMMA}")
    print(f"  Tau              : {TAU}")
    print(f"  Alpha (init)     : {ALPHA}")
    print(f"  Reward Scale     : {Reward_Scale}")
    print(f"  LR (Q)           : {Q_LEARNING_RATE}")
    print(f"  LR (Policy)      : {POLICY_LEARNING_RATE}")
    print(f"  LR (Alpha)       : {Learning_Rate}")
    print(f"  Grad Clip        : {Grdient_clip_max_norm}")
    print(f"  Max Episodes     : {MAX_EPISODES}")
    print(f"  Max Steps/Ep     : {MAX_STEPS_PER_EP}")
    print(f"  Buffer Size      : {BUFFER_SIZE}")
    print(f"  Eval Interval    : {EVAL_INTERVAL}")
    print(f"  Checkpoint Intv  : {CHECKPOINT_INTERVAL}")
    print(f"  Model Dir        : {run_model_dir}")
    print(f"  Python           : {platform.python_version()}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA             : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"{'═' * 60}\n")


def learner_worker(sac_trainer, total_steps_ref, writer):
    """Background training thread."""
    print("[THREAD] Background Learner Started.")
    training_started = False

    while rclpy.ok() and train_event.is_set():
        while not data_queue.empty():
            transition = data_queue.get()
            sac_trainer.replay_buffer.push(*transition)

        current_step = total_steps_ref[0]

        if len(sac_trainer.replay_buffer) >= BATCH_SIZE:
            if not training_started:
                train_metrics["total_updates"] = current_step
                training_started = True
                print(f"[THREAD] Buffer ready. Starting training at step {current_step}.")

            if train_metrics["total_updates"] < current_step:
                q1, q2, pi, alpha_val = sac_trainer.update(BATCH_SIZE)
                train_metrics["total_updates"] += 1

                if train_metrics["total_updates"] % TENSORBOARD_LOG_INTERVAL == 0:
                    writer.add_scalar('Loss/Critic_Q1', q1, train_metrics["total_updates"])
                    writer.add_scalar('Loss/Critic_Q2', q2, train_metrics["total_updates"])
                    writer.add_scalar('Loss/Actor', pi, train_metrics["total_updates"])
                    writer.add_scalar('Alpha', alpha_val, train_metrics["total_updates"])

                if train_metrics["total_updates"] % CONSOLE_LOG_INTERVAL == 0:
                    print(f"[LEARNER] Update {train_metrics['total_updates']} | "
                          f"Q1: {q1:.4f} | Pi: {pi:.4f} | α: {alpha_val:.4f}")
            else:
                time.sleep(TRAIN_SLEEP_TIME)
        else:
            time.sleep(BUFFER_WAIT_TIME)


def main():
    parser = argparse.ArgumentParser(description="SAC-MLP Training Suite")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if not args.train and not args.test:
        args.train = True

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[SEED] Set to: {args.seed}")

    rclpy.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    training_start_time = time.time()

    # Output Directory
    benchmark_run_id = os.environ.get("BENCHMARK_RUN_ID", None)
    if benchmark_run_id:
        run_model_dir = os.path.join(MODEL_DIR, benchmark_run_id)
    elif not args.test:
        run_model_dir = os.path.join(MODEL_DIR, f"mlp_{timestamp}")
    else:
        run_model_dir = MODEL_DIR

    os.makedirs(run_model_dir, exist_ok=True)

    # Dual Logger
    log_filename = f"train_log_{timestamp}.txt" if not args.test else f"test_log_{timestamp}.txt"
    log_filepath = os.path.join(run_model_dir, log_filename)
    sys.stdout = DualLogger(log_filepath)
    sys.stderr = sys.stdout
    print(f"Terminal logs saved to: {log_filepath}")

    # Environment
    env = PGRCEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    low = np.array([LINEAR_MIN, ANGULAR_MIN])
    high = np.array([LINEAR_MAX, ANGULAR_MAX])
    action_scale = torch.FloatTensor((high - low) / 2.0).to(device)
    action_bias = torch.FloatTensor((high + low) / 2.0).to(device)

    replay_buffer = ReplayBufferSimple(BUFFER_SIZE, state_dim, action_dim, device)

    sac_trainer = SAC_Trainer_MLP(
        replay_buffer, state_dim, action_dim, HIDDEN_DIM, action_scale,
        ALPHA, GAMMA, TAU, Learning_Rate, Reward_Scale,
        Q_LEARNING_RATE, POLICY_LEARNING_RATE, Grdient_clip_max_norm,
        device, action_bias=action_bias
    )

    total_steps = [0]
    best_eval_reward = -float('inf')
    best_eval_success_rate = -1.0
    best_eval_model_path = None

    # Print header
    print_training_header(args, device, state_dim, action_dim, run_model_dir)

    if args.test:
        path = args.model_path if args.model_path else os.path.join(run_model_dir, "best_model")
        sac_trainer.load_model(path)
        is_training = False
    else:
        tb_tag = benchmark_run_id if benchmark_run_id else f"mlp_{timestamp}"
        writer = SummaryWriter(f"{LOG_BASE_DIR}{tb_tag}")
        is_training = True
        threading.Thread(target=learner_worker, args=(sac_trainer, total_steps, writer), daemon=True).start()

    interrupted = False
    eps = 0

    try:
        if not is_training:
            env.is_evaluating = True

        for eps in range(1, MAX_EPISODES + 1):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(MAX_STEPS_PER_EP):
                total_steps[0] += 1
                action = sac_trainer.get_action(state, deterministic=(not is_training))
                next_state, reward, done, truncated, _ = env.step(action, max_steps=MAX_STEPS_PER_EP)
                episode_reward += reward

                if is_training:
                    data_queue.put((state, action, reward, next_state, float(done)))

                state = next_state
                if done or truncated:
                    break

            result_tag = env.last_termination_reason.upper()
            s_idx = env.current_start_idx
            g_idx = env.current_goal_idx
            g_type = env.current_goal_type

            if is_training:
                writer.add_scalar('Train/Reward', episode_reward, eps)
                writer.add_scalar('Train/Steps', step + 1, eps)
                train_success = 1.0 if result_tag == "GOAL_REACHED" else 0.0
                writer.add_scalar('Train/Success', train_success, eps)

                print(f" TRAINING Ep {eps:4d} | Steps: {step+1:3d} | Total: {total_steps[0]:7d} | "
                      f"Reward: {episode_reward:8.2f} | Path: {s_idx}->{g_idx} ({g_type}) | Result: {result_tag}")

                # Evaluation
                if eps % EVAL_INTERVAL == 0 and len(replay_buffer) >= BATCH_SIZE:
                    eval_rewards = []
                    eval_successes = 0
                    train_event.clear()
                    env.is_evaluating = True
                    env.last_termination_reason = "EVAL_START"

                    eval_goals = env.active_goals * 2
                    current_eval_episodes = len(eval_goals)
                    print(f"\n--- Evaluation at Episode {eps} ({current_eval_episodes} eps) ---")

                    for eval_ep_num, goal_coords in enumerate(eval_goals):
                        e_state, _ = env.reset(options={'eval_goal_coords': goal_coords})
                        e_reward = 0
                        for e_step in range(MAX_STEPS_PER_EP):
                            e_action = sac_trainer.get_action(e_state, deterministic=True)
                            e_state, e_r, e_done, e_trunc, _ = env.step(e_action, max_steps=MAX_STEPS_PER_EP)
                            e_reward += e_r
                            if e_done or e_trunc:
                                if env.last_termination_reason == "goal_reached":
                                    eval_successes += 1
                                break
                        eval_rewards.append(e_reward)

                    env.is_evaluating = False
                    env.last_termination_reason = "TRAIN_RESUME"
                    train_event.set()

                    avg_reward = np.mean(eval_rewards)
                    success_rate = (eval_successes / current_eval_episodes) * 100
                    print(f"Summary | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}%\n")

                    writer.add_scalar('Eval/AvgReward', avg_reward, eps)
                    writer.add_scalar('Eval/SuccessRate', success_rate, eps)

                    if success_rate > best_eval_success_rate or \
                       (success_rate == best_eval_success_rate and avg_reward > best_eval_reward):
                        best_eval_success_rate = success_rate
                        best_eval_reward = avg_reward

                        if best_eval_model_path:
                            for suffix in ['_q1.pth', '_q2.pth', '_policy.pth', '_target_q1.pth', '_target_q2.pth']:
                                f = best_eval_model_path + suffix
                                if os.path.exists(f):
                                    os.remove(f)

                        best_eval_model_path = os.path.join(run_model_dir, f"best_eval_model_ep_{eps}")
                        sac_trainer.save_model(best_eval_model_path)
                        print(f"** New best evaluation model saved: best_eval_model_ep_{eps} (Success: {success_rate:.1f}%, Reward: {avg_reward:.2f}) **")

                if eps % CHECKPOINT_INTERVAL == 0:
                    sac_trainer.save_model(os.path.join(run_model_dir, f"model_ep_{eps}"))
            else:
                print(f"[TEST] Ep {eps:4d} | Steps: {step+1:3d} | Reward: {episode_reward:8.2f} | Result: {result_tag}")

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving...")
    finally:
        final_path = os.path.join(run_model_dir, "final_trained_model")
        sac_trainer.save_model(final_path)

        if is_training:
            writer.close()
        env.close()

        training_duration = time.time() - training_start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        seconds = int(training_duration % 60)

        print(f"\n{'═' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'═' * 60}")
        print(f"  Status           : {'INTERRUPTED' if interrupted else 'COMPLETED'}")
        print(f"  Total Episodes   : {eps}")
        print(f"  Total Steps      : {total_steps[0]}")
        print(f"  Total Updates    : {train_metrics['total_updates']}")
        print(f"  Wall-Clock Time  : {hours}h {minutes}m {seconds}s")
        print(f"  Best Eval SR     : {best_eval_success_rate:.1f}%")
        print(f"  Best Eval Reward : {best_eval_reward:.2f}")
        print(f"  Final Model      : {final_path}")
        print(f"{'═' * 60}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()