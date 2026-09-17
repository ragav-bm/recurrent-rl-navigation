#!/usr/bin/env python3
from torch import device
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
from collections import deque
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

# Ensure we can import from package root (src) when running from workspace root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Internal imports
from sac_agent.sac_v2_lstm_R import SAC_Trainer 
from sac_agent.common.buffers import ReplayBufferLSTMPER, ReplayBufferLSTM
from envs.pgrc_env_map_goal import PGRCEnv

from config.config import *

data_queue = Queue(maxsize=20)
train_metrics = {"q1_loss": 0.0, "q2_loss": 0.0, "pi_loss": 0.0, "Alpha": 0.0, "total_updates": 0}

train_event = threading.Event()
train_event.set()


def print_training_header(args, device, state_dim, action_dim, run_model_dir):
    """Print a machine-parseable header for the results collector."""
    print(f"\n{'═' * 60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'═' * 60}")
    print(f"  Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Algorithm        : SAC")
    print(f"  Architecture     : LSTM")
    print(f"  Buffer           : {args.buffer_type if hasattr(args, 'buffer_type') else 'per'}")
    print(f"  Seed             : {args.seed}")
    print(f"  Device           : {device}")
    print(f"  GPU              : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  State dim        : {state_dim}")
    print(f"  Action dim       : {action_dim}")
    print(f"  Hidden dim       : {HIDDEN_DIM}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Seq len          : {SEQ_LEN}")
    print(f"  Burn-in len      : {BURNIN_LEN}")
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
    print(f"  Min Steps/Ep     : {MIN_STEPS_PER_EP}")
    print(f"  Buffer Size      : {BUFFER_SIZE}")
    print(f"  PER Alpha        : {per_alpha}")
    print(f"  PER Beta Start   : {BETA_START}")
    print(f"  PER Beta Frames  : {BETA_FRAMES}")
    print(f"  Eval Interval    : {EVAL_INTERVAL}")
    print(f"  Checkpoint Intv  : {CHECKPOINT_INTERVAL}")
    print(f"  Model Dir        : {run_model_dir}")
    print(f"  Python           : {platform.python_version()}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA             : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"{'═' * 60}\n")


def learner_worker(sac_trainer, total_steps_ref, writer, train_event):
    """Background thread worker for training the SAC model."""
    print("[THREAD] Background Learner Started.")
    training_started = False

    while rclpy.ok():
        train_event.wait()
        
        while not data_queue.empty():
            pkg = data_queue.get()
            sac_trainer.replay_buffer.push_all_state(*pkg)
        
        current_global_step = total_steps_ref[0]
        
        if len(sac_trainer.replay_buffer) >= BATCH_SIZE:
            if not training_started:
                train_metrics["total_updates"] = current_global_step
                training_started = True
                print(f"\n[THREAD] Buffer filled with {BATCH_SIZE} episodes!")
                print(f"[THREAD] Syncing updates to Step {current_global_step}. Starting 1:1 training...\n")

            if train_metrics["total_updates"] < current_global_step:
                beta = min(1.0, BETA_START + current_global_step * (1.0 - BETA_START) / BETA_FRAMES)
                
                q1, q2, pi, auto_alpha, q1_lr, q2_lr, policy_lr, alpha_lr = sac_trainer.update(
                    batch_size=BATCH_SIZE, beta=beta, seq_len=SEQ_LEN, burnin_len=BURNIN_LEN
                )

                train_metrics["total_updates"] += 1

                if train_metrics["total_updates"] % TENSORBOARD_LOG_INTERVAL == 0:
                    writer.add_scalar('Loss/Critic_Q1', q1, train_metrics["total_updates"])
                    writer.add_scalar('Loss/Critic_Q2', q2, train_metrics["total_updates"])
                    writer.add_scalar('Loss/Actor_Policy', pi, train_metrics["total_updates"])
                    writer.add_scalar('ALPHA Tuning', auto_alpha, train_metrics["total_updates"])
                    writer.add_scalar('Learning_Rate/Q1_LR', q1_lr, train_metrics["total_updates"])
                    writer.add_scalar('Learning_Rate/Q2_LR', q2_lr, train_metrics["total_updates"])
                    writer.add_scalar('Learning_Rate/Policy_LR', policy_lr, train_metrics["total_updates"])
                    writer.add_scalar('Learning_Rate/Alpha_LR', alpha_lr, train_metrics["total_updates"])
                
                if train_metrics["total_updates"] % CONSOLE_LOG_INTERVAL == 0:
                    print(f"[LEARNER] OK! Update: {train_metrics['total_updates']} | "
                          f"Env Step: {current_global_step} | "
                          f"Q1 Loss: {q1:.4f} | Pi Loss: {pi:.4f} | Alpha: {auto_alpha:.4f}")
            else:
                time.sleep(TRAIN_SLEEP_TIME) 
        else:
            time.sleep(BUFFER_WAIT_TIME)


def main():
    """Main entry point for SAC-LSTM training."""
    parser = argparse.ArgumentParser(description="SAC-LSTM Training Suite")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--buffer-type', type=str, default='per', choices=['per', 'uniform'],
                        help="'per' = Prioritized Replay, 'uniform' = Standard Replay")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    args = parser.parse_args()

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
    
    buffer_tag = f"lstm_{args.buffer_type}"
    
    benchmark_run_id = os.environ.get("BENCHMARK_RUN_ID", None)
    if benchmark_run_id:
        run_model_dir = os.path.join(MODEL_DIR, benchmark_run_id)
    elif not args.test:
        run_model_dir = os.path.join(MODEL_DIR, f"{buffer_tag}_{timestamp}")
    else:
        run_model_dir = MODEL_DIR

    os.makedirs(run_model_dir, exist_ok=True)

    # Dual Logger
    log_filename = f"train_log_{timestamp}.txt" if not args.test else f"test_log_{timestamp}.txt"
    log_filepath = os.path.join(run_model_dir, log_filename)
    sys.stdout = DualLogger(log_filepath)
    sys.stderr = sys.stdout
    print(f"Terminal logs are being saved to: {log_filepath}")
    
    env = PGRCEnv()
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low = env.action_space.low
    high = env.action_space.high

    if args.buffer_type == 'per':
        replay_buffer = ReplayBufferLSTMPER(BUFFER_SIZE, max_priority, epsilon, per_alpha, device)
        print("[BUFFER] Using PER (Prioritized Experience Replay)")
    else:
        replay_buffer = ReplayBufferLSTM(BUFFER_SIZE, device)
        print("[BUFFER] Using Uniform Replay (No PER)")

    linear_low, linear_high = LINEAR_MIN, LINEAR_MAX
    angular_low, angular_high = ANGULAR_MIN, ANGULAR_MAX
    low = np.array([linear_low, angular_low])
    high = np.array([linear_high, angular_high])

    action_scale = torch.FloatTensor((high - low) / 2.0).to(device)
    action_bias = torch.FloatTensor((high + low) / 2.0).to(device)
        
    sac_trainer = SAC_Trainer(
        replay_buffer, state_dim, action_dim, HIDDEN_DIM, action_scale,  
        ALPHA, GAMMA, TAU, Learning_Rate, Reward_Scale, Q_LEARNING_RATE,
        POLICY_LEARNING_RATE, Grdient_clip_max_norm, device, action_bias=action_bias
    )

    total_steps = [0] 
    best_eval_model_path = None
    best_eval_reward = -float('inf')
    best_eval_success_rate = -1.0

    # ══════════════════════════════════════════════════════════
    # PRINT TRAINING HEADER (for collector & conference paper)
    # ══════════════════════════════════════════════════════════
    print_training_header(args, device, state_dim, action_dim, run_model_dir)

    if args.test:
        path = args.model_path if args.model_path else os.path.join(run_model_dir, "best_eval_model")
        sac_trainer.load_model(path)
        is_training = False
    else:
        writer = SummaryWriter(f"{LOG_BASE_DIR}{buffer_tag}_{timestamp}")
        is_training = True
        threading.Thread(
            target=learner_worker,
            args=(sac_trainer, total_steps, writer, train_event),
            daemon=True
        ).start()

    interrupted = False
    eps = 0

    try:
        h_in = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
        last_action = np.zeros(action_dim, dtype=np.float32)

        if not is_training:
            env.is_evaluating = True
            print("\n--- RUNNING IN TEST MODE ---")

        for eps in range(1, MAX_EPISODES + 1):
            state, _ = env.reset()
            if not (env.continuous_mode and env.last_termination_reason == "goal_reached"):
                h_in = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
                last_action = np.zeros(action_dim, dtype=np.float32)
            else:
                h_in = (h_in[0].detach(), h_in[1].detach())
                    
            episode_reward = 0
            episode_data = {'s': [], 'a': [], 'la': [], 'r': [], 'ns': [], 'd': [], 'h': []}

            for step in range(MAX_STEPS_PER_EP):
                total_steps[0] += 1
                h_cpu = (h_in[0].detach().cpu(), h_in[1].detach().cpu())
                episode_data['h'].append(h_cpu)
                
                with torch.no_grad():
                    action_np, h_in = sac_trainer.policy_net.get_action(
                        state, last_action, h_in, deterministic=(not is_training)
                    )
                
                next_state, reward, done, truncated, _ = env.step(action_np, max_steps=MAX_STEPS_PER_EP)
                
                if is_training:
                    episode_data['s'].append(state)
                    episode_data['a'].append(action_np)
                    episode_data['la'].append(last_action)
                    episode_data['r'].append(reward)
                    episode_data['ns'].append(next_state)
                    episode_data['d'].append(done)

                state, last_action, episode_reward = next_state, action_np, episode_reward + reward
                if done or truncated: 
                    break

            # Metadata
            if env.continuous_mode and env.last_termination_reason == "goal_reached":
                reset_status = "CONTINUOUS"
            else:
                reset_status = "RESET"

            s_idx = env.current_start_idx
            g_idx = env.current_goal_idx
            g_type = env.current_goal_type
            result_tag = env.last_termination_reason.upper()

            if is_training:
                is_too_short = (step + 1 < MIN_STEPS_PER_EP)
                
                if not is_too_short:
                    pkg = (
                        episode_data['h'], episode_data['s'], episode_data['a'], 
                        episode_data['la'], episode_data['r'], episode_data['ns'], 
                        episode_data['d']
                    )
                    data_queue.put(pkg)
                else:
                    print(f" >>> [SKIP] Ep {eps}: Too short ({step+1} steps) - Discarded.")

                current_global_step = total_steps[0]
                writer.add_scalar('Train/Reward_vs_Episode', episode_reward, eps)
                writer.add_scalar('Train/Steps_vs_Episode', step + 1, eps)
                train_success_metric = 1.0 if result_tag == "GOAL_REACHED" else 0.0
                writer.add_scalar('Train/Success_vs_Episode', train_success_metric, eps)

                print(f" TRAINING Ep {eps:4d} | Steps: {step+1:3d} | Total: {current_global_step:7d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Path: {s_idx}->{g_idx} ({g_type}) | "
                      f"Result: {result_tag:12} | Mode: {reset_status}")

                # Evaluation
                if eps % EVAL_INTERVAL == 0 and len(sac_trainer.replay_buffer) >= BATCH_SIZE:
                    eval_rewards, eval_successes = [], 0
                    train_event.clear()
                    env.is_evaluating = True
                    env.last_termination_reason = "EVAL_START"

                    eval_goals_to_test = env.active_goals * 2
                    current_eval_episodes = len(eval_goals_to_test)
                    print(f"\n--- Starting Evaluation for Episode {eps} ({current_eval_episodes} episodes) ---")

                    e_h = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
                    e_last_a = np.zeros(action_dim, dtype=np.float32)

                    for eval_ep_num, goal_coords_to_test in enumerate(eval_goals_to_test):
                        e_state, _ = env.reset(options={'eval_goal_coords': goal_coords_to_test})
                        
                        if not (env.continuous_mode and env.last_termination_reason == "goal_reached"):
                            e_h = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
                            e_last_a = np.zeros(action_dim, dtype=np.float32)
                            
                        e_reward = 0
                        for e_step in range(MAX_STEPS_PER_EP):
                            with torch.no_grad():
                                e_action, e_h = sac_trainer.policy_net.get_action(e_state, e_last_a, e_h, deterministic=True)
                            e_n_state, e_r, e_done, e_trunc, _ = env.step(e_action, max_steps=MAX_STEPS_PER_EP)
                            e_reward += e_r
                            e_state, e_last_a = e_n_state, e_action
                            if e_done or e_trunc:
                                res_tag = env.last_termination_reason.upper()
                                if res_tag == "GOAL_REACHED":
                                    eval_successes += 1
                                break
                        eval_rewards.append(e_reward)
                    
                    env.is_evaluating = False
                    env.last_termination_reason = "TRAIN_RESUME"
                    train_event.set()

                    avg_e_reward = np.mean(eval_rewards)
                    success_rate = (eval_successes / current_eval_episodes) * 100
                    print(f"Summary | Avg Reward: {avg_e_reward:.2f} | Success Rate: {success_rate:.1f}%\n")

                    writer.add_scalar('Eval/AvgReward', avg_e_reward, eps)
                    writer.add_scalar('Eval/SuccessRate', success_rate, eps)

                    if success_rate > best_eval_success_rate or \
                       (success_rate == best_eval_success_rate and avg_e_reward > best_eval_reward):
                        best_eval_success_rate = success_rate
                        best_eval_reward = avg_e_reward

                        if best_eval_model_path:
                            try:
                                suffixes = ['_q1.pth', '_q2.pth', '_policy.pth', '_target_q1.pth', '_target_q2.pth', '_target_policy.pth']
                                for suffix in suffixes:
                                    old_file = best_eval_model_path + suffix
                                    if os.path.exists(old_file):
                                        os.remove(old_file)
                            except OSError as e:
                                print(f"[Warning] Error removing old best model: {e}")

                        new_best_name = f"best_eval_model_ep_{eps}"
                        best_eval_model_path = os.path.join(run_model_dir, new_best_name)
                        sac_trainer.save_model(best_eval_model_path)
                        print(f"** New best evaluation model saved: {new_best_name} (Success: {success_rate:.1f}%, Reward: {avg_e_reward:.2f}) **")

                if eps % CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = os.path.join(run_model_dir, f"model_ep_{eps}")
                    sac_trainer.save_model(checkpoint_path)
                    print(f"Intermediate checkpoint saved: {checkpoint_path}")
            else:
                print(f"[TEST] Ep {eps:4d} | Steps: {step+1:3d} | Reward: {episode_reward:8.2f} | "
                      f"Path: {s_idx}->{g_idx} ({g_type}) | Result: {result_tag:12} | Mode: {reset_status}")
                
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving final model...")
    finally:
        final_path = os.path.join(run_model_dir, "final_trained_model")
        sac_trainer.save_model(final_path)
        print(f"Final model saved at: {final_path}")
        
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