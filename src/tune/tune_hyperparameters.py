#!/usr/bin/env python3
import optuna
import argparse
import rclpy
import time
import os
import sys
import pathlib
import torch
import numpy as np
import threading
from queue import Queue
from collections import deque

# Ensure we can import from package root (src) when running from workspace root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Internal imports
from sac_agent.sac_v2_lstm_R import SAC_Trainer
from sac_agent.common.buffers import ReplayBufferLSTMPER
from envs.pgrc_env_map_goal import PGRCEnv
from config.config import * # Import existing constants, we will override some

# --- Constants for Tuning ---
TUNING_MAX_EPISODES = 400       # Number of episodes to run for each trial
TUNING_EVAL_INTERVAL = 50      # Evaluate every N episodes
TUNING_MAX_STEPS_PER_EP = 700   # Max steps per episode during trial

def objective(trial):
    """
    The Optuna objective function. Each call to this function is one "trial"
    with a specific set of hyperparameters.
    """
    # --- 1. Suggest Hyperparameters ---
    # FOCUSED SEARCH: Tuning only learning rates. Other parameters are fixed to best known values.
    
    base_lr = trial.suggest_float('base_lr', 3e-4, 5e-4, log=True)

    actor_scale = trial.suggest_float('actor_scale', 0.8, 1.3)
    critic_scale = trial.suggest_float('critic_scale', 0.6, 1.2)
    alpha_scale = trial.suggest_float('alpha_scale', 0.5, 1.0)
    
    params = {
        # Fixed parameters based on previous best trials
        'hidden_dim': 512,
        'batch_size': 32,
        'seq_len': 32,
        'burnin_len': 16,
        'gamma': 0.973,
        'tau': 0.0038,
        'grad_clip': 1.6,
        # Tuning only the learning rates
        # 'lr_critic': trial.suggest_float('lr_critic', 1e-4, 8e-4, log=True),
        # 'lr_actor': trial.suggest_float('lr_actor', 1e-4, 8e-4, log=True),
        # 'lr_alpha': trial.suggest_float('lr_alpha', 1e-4, 8e-4, log=True),

         # Derived LRs
        'lr_actor': base_lr * actor_scale,
        'lr_critic': base_lr * critic_scale,
        'lr_alpha': base_lr * alpha_scale,
    }

    # --- Setup for this trial (isolated) ---
    data_queue = Queue(maxsize=20)
    train_event = threading.Event()
    train_event.set()

    # This worker is defined inside the objective to capture trial-specific params
    def learner_worker(sac_trainer, total_steps_ref, train_event, local_params):
        print(f"[TRIAL-{trial.number}] Background Learner Started.")
        training_started = False
        train_metrics = {"total_updates": 0}

        while rclpy.ok() and train_event.is_set():
            while not data_queue.empty():
                pkg = data_queue.get()
                sac_trainer.replay_buffer.push_all_state(*pkg)

            current_global_step = total_steps_ref[0]

            if len(sac_trainer.replay_buffer) >= local_params['batch_size']:
                if not training_started:
                    train_metrics["total_updates"] = current_global_step
                    training_started = True
                    print(f"\n[TRIAL-{trial.number}] Buffer filled. Starting training...")

                if train_metrics["total_updates"] < current_global_step:
                    beta = min(1.0, BETA_START + current_global_step * (1.0 - BETA_START) / BETA_FRAMES)
                    sac_trainer.update(
                        batch_size=local_params['batch_size'], beta=beta,
                        seq_len=local_params['seq_len'], burnin_len=local_params['burnin_len']
                    )
                    train_metrics["total_updates"] += 1
                else:
                    time.sleep(TRAIN_SLEEP_TIME)
            else:
                time.sleep(BUFFER_WAIT_TIME)
        print(f"[TRIAL-{trial.number}] Learner thread finished.")

    # --- 2. Run Training & Evaluation ---
    best_eval_success_rate = -1.0
    try:
        rclpy.init()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = PGRCEnv()
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

        # Action space setup
        low = np.array([LINEAR_MIN, ANGULAR_MIN])
        high = np.array([LINEAR_MAX, ANGULAR_MAX])
        action_scale = torch.FloatTensor((high - low) / 2.0).to(device)
        action_bias = torch.FloatTensor((high + low) / 2.0).to(device)

        replay_buffer = ReplayBufferLSTMPER(BUFFER_SIZE, max_priority, epsilon, per_alpha, device)

        sac_trainer = SAC_Trainer(
            replay_buffer, state_dim, action_dim, params['hidden_dim'], action_scale,
            ALPHA, params['gamma'], params['tau'], params['lr_alpha'], Reward_Scale,
            params['lr_critic'], params['lr_actor'], params['grad_clip'], device, action_bias=action_bias
        )

        total_steps = [0]
        learner_thread = threading.Thread(target=learner_worker, args=(sac_trainer, total_steps, train_event, params), daemon=True)
        learner_thread.start()

        for eps in range(1, TUNING_MAX_EPISODES + 1):
            state, _ = env.reset()
            h_in = (torch.zeros([1, 1, params['hidden_dim']]).to(device), torch.zeros([1, 1, params['hidden_dim']]).to(device))
            episode_data = {'s': [], 'a': [], 'la': [], 'r': [], 'ns': [], 'd': [], 'h': []}
            episode_reward = 0.0
            last_action = np.zeros(action_dim, dtype=np.float32)

            for step in range(TUNING_MAX_STEPS_PER_EP):
                total_steps[0] += 1
                h_cpu = (h_in[0].detach().cpu(), h_in[1].detach().cpu())
                episode_data['h'].append(h_cpu)
                action_np, h_in = sac_trainer.policy_net.get_action(state, last_action, h_in, deterministic=False)

                next_state, reward, done, truncated, _ = env.step(action_np, max_steps=TUNING_MAX_STEPS_PER_EP)
                episode_reward += reward

                episode_data['s'].append(state); episode_data['a'].append(action_np)
                episode_data['la'].append(last_action); episode_data['r'].append(reward)
                episode_data['ns'].append(next_state); episode_data['d'].append(done)

                state, last_action = next_state, action_np
                if done or truncated:
                    break

            # --- Add per-episode logging for better feedback ---
            result_tag = env.last_termination_reason.upper()
            current_global_step = total_steps[0]
            print(f"[TRIAL-{trial.number}] Ep {eps:3d} | Steps: {step+1:3d} | Total: {current_global_step:6d} | "
                  f"Reward: {episode_reward:8.2f} | Result: {result_tag:12}")

            is_short_success = (step + 1 < MIN_STEPS_PER_EP) and (env.last_termination_reason.upper() == "GOAL_REACHED")
            if not is_short_success:
                pkg = (episode_data['h'], episode_data['s'], episode_data['a'], episode_data['la'],
                       episode_data['r'], episode_data['ns'], episode_data['d'])
                data_queue.put(pkg)

            # --- Evaluation Block ---
            if eps % TUNING_EVAL_INTERVAL == 0 and len(sac_trainer.replay_buffer) >= params['batch_size']:
                eval_successes = 0
                env.is_evaluating = True

                eval_goals_to_test = env.active_goals * 2 # Test each goal twice
                current_eval_episodes = len(eval_goals_to_test)
                print(f"\n[TRIAL-{trial.number}] Starting Evaluation for Episode {eps}...")

                for eval_ep_num, goal_coords_to_test in enumerate(eval_goals_to_test):
                    e_state, _ = env.reset(options={'eval_goal_coords': goal_coords_to_test})
                    e_h = (torch.zeros([1, 1, params['hidden_dim']]).to(device), torch.zeros([1, 1, params['hidden_dim']]).to(device))
                    e_last_a = np.zeros(action_dim, dtype=np.float32)
                    e_reward_total = 0.0

                    for e_step in range(TUNING_MAX_STEPS_PER_EP):
                        e_action, e_h = sac_trainer.policy_net.get_action(e_state, e_last_a, e_h, deterministic=True)
                        e_n_state, e_r, e_done, e_trunc, _ = env.step(e_action, max_steps=TUNING_MAX_STEPS_PER_EP)
                        e_reward_total += e_r
                        e_state, e_last_a = e_n_state, e_action

                        if e_done or e_trunc:
                            res_tag = env.last_termination_reason.upper()
                            if res_tag == "GOAL_REACHED":
                                eval_successes += 1
                                
                            e_reset_status = "CONTINUOUS" if (env.continuous_mode and env.last_termination_reason == "goal_reached") else "RESET"

                            s_idx = env.current_start_idx
                            g_idx = eval_ep_num % len(env.active_goals)
                            print(f"[TRIAL-{trial.number}] EVAL {eval_ep_num+1:3d}/{current_eval_episodes} | "
                                  f"Steps: {e_step+1:3d} | Reward: {e_reward_total:8.2f} | Path: {s_idx}->{g_idx} | Result: {res_tag:12} | Mode: {e_reset_status}")
                            break

                env.is_evaluating = False
                success_rate = (eval_successes / current_eval_episodes) * 100
                print(f"[TRIAL-{trial.number}] Eval @ Ep {eps} | Success Rate: {success_rate:.1f}%")

                if success_rate > best_eval_success_rate:
                    best_eval_success_rate = success_rate

                # --- Optuna Pruning ---
                # Report intermediate results to Optuna and check if we should prune.
                trial.report(success_rate, eps)
                if trial.should_prune():
                    print(f"[TRIAL-{trial.number}] Pruned at episode {eps}.")
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
        # Re-raise the exception so Optuna correctly registers the state as PRUNED
        raise
    except Exception as e:
        print(f"[TRIAL-{trial.number}] An error occurred: {e}")
        # In case of an error, return a very low value so Optuna knows this trial failed.
        return -1.0
    finally:
        # --- 3. Cleanup ---
        # Stop the learner thread and wait for it to finish safely
        train_event.clear()
        if 'learner_thread' in locals():
            learner_thread.join(timeout=5.0)
            
        # This block ensures that ROS and the environment are shut down correctly
        # for every trial, even if it's pruned or fails.
        if 'env' in locals() and env is not None:
            env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"[TRIAL-{trial.number}] Trial finished. Best success rate: {best_eval_success_rate:.1f}%")

    return best_eval_success_rate


if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna for SAC-LSTM.")
    parser.add_argument('--study-name', type=str, default="sac-lstm-lr-tuning", help="Name for the Optuna study.")
    parser.add_argument('--n-trials', type=int, default=50, help="Number of trials to run.")
    parser.add_argument('--storage', type=str, default="sqlite:///sac_tuning.db", help="Database URL for Optuna storage.")
    args = parser.parse_args()

    # --- Create and Run the Study ---
    # Using a database (like SQLite) allows you to pause and resume the study.
    # The pruner stops unpromising trials early.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=TUNING_EVAL_INTERVAL, interval_steps=TUNING_EVAL_INTERVAL)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize", # We want to maximize the success rate
        pruner=pruner
    )

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")

    # --- Print Results ---
    print("\n" + "="*50)
    print("TUNING COMPLETE")
    print(f"Study: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print(f"Pruned trials: {len(pruned_trials)}")
    print(f"Complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    print("\n--- Best Trial ---")
    print(f"  Value (Success Rate): {best_trial.value:.2f}%")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- Save best parameters to a file ---
    best_params_file = "best_hyperparameters.txt"
    with open(best_params_file, "w") as f:
        f.write(f"Best trial for study: {study.study_name}\n")
        f.write(f"Success Rate: {best_trial.value:.2f}%\n")
        f.write("\n--- Hyperparameters ---\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest parameters saved to {best_params_file}")

    # You can also visualize the results if you have plotly installed:
    # `pip install plotly`
    # from optuna.visualization import plot_optimization_history, plot_param_importances
    #
    # fig = plot_optimization_history(study)
    # fig.show()
    #
    # fig2 = plot_param_importances(study)
    # fig2.show()