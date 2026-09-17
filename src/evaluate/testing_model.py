#!/usr/bin/env python3
import argparse
import os
import time
import torch
import numpy as np
import rclpy

import pathlib
import sys

# Ensure we can import from package root (src) when running from workspace root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Internal imports
from sac_agent.sac_v2_lstm_R import SAC_Trainer 
from sac_agent.common.buffers import ReplayBufferLSTMPER 
from envs.pgrc_env_map_goal import PGRCEnv

from config.config import *


"""
Automated Model Testing Suite for Mapless RL Navigation (SAC-LSTM).

Usage:
    python3 src/evaluate/testing_model.py --model_dir ./models/sac_lstm_checkpoints/ --models best_eval_model_ep_900 --episodes 50
"""

def main():
    parser = argparse.ArgumentParser(description="Automated Model Testing Suite")
    parser.add_argument('--model_dir', type=str, default=TEST_MODEL_DIR, 
                        help='Base directory where the models are located.')
    parser.add_argument('--models', nargs='+', default=TEST_MODELS, 
                        help='List of model name prefixes (e.g., model_ep_1300 model_ep_1400)')
    parser.add_argument('--episodes', type=int, default=TEST_EPISODES, 
                        help='Number of random episodes to test each model.')
    args = parser.parse_args()

    # Initialize ROS 2 and PyTorch
    rclpy.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Automated Testing Suite on {device} ---")
    
    # Initialize Environment
    env = PGRCEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Setup Action Scaling
    linear_low, linear_high = LINEAR_MIN, LINEAR_MAX
    angular_low, angular_high = ANGULAR_MIN, ANGULAR_MAX
    low = np.array([linear_low, angular_low])
    high = np.array([linear_high, angular_high])

    action_scale = torch.FloatTensor((high - low) / 2.0).to(device)
    action_bias = torch.FloatTensor((high + low) / 2.0).to(device)
    
    # Evaluation does not require replay buffer allocation
    replay_buffer = None
    sac_trainer = SAC_Trainer(
        replay_buffer, state_dim, action_dim, HIDDEN_DIM, action_scale,  
        ALPHA, GAMMA, TAU, Learning_Rate, Reward_Scale, Q_LEARNING_RATE,
        POLICY_LEARNING_RATE, Grdient_clip_max_norm, device, action_bias=action_bias
    )
    
    # Dictionary to store final metrics for the report
    test_results = {}

    for model_name in args.models:
        model_path = os.path.join(args.model_dir, model_name)
        print(f"\n{'='*60}\n>>> Loading and Testing Model: {model_name}\n{'='*60}")
        
        try:
            sac_trainer.load_model(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model '{model_name}'. Skipping... \nDetails: {e}")
            continue
            
        # --- Random Episode Testing Setup ---
        # Use evaluation settings (e.g., for start pose) but with random goals by not passing explicit coordinates.
        env.is_evaluating = True 
        print(f"Model will be tested for {args.episodes} random episodes.")

        all_rewards = []
        all_successes = 0
        
        # Initialize memory outside the loop
        h_in = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
        last_action = np.zeros(action_dim, dtype=np.float32)

        for ep_num in range(1, args.episodes + 1):
            # Reset without options to trigger the environment's internal random goal selection.
            state, _ = env.reset()
            
            if not (env.continuous_mode and env.last_termination_reason == "goal_reached"):
                h_in = (torch.zeros([1, 1, HIDDEN_DIM]).to(device), torch.zeros([1, 1, HIDDEN_DIM]).to(device))
                last_action = np.zeros(action_dim, dtype=np.float32)
                
            episode_reward = 0
            
            for step in range(TESTING_MAX_STEPS_PER_EP):
                # Deterministic action for evaluation
                # action_np, h_in = sac_trainer.policy_net.get_action(state, last_action, h_in, deterministic=True)
                with torch.no_grad():
                    action_np, h_in = sac_trainer.policy_net.get_action(
                        state, last_action, h_in, deterministic=True
                    )
                next_state, reward, done, truncated, _ = env.step(action_np, max_steps=TESTING_MAX_STEPS_PER_EP)
                episode_reward += reward
                
                state = next_state
                last_action = action_np
                
                if done or truncated:
                    res_tag = env.last_termination_reason.upper()
                    
                    # Check if the episode was a successful goal arrival
                    if done and env.last_termination_reason == "goal_reached": 
                        all_successes += 1
                    
                    # Get path info from the environment for detailed logging
                    s_idx = env.current_start_idx
                    g_idx = env.current_goal_idx
                    g_type = env.current_goal_type

                    goal_pos = env.goal_pose.pose.position
                    goal_str = f"{goal_pos.x:.1f},{goal_pos.y:.1f}"
                    print(f"[{model_name} | Ep {ep_num:3d}/{args.episodes}] Steps: {step+1:3d} | Reward: {episode_reward:8.2f} | "
                          f"Path: {s_idx}->{g_idx} ({g_type}) @ {goal_str} | Result: {res_tag:12}")
                    break
                    
            all_rewards.append(episode_reward)
            
        # Calculate and store summary stats
        avg_reward = np.mean(all_rewards)
        success_rate = (all_successes / args.episodes) * 100
        test_results[model_name] = {'avg_reward': avg_reward, 'success_rate': success_rate}
        print(f"\n--- Summary for {model_name} ---")
        print(f"Average Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}%")

    # Final Report Output
    print("\n" + "="*60)
    print("FINAL AUTOMATED TESTING REPORT")
    print("="*60)
    for model_name, res in test_results.items():
        print(f"Model: {model_name:20s} | Avg Reward: {res['avg_reward']:8.2f} | Success Rate: {res['success_rate']:5.1f}%")
    print("="*60)

    # --- Write report to a text file ---
    report_path = os.path.join(args.model_dir, "evaluation_summary.txt")
    try:
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FINAL AUTOMATED TESTING REPORT\n")
            f.write("="*60 + "\n")
            for model_name, res in test_results.items():
                f.write(f"Model: {model_name:20s} | Avg Reward: {res['avg_reward']:8.2f} | Success Rate: {res['success_rate']:5.1f}%\n")
            f.write("="*60 + "\n")
        print(f"\nEvaluation summary saved to: {report_path}")
    except Exception as e:
        print(f"\n[ERROR] Could not write report to file: {e}")

    # Clean up ROS 2 gracefully
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()