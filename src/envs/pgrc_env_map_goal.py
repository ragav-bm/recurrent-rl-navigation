#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import math
import random
from tf_transformations import euler_from_quaternion
from rl_training.srv import ResetRobotPose, SetGoalAndObstaclePose 
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
import subprocess

from rclpy.parameter import Parameter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import *

# -------------------- Utility Functions -------------------- #
def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def call_service_synchronously(node, client, request, timeout_sec=1.0, max_retries=3):
    """Calls a service and aggressively retries if Gazebo times out."""
    for attempt in range(max_retries):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        
        if future.done():
            return future.result()
            
        node.get_logger().warn(f"Service call timeout on {client.srv_name} (Attempt {attempt+1}/{max_retries}). Retrying...")
        
    node.get_logger().error(f"CRITICAL: Service {client.srv_name} failed after {max_retries} attempts!")
    return None

# -------------------- Environment -------------------- #
class PGRCEnv(Node, gym.Env):
    def __init__(self, node_name="pgrc_env", history_len=HISTORY_LEN, reward_kwargs=None):
        """
        Initializes the PGRC Environment using ROS 2 and Gymnasium.
        
        Args:
            node_name (str): Name of the ROS 2 node.
            history_len (int): Length of the LiDAR history queue.
            reward_kwargs (dict, optional): Dynamic reward tuning parameters.
        """
        Node.__init__(self, node_name) 

        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self.history_len = history_len


        # --- ROS Setup ---
        self.lidar_history = deque([np.zeros(LIDAR_SAMPLES, dtype=np.float32)]*history_len, maxlen=history_len)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/rl_goal', 10)
        self.obs_pub = self.create_publisher(Float32MultiArray, '/rl/processed_obs', 10)
                

        # Service Clients
        self.reset_client = self.create_client(ResetRobotPose, 'reset_robot_pose')
        self.set_entities_client = self.create_client(SetGoalAndObstaclePose, 'set_goal_and_obstacle_pose')
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

        # --- Robot State & Spaces ---
        self.odom_pose = None
        self.odom_vel = None
        self.v_max, self.w_max = LINEAR_MAX, ANGULAR_MAX
        self.collision_dist = COLLISION_DIST
        self.safe_distance = SAFE_DIST
   
        obs_dim = (LIDAR_SAMPLES * history_len) + METRICS_DIM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=ACTION_SPACE_LOW, high=ACTION_SPACE_HIGH, shape=(2,), dtype=np.float32)

        # --- Debug Tracking ---
        self.episode_num = 0
        self.cum_reward = 0.0
        self.step_count = 0
        
        # --- Reward Parameters (Dynamic for Tuning) ---
        reward_kwargs = reward_kwargs or {}
        self.progress_factor = reward_kwargs.get('progress_factor', PROGRESS_REWARD_FACTOR)
        self.dist_weight = reward_kwargs.get('dist_weight', DISTANCE_REWARD_WEIGHT)
        self.time_penalty = reward_kwargs.get('time_penalty', TIME_PENALTY)
        self.safety_factor = reward_kwargs.get('safety_factor', SAFETY_FACTOR)
        self.closing_speed_factor = reward_kwargs.get('closing_speed_factor', CLOSING_SPEED_FACTOR)
        self.backward_reward_factor = reward_kwargs.get('backward_reward_factor', BACKWARD_REWARD_FACTOR)

        self.prev_goal_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.min_lidar_range = MAX_RANGE
        self.goal_pose = None
        self.prev_min_lidar_range = None

        # --- Goal Management based on Config ---
        self.goal_lists = {
            'TURTLEBOT_WORLD_GOALS': TURTLEBOT_WORLD_GOALS,
            'DQN4_CURRICULUM_GOALS': DQN4_CURRICULUM_GOALS,
            'DYNA_WORLD_GOALS': DYNA_WORLD_GOALS,
        }
        self.active_goals = self.goal_lists[ACTIVE_GOALS_KEY]
        self.goal_setup_type = GOAL_SETUP_TYPE

        self.random_reset = RANDOM_RESET
        self.reset_radius = RESET_RADIUS
        self.start_poses = [(0.0, 0.0, 0.0)]
        self.is_dynamic_env = IS_DYNAMIC_ENV

        reset_service_name = '/reset_simulation' if self.is_dynamic_env else '/reset_world'
        self.reset_sim_client = self.create_client(Empty, reset_service_name)
        self.get_logger().info(f"Environment configured with reset service: '{reset_service_name}'")
        self.prev_goal_idx = -1

        self.current_start_idx = -1
        self.current_goal_idx = -1
        self.current_goal_type = "NONE"
        self.current_start_type = "NONE"
        self.new_odom_ready = False
        self.new_scan_ready = False

        while not self.reset_sim_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(f"Waiting for '{reset_service_name}' service to be available...")

        try:
            # This runs a system command to see if the GLX (Graphics) is using NVIDIA
            gpu_info = subprocess.check_output("glxinfo | grep 'OpenGL vendor'", shell=True).decode()
            print(f"--- GAZEBO GRAPHICS CHECK ---")
            print(f"Renderer: {gpu_info.strip()}")
        except:
            print("Could not verify Gazebo GPU usage.")

        self.continuous_mode = CONTINUOUS_MODE
        self.last_termination_reason = "Initiated"

        self.current_goal_idx = -1
        self.current_goal_type = "None"
        self.is_evaluating = False # Flag to indicate if the environment is in evaluation mode
        self.eval_random_start = EVAL_RANDOM_START # Use specific setting for evaluation starts

    def odom_callback(self, msg):
        """
        Callback for odometry messages.
        
        Args:
            msg (nav_msgs.msg.Odometry): The incoming odometry message.
        """
        self.odom_pose = msg.pose.pose
        self.odom_vel = msg.twist.twist
        self.new_odom_ready = True # Flag tripped!

    def laser_callback(self, msg):
        """
        Callback for laser scan messages.
        Processes and normalizes LiDAR data, appending it to the history queue.
        
        Args:
            msg (sensor_msgs.msg.LaserScan): The incoming laser scan message.
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=MIN_RANGE)
        self.min_lidar_range = np.min(ranges) 
        indices = np.linspace(0, len(ranges)-1, LIDAR_SAMPLES, dtype=int)
        normalized_ranges = np.clip(ranges[indices], MIN_RANGE, MAX_RANGE) / MAX_RANGE
        self.lidar_history.append(normalized_ranges) # Scale to 0 (close) to 1 (far)
        self.new_scan_ready = True # Flag tripped!



    def compute_reward(self, current_action):
        """
        Computes the reward for the current step, tuned for dynamic environments.
        
        Args:
            current_action (np.ndarray): The action applied at the current step.
            
        Returns:
            tuple: (reward (float), done (bool))
        """
        # 1. Terminal States
        if self.min_lidar_range < self.collision_dist:
            return COLLISION_REWARD, True # e.g., -20.0
            
        dx = self.odom_pose.position.x - self.goal_pose.pose.position.x
        dy = self.odom_pose.position.y - self.goal_pose.pose.position.y
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        
        if dist_to_goal < GOAL_REACHED_DIST:
            return GOAL_REWARD, True # e.g., 50.0
            
        # === Reward Shaping for Dynamic Navigation ===

        # 2. Progress Reward: Encourages moving towards the goal.
        delta_dist = (self.prev_goal_dist - dist_to_goal) if self.prev_goal_dist is not None else 0.0
        progress_reward = self.progress_factor * np.clip(delta_dist, PROGRESS_REWARD_CLIP_MIN, PROGRESS_REWARD_CLIP_MAX)
        self.prev_goal_dist = dist_to_goal

        # 3. Distance Reward: A smooth reward based on proximity to the goal.
        distance_reward = 1.0 - np.tanh(DISTANCE_REWARD_FACTOR * dist_to_goal)

        # 4. Safety Penalty (Dynamic-Aware): Penalizes getting too close to obstacles.
        # This penalty is non-linear, increasing sharply near the collision distance.
        safety_penalty = 0.0
        if self.min_lidar_range < self.safe_distance:
            # This creates a penalty from 0 (at safe_distance) to -1 (at collision_dist)
            normalized_danger = (self.safe_distance - self.min_lidar_range) / (self.safe_distance - self.collision_dist)
            # Using a squared term makes the penalty much stronger when very close.
            # A new config parameter DYNAMIC_SAFETY_PENALTY_FACTOR could be introduced for tuning.
            safety_penalty = self.safety_factor * (normalized_danger ** 2)

        # 5. Closing Speed Penalty (NEW): Penalizes rapidly approaching an obstacle.
        # This helps the agent learn to anticipate collisions with moving objects.
        closing_speed_penalty = 0.0
        if self.prev_min_lidar_range is not None:
            delta_min_lidar = self.min_lidar_range - self.prev_min_lidar_range
            # Penalize if the distance to the nearest obstacle is decreasing
            if delta_min_lidar < 0:
                # A new config parameter CLOSING_SPEED_PENALTY_FACTOR could be introduced for tuning.
                # The factor '5.0' is chosen to be significant but not overpowering.
                closing_speed_penalty = self.closing_speed_factor * delta_min_lidar # delta_min_lidar is negative, so this is a penalty
        
        # Update for the next step. This must be initialized to None in reset().
        self.prev_min_lidar_range = self.min_lidar_range

        # 6. Backward Motion: The penalty is removed to allow for evasive maneuvers.
        # The progress reward already discourages moving away from the goal.
        backward_penalty = 0.0
        if current_action[0] < 0.0 and self.min_lidar_range < self.safe_distance:
            backward_penalty = self.backward_reward_factor


        # 7. Constant Time Penalty: Encourages finishing the episode quickly.
        # time_penalty = TIME_PENALTY

        # --- Total Reward ---
        reward = (
            progress_reward +
            self.dist_weight * distance_reward +
            self.time_penalty +
            safety_penalty +
            closing_speed_penalty +
            backward_penalty
        )
        
        self.prev_action = current_action.copy()
        return reward, False



    def step(self, action, max_steps=800):
        """
        Applies an action to the environment and advances one step.
        
        Args:
            action (np.ndarray): Array containing linear and angular velocity commands.
            max_steps (int): Maximum number of steps per episode.
            
        Returns:
            tuple: (obs (np.ndarray), reward (float), done (bool), truncated (bool), info (dict))
        """
        self.step_count += 1
        twist = Twist()
        twist.linear.x = float(action[0])  # Scaled linear velocity [m/s]
        twist.angular.z = float(action[1]) # Scaled angular velocity [rad/s]
        self.cmd_pub.publish(twist)
        
        # Action duration timing
        start = self.get_clock().now()
        while (self.get_clock().now() - start).nanoseconds / NANO_TO_SEC < FIXED_ACTION_DURATION:
            rclpy.spin_once(self, timeout_sec=SPIN_WAIT_TIME)

        obs = self._get_obs()
        reward, done = self.compute_reward(action)
        self.cum_reward += reward

        truncated =self.step_count >= max_steps
        if done:
            if self.min_lidar_range < self.collision_dist:
                self.last_termination_reason = "Collision" # It hit a wall
            else:
                self.last_termination_reason = "goal_reached"
                
        elif truncated:
            # print("Episode Truncated: Max Steps Reached")
            self.last_termination_reason = "truncated"

        # Only add a velocity bonus if the episode is not yet done.
        if not done and twist.linear.x > LINEAR_VELOCITY_THRESHOLD:
            reward += STEP_VELOCITY_REWARD * twist.linear.x


        obs_msg = Float32MultiArray()
        obs_msg.data = [float(x) for x in obs] 

        # Publish it for the monitor to see
        self.obs_pub.publish(obs_msg)
            
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        import time
        wait_start = time.perf_counter()
        while not (self.new_scan_ready and self.new_odom_ready) or self.goal_pose is None:
            rclpy.spin_once(self, timeout_sec=SPIN_TIMEOUT)
            if time.perf_counter() - wait_start > 1.0:
                self.get_logger().error("WATCHDOG: Stuck waiting for /scan. Forcing Gazebo UNPAUSE!")
                call_service_synchronously(self, self.unpause_client, Empty.Request())
                wait_start = time.perf_counter()

        self.new_odom_ready = False
        self.new_scan_ready = False

        lidar_obs = np.concatenate(list(self.lidar_history)).astype(np.float32)

        if not self.is_evaluating:
            noise = np.random.normal(0, LIDAR_NOISE_STD, lidar_obs.shape).astype(np.float32)
            lidar_obs = np.clip(lidar_obs + noise, 0.0, 1.0)

        robot_x, robot_y = self.odom_pose.position.x, self.odom_pose.position.y
        goal_x, goal_y = self.goal_pose.pose.position.x, self.goal_pose.pose.position.y
        goal_dist = np.hypot(goal_x - robot_x, goal_y - robot_y)
        q = self.odom_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        goal_angle = normalize_angle(math.atan2(goal_y - robot_y, goal_x - robot_x) - yaw)
        is_first_step = 1.0 if self.step_count <= 1 else 0.0

        # NEW: Extract obstacle-aware features from current lidar
        current_lidar = np.array(self.lidar_history[-1])
        min_range = np.min(current_lidar)
        min_range_idx = np.argmin(current_lidar)
        # Angle of closest obstacle relative to robot heading
        min_range_angle = (min_range_idx / LIDAR_SAMPLES) * 2 * np.pi - np.pi

        # NEW: Temporal difference — detect moving obstacles
        if len(self.lidar_history) >= 2:
            prev_lidar = np.array(self.lidar_history[-2])
            lidar_delta = np.min(current_lidar) - np.min(prev_lidar)  # negative = approaching
        else:
            lidar_delta = 0.0

        obs_metrics = np.array([
            goal_dist / MAX_RANGE,                          # 1. normalized distance to goal
            goal_angle / np.pi,                             # 2. normalized angle to goal
            min_range,                                      # 3. closest obstacle (already normalized)
            self.odom_vel.linear.x / self.v_max,           # 4. linear velocity
            self.odom_vel.angular.z / self.w_max,          # 5. angular velocity
            is_first_step,                                  # 6. first step flag
            min_range_angle / np.pi,                        # 7. NEW: angle to closest obstacle
            lidar_delta,                                    # 8. NEW: obstacle approach rate
            self.prev_action[0],                           # 9. NEW: previous linear velocity
            self.prev_action[1] / self.w_max,             # 10. NEW: previous angular velocity
        ], dtype=np.float32)

        return np.concatenate([lidar_obs, obs_metrics])

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.
        
        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional options for resetting.
            
        Returns:
            tuple: (obs (np.ndarray), info (dict))
        """
        super().reset(seed=seed, options=options) # Pass options to parent reset method

        # Determine if this is a "soft" reset (continue from current pos) or a "hard" reset (teleport).
        is_hard_reset = not (self.continuous_mode and self.last_termination_reason == "goal_reached")

        # --- World/Simulation Reset ---
        # A hard reset is needed for collisions, timeouts, or the very first run.
        # A soft reset (in continuous mode after reaching a goal) skips this.
        if is_hard_reset:
            if self.is_dynamic_env: # For dynamic worlds, a full simulation reset is required.
                # For dynamic worlds, a full simulation reset is required.
                call_service_synchronously(self, self.reset_sim_client, Empty.Request(), timeout_sec=5.0)
                call_service_synchronously(self, self.pause_client, Empty.Request())
            else:
                # For static worlds, we can pause first and then reset models.
                call_service_synchronously(self, self.pause_client, Empty.Request())
                call_service_synchronously(self, self.reset_sim_client, Empty.Request())
        
        # Standard Reset Params
        self.episode_num += 1
        self.cum_reward, self.step_count = 0.0, 0
        self.new_scan_ready = False
        self.new_odom_ready = False
        if is_hard_reset:
            self.lidar_history = deque([np.zeros(LIDAR_SAMPLES, dtype=np.float32)]*self.history_len, maxlen=self.history_len)
            self.prev_min_lidar_range = None
            self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        req_reset = ResetRobotPose.Request()
        # Default values, will be overwritten by goal selection logic
        gx, gy, qz, qw = 0.0, 0.0, 0.0, 1.0 

        # ---------------------------------------------------------
        # GOAL & INDEX SELECTION LOGIC
        # ---------------------------------------------------------
        # helper to pick an index different from prev if possible
        def _pick_index(length, avoid_idx=None):
            if avoid_idx is None or length <= 1:
                return random.randrange(length)
            choices = [i for i in range(length) if i != avoid_idx]
            return random.choice(choices) if choices else avoid_idx

        avoid_same_goal = not is_hard_reset
        
        eval_goal_coords = options.get('eval_goal_coords') if options else None
        eval_start_coords = options.get('eval_start_coords') if options else None
        
        if self.goal_setup_type == 'PAIRED':
            # MODE: Normal Goal Pairs
            # Determine indices based on previous termination
            if avoid_same_goal:
                # CONTINUOUS: Start where we just finished
                idx_start = self.current_goal_idx
                # Pick a new goal that is definitely NOT our current position
                idx_goal = _pick_index(len(self.active_goals), avoid_idx=idx_start)
            else:
                # RESET: Pick a fresh pair and teleport
                idx_start, idx_goal = random.sample(range(len(self.active_goals)), 2)

            # Store for the logger
            self.current_start_idx = idx_start
            self.current_goal_idx = idx_goal
            self.current_goal_type = "NORMAL"

            # Get actual coordinates
            start_pose = self.active_goals[idx_start]
            goal_pose = self.active_goals[idx_goal]
            
            # Prepare service request
            req_reset.x, req_reset.y, req_reset.z, req_reset.yaw = start_pose
            gx, gy, qz, qw = goal_pose
            
            self.current_goal_idx = idx_goal
            self.current_goal_type = "NORMAL"
        else:
            # MODE: Single List (Curriculum, Dynamic, etc.)

            # ---- START POSE TRACKING ----
            if avoid_same_goal:
                # CONTINUOUS MODE: The start index for logging is the goal we just reached.
                # The robot's physical pose is NOT reset because is_hard_reset is False.
                self.current_start_idx = self.current_goal_idx
                self.current_start_type = "CONTINUOUS"
            elif self.random_reset:
                # HARD RESET to a random pose within the radius.
                r = self.reset_radius * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                req_reset.x, req_reset.y = r * math.cos(theta), r * math.sin(theta)
                req_reset.yaw = random.uniform(-math.pi, math.pi)
                self.current_start_idx = -1 # -1 signifies a random start, not from a goal index.
                self.current_start_type = "RANDOM"
            else:
                # HARD RESET to the fixed origin (0,0).
                req_reset.x, req_reset.y, req_reset.z, req_reset.yaw = 0.0, 0.0, 0.0, 0.0
                self.current_start_idx = 0 # 0 signifies the origin.
                self.current_start_type = "FIXED_ORIGIN"

            # ---- GOAL SELECTION (SIMPLIFIED) ----
            # For a continuous run, avoid_same_goal is True, so we pick a new goal different from the current one.
            # For a hard reset, avoid_same_goal is False, so any goal can be picked.
            self.current_goal_idx = _pick_index(len(self.active_goals), avoid_idx=self.current_goal_idx if avoid_same_goal else None)
            self.current_goal_type = ACTIVE_GOALS_KEY.replace("_GOALS", "") # e.g., "CURRICULUM"
            gx, gy, qz, qw = self.active_goals[self.current_goal_idx]
        
        # Override goal selection if in evaluation mode with explicit coordinates
        if self.is_evaluating and eval_goal_coords is not None:
            gx, gy, qz, qw = eval_goal_coords
            
            # Robot start pose for evaluation:
            if eval_start_coords is not None:
                # Force hard teleport to the specific evaluation start point to decouple episodes
                req_reset.x, req_reset.y, req_reset.z, req_reset.yaw = eval_start_coords
                self.current_start_type = "EXPLICIT_EVAL_START"
                self.current_start_idx = -1
                is_hard_reset = True
            elif is_hard_reset:
                if self.eval_random_start:
                    r = self.reset_radius * math.sqrt(random.random())
                    theta = random.random() * 2 * math.pi
                    req_reset.x, req_reset.y = r * math.cos(theta), r * math.sin(theta)
                    req_reset.yaw = random.uniform(-math.pi, math.pi)
                    self.current_start_idx = -1
                    self.current_start_type = "RANDOM_EVAL"
                else:
                    req_reset.x, req_reset.y, req_reset.z, req_reset.yaw = 0.0, 0.0, 0.0, 0.0
                    self.current_start_type = "FIXED_ORIGIN_EVAL"
                    self.current_start_idx = 0
            else:
                self.current_start_type = "CONTINUOUS_EVAL"
            
            self.current_goal_idx = -1 # Indicate explicit coordinates, not from a list index
            self.current_goal_type = "EVAL_EXPLICIT"

        # ---------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------
        # Teleport robot if it's a hard reset
        if is_hard_reset:
            call_service_synchronously(self, self.reset_client, req_reset)
        
        req_set = SetGoalAndObstaclePose.Request()
        req_set.goal_x, req_set.goal_y = gx, gy
        call_service_synchronously(self, self.set_entities_client, req_set)

        # Unpause physics if it's a hard reset
        # This ensures the simulation is running after a reset.
        if is_hard_reset:
            call_service_synchronously(self, self.unpause_client, Empty.Request())

        self.goal_pose = PoseStamped()
        self.goal_pose.pose.position.x, self.goal_pose.pose.position.y = gx, gy
        
        # Sync Odom
        self.odom_pose = None 
        self.odom_vel = None


        while self.odom_pose is None  or self.odom_vel is None:
            rclpy.spin_once(self, timeout_sec=SPIN_TIMEOUT)
        
        self.prev_goal_dist = np.hypot(gx - self.odom_pose.position.x, gy - self.odom_pose.position.y) # Update prev_goal_dist for reward calculation
        
        # Only track prev_goal_idx for training mode's random goal selection
        self.prev_goal_idx = self.current_goal_idx

        obs = self._get_obs()
        
        obs_msg = Float32MultiArray()
        obs_msg.data = [float(x) for x in obs]
        self.obs_pub.publish(obs_msg)
        


        return obs, {}

    def close(self):
        """
        Closes the environment and shuts down the ROS node.
        
        Returns:
            None
        """
        self.cmd_pub.publish(Twist())
        self.destroy_node()