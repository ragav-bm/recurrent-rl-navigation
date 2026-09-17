"""
Global Configuration for Mapless RL Navigation (SAC + LSTM).
Centralizes environment settings, reward shaping parameters, network architecture, and training hyperparameters.
"""
import os

# Dynamically resolve workspace root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ==============================================================================
# 1. Environment Selection & Target Coordinates
# ==============================================================================
# Active environment key: 'TURTLEBOT_WORLD', 'DQN4_CURRICULUM', 'DYNA_WORLD'
ENV_NAME = 'DQN4_CURRICULUM'

# Goal definitions across benchmark worlds
TURTLEBOT_WORLD_GOALS = [
    (-1.920,  0.556,  0.000,  1.000),  # cylinder_0
    ( 0.533,  0.579,  0.000,  1.000),  # cylinder_1
    ( 2.028, -0.567,  0.000,  1.000),  # cylinder_2
    ( 0.337, -1.997,  0.000,  1.000),  # cylinder_3
    (-1.524, -1.555,  0.014,  0.999),  # cylinder_4
    (-0.547,  1.479,  0.207,  0.978),  # cylinder_5
    (-0.539, -0.566,  0.001,  1.000),  # cylinder_6
    ( 1.721,  0.624,  0.000,  1.000),  # cylinder_7
    ( 0.637,  1.882,  0.000,  1.000),  # cylinder_8
    ( 0.462, -0.626, -0.008,  1.000),  # cylinder_9
]

DQN4_CURRICULUM_GOALS = [
    (1.897, 1.970, 0.414, 0.910),
    (1.998, -1.936, -0.318, 0.948),
    (-1.398, 1.361, 0.915, 0.403),
    (-1.866, -1.997, 0.998, 0.066),
    (1.986, -1.039, 0.002, 1.000),
    (-1.631, 0.122, 1.000, 0.014),
    (0.591, 1.404, 0.703, 0.711),
    (1.197, 0.767, 0.198, 0.980),
    (1.182, -0.027, -0.154, 0.988),
]

DYNA_WORLD_GOALS = [
    (-6.715,  6.903, 0.500, 1.000),
    ( 3.167,  1.029, 0.500, 1.000),
    (-2.347,  5.797, 0.500, 1.000),
    (-4.205, -1.917, 0.500, 1.000),
    ( 7.423, -7.637, 0.500, 1.000),
    ( 3.305, -5.103, 0.500, 1.000),
    (-7.866, -2.019, 0.500, 1.000),
    ( 1.209,  5.887, 0.500, 1.000),
    ( 6.837,  7.604, 0.500, 1.000),
    (-0.436, -7.882, 0.500, 1.000),
    (-7.539, -7.541, 0.500, 1.000),
]

ENV_CONFIGS = {
    'TURTLEBOT_WORLD': {
        'IS_DYNAMIC_ENV': False,
        'CONTINUOUS_MODE': True,
        'RANDOM_RESET': True,
        'RESET_RADIUS': 0.4,
        'GOALS_KEY': 'TURTLEBOT_WORLD_GOALS',
        'GOAL_SETUP_TYPE': 'PAIRED'
    },
    'DQN4_CURRICULUM': {
        'IS_DYNAMIC_ENV': True,
        'CONTINUOUS_MODE': True,
        'RANDOM_RESET': True,
        'RESET_RADIUS': 0.4,
        'GOALS_KEY': 'DQN4_CURRICULUM_GOALS',
        'GOAL_SETUP_TYPE': 'SINGLE_LIST'
    },
    'DYNA_WORLD': {
        'IS_DYNAMIC_ENV': True,
        'CONTINUOUS_MODE': True,
        'RANDOM_RESET': True,
        'RESET_RADIUS': 1.0,
        'GOALS_KEY': 'DYNA_WORLD_GOALS',
        'GOAL_SETUP_TYPE': 'SINGLE_LIST'
    }
}

SELECTED_ENV_CONFIG = ENV_CONFIGS[ENV_NAME]
IS_DYNAMIC_ENV = SELECTED_ENV_CONFIG['IS_DYNAMIC_ENV']
CONTINUOUS_MODE = SELECTED_ENV_CONFIG['CONTINUOUS_MODE']
RANDOM_RESET = SELECTED_ENV_CONFIG['RANDOM_RESET']
RESET_RADIUS = SELECTED_ENV_CONFIG['RESET_RADIUS']
ACTIVE_GOALS_KEY = SELECTED_ENV_CONFIG['GOALS_KEY']
GOAL_SETUP_TYPE = SELECTED_ENV_CONFIG['GOAL_SETUP_TYPE']

# ==============================================================================
# 2. Sensor & Actuator Limits
# ==============================================================================
HISTORY_LEN = 1
LIDAR_SAMPLES = 180
METRICS_DIM = 6
MAX_RANGE = 3.5
MIN_RANGE = 0.12
FIXED_ACTION_DURATION = 0.05
LIDAR_NOISE_STD = 0.01

LINEAR_MIN = -0.05
LINEAR_MAX = 0.22
ANGULAR_MAX = 3.5
ANGULAR_MIN = -3.5

ACTION_SPACE_LOW = -1.0
ACTION_SPACE_HIGH = 1.0

OBSTACLE_RESET_POSES = [(-1.86, -2.00, 1.94, 1.98)]
GOAL_RADIUS = 2.0
Y_THRESHOLD = 1.6
MAX_HIGH_Y_RATIO = 0.7
NUM_GOALS = 50
NUM_STARTS = 50
DEBUG_COLLISION_TIME = 2.0

# ==============================================================================
# 3. Reward Function Parameters
# ==============================================================================
COLLISION_DIST = 0.15
SAFE_DIST = 0.35
GOAL_REACHED_DIST = 0.35

COLLISION_REWARD = -50.0
GOAL_REWARD = 50.0

PROGRESS_REWARD_FACTOR = 7.0
PROGRESS_REWARD_CLIP_MIN = -0.05
PROGRESS_REWARD_CLIP_MAX = 0.05
DISTANCE_REWARD_FACTOR = 1.5
DISTANCE_REWARD_WEIGHT = 0.5135

TIME_PENALTY = -0.01716
SAFETY_FACTOR = -0.5102
CLOSING_SPEED_FACTOR = 3.0862
BACKWARD_REWARD_FACTOR = 0.00686

STEP_VELOCITY_REWARD = 0.03
LINEAR_VELOCITY_THRESHOLD = 0.05

# ==============================================================================
# 4. Neural Network & SAC Hyperparameters
# ==============================================================================
HIDDEN_DIM = 512
BATCH_SIZE = 64
SEQ_LEN = 16
BURNIN_LEN = 8
DEFAULT_SEQ_LEN = SEQ_LEN
DEFAULT_BURNIN_LEN = BURNIN_LEN
MIN_STEPS_PER_EP = SEQ_LEN + BURNIN_LEN

ALPHA = 0.01
GAMMA = 0.99
TAU = 0.0038
Reward_Scale = 0.05

# Learning rates optimized via Optuna tuning
Learning_Rate = 8e-5        # Temperature Alpha learning rate
Q_LEARNING_RATE = 8e-5      # Critic learning rate
POLICY_LEARNING_RATE = 8e-5 # Actor learning rate
Grdient_clip_max_norm = 1.6

TARGET_ENTROPY = -1.0
INIT_W = 3e-3
LOG_STD_MIN = -5
LOG_STD_MAX = 2
POLICY_EPSILON = 1e-4
LINEAR_BIAS_INIT = 0.01
ACTION_RANGE_DEFAULT = 1.0
NOISE_SCALE_DEFAULT = 1.0

# Learning rate scheduler settings
LR_SCHEDULER_T0 = 5000
LR_SCHEDULER_TOTAL_UPDATE = 400000
LR_SCHEDULER_T_MULT = 1
LR_SCHEDULER_ETA_MIN = 1e-6
LR_SCHEDULER_START_FACTOR = 0.1
ALPHA_MIN = 0.015
ALPHA_MAX = 0.03

# ==============================================================================
# 5. Replay Buffer & Training Loop Configuration
# ==============================================================================
MAX_EPISODES = 2000
MAX_STEPS_PER_EP = 1200
BUFFER_SIZE = 400000
WARMUP_EPISODES = 150

# Prioritized Experience Replay (PER)
max_priority = 1.0
epsilon = 1e-4
per_alpha = 0.6
BETA_START = 0.4
BETA_FRAMES = 300000

# Evaluation and check-pointing
EVAL_INTERVAL = 50
EVAL_EPISODES = 10
EVAL_RANDOM_START = False
SUCCESS_REWARD_MIN = 40.0
CHECKPOINT_INTERVAL = 100
TENSORBOARD_LOG_INTERVAL = 10
CONSOLE_LOG_INTERVAL = 1000

# Standalone Testing Configuration
TEST_MODEL_DIR = os.path.join(BASE_DIR, "models/sac_lstm_checkpoints/")
TEST_MODELS = ["best_eval_model_ep_900", "model_ep_1150"]
TEST_EPISODES = 50
TESTING_MAX_STEPS_PER_EP = 800

# Logging & Checkpoint Paths
MODEL_DIR = os.path.join(BASE_DIR, "models/sac_lstm_checkpoints/")
LOG_BASE_DIR = os.path.join(BASE_DIR, "runs/sac_lstm_integrated/")

# System Timing & ROS 2 Constants
SPIN_TIMEOUT = 0.05
CONTROL_HZ = 0.05
NANO_TO_SEC = 1e9
SPIN_WAIT_TIME = 0.001
HARD_GOAL_PROB = 0.9
TRAIN_SLEEP_TIME = 0.005
BUFFER_WAIT_TIME = 0.1

