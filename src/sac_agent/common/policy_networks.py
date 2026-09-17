import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import INIT_W, LOG_STD_MIN, LOG_STD_MAX, POLICY_EPSILON, ACTION_RANGE_DEFAULT, NOISE_SCALE_DEFAULT

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        """
        Initialize the base policy network.
        
        Args:
            state_space: The environment's observation space.
            action_space: The environment's action space.
            action_range (float): Scaling factor for action outputs.
        """
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        """Forward pass. To be overridden by subclasses."""
        pass
    
    def evaluate(self):
        """Evaluate state. To be overridden by subclasses."""
        pass 
    
    def get_action(self):
        """Get action for interaction. To be overridden by subclasses."""
        pass

    def sample_action(self,):
        """
        Sample a random action within the action range.
        
        Returns:
            np.ndarray: Randomly sampled action.
        """
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()

class DPG_PolicyNetwork(PolicyNetworkBase):
    """
    Deterministic policy gradient network
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W):
        """
        Initialize the DPG policy network.
        
        Args:
            state_space: State space of the environment.
            action_space: Action space of the environment.
            hidden_dim (int): Number of hidden units.
            action_range (float): Action output scaling factor.
            init_w (float): Initialization weight limit.
        """
        super().__init__(state_space, action_space, action_range)
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Current state tensor.
            
        Returns:
            torch.Tensor: Computed deterministic action.
        """
        activation=F.relu
        x = activation(self.linear1(state)) 
        x = activation(self.linear2(x))
        x = torch.tanh(self.linear3(x)).clone() # need clone to prevent in-place operation (which cause gradients not be drived)
        # x = self.linear3(x) # for simplicity, no restriction on action range

        return x

    def evaluate(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        
        Args:
            state (torch.Tensor): Current state tensor.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            torch.Tensor: Evaluated action with optional noise.
        '''
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action = self.action_range*action+noise
        return action


    def get_action(self, state, noise_scale=NOISE_SCALE_DEFAULT):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        
        Args:
            state (np.ndarray): Current state array.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            np.ndarray: Sampled action mapped to CPU.
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.mean_linear.weight.device) # state dim: (N, dim of state)
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0]

    def sample_action(self):
        """
        Sample a random action from a standard normal distribution.
        
        Returns:
            np.ndarray: Randomly sampled action scaled by action range.
        """
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()

class DPG_PolicyNetworkLSTM(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W):
        """
        Initialize LSTM DPG policy network.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_dim (int): Number of hidden units.
            action_range (float): Action output scaling factor.
            init_w (float): Initialization bound for output weights.
        """
        super().__init__(state_space, action_space, action_range)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        
        Args:
            state (torch.Tensor): Current state sequence.
            last_action (torch.Tensor): Previous action sequence.
            hidden_in (tuple): Initial LSTM hidden state.
            
        Returns:
            tuple: Computed action tensor, updated LSTM hidden state.
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # branch 1
        fc_branch = activation(self.linear1(state)) 
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        lstm_branch,  lstm_hidden = self.lstm1(lstm_branch, hidden_in)    # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1)   
        x = activation(self.linear3(merged_branch))
        x = torch.tanh(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        
        Args:
            state (torch.Tensor): Sequence of states.
            last_action (torch.Tensor): Sequence of prior actions.
            hidden_in (tuple): Initial hidden state.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            tuple: (Action tensor with noise, Updated hidden state)
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action = self.action_range*action+noise
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in,  noise_scale=NOISE_SCALE_DEFAULT):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        
        Args:
            state (np.ndarray): Current step state.
            last_action (np.ndarray): Current step previous action.
            hidden_in (tuple): Initial hidden state.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            tuple: (Sampled action as numpy array, Updated hidden state)
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device) # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        """
        Samples a random action utilizing standard normal distribution.
        
        Returns:
            np.ndarray: Output action.
        """
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()


class DPG_PolicyNetworkLSTM2(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W):
        """
        Initialize single branch LSTM DPG policy network.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_dim (int): Number of hidden units.
            action_range (float): Action output scaling factor.
        """
        super().__init__(state_space, action_space, action_range)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        
        Args:
            state (torch.Tensor): Current state sequence.
            last_action (torch.Tensor): Previous action sequence.
            hidden_in (tuple): Initial LSTM hidden state.
            
        Returns:
            tuple: Action tensor, updated LSTM hidden state.
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # single branch
        x = torch.cat([state, last_action], -1)
        x = activation(self.linear1(x))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        x,  lstm_hidden = self.lstm1(x, hidden_in)    # no activation after lstm
        x = activation(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        
        Args:
            state (torch.Tensor): Sequence of states.
            last_action (torch.Tensor): Sequence of prior actions.
            hidden_in (tuple): Hidden state input.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            tuple: Actions with noise, Output hidden state.
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action = self.action_range*action+noise
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in,  noise_scale=NOISE_SCALE_DEFAULT):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        
        Args:
            state (np.ndarray): Current step state.
            last_action (np.ndarray): Current step previous action.
            hidden_in (tuple): Initial hidden state.
            noise_scale (float): Scale of exploration noise.
            
        Returns:
            tuple: NumPy array representing chosen action, Output hidden state.
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device) # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(action.device)
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        """
        Samples a random action.
        
        Returns:
            np.ndarray: Randomly generated action.
        """
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()


        
class TD3_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        """
        Initialize the TD3 Policy Network.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_size (int): Size of hidden layers.
            action_range (float): Action output scaling factor.
            log_std_min (float): Minimum log standard deviation.
            log_std_max (float): Maximum log standard deviation.
        """
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output_linear = nn.Linear(hidden_size, self._action_dim)
        self.output_linear.weight.data.uniform_(-init_w, init_w)
        self.output_linear.bias.data.uniform_(-init_w, init_w)

        
    def forward(self, state):
        """
        Forward pass through the TD3 policy network.
        
        Args:
            state (torch.Tensor): Current state tensor.
        Returns:
            torch.Tensor: Computed action output.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        output  = torch.tanh(self.output_linear(x))
        return output
    
    def evaluate(self, state, eval_noise_scale, epsilon=POLICY_EPSILON):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        
        Args:
            state (torch.Tensor): Current state.
            eval_noise_scale (float): Scale factor for evaluation noise.
            epsilon (float): Small constant to avoid log(0).
            
        Returns:
            torch.Tensor: Evaluated action with clamped noise.
        '''
        action = self.forward(state)
        
        ''' add noise '''
        normal = Normal(0, 1)
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = self.action_range*action + noise.to(action.device)

        return action
        
    
    def get_action(self, state, explore_noise_scale):
        '''
        generate action for interaction with env
        
        Args:
            state (np.ndarray): Current state.
            explore_noise_scale (float): Scale of exploration noise.
            
        Returns:
            np.ndarray: Computed noisy action on CPU.
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.mean_linear.weight.device)
        action = self.forward(state)

        action = action.detach().cpu().numpy()[0] 

        ''' add noise '''
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action = self.action_range*action + noise.numpy()

        return action



class SAC_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        """
        Initialize SAC Policy Network predicting mean and standard deviation.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_size (int): Dimensions of hidden layers.
            action_range (float): Magnitude limit for action representation.
            log_std_min (float): Minimum threshold for log_std.
            log_std_max (float): Maximum threshold for log_std.
        """
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        """
        Forward pass computing mean and log_std for action distribution.
        
        Args:
            state (torch.Tensor): Current state.
        Returns:
            tuple: Mean and Clamped Log_std of the distribution.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=POLICY_EPSILON):
        '''
        generate sampled action with state as input wrt the policy network;
        
        Args:
            state (torch.Tensor): Observations.
            epsilon (float): Smoothing factor.
            
        Returns:
            tuple: Contains (action, log_prob, z, mean, log_std).
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(mean.device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(mean.device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=True):
        """
        Produce an action for env interaction.
        
        Args:
            state (np.ndarray): State array.
            deterministic (bool): If True, returns mean action, otherwise samples.
        Returns:
            np.ndarray: Computed action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.mean_linear.weight.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(mean.device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action



class SAC_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range, action_bias, init_w=INIT_W, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        """
        Initialize SAC Policy Network with LSTM.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_size (int): Size of LSTM and FC hidden layers.
            action_range (torch.Tensor): Range tensor for actions.
            action_bias (torch.Tensor): Bias tensor for actions.
            log_std_min (float): Min log std.
        """
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = action_range # [0.11, 2.5]
        self.action_bias = action_bias   # [0.11, 0.0]

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        
        Args:
            state (torch.Tensor): Sequence of states.
            last_action (torch.Tensor): Sequence of prior actions.
            hidden_in (tuple): Initial LSTM hidden state.
            
        Returns:
            tuple: (mean, log_std, lstm_hidden)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        # fc_branch = F.relu(self.ln1(self.linear1(state)))
        fc_branch = F.relu(self.linear1(state))

        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))

        # lstm_branch = F.relu(self.ln2(self.linear2(lstm_branch)))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, lstm_hidden
    
    def evaluate(self, state, last_action, hidden_in, epsilon=POLICY_EPSILON):
        """
        Generates sampled sequence action for network evaluation.
        
        Args:
            state (torch.Tensor): Sequence of states.
            last_action (torch.Tensor): Sequence of previous actions.
            hidden_in (tuple): LSTM hidden state.
        Returns:
            tuple: Contains (action, log_prob, z, mean, log_std, hidden_out).
        """
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(mean.device)
        action_0 = torch.tanh(mean + std * z) # Squash to [-1, 1]
        
        # Scale and Bias transformation to map to [0.0, 0.22] and [-2.5, 2.5]
        action = self.action_scale * action_0 + self.action_bias
        
        # Jacobian adjustment for log_prob calculation
        log_prob = Normal(mean, std).log_prob(mean + std * z) - \
                   torch.log(1. - action_0.pow(2) + epsilon) - \
                   torch.log(self.action_scale)
        
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        """
        Generates action for single step environment interaction.
        
        Args:
            state (np.ndarray): Current state.
            last_action (np.ndarray): Previous action.
            hidden_in (tuple): LSTM hidden state.
        Returns:
            tuple: (action array, hidden_out tuple)
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.action_scale.device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.action_scale.device)
        
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        
        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            std = log_std.exp()
            z = torch.randn_like(mean).to(mean.device)
            action_0 = torch.tanh(mean + std * z)
            
        # Map output to correct physical range
        action = self.action_scale * action_0 + self.action_bias
        return action.detach().cpu().numpy()[0][0], hidden_out
    
    # def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
    #     '''
    #     generate sampled action with state as input wrt the policy network;
    #     '''
    #     mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
    #     std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
    #     normal = Normal(0, 1)
    #     z = normal.sample(mean.shape).to(mean.device)

    #     action_0 = torch.tanh(mean + std * z)  # TanhNormal distribution as actions; reparameterization trick
    #     action = self.action_range * action_0
    #     # log_prob = Normal(mean, std).log_prob(mean + std * z.to(mean.device)) - torch.log(
    #     #     1. - action_0.pow(2) + epsilon) - np.log(self.action_range)


    #     log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(
    #         1. - action_0.pow(2) + epsilon) - torch.log(self.action_range)
        
    #     # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
    #     # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
    #     # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
    #     log_prob = log_prob.sum(dim=-1, keepdim=True)
    #     return action, log_prob, z, mean, log_std, hidden_out
    # 
    # def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
    #     mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
    #     std = log_std.exp()
        
    #     normal = Normal(0, 1)
    #     z = normal.sample(mean.shape).to(mean.device)
    #     action_0 = torch.tanh(mean + std * z) # The raw [-1, 1] output
        
    #     # Apply the transformation
    #     action = self.action_scale * action_0 + self.action_bias
        
    #     # Correct log_prob for the change of variables
    #     # Subtracting log(action_scale) is required because of the scaling
    #     log_prob = Normal(mean, std).log_prob(mean + std * z) - \
    #                torch.log(1. - action_0.pow(2) + epsilon) - \
    #                torch.log(self.action_scale)
        
    #     log_prob = log_prob.sum(dim=-1, keepdim=True)
    #     return action, log_prob, z, mean, log_std, hidden_out
    
    # # def get_action(self, state, last_action, hidden_in, deterministic=True):
    # #     state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)  # increase 2 dims to match with training data
    # #     last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)
    # #     mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
    # #     std = log_std.exp()
        
    # #     normal = Normal(0, 1)
    # #     z = normal.sample(mean.shape).to(mean.device)

    # #     # Calculation on GPU
    # #     action_gpu = self.action_range * torch.tanh(mean + std * z)

    # #     if deterministic:
    # #         # Deterministic (Evaluation): Calculate, detach, move to CPU, then numpy
    # #         action = (self.action_range * torch.tanh(mean)).detach().cpu().numpy()
    # #     else:
    # #         # Stochastic (Training): Move the sampled action to CPU, then numpy
    # #         action = action_gpu.detach().cpu().numpy()
            
    # #     return action[0][0], hidden_out

    # def get_action(self, state, last_action, hidden_in, deterministic=True):
    #     state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.action_scale.device)
    #     last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.action_scale.device)
        
    #     mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        
    #     if deterministic:
    #         action_0 = torch.tanh(mean)
    #     else:
    #         std = log_std.exp()
    #         z = torch.randn_like(mean).to(mean.device)
    #         action_0 = torch.tanh(mean + std * z)
            
    #     action = self.action_scale * action_0 + self.action_bias
    #     return action.detach().cpu().numpy()[0][0], hidden_out

    
class SAC_PolicyNetworkGRU(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=ACTION_RANGE_DEFAULT, init_w=INIT_W, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        """
        Initialize SAC Policy Network with GRU.
        
        Args:
            state_space: State space.
            action_space: Action space.
            hidden_size (int): Dimension of hidden layers.
            action_range (float): Action output magnitude limit.
        """
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.lstm1 = nn.GRU(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        
        Args:
            state (torch.Tensor): Current state sequence.
            last_action (torch.Tensor): Last action sequence.
            hidden_in (torch.Tensor): Initial GRU hidden state.
            
        Returns:
            tuple: (mean, log_std, lstm_hidden)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, lstm_hidden
    
    def evaluate(self, state, last_action, hidden_in, epsilon=POLICY_EPSILON):
        '''
        generate sampled action with state as input wrt the policy network;
        
        Args:
            state (torch.Tensor): State sequence.
            last_action (torch.Tensor): Previous action sequence.
            hidden_in (torch.Tensor): Initial GRU hidden state.
            epsilon (float): Smoothing value to avoid NaNs.
            
        Returns:
            tuple: (action, log_prob, z, mean, log_std, hidden_out)
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(mean.device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(mean.device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        """
        Calculate action output for interactions using GRU.
        
        Args:
            state (np.ndarray): Environment state.
            last_action (np.ndarray): Previous action.
            hidden_in (torch.Tensor): GRU Hidden state.
        Returns:
            tuple: NumPy action array and new hidden state.
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.mean_linear.weight.device)
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(mean.device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy() if deterministic else \
        action.detach().cpu().numpy()
        return action[0][0], hidden_out
