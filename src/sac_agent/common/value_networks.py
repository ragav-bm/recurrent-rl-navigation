import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .initialize import *


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        """
        Initializes the base value network.
        
        Args:
            state_space: Observation space definitions.
            activation: Activation function to be used.
        """
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  

        self.activation = activation

    def forward(self):
        """Forward pass to calculate value. Overridden by subclasses."""
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        """
        Initializes the base Q-network.
        
        Args:
            state_space: Dimensionality of environment observations.
            action_space: Dimensionality of environment actions.
            activation: Activation function reference.
        """
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]


class ValueNetwork(ValueNetworkBase):
    def __init__(self, state_space, hidden_dim, activation=F.relu, output_activation=None):
        """
        Standard Multilayer Perceptron Value Network.
        
        Args:
            state_space: Description of state dimensions.
            hidden_dim (int): Number of hidden units.
            activation (Callable): Activation function.
            output_activation (Callable, optional): Output activation function.
        """
        super().__init__(state_space, activation)

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state):
        """
        Forward pass for the Value network.
        
        Args:
            state (torch.Tensor): Current state input.
        Returns:
            torch.Tensor: Computed State-Value (V).
        """
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x        


class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        """
        Standard Multilayer Perceptron Q-Network.
        
        Args:
            state_space: Description of state dimensions.
            action_space: Description of action dimensions.
            hidden_dim (int): Number of hidden layer units.
            activation (Callable): Used activation function.
            output_activation (Callable, optional): Activation applied at the output.
        """
        super().__init__(state_space, action_space, activation)

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action):
        """
        Forward pass to compute Action-Value (Q).
        
        Args:
            state (torch.Tensor): Observation state.
            action (torch.Tensor): Given action.
        Returns:
            torch.Tensor: Computed Q value.
        """
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x        

class QNetworkLSTM(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        """
        LSTM-based Q-Network following a two-branch architecture.
        
        Args:
            state_space: Environment state space.
            action_space: Environment action space.
            hidden_dim (int): Hidden representation dimensions.
            activation (Callable): ReLu, Tanh, etc.
            output_activation (Callable, optional): Activation for Q value.
        """
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        
        Args:
            state (torch.Tensor): State sequence.
            action (torch.Tensor): Action sequence.
            last_action (torch.Tensor): Previous action sequence.
            hidden_in (tuple): Initial LSTM hidden state.
            
        Returns:
            tuple: (Computed Q value sequence, Updated hidden state)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
 
        # fc_branch = self.activation(self.ln1(self.linear1(fc_branch)))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1) 
        lstm_branch = self.activation(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim

        # lstm_branch = self.activation(self.ln2(self.linear2(lstm_branch)))  # linear layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        """
        LSTM-based Q-Network following a single-branch architecture.
        
        Args:
            state_space: Space of observations.
            action_space: Space of actions.
            hidden_dim (int): Number of hidden units.
            activation (Callable): Activation function for fully connected layers.
            output_activation (Callable, optional): Not used explicitly unless specified.
        """
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+2*self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        
        Args:
            state (torch.Tensor): Sequential states.
            action (torch.Tensor): Sequential actions.
            last_action (torch.Tensor): Sequential previous actions.
            hidden_in (tuple): Input hidden state vector.
            
        Returns:
            tuple: Computed Q-value estimates and resulting LSTM state.
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action, last_action], -1) 
        x = self.activation(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   


class QNetworkGRU(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        """
        GRU-based two branch Q-Network architecture.
        
        Args:
            state_space: Array structure describing states.
            action_space: Array structure describing actions.
            hidden_dim (int): LSTM hidden dimension.
            activation (Callable): Function such as ReLu or LeakyReLu.
            output_activation (Callable, optional): Final layer output transform.
        """
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        
        Args:
            state (torch.Tensor): State sequence tensor.
            action (torch.Tensor): Action sequence tensor.
            last_action (torch.Tensor): Prior action sequence tensor.
            hidden_in (torch.Tensor): Input GRU hidden vector.
            
        Returns:
            tuple: (Computed sequential Q values, Next GRU hidden state)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.cat([state, action], -1) 
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1) 
        lstm_branch = self.activation(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   
