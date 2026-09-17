import numpy as np
import torch
from typing import List, Tuple

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import DEFAULT_SEQ_LEN, DEFAULT_BURNIN_LEN

class ReplayBufferLSTMPER:
    def __init__(self, buffer_size,max_priority,epsilon,per_alpha, device):
        self.capacity = int(buffer_size)
        self.max_priority = max_priority
        self.epsilon = epsilon
        self.device = device
        self.per_alpha = per_alpha  
        
        self.priorities = np.zeros(self.capacity, dtype=np.float32) 
        self.buffer = [] 
        self.position = 0

    def push(self, hidden_in, hidden_out, s_list, a_list, la_list, r_list, ns_list, d_list):
        """ Stores full episode lists into the buffer """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # episode is stored as a tuple of lists
        self.buffer[self.position] = (hidden_in, hidden_out, s_list, a_list, la_list, r_list, ns_list, d_list)
        self.priorities[self.position] = self.max_priority
        self.position = int((self.position + 1) % self.capacity)

    def push_all_state(self, h_list, s_list, a_list, la_list, r_list, ns_list, d_list):
        """ Stores full episode data into the buffer """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Store the entire list of hidden states
        self.buffer[self.position] = (h_list, s_list, a_list, la_list, r_list, ns_list, d_list)
        self.priorities[self.position] = self.max_priority
        self.position = int((self.position + 1) % self.capacity)

    def sample_with_burnin(self, batch_size, beta, seq_len=DEFAULT_SEQ_LEN, burnin_len=DEFAULT_BURNIN_LEN):
        """ 
        Slices a sub-sequence from episodes. 
        Drastically reduces GPU training time from ~0.9s to ~0.05s.
        """
        current_size = len(self.buffer)
        total_len = seq_len + burnin_len
        
        # 1. Prioritized Sampling Logic
        priorities_base = self.priorities[:current_size] 
        probabilities = (priorities_base ** self.per_alpha) / (priorities_base ** self.per_alpha).sum()
        indices = np.random.choice(range(current_size), batch_size, p=probabilities, replace=False)
        # indices = np.random.choice(range(current_size), batch_size, p=probabilities, replace=True)      
        # Importance Sampling Weights
        w_i_raw = (1.0 / (current_size * probabilities[indices])) ** beta
        is_weights = torch.FloatTensor(w_i_raw / w_i_raw.max()).to(self.device)

        s_batch, a_batch, r_batch, ns_batch, d_batch, la_batch = [],[],[],[],[],[]
        hi_batch, ci_batch = [], []

        for idx in indices:
            ep = self.buffer[idx] # (h_in, h_out, s_list, a_list, ...)
            ep_len = len(ep[2]) # Length of the state list

            # 2. Pick a random window within the episode
            if ep_len > total_len:
                start = np.random.randint(0, ep_len - total_len)
                end = start + total_len
            else:
                start, end = 0, ep_len

            # 3. Slice the lists [start:end]
            s_batch.append(ep[2][start:end])
            a_batch.append(ep[3][start:end])
            la_batch.append(ep[4][start:end])
            r_batch.append(ep[5][start:end])
            ns_batch.append(ep[6][start:end])
            d_batch.append(ep[7][start:end])
            
            # Use initial hidden state of the episode (hi_tup is ep[0])
            # For deeper burn-in accuracy, you'd store hidden states per step, 
            # but using the start-of-episode h_in is standard for basic burn-in.
            hi_batch.append(ep[0][0].squeeze(0)) # h
            ci_batch.append(ep[0][1].squeeze(0)) # c

        # 4. Helper to pad and convert to tensor
        def to_tensor(data_list, is_done_mask=False, is_reward=False):
            padded = []
            for item in data_list:
                item_np = np.array(item)
                # Pad if the episode slice is shorter than total_len
                if len(item_np) < total_len:
                    diff = total_len - len(item_np)
                    if is_done_mask:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=1.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=1.0)
                    elif is_reward:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=0.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=0.0)
                    else:
                        # EDGE pad states and actions to prevent training the actor on zeros!
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='edge')
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='edge')
                padded.append(item_np)
            return torch.FloatTensor(np.array(padded)).to(self.device)

        h_in = torch.stack(hi_batch, dim=1).to(self.device)
        c_in = torch.stack(ci_batch, dim=1).to(self.device)

        return (h_in, c_in), \
               to_tensor(s_batch), to_tensor(a_batch), to_tensor(la_batch), \
               to_tensor(r_batch, is_reward=True), to_tensor(ns_batch), to_tensor(d_batch, is_done_mask=True), \
               is_weights, indices

    def sample_with_burnin_all_state(self, batch_size, beta, seq_len=32, burnin_len=8):
        """ 
        Slices a sub-sequence from episodes. 
        Fixed: Now retrieves the hidden state corresponding to the start of the slice.
        """
        current_size = len(self.buffer)
        total_len = seq_len + burnin_len
        
        # 1. Prioritized Sampling Logic
        priorities_base = self.priorities[:current_size] 
        probabilities = (priorities_base ** self.per_alpha) / (priorities_base ** self.per_alpha).sum()
        indices = np.random.choice(range(current_size), batch_size, p=probabilities, replace=False)
        
        # Importance Sampling Weights
        w_i_raw = (1.0 / (current_size * probabilities[indices])) ** beta
        is_weights = torch.FloatTensor(w_i_raw / w_i_raw.max()).to(self.device)

        s_batch, a_batch, r_batch, ns_batch, d_batch, la_batch = [], [], [], [], [], []
        hi_batch, ci_batch = [], []

        for idx in indices:
            ep = self.buffer[idx] # Expects: (h_list, s_list, a_list, la_list, r_list, ns_list, d_list)
            ep_len = len(ep[1]) # Based on length of s_list

            # 2. Pick a random window within the episode
            if ep_len > total_len:
                start = np.random.randint(0, ep_len - total_len)
                end = start + total_len
            else:
                start, end = 0, ep_len

            # 3. Slice the lists [start:end]
            # Note: Indexing shifts to accommodate the new h_list at index 0
            s_batch.append(ep[1][start:end])
            a_batch.append(ep[2][start:end])
            la_batch.append(ep[3][start:end])
            r_batch.append(ep[4][start:end])
            ns_batch.append(ep[5][start:end])
            d_batch.append(ep[6][start:end])
            
            # THE SOLUTION: Retrieve the hidden state (h, c) from the 'start' index
            # ep[0] is the h_list containing (h_tensor, c_tensor) per step
            h_at_start, c_at_start = ep[0][start]
            hi_batch.append(h_at_start.squeeze(0)) # Shape: (Layers, Hidden)
            ci_batch.append(c_at_start.squeeze(0))

        # 4. Helper to pad and convert to tensor
        def to_tensor(data_list, is_done_mask=False, is_reward=False):
            padded = []
            for item in data_list:
                item_np = np.array(item)
                # Pad if the episode slice is shorter than total_len
                if len(item_np) < total_len:
                    diff = total_len - len(item_np)
                    if is_done_mask:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=1.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=1.0)
                    elif is_reward:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=0.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=0.0)
                    else:
                        # EDGE pad states and actions to prevent training the actor on zeros!
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='edge')
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='edge')
                padded.append(item_np)
            return torch.FloatTensor(np.array(padded)).to(self.device)

        # Process hidden states to (Layers, Batch, Hidden)
        h_in = torch.stack(hi_batch, dim=1).to(self.device)
        c_in = torch.stack(ci_batch, dim=1).to(self.device)

        return (h_in, c_in), \
               to_tensor(s_batch), to_tensor(a_batch), to_tensor(la_batch), \
               to_tensor(r_batch, is_reward=True), to_tensor(ns_batch), to_tensor(d_batch, is_done_mask=True), \
               is_weights, indices

    def update_priorities(self, indices, errors):
        self.priorities[indices] = np.abs(errors) + self.epsilon
        self.max_priority = max(self.max_priority, self.priorities[indices].max())

    def __len__(self): return len(self.buffer)

class ReplayBufferLSTM:
    """
    Same interface as ReplayBufferLSTMPER, but uses UNIFORM sampling.
    No priorities, no importance sampling weights correction.
    This is the SAC+LSTM (no PER) baseline.
    """
    def __init__(self, buffer_size, device):
        self.capacity = int(buffer_size)
        self.device = device
        self.buffer = []
        self.position = 0

    def push_all_state(self, h_list, s_list, a_list, la_list, r_list, ns_list, d_list):
        """Same interface as ReplayBufferLSTMPER.push_all_state"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (h_list, s_list, a_list, la_list, r_list, ns_list, d_list)
        self.position = int((self.position + 1) % self.capacity)

    def sample_with_burnin_all_state(self, batch_size, beta=None, seq_len=DEFAULT_SEQ_LEN, burnin_len=DEFAULT_BURNIN_LEN):
        """
        Same return signature as ReplayBufferLSTMPER.sample_with_burnin_all_state.
        beta is IGNORED (accepted for interface compatibility).
        is_weights are all 1.0 (uniform = no correction needed).
        """
        current_size = len(self.buffer)
        total_len = seq_len + burnin_len

        # UNIFORM sampling (the only difference from PER)
        if batch_size > current_size:
            indices = np.random.choice(range(current_size), batch_size, replace=True)
        else:
            indices = np.random.choice(range(current_size), batch_size, replace=False)

        # All weights = 1.0 (no importance sampling correction)
        is_weights = torch.ones(batch_size).to(self.device)

        s_batch, a_batch, r_batch, ns_batch, d_batch, la_batch = [], [], [], [], [], []
        hi_batch, ci_batch = [], []

        for idx in indices:
            ep = self.buffer[idx]
            ep_len = len(ep[1])

            if ep_len > total_len:
                start = np.random.randint(0, ep_len - total_len)
                end = start + total_len
            else:
                start, end = 0, ep_len

            s_batch.append(ep[1][start:end])
            a_batch.append(ep[2][start:end])
            la_batch.append(ep[3][start:end])
            r_batch.append(ep[4][start:end])
            ns_batch.append(ep[5][start:end])
            d_batch.append(ep[6][start:end])

            h_at_start, c_at_start = ep[0][start]
            hi_batch.append(h_at_start.squeeze(0))
            ci_batch.append(c_at_start.squeeze(0))

        def to_tensor(data_list, is_done_mask=False, is_reward=False):
            padded = []
            for item in data_list:
                item_np = np.array(item)
                if len(item_np) < total_len:
                    diff = total_len - len(item_np)
                    if is_done_mask:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=1.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=1.0)
                    elif is_reward:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='constant', constant_values=0.0)
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='constant', constant_values=0.0)
                    else:
                        if item_np.ndim > 1:
                            item_np = np.pad(item_np, ((0, diff), (0, 0)), mode='edge')
                        else:
                            item_np = np.pad(item_np, (0, diff), mode='edge')
                padded.append(item_np)
            return torch.FloatTensor(np.array(padded)).to(self.device)

        h_in = torch.stack(hi_batch, dim=1).to(self.device)
        c_in = torch.stack(ci_batch, dim=1).to(self.device)

        return (h_in, c_in), \
               to_tensor(s_batch), to_tensor(a_batch), to_tensor(la_batch), \
               to_tensor(r_batch, is_reward=True), to_tensor(ns_batch), \
               to_tensor(d_batch, is_done_mask=True), \
               is_weights, indices

    def update_priorities(self, indices, errors):
        """No-op. PER interface compatibility."""
        pass

    def __len__(self):
        return len(self.buffer)
    


class ReplayBufferSimple:
    """Standard replay buffer for non-recurrent SAC (MLP)."""
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.capacity = int(buffer_size)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size