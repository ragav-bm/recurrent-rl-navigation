#!/usr/bin/env python3
"""
SAC-Discrete with MLP (no memory) for discrete action spaces.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscretePolicyNetworkMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits_head(x)
        action_probs = F.softmax(logits, dim=-1)
        log_action_probs = torch.log(torch.clamp(action_probs, min=1e-8))
        return action_probs, log_action_probs

    def get_action(self, state, deterministic=False):
        device = next(self.parameters()).device
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state_t)
        action_probs = action_probs.squeeze(0)

        if deterministic:
            return torch.argmax(action_probs).item()
        else:
            return Categorical(action_probs).sample().item()


class DiscreteQNetworkMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q_head(x)


class SAC_Discrete_Trainer_MLP:
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim,
                 alpha, gamma, tau, alpha_lr, reward_scale,
                 q_lr, policy_lr, grad_clip, device,
                 target_entropy_ratio=0.98):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.total_updates = 0

        self.policy_net = DiscretePolicyNetworkMLP(
            state_dim, action_dim, hidden_dim).to(device)
        self.q_net1 = DiscreteQNetworkMLP(state_dim, action_dim, hidden_dim).to(device)
        self.q_net2 = DiscreteQNetworkMLP(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net1 = DiscreteQNetworkMLP(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net2 = DiscreteQNetworkMLP(state_dim, action_dim, hidden_dim).to(device)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.target_entropy = np.log(action_dim) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)
        self.q1_optimizer = torch.optim.Adam(
            self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(
            self.q_net2.parameters(), lr=q_lr)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr)

    def get_action(self, state, deterministic=False):
        return self.policy_net.get_action(state, deterministic)

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)
        # actions: (batch, action_dim) one-hot → convert to indices
        action_indices = torch.argmax(actions, dim=-1)  # (batch,)

        with torch.no_grad():
            next_probs, next_log_probs = self.policy_net(next_states)
            target_q1 = self.target_q_net1(next_states)
            target_q2 = self.target_q_net2(next_states)
            target_q_min = torch.min(target_q1, target_q2)

            target_v = (next_probs * (
                target_q_min - self.alpha * next_log_probs
            )).sum(dim=-1)

            q_target = (rewards.squeeze() * self.reward_scale +
                       (1.0 - dones.squeeze()) * self.gamma * target_v)

        current_q1 = self.q_net1(states)
        current_q2 = self.q_net2(states)
        q1_taken = current_q1.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1)
        q2_taken = current_q2.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1)

        q1_loss = F.mse_loss(q1_taken, q_target)
        q2_loss = F.mse_loss(q2_taken, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.grad_clip)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.grad_clip)
        self.q2_optimizer.step()

        # Policy update
        action_probs, log_action_probs = self.policy_net(states)
        with torch.no_grad():
            q1_pi = self.q_net1(states)
            q2_pi = self.q_net2(states)
            min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (action_probs * (
            self.alpha * log_action_probs - min_q_pi
        )).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        # Alpha update
        with torch.no_grad():
            entropy = -(action_probs * log_action_probs).sum(dim=-1).mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Target update
        for tp, p in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.total_updates += 1

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, "policy.pt"))
        torch.save(self.q_net1.state_dict(), os.path.join(path, "q1.pt"))
        torch.save(self.q_net2.state_dict(), os.path.join(path, "q2.pt"))

    def load_model(self, path):
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, "policy.pt"), map_location=self.device))
        self.q_net1.load_state_dict(
            torch.load(os.path.join(path, "q1.pt"), map_location=self.device))
        self.q_net2.load_state_dict(
            torch.load(os.path.join(path, "q2.pt"), map_location=self.device))