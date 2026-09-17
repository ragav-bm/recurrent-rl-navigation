#!/usr/bin/env python3
"""
SAC-Discrete with LSTM for discrete action spaces.
Uses categorical policy instead of Gaussian.
Q-networks output Q-values for ALL actions (no action input).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════
# POLICY NETWORK (Categorical)
# ═══════════════════════════════════════════════════════════

class DiscretePolicyNetworkLSTM(nn.Module):
    """
    Outputs action probabilities via softmax over logits.
    LSTM processes (state, last_action) sequences.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lstm_hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        input_dim = state_dim + action_dim  # state + last_action (one-hot)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, last_action, hidden_state):
        """
        Args:
            state: (batch, seq_len, state_dim)
            last_action: (batch, seq_len, action_dim) one-hot
            hidden_state: tuple(h, c) each (1, batch, lstm_hidden_dim)
        Returns:
            action_probs: (batch, seq_len, action_dim)
            log_action_probs: (batch, seq_len, action_dim)
            hidden_state: updated (h, c)
        """
        x = torch.cat([state, last_action], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x, hidden_state = self.lstm(x, hidden_state)
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.logits_head(x)

        # Stable softmax
        action_probs = F.softmax(logits, dim=-1)
        # Clamp to avoid log(0)
        action_probs_clamped = torch.clamp(action_probs, min=1e-8)
        log_action_probs = torch.log(action_probs_clamped)

        return action_probs, log_action_probs, hidden_state

    def get_action(self, state, last_action, hidden_state, deterministic=False):
        """
        Single-step action selection for environment interaction.
        Returns:
            action: int (discrete action index)
            hidden_state: updated (h, c)
        """
        device = next(self.parameters()).device
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        la_t = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(device)

        action_probs, _, hidden_state = self.forward(state_t, la_t, hidden_state)
        action_probs = action_probs.squeeze(0).squeeze(0)  # (action_dim,)

        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()

        return action, hidden_state

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device))


# ═══════════════════════════════════════════════════════════
# Q-NETWORK (Outputs Q-values for ALL actions)
# ═══════════════════════════════════════════════════════════

class DiscreteQNetworkLSTM(nn.Module):
    """
    Q(state) → [Q(s,a0), Q(s,a1), ..., Q(s,an)]
    No action input needed — outputs Q-values for all actions at once.
    Uses last_action as temporal context for LSTM.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lstm_hidden_dim):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        input_dim = state_dim + action_dim  # state + last_action (temporal context)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)  # Q per action

    def forward(self, state, last_action, hidden_state):
        """
        Returns:
            q_values: (batch, seq_len, action_dim) — Q for each action
            hidden_state: updated (h, c)
        """
        x = torch.cat([state, last_action], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x, hidden_state = self.lstm(x, hidden_state)
        x = F.relu(self.ln2(self.fc2(x)))
        q_values = self.q_head(x)
        return q_values, hidden_state


# ═══════════════════════════════════════════════════════════
# SAC-DISCRETE TRAINER
# ═══════════════════════════════════════════════════════════

class SAC_Discrete_Trainer:
    """
    SAC for discrete actions with LSTM + PER + Burn-in support.
    Key differences from continuous SAC:
      - Policy: Categorical (softmax) instead of Gaussian
      - Q-networks: output vector (all actions) instead of scalar
      - No reparameterization trick — use expectation over all actions
      - Target entropy: log(|A|) * ratio instead of -|A|
    """
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
        self.hidden_dim = hidden_dim
        self.replay_buffer = replay_buffer
        self.total_updates = 0

        # ─── Networks ─────────────────────────────────
        self.policy_net = DiscretePolicyNetworkLSTM(
            state_dim, action_dim, hidden_dim, hidden_dim).to(device)

        self.q_net1 = DiscreteQNetworkLSTM(
            state_dim, action_dim, hidden_dim, hidden_dim).to(device)
        self.q_net2 = DiscreteQNetworkLSTM(
            state_dim, action_dim, hidden_dim, hidden_dim).to(device)

        self.target_q_net1 = DiscreteQNetworkLSTM(
            state_dim, action_dim, hidden_dim, hidden_dim).to(device)
        self.target_q_net2 = DiscreteQNetworkLSTM(
            state_dim, action_dim, hidden_dim, hidden_dim).to(device)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # ─── Target Entropy ──────────────────────────
        # For discrete: target_entropy = log(|A|) * ratio
        self.target_entropy = np.log(action_dim) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        # ─── Optimizers ──────────────────────────────
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)
        self.q1_optimizer = torch.optim.Adam(
            self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(
            self.q_net2.parameters(), lr=q_lr)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, beta=0.4, seq_len=8, burnin_len=4):
        """
        SAC-Discrete update with sequence sampling, burn-in, and PER.
        """
        # ─── Sample from buffer ──────────────────────
        # FIX: Correct unpacking order — buffer returns (h, s, a, la, r, ns, d, is_weights, indices)
        h_states, states, actions, last_actions, rewards, next_states, dones, is_weights, indices = \
            self.replay_buffer.sample_with_burnin_all_state(batch_size, beta, seq_len, burnin_len)

        total_len = burnin_len + seq_len
        b = states.shape[0]

        # ═══════════════════════════════════════════════
        # BURN-IN: Warm up LSTM hidden states
        # ═══════════════════════════════════════════════
        if burnin_len > 0:
            burnin_s = states[:, :burnin_len, :]
            burnin_la = last_actions[:, :burnin_len, :]

            with torch.no_grad():
                _, _, h_policy = self.policy_net(burnin_s, burnin_la, h_states)
                _, h_q1 = self.q_net1(burnin_s, burnin_la, h_states)
                _, h_q2 = self.q_net2(burnin_s, burnin_la, h_states)
                _, h_tq1 = self.target_q_net1(burnin_s, burnin_la, h_states)
                _, h_tq2 = self.target_q_net2(burnin_s, burnin_la, h_states)
        else:
            h_policy = h_states
            h_q1 = h_states
            h_q2 = h_states
            h_tq1 = h_states
            h_tq2 = h_states

        # Detach hidden states for training portion
        h_policy = (h_policy[0].detach(), h_policy[1].detach())
        h_q1 = (h_q1[0].detach(), h_q1[1].detach())
        h_q2 = (h_q2[0].detach(), h_q2[1].detach())
        h_tq1 = (h_tq1[0].detach(), h_tq1[1].detach())
        h_tq2 = (h_tq2[0].detach(), h_tq2[1].detach())

        # ═══════════════════════════════════════════════
        # TRAINING SEQUENCE (after burn-in)
        # ═══════════════════════════════════════════════
        train_s = states[:, burnin_len:, :]            # (batch, seq_len, state_dim)
        train_ns = next_states[:, burnin_len:, :]      # (batch, seq_len, state_dim)
        train_la = last_actions[:, burnin_len:, :]     # (batch, seq_len, action_dim)
        train_a = actions[:, burnin_len:, :]           # (batch, seq_len, action_dim) one-hot
        train_r = rewards[:, burnin_len:]              # (batch, seq_len)
        train_d = dones[:, burnin_len:]                # (batch, seq_len)

        # Convert one-hot actions to integer indices
        train_a_idx = torch.argmax(train_a, dim=-1)    # (batch, seq_len) integers

        # The action taken IS the next step's last_action
        train_next_la = train_a

        # ─── Q-Network Update ────────────────────────
        with torch.no_grad():
            # Policy probabilities for next states
            next_probs, next_log_probs, _ = self.policy_net(
                train_ns, train_next_la, h_policy)

            # Target Q-values for next states
            target_q1, _ = self.target_q_net1(train_ns, train_next_la, h_tq1)
            target_q2, _ = self.target_q_net2(train_ns, train_next_la, h_tq2)
            target_q_min = torch.min(target_q1, target_q2)

            # Soft state value: V(s') = Σ π(a|s') [Q(s',a) - α log π(a|s')]
            target_v = (next_probs * (
                target_q_min - self.alpha * next_log_probs
            )).sum(dim=-1)  # (batch, seq_len)

            # TD target
            q_target = (train_r * self.reward_scale +
                       (1.0 - train_d) * self.gamma * target_v)

        # Current Q-values for all actions
        current_q1, _ = self.q_net1(train_s, train_la, h_q1)  # (batch, seq_len, action_dim)
        current_q2, _ = self.q_net2(train_s, train_la, h_q2)

        # Select Q-values for actions actually taken
        a_idx = train_a_idx.unsqueeze(-1)  # (batch, seq_len, 1)
        q1_taken = current_q1.gather(-1, a_idx).squeeze(-1)  # (batch, seq_len)
        q2_taken = current_q2.gather(-1, a_idx).squeeze(-1)

        # PER-weighted loss
        td_error1 = q1_taken - q_target
        td_error2 = q2_taken - q_target

        # FIX: Ensure is_weights is a tensor before calling unsqueeze
        if isinstance(is_weights, np.ndarray):
            is_weights = torch.FloatTensor(is_weights).to(self.device)

        weights = is_weights.unsqueeze(-1).expand_as(td_error1)  # (batch, seq_len)

        q1_loss = (weights * td_error1.pow(2)).mean()
        q2_loss = (weights * td_error2.pow(2)).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.grad_clip)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.grad_clip)
        self.q2_optimizer.step()

        # ─── Policy Update ───────────────────────────
        # Re-compute action probs (with gradient this time)
        action_probs, log_action_probs, _ = self.policy_net(
            train_s, train_la, h_policy)

        with torch.no_grad():
            q1_pi, _ = self.q_net1(train_s, train_la, h_q1)
            q2_pi, _ = self.q_net2(train_s, train_la, h_q2)
            min_q_pi = torch.min(q1_pi, q2_pi)  # (batch, seq_len, action_dim)

        # Policy loss: Σ π(a|s) [α log π(a|s) - Q(s,a)]
        policy_loss = (action_probs * (
            self.alpha * log_action_probs - min_q_pi
        )).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        # ─── Alpha Update ────────────────────────────
        # Entropy: H = -Σ π(a) log π(a)
        with torch.no_grad():
            entropy = -(action_probs * log_action_probs).sum(dim=-1).mean()

        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # ─── Update PER priorities ───────────────────
        priorities = (td_error1.detach().abs().mean(dim=-1) + 1e-6).cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # ─── Soft-update targets ─────────────────────
        for target_param, param in zip(
                self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(
                self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

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