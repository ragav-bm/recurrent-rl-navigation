import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import (
    ALPHA_MIN, ALPHA_MAX, TARGET_ENTROPY,
    LR_SCHEDULER_START_FACTOR, LR_SCHEDULER_ETA_MIN
)


class SAC_Trainer_MLP():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim,
                 action_scale, alpha, gamma, tau, alpha_lr,
                 reward_scale, q_lr, policy_lr, grad_clip,
                 device, action_bias, writer=None):
        """
        SAC Trainer using standard MLP (no LSTM/recurrence).
        """
        self.replay_buffer = replay_buffer
        self.device = device
        self.writer = writer
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.gradient_clip_norm = grad_clip
        self.action_scale = action_scale
        self.action_bias = action_bias

        # --- Build Networks ---
        from sac_agent.common.value_networks import QNetwork
        from sac_agent.common.policy_networks import SAC_PolicyNetwork

        class DimPlaceholder:
            def __init__(self, dim):
                self.shape = (dim,)

        state_space_ph = DimPlaceholder(state_dim)
        action_space_ph = DimPlaceholder(action_dim)

        # Use the simple MLP Q-network and policy
        self.q_net1 = QNetwork(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.q_net2 = QNetwork(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.target_q_net1 = QNetwork(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.target_q_net2 = QNetwork(state_space_ph, action_space_ph, hidden_dim).to(device)

        # SAC_PolicyNetwork uses action_range as a float scalar
        self.policy_net = SAC_PolicyNetwork(
            state_space_ph, action_space_ph, hidden_dim, action_range=1.0
        ).to(device)

        # Sync targets
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # --- Optimizers ---
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # --- Alpha (entropy temperature) ---
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = TARGET_ENTROPY if TARGET_ENTROPY is not None else -action_dim
        self.alpha = torch.clamp(self.log_alpha.exp(), min=ALPHA_MIN, max=ALPHA_MAX)

        self.total_updates = 0

    def update(self, batch_size):
        """Perform one SAC update step."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Scale rewards
        rewards = rewards * self.reward_scale

        # --- Alpha Update ---
        # Get current policy actions and log_probs
        new_actions, log_pi, _, _, _ = self.policy_net.evaluate(states)
        # Scale actions to physical range
        scaled_new_actions = self.action_scale * new_actions + self.action_bias

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(np.log(ALPHA_MIN), np.log(ALPHA_MAX))
        self.alpha = torch.clamp(self.log_alpha.exp().detach(), min=ALPHA_MIN, max=ALPHA_MAX)

        # --- Critic Update ---
        # Scale the stored actions to physical range for Q input
        scaled_actions = self.action_scale * actions + self.action_bias

        curr_q1 = self.q_net1(states, scaled_actions)
        curr_q2 = self.q_net2(states, scaled_actions)

        with torch.no_grad():
            next_new_actions, next_log_pi, _, _, _ = self.policy_net.evaluate(next_states)
            scaled_next_actions = self.action_scale * next_new_actions + self.action_bias

            target_q1 = self.target_q_net1(next_states, scaled_next_actions)
            target_q2 = self.target_q_net2(next_states, scaled_next_actions)
            target_q = torch.min(target_q1, target_q2)
            expected_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_pi)

        q1_loss = F.smooth_l1_loss(curr_q1, expected_q)
        q2_loss = F.smooth_l1_loss(curr_q2, expected_q)

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.gradient_clip_norm)
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.gradient_clip_norm)
        self.q_optimizer2.step()

        # --- Actor Update ---
        for p in self.q_net1.parameters():
            p.requires_grad = False
        for p in self.q_net2.parameters():
            p.requires_grad = False

        q1_pi = self.q_net1(states, scaled_new_actions)
        q2_pi = self.q_net2(states, scaled_new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_norm)
        self.policy_optimizer.step()

        for p in self.q_net1.parameters():
            p.requires_grad = True
        for p in self.q_net2.parameters():
            p.requires_grad = True

        # --- Soft Update Targets ---
        self._soft_update(self.q_net1, self.target_q_net1)
        self._soft_update(self.q_net2, self.target_q_net2)

        self.total_updates += 1

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), self.alpha.item()

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_action(self, state, deterministic=False):
        """Get action from the policy (no hidden state needed)."""
        action = self.policy_net.get_action(state, deterministic=deterministic)
        # Scale from [-1, 1] to physical range
        scaled_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return scaled_action

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path + '_q1.pth')
        torch.save(self.q_net2.state_dict(), path + '_q2.pth')
        torch.save(self.policy_net.state_dict(), path + '_policy.pth')
        torch.save(self.target_q_net1.state_dict(), path + '_target_q1.pth')
        torch.save(self.target_q_net2.state_dict(), path + '_target_q2.pth')
        print(f"Models saved to {path}")

    def load_model(self, path):
        device = self.device
        self.policy_net.load_state_dict(torch.load(path + '_policy.pth', map_location=device))
        self.q_net1.load_state_dict(torch.load(path + '_q1.pth', map_location=device))
        self.q_net2.load_state_dict(torch.load(path + '_q2.pth', map_location=device))
        self.target_q_net1.load_state_dict(torch.load(path + '_target_q1.pth', map_location=device))
        self.target_q_net2.load_state_dict(torch.load(path + '_target_q2.pth', map_location=device))
        print(f"Models loaded from {path}")