import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import LR_SCHEDULER_T0, LR_SCHEDULER_TOTAL_UPDATE, LR_SCHEDULER_T_MULT, LR_SCHEDULER_ETA_MIN, LR_SCHEDULER_START_FACTOR, ALPHA_MIN, ALPHA_MAX, TARGET_ENTROPY

class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_scale, alpha,gamma,tau, alpa_learning_rate ,Reward_Scale,Q_LEARNING_RATE,POLICY_LEARNING_RATE,Grdient_clip_max_norm, device, action_bias, writer=None, warmup_updates=LR_SCHEDULER_T0, total_updates=LR_SCHEDULER_TOTAL_UPDATE):
        """
        Initializes the Soft Actor-Critic (SAC) Trainer with LSTM networks.
        
        Args:
            replay_buffer: Experience replay buffer supporting sequences.
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            hidden_dim (int): Dimensionality of hidden layers.
            action_scale (torch.Tensor): Scaling factor for actions.
            alpha (float): Initial entropy temperature parameter.
            gamma (float): Discount factor.
            tau (float): Polyak averaging parameter for target networks.
            alpa_learning_rate (float): Learning rate for alpha.
            Reward_Scale (float): Reward scaling factor.
            Q_LEARNING_RATE (float): Learning rate for critic networks.
            POLICY_LEARNING_RATE (float): Learning rate for actor network.
            Grdient_clip_max_norm (float): Gradient clipping maximum norm.
            device (torch.device): Device to run computations on (CPU/GPU).
            action_bias (torch.Tensor): Bias applied to actions.
            writer (SummaryWriter, optional): TensorBoard writer.
            warmup_updates (int): Number of steps for learning rate warmup.
            total_updates (int): Total expected training steps.
        """
        self.replay_buffer = replay_buffer
        self.device = device
        self.writer = writer
        
        # Hyperparameters
        # self.alpha = alpha  # Entropy coefficient
        self.gamma = gamma
        self.tau = tau
        self.alpa_learning_rate = alpa_learning_rate
        self.reward_scale = Reward_Scale
        self.Q_learning_rate = Q_LEARNING_RATE
        self.policy_learning_rate = POLICY_LEARNING_RATE
        self.gradient_clip_norm = Grdient_clip_max_norm # New gradient clipping norm
        self.init_alpha = alpha
        self.action_scale = action_scale
        self.action_bias = action_bias

        # 1. Initialize Networks (Using names consistently)
        from sac_agent.common.value_networks import QNetworkLSTM
        from sac_agent.common.policy_networks import SAC_PolicyNetworkLSTM
        
        # We'll use DimPlaceholder or just pass the int if your classes support it
        class DimPlaceholder:
            def __init__(self, dim): self.shape = (dim,)
        
        state_space_ph = DimPlaceholder(state_dim)
        action_space_ph = DimPlaceholder(action_dim)

        self.q_net1 = QNetworkLSTM(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.q_net2 = QNetworkLSTM(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.target_q_net1 = QNetworkLSTM(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.target_q_net2 = QNetworkLSTM(state_space_ph, action_space_ph, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM(state_space_ph, action_space_ph, hidden_dim,  self.action_scale, self.action_bias).to(device)
        self.target_policy_net = SAC_PolicyNetworkLSTM(state_space_ph, action_space_ph, hidden_dim,self.action_scale, self.action_bias).to(device)
        # Sync Target Networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict()) # Sync policy

        # 2. Optimizers
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.Q_learning_rate) # Faster Critic
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.Q_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate) # Slower Actor

        # 3. Learning Rate Schedulers (Linear Warmup + Cosine Annealing)
        self.warmup_updates = warmup_updates
        self.max_updates = total_updates

        self.log_alpha = torch.tensor([np.log(self.init_alpha)], requires_grad=True, device=device)         
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpa_learning_rate)

        self.alpha_scheduler = SequentialLR(
            self.alpha_optimizer,
            schedulers=[
                LinearLR(self.alpha_optimizer, start_factor=LR_SCHEDULER_START_FACTOR, total_iters=warmup_updates),
                CosineAnnealingLR(self.alpha_optimizer, T_max=total_updates - warmup_updates, eta_min=LR_SCHEDULER_ETA_MIN)
            ],
            milestones=[warmup_updates]
        )
        
         # Q-network schedulers
        self.q_scheduler1 = SequentialLR(
            self.q_optimizer1,
            [LinearLR(self.q_optimizer1, start_factor=LR_SCHEDULER_START_FACTOR, end_factor=1.0, total_iters=warmup_updates),
             CosineAnnealingLR(self.q_optimizer1, T_max=total_updates - warmup_updates, eta_min=LR_SCHEDULER_ETA_MIN)],
            milestones=[warmup_updates]
        )
        self.q_scheduler2 = SequentialLR(
            self.q_optimizer2,
            [LinearLR(self.q_optimizer2, start_factor=LR_SCHEDULER_START_FACTOR, end_factor=1.0, total_iters=warmup_updates),
             CosineAnnealingLR(self.q_optimizer2, T_max=total_updates - warmup_updates, eta_min=LR_SCHEDULER_ETA_MIN)],
            milestones=[warmup_updates]
        )
        
        # Policy network scheduler
        self.policy_scheduler = SequentialLR(
            self.policy_optimizer,
            [LinearLR(self.policy_optimizer, start_factor=LR_SCHEDULER_START_FACTOR, end_factor=1.0, total_iters=warmup_updates),
             CosineAnnealingLR(self.policy_optimizer, T_max=total_updates - warmup_updates, eta_min=LR_SCHEDULER_ETA_MIN)],
            milestones=[warmup_updates]
        )


        self.target_entropy = TARGET_ENTROPY if TARGET_ENTROPY is not None else -action_dim
        self.alpha = torch.clamp(self.log_alpha.exp(), min=ALPHA_MIN, max=ALPHA_MAX) # Keep this
        # 4. Update counter for schedulers
        self.total_updates = 0

    def update(self, batch_size, beta, seq_len=64, burnin_len=20):
            """
            Performs one step of policy and value network updates using a batch of sampled transitions.
            
            Args:
                batch_size (int): Number of samples in the batch.
                beta (float): Importance sampling parameter for PER.
                seq_len (int): Length of the recurrent sequence.
                burnin_len (int): Burn-in length for RNN hidden states.
                
            Returns:
                tuple: (q1_loss, q2_loss, policy_loss, alpha_value, q1_lr, q2_lr, policy_lr, alpha_lr)
            """
            import torch.nn.functional as F # Imported here for Huber Loss
            
            # 1. Sample trajectory and the hidden state at the START of burn-in
            h_in, s, a, la, r, ns, d, weights, indices = self.replay_buffer.sample_with_burnin_all_state(
                batch_size, beta, seq_len=seq_len, burnin_len=burnin_len
            )

            # ------------------------------------------------------------------
            # 1. Entropy (Alpha) Update
            # ------------------------------------------------------------------
            # Unroll Online Policy to get current log probabilities
            new_a_pi, log_pi_curr, _, _, _, _ = self.policy_net.evaluate(s, la, h_in)
            log_pi_curr_sliced = log_pi_curr[:, burnin_len:, :]

            alpha_loss = -(self.log_alpha * (log_pi_curr_sliced + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # self.alpha_scheduler.step() 
            # self.alpha_scheduler.step() 
            
            # Prevent the underlying log_alpha from drifting into extreme negative/positive values
            with torch.no_grad():
                self.log_alpha.clamp_(np.log(ALPHA_MIN), np.log(ALPHA_MAX))

            self.alpha = torch.clamp(self.log_alpha.exp().detach(), min=ALPHA_MIN, max=ALPHA_MAX)

            # ------------------------------------------------------------------
            # 2. Critic Update (Independent Unrolling)
            # ------------------------------------------------------------------
            # Online Critics unroll independently from h_in
            curr_q1, _ = self.q_net1(s, a, la, h_in)
            curr_q2, _ = self.q_net2(s, a, la, h_in)
            
            curr_q1_sliced = curr_q1[:, burnin_len:, :]
            curr_q2_sliced = curr_q2[:, burnin_len:, :]

            with torch.no_grad():
                # Use TARGET Policy to get next actions
                new_a_next, log_pi_next, _, _, _, _ = self.target_policy_net.evaluate(ns, a, h_in)
                
                # Target Critics unroll from h_in
                target_q1, _ = self.target_q_net1(ns, new_a_next, a, h_in)
                target_q2, _ = self.target_q_net2(ns, new_a_next, a, h_in)
                target_q = torch.min(target_q1, target_q2)
                
                target_q_sliced = target_q[:, burnin_len:, :]
                log_pi_next_sliced = log_pi_next[:, burnin_len:, :]
                
                r_sliced = r[:, burnin_len:].unsqueeze(-1) if r.dim() == 2 else r[:, burnin_len:, :]
                d_sliced = d[:, burnin_len:].unsqueeze(-1) if d.dim() == 2 else d[:, burnin_len:, :]

                # Bellman Equation
                scaled_r = r_sliced * self.reward_scale
                expected_q = scaled_r + (1 - d_sliced) * self.gamma * (target_q_sliced - self.alpha * log_pi_next_sliced)
            
            # Importance Sampling Weights
            w = weights.view(-1, 1, 1)
            
            # --- HUBER LOSS IMPLEMENTATION ---
            # Replaced .pow(2) with Smooth L1 Loss to prevent gradient explosions
            q1_loss = (w * F.smooth_l1_loss(curr_q1_sliced, expected_q, reduction='none')).mean()
            q2_loss = (w * F.smooth_l1_loss(curr_q2_sliced, expected_q, reduction='none')).mean()

            # Optimize Critics
            self.q_optimizer1.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.gradient_clip_norm)
            self.q_optimizer1.step()
            self.q_scheduler1.step() 

            self.q_optimizer2.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.gradient_clip_norm)
            self.q_optimizer2.step()
            self.q_scheduler2.step() 

            # ------------------------------------------------------------------
            # 3. Actor Optimization (Independent Unrolling)
            # ------------------------------------------------------------------
            # Re-evaluate Q with the current Policy actions
            # Freeze Q-networks to prevent unnecessary gradient computation
            for p in self.q_net1.parameters(): p.requires_grad = False
            for p in self.q_net2.parameters(): p.requires_grad = False

            q1_pi, _ = self.q_net1(s, new_a_pi, la, h_in)
            q2_pi, _ = self.q_net2(s, new_a_pi, la, h_in)
            min_q_pi_sliced = torch.min(q1_pi, q2_pi)[:, burnin_len:, :]

            policy_loss = (self.alpha * log_pi_curr_sliced - min_q_pi_sliced).mean()
            

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_norm)
            self.policy_optimizer.step()
            self.policy_scheduler.step()

            # Unfreeze Q-networks
            for p in self.q_net1.parameters(): p.requires_grad = True
            for p in self.q_net2.parameters(): p.requires_grad = True

            # ------------------------------------------------------------------
            # 4. Maintenance
            # ------------------------------------------------------------------
            with torch.no_grad():
                td_error = torch.max(torch.abs(curr_q1_sliced - expected_q), 
                                    torch.abs(curr_q2_sliced - expected_q)).mean(dim=1).flatten()
                self.replay_buffer.update_priorities(indices, td_error.cpu().numpy())

            self._soft_update(self.q_net1, self.target_q_net1)
            self._soft_update(self.q_net2, self.target_q_net2)
            self._soft_update(self.policy_net, self.target_policy_net) 
            
            # Increment update counter
            self.total_updates += 1
            
            # Return all relevant metrics for logging in the main training loop
            return (
                q1_loss.item(), q2_loss.item(), policy_loss.item(), self.alpha.item(),
                self.q_optimizer1.param_groups[0]['lr'],
                self.q_optimizer2.param_groups[0]['lr'],
                self.policy_optimizer.param_groups[0]['lr'],
                self.alpha_optimizer.param_groups[0]['lr']
            )


    def _soft_update(self, net, target_net):
        """
        Soft updates target network parameters towards primary network parameters.
        
        Args:
            net (nn.Module): Primary network.
            target_net (nn.Module): Target network.
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        """
        Saves the current state dictionaries of all models to disk.
        
        Args:
            path (str): Base file path prefix.
            
        Returns:
            None
        """
        torch.save(self.q_net1.state_dict(), path + '_q1.pth')
        torch.save(self.q_net2.state_dict(), path + '_q2.pth')
        torch.save(self.policy_net.state_dict(), path + '_policy.pth')
        torch.save(self.target_q_net1.state_dict(), path + '_target_q1.pth')
        torch.save(self.target_q_net2.state_dict(), path + '_target_q2.pth')
        torch.save(self.target_policy_net.state_dict(), path + '_target_policy.pth') # Add this
        print(f"Models saved to {path}")


    def load_model(self, path):
        """
        Loads pre-trained model weights from disk.
        
        Args:
            path (str): Base file path prefix to load from.
            
        Returns:
            None
        """
        print(f"--- Loading models from {path} ---")
        device = self.device # Ensure we load to the correct device
        
        # 1. Load Primary Networks
        self.policy_net.load_state_dict(torch.load(path + '_policy.pth', map_location=device))
        self.q_net1.load_state_dict(torch.load(path + '_q1.pth', map_location=device))
        self.q_net2.load_state_dict(torch.load(path + '_q2.pth', map_location=device))
        self.target_q_net1.load_state_dict(torch.load(path + '_target_q1.pth', map_location=device))
        self.target_q_net2.load_state_dict(torch.load(path + '_target_q2.pth', map_location=device))
        self.target_policy_net.load_state_dict(torch.load(path + '_target_policy.pth', map_location=device))
        print("Target networks loaded from disk.")
        print("Successfully loaded all models for training/testing.")