# Algorithmic Methodology & Theoretical Formulation

This document details the mathematical formulation, network architecture, and reward engineering behind the **SAC + LSTM + R2D2 Burn-In + PER** mapless navigation system.

---

## 1. POMDP Formulation & Recurrent SAC Architecture

Mapless navigation in dynamic environments is modeled as a **Partially Observable Markov Decision Process (POMDP)** defined by $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \gamma)$. A single 2D LiDAR scan $o_t \in \mathcal{O}$ captures instantaneous obstacle ranges, leaving obstacle velocities unobservable from a single frame.

### R2D2-Style Sequence Burn-In
Training recurrent policies off-policy with Experience Replay introduces **recurrent state staleness** (stored hidden states $(h_t, c_t)$ drift as network weights $\theta$ update) and **zero-padding bias**. Our implementation (`src/sac_agent/sac_v2_lstm_R.py` & `src/sac_agent/common/buffers.py`) resolves both issues:

1. **Sub-Sequence Sampling ($L_{\text{total}} = L_{\text{burn}} + L_{\text{seq}} = 16 + 32 = 48$):**
   - Episodes are stored with step-wise LSTM hidden states $(h_t, c_t)$.
   - During each gradient update, a contiguous slice of $48$ steps is sampled using Prioritized Experience Replay ($\alpha_{\text{PER}} = 0.6$, importance sampling $\beta \to 1.0$).
2. **Burn-In Phase ($L_{\text{burn}} = 16$, No Gradients):**
   - The first $16$ steps are unrolled through the Actor and Twin-Critic LSTMs under `torch.no_grad()` starting from the stored $(h_{\text{start}}, c_{\text{start}})$, producing a refreshed hidden state $(h_{\text{burn}}, c_{\text{burn}})$ aligned with current network weights.
3. **Training Phase ($L_{\text{seq}} = 32$, BPTT):**
   - Critic TD-errors and Actor policy gradients are computed exclusively over the remaining $32$ steps, reducing per-batch GPU update latency from **~0.90s (full-episode replay) to ~0.05s (~18× speedup)**.
4. **Edge-Repetition Padding & Episode Filtering:**
   - Episodes shorter than $L_{\text{total}} = 48$ steps are discarded from replay, and boundary slices are edge-padded rather than zero-padded to avoid corrupting LSTM input/forget gates.

---

## 2. Observation & Action Spaces (`src/envs/pgrc_env_map_goal.py`)

### Observation Vector ($\mathbf{o}_t \in \mathbb{R}^{186}$)
- **Normalized 2D LiDAR (`180-D`):** Downsampled uniformly from $360^\circ$ laser scans, clipped to $[d_{\min}, d_{\max}] = [0.12\text{ m}, 3.5\text{ m}]$, and normalized to $[0, 1]$. During training, zero-mean Gaussian noise ($\sigma = 0.01$) is injected for sim-to-real robustness.
- **Ego & Goal Metrics (`6-D`):**
  1. Normalized goal distance: $d_{\text{goal}} / 3.5$
  2. Normalized relative heading to goal: $\theta_{\text{goal}} / \pi \in [-1, 1]$
  3. Minimum obstacle range: $d_{\min} / 3.5$
  4. Normalized linear velocity: $v_t / v_{\max}$
  5. Normalized angular velocity: $\omega_t / \omega_{\max}$
  6. Episode initial-step indicator: $\mathbb{I}(t \le 1) \in \{0, 1\}$
- **Past-Action Conditioning:** The previous action $\mathbf{a}_{t-1} \in \mathbb{R}^2$ is concatenated with $\mathbf{o}_t$ inside the LSTM recurrent branch to capture actuation dynamics.

### Continuous Action Space ($\mathbf{a}_t \in \mathbb{R}^2$)
- Linear velocity: $v_t \in [-0.05, 0.22]\text{ m/s}$ (allows slight reversing maneuvers in tight corners)
- Angular velocity: $\omega_t \in [-2.5, 2.5]\text{ rad/s}$
- Control frequency: $20\text{ Hz}$ ($\Delta t = 0.05\text{ s}$)

---

## 3. Dynamic-Aware Reward Function

Terminal transitions yield $r_{\text{goal}} = +50.0$ when $d_{\text{goal}} < 0.35\text{ m}$ or $r_{\text{col}} = -50.0$ when $d_{\min} < 0.15\text{ m}$. Non-terminal steps use dense dynamic-aware shaping:

$$\mathcal{R}_t = r_{\text{progress}} + w_d \, r_{\text{distance}} + r_{\text{safety}} + r_{\text{closing}} + r_{\text{time}} + r_{\text{vel}}$$

| Term | Mathematical Formulation | Config Weight / Parameter | Purpose |
| :--- | :--- | :---: | :--- |
| **Goal Progress** | $k_p \cdot \text{clip}(d_{\text{goal}}^{t-1} - d_{\text{goal}}^t, -0.05, 0.05)$ | $k_p = 7.0$ | Encourages direct progress toward the target |
| **Proximity Shaping** | $w_d \cdot (1 - \tanh(1.5 \, d_{\text{goal}}^t))$ | $w_d = 0.5135$ | Provides smooth gradient near the goal |
| **Quadratic Safety Barrier** | $k_{\text{safe}} \left(\frac{d_{\text{safe}} - d_{\min}^t}{d_{\text{safe}} - d_{\text{col}}}\right)^2 \mathbb{I}(d_{\min}^t < d_{\text{safe}})$ | $k_{\text{safe}} = -0.5102$, $d_{\text{safe}} = 0.35\text{ m}$ | Non-linearly penalizes proximity to obstacles |
| **Closing-Speed Penalty** | $k_{\text{close}} \cdot \min(0, \, d_{\min}^t - d_{\min}^{t-1})$ | $k_{\text{close}} = 3.0862$ | Penalizes rapid approach toward moving obstacles |
| **Time Penalty** | Constant per-step penalty | $r_{\text{time}} = -0.01716$ | Incentivizes time-optimal trajectories |
| **Forward Velocity Bonus** | $k_v \cdot v_t \, \mathbb{I}(v_t > 0.05\text{ m/s})$ | $k_v = 0.01$ | Prevents stationary hesitation |

### Continuous Multi-Goal Curriculum (`CONTINUOUS_MODE`)
When an episode terminates via `goal_reached`, the environment executes a **soft reset**: a new goal coordinate is sampled without teleporting the robot or resetting moving obstacles, and the LSTM hidden state $(h_t, c_t)$ is detached from the computation graph (`h.detach()`) and carried over to the next goal.

---

## 4. Full Per-Seed POPGym & PO-MuJoCo Benchmark Results

### 4.1 POPGym Memory Benchmark (78 Runs)

#### `SAC + LSTM + PER`
| Task | Difficulty | Seed 42 | Seed 123 | Seed 456 | Steps (approx) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RepeatPrevious | Easy | **1.0** | **1.0** | **1.0** | 255K |
| PositionOnlyCartPole | Easy | 0.4 | **0.9** | **0.6** | ~107–118K |
| PositionOnlyCartPole | Medium | **0.7** | **0.4** | 0.1 | ~111–122K |
| HigherLower | Easy | **0.5** | **0.5** | **0.5** | 255K |
| HigherLower | Medium | **0.5** | **0.5** | **0.5** | 515K |

#### `SAC + LSTM + Uniform`
| Task | Difficulty | Seed 42 | Seed 123 | Seed 456 | Steps (approx) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RepeatPrevious | Easy | **1.0** | **1.0** | **1.0** | 255K |
| PositionOnlyCartPole | Easy | 0.6 | 0.7 | 0.6 | ~107–118K |
| PositionOnlyCartPole | Medium | 0.6 | 0.4 | 0.5 | ~107–120K |
| HigherLower | Easy | 0.5 | 0.5 | 0.5 | 255K |

#### `SAC + MLP` (Feedforward Baseline)
| Task | Difficulty | Seed 42 | Seed 123 | Seed 456 | Steps (approx) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PositionOnlyCartPole | Easy | 0.2 | 0.2 | 0.2 | ~113–119K |
| PositionOnlyCartPole | Medium | 0.1 | 0.1 | 0.1 | ~109–113K |

---

### 4.2 PO-MuJoCo Continuous Control Benchmark (3M Steps, Velocity Masked)

#### `SAC + LSTM + PER`
| Task | Seed 42 | Seed 123 | Seed 456 | Mean Return |
| :--- | :---: | :---: | :---: | :---: |
| **HalfCheetah-v4 (PO)** | **14,900** | **7,900** | **12,709** | **11,836** |
| **Ant-v4 (PO)** | **3,501** | **3,392** | **1,822** | **2,905** |

#### `SAC + LSTM + Uniform`
| Task | Seed 42 | Seed 123 | Seed 456 | Mean Return |
| :--- | :---: | :---: | :---: | :---: |
| **HalfCheetah-v4 (PO)** | 9,201 | 7,422 | 7,921 | 8,181 |
| **Ant-v4 (PO)** | 1,791 | 2,461 | 1,623 | 1,958 |

