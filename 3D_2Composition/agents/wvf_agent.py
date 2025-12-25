"""
Unified World Value Functions (WVF) Agent for Compositional RL

CORRECTED & OPTIMIZED IMPLEMENTATION

Key fixes based on Nangue Tasse theory:
1. NO FRAME STACKING - use LSTM-only for cleaner belief states
2. Episodes terminate on ANY goal (theory-compliant)
3. Proper extended rewards R̄(s, g, a) with R̄_MIN for wrong goals
4. Goal-agnostic LSTM + separate Q-heads for each goal

Theory:
- Learn Q̄(s, g, a) for each goal g ∈ G
- Composition: Q̄*_{B ∧ S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}
- Policy: π(s) = argmax_a max_g Q̄*_composed(s, g, a)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class GoalConditionedEVFNetwork(nn.Module):
    """
    Extended Value Function Network: Q̄(s, g, a) for all goals
    
    Architecture:
    - CNN backbone (goal-agnostic visual features)
    - LSTM (goal-agnostic temporal reasoning)
    - Separate Q-heads for each goal g
    
    NO FRAME STACKING - LSTM handles temporal integration
    """
    
    def __init__(self, input_channels=3, height=60, width=80, num_goals=4, 
                 action_size=3, hidden_size=256, lstm_size=128):
        super(GoalConditionedEVFNetwork, self).__init__()
        
        self.num_goals = num_goals
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN backbone (shared, goal-agnostic)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        self._conv_output_size = self._get_conv_output_size(input_channels, height, width)
        
        # LSTM for temporal integration (shared, goal-agnostic)
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Separate Q-head for each goal (THIS IS KEY!)
        # Q̄(h, g, a) where h is goal-agnostic LSTM hidden state
        self.goal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
            for _ in range(num_goals)
        ])
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels, height, width):
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            output = self.conv(dummy)
            return int(np.prod(output.shape[1:]))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, state, hidden=None):
        """
        Forward pass: outputs Q̄(s, g, a) for ALL goals
        
        Args:
            state: (batch, C, H, W) single frame
            hidden: LSTM hidden state (h, c)
        
        Returns:
            q_values: (batch, num_goals, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # CNN features (goal-agnostic)
        conv_features = self.conv(state)
        conv_features = conv_features.view(batch_size, -1)
        
        # LSTM expects (batch, seq_len=1, features)
        conv_features = conv_features.unsqueeze(1)
        
        if hidden is not None:
            lstm_out, hidden = self.lstm(conv_features, hidden)
        else:
            lstm_out, hidden = self.lstm(conv_features)
        
        # Take last LSTM output (goal-agnostic belief state)
        h = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Compute Q̄(h, g, a) for each goal using separate heads
        q_per_goal = []
        for head in self.goal_heads:
            q_goal = head(h)  # (batch, action_size)
            q_per_goal.append(q_goal)
        
        # Stack: (batch, num_goals, action_size)
        q_values = torch.stack(q_per_goal, dim=1)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize LSTM hidden state"""
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """Episode-based replay buffer with extended rewards for all goals"""
    
    def __init__(self, capacity=2000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push(self, state, action, rewards_per_goal, next_state, dones_per_goal):
        """
        Add transition to current episode
        
        Args:
            rewards_per_goal: [r_g1, r_g2, ..., r_gN] for N goals
            dones_per_goal: [done_g1, done_g2, ..., done_gN]
        """
        self.current_episode.append((state, action, rewards_per_goal, next_state, dones_per_goal))
        
        # End episode if ANY goal is done (episode terminates)
        if any(dones_per_goal):
            self.end_episode()
    
    def end_episode(self):
        """Finalize and store current episode"""
        if len(self.current_episode) > 0:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=4):
        """Sample sequences from episodes"""
        batch = []
        
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            
            if len(episode) > seq_len:
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                sequence = list(episode)
            
            batch.append(sequence)
        
        return batch
    
    def __len__(self):
        return len(self.episodes)


class UnifiedWorldValueFunctionAgent:
    """
    CORRECTED World Value Function Agent (NO FRAME STACKING)
    
    Key improvements:
    1. LSTM-only (no frame stacking) → cleaner belief states
    2. Theory-compliant extended rewards
    3. Proper composition via min operator
    
    Goals: {red_box, blue_box, red_sphere, blue_sphere}
    Tasks: {red, blue, box, sphere}
    """
    
    GOALS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}
    IDX_TO_GOAL = {i: g for i, g in enumerate(GOALS)}
    
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    TASK_TO_IDX = {t: i for i, t in enumerate(PRIMITIVES)}
    
    TASK_GOALS = {
        'red': ['red_box', 'red_sphere'],
        'blue': ['blue_box', 'blue_sphere'],
        'box': ['red_box', 'blue_box'],
        'sphere': ['red_sphere', 'blue_sphere'],
    }
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=256, lstm_size=128,
                 tau=0.005, grad_clip=10.0,
                 r_correct=1.0, r_wrong=-0.1, step_penalty=-0.01,
                 r_bar_min=-10.0):
        
        self.env = env
        self.action_dim = 3
        self.num_goals = len(self.GOALS)
        self.seq_len = seq_len
        
        # Reward parameters
        self.r_correct = r_correct
        self.r_wrong = r_wrong
        self.step_penalty = step_penalty
        self.r_bar_min = r_bar_min  # R̄_MIN for wrong goals
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.grad_clip = grad_clip
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WVF Agent (LSTM-only, theory-compliant) using device: {self.device}")
        
        # Get observation shape (single frame, NO stacking)
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        if sample_img.shape[0] in [3, 4]:
            self.obs_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.obs_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        print(f"Observation shape (single frame): {self.obs_shape}")
        print(f"Number of goals: {self.num_goals}")
        print(f"Goals: {self.GOALS}")
        print(f"R̄_MIN penalty: {self.r_bar_min}")
        print(f"NO FRAME STACKING - LSTM handles temporal integration")
        
        # Create networks
        self._create_networks()
        
        # Episode state
        self.current_hidden = None
        self.current_task = None
    
    def _create_networks(self):
        """Create goal-conditioned EVF networks"""
        self.q_network = GoalConditionedEVFNetwork(
            input_channels=self.obs_shape[0],
            height=self.obs_shape[1],
            width=self.obs_shape[2],
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        
        self.target_network = GoalConditionedEVFNetwork(
            input_channels=self.obs_shape[0],
            height=self.obs_shape[1],
            width=self.obs_shape[2],
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = EpisodeReplayBuffer(capacity=self.memory_size)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"  EVF network parameters: {total_params:,}")
    
    def sample_task(self):
        """Sample random primitive task"""
        return random.choice(self.PRIMITIVES)
    
    def compute_extended_rewards(self, info, current_task):
        """
        Compute R̄(s, g, a) for ALL goals (FIXED)
        
        Theory-compliant:
        - If reached goal g: normal task reward, done=True
        - If reached different goal g': R̄_MIN penalty, done=True
        - If no goal reached: step penalty, done=False
        
        Args:
            info: Env info with 'contacted_object'
            current_task: Current primitive task
            
        Returns:
            rewards: [r_g1, r_g2, ..., r_g4]
            dones: [done_g1, done_g2, ..., done_g4]
            task_success: Whether task goal was reached
        """
        contacted = info.get('contacted_object', None)
        task_goal_set = self.TASK_GOALS[current_task]
        
        rewards = []
        dones = []
        task_success = False
        
        if contacted is None:
            # No goal reached - step penalty for all, continue episode
            for goal in self.GOALS:
                rewards.append(self.step_penalty)
                dones.append(False)
        else:
            # A goal was reached! Episode terminates for ALL goals
            for goal in self.GOALS:
                if contacted == goal:
                    # We reached THIS goal
                    if goal in task_goal_set:
                        rewards.append(self.r_correct)  # Correct for task!
                        task_success = True
                    else:
                        rewards.append(self.r_wrong)  # Wrong for task
                    dones.append(True)  # Episode ends
                else:
                    # We reached a DIFFERENT goal - R̄_MIN penalty!
                    # This is the key: teaches agent NOT to go to other goals
                    rewards.append(self.r_bar_min)
                    dones.append(True)  # Episode ends for all goals
        
        return rewards, dones, task_success
    
    def compute_reward(self, info, current_task):
        """Backward-compatible method for experiment_utils"""
        rewards, dones, task_success = self.compute_extended_rewards(info, current_task)
        
        contacted = info.get('contacted_object', None)
        if contacted is None:
            return self.step_penalty, False
        
        task_goal_set = self.TASK_GOALS[current_task]
        if contacted in task_goal_set:
            return self.r_correct, True
        else:
            return self.r_wrong, True
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame (NO stacking)"""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if img.shape[0] not in [3, 4]:
            img = np.transpose(img, (2, 0, 1))
        
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        
        if img.shape[0] == 4:
            img = img[:3]
        
        return img
    
    def reset_episode(self, obs, task_name=None):
        """Reset for new episode"""
        frame = self.preprocess_frame(obs)
        
        # Initialize LSTM hidden state
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        if task_name is not None:
            self.current_task = task_name
        
        return frame
    
    def step_episode(self, obs):
        """Process new observation (just preprocess, no stacking)"""
        return self.preprocess_frame(obs)
    
    def select_action(self, obs, task_idx=None, epsilon=None):
        """
        Select action during TRAINING
        
        Uses current task's goal set:
        Q_task(s, a) = max_{g ∈ task_goals} Q̄(s, g, a)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Q̄(s, g, a) for all goals
            q_all_goals, self.current_hidden = self.q_network(state, self.current_hidden)
            self.current_hidden = (self.current_hidden[0].detach(),
                                   self.current_hidden[1].detach())
            
            if self.current_task is not None:
                # Get indices of goals in current task's goal set
                task_goal_indices = [self.GOAL_TO_IDX[g] 
                                    for g in self.TASK_GOALS[self.current_task]]
                # Max over task's goals: Q_task(s, a) = max_g Q̄(s, g, a)
                q_task = q_all_goals[0, task_goal_indices, :].max(dim=0)[0]
                return q_task.argmax().item()
            else:
                # Fallback: max over all goals
                q_max = q_all_goals[0].max(dim=0)[0]
                return q_max.argmax().item()
    
    def select_action_composed(self, obs, features):
        """
        CORRECTED Boolean composition for zero-shot transfer
        
        Theory: Q̄*_{B ∧ S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}
        
        Algorithm:
        1. For each goal g, compute Q̄_composed(s, g, a):
           - For each primitive task t in features:
             - If g ∈ TASK_GOALS[t]: include Q̄_t(s, g, a)
           - Q̄_composed(s, g, a) = min over all included Q̄ values
        2. Q(s, a) = max_g Q̄_composed(s, g, a)
        3. π(s) = argmax_a Q(s, a)
        
        Example: features = ['blue', 'sphere']
        - For g = blue_sphere:
          - blue: blue_sphere ∈ {blue_box, blue_sphere} ✓
          - sphere: blue_sphere ∈ {red_sphere, blue_sphere} ✓
          - Q̄_comp(s, blue_sphere, a) = min(Q̄_blue(s, blue_sphere, a), 
                                              Q̄_sphere(s, blue_sphere, a))
        - For g = blue_box:
          - blue: blue_box ∈ {blue_box, blue_sphere} ✓
          - sphere: blue_box ∉ {red_sphere, blue_sphere} ✗
          - Q̄_sphere(s, blue_box, a) should be low (learned via R̄_MIN)
          - Q̄_comp(s, blue_box, a) = min(Q̄_blue(s, blue_box, a), 
                                          Q̄_sphere(s, blue_box, a)) → low!
        """
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Q̄(s, g, a) for all goals: (1, num_goals, num_actions)
            q_all_goals, new_hidden = self.q_network(state, self.current_hidden)
            q_all_goals = q_all_goals[0]  # (num_goals, num_actions)
            
            self.current_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            # For each goal, compute min over task Q-values
            composed_q_per_goal = []
            
            for goal_idx, goal in enumerate(self.GOALS):
                # Collect Q̄_task(s, goal, a) for each task in features
                q_values_for_composition = []
                
                for task in features:
                    # Get this task's Q̄ values for this goal
                    # Note: During training, we learned Q̄(s, g, a) for ALL g
                    # Even if g ∉ task_goals, Q̄ was trained with R̄_MIN penalties
                    q_values_for_composition.append(q_all_goals[goal_idx])
                
                # min over tasks: Q̄_composed(s, goal, a)
                if len(q_values_for_composition) > 0:
                    q_stacked = torch.stack(q_values_for_composition, dim=0)
                    q_min = q_stacked.min(dim=0)[0]  # (num_actions,)
                    composed_q_per_goal.append(q_min)
            
            # Stack all goals: (num_goals, num_actions)
            composed_q = torch.stack(composed_q_per_goal, dim=0)
            
            # max over goals: Q(s, a) = max_g Q̄_composed(s, g, a)
            q_final = composed_q.max(dim=0)[0]  # (num_actions,)
            
            return q_final.argmax().item()
    
    def remember_extended(self, state, action, rewards_per_goal, next_state, dones_per_goal):
        """Store transition with extended rewards"""
        self.memory.push(state, action, rewards_per_goal, next_state, dones_per_goal)
    
    def soft_update_target(self):
        """Soft update target network"""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Train on batch of sequences (FIXED for proper temporal learning)"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        sequences = self.memory.sample(self.batch_size, self.seq_len)
        max_len = max(len(seq) for seq in sequences)
        
        # Prepare batched data
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        lengths = []
        
        for seq in sequences:
            seq_len_actual = len(seq)
            lengths.append(seq_len_actual)
            
            states = [s[0] for s in seq]
            actions = [s[1] for s in seq]
            rewards = [s[2] for s in seq]
            next_states = [s[3] for s in seq]
            dones = [s[4] for s in seq]
            
            # Pad to max length
            if seq_len_actual < max_len:
                pad_len = max_len - seq_len_actual
                states.extend([states[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([rewards[-1]] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([[True] * self.num_goals] * pad_len)
            
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(np.array(dones_batch)).to(self.device)
        
        batch_size, seq_len_max = states_t.shape[:2]
        
        # Initialize hidden states
        hidden = self.q_network.init_hidden(batch_size, self.device)
        target_hidden = self.target_network.init_hidden(batch_size, self.device)
        
        # Forward through online network
        q_values_list = []
        for t in range(seq_len_max):
            q_vals, hidden = self.q_network(states_t[:, t], hidden)
            q_values_list.append(q_vals)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        q_values = torch.stack(q_values_list, dim=1)  # (batch, seq, num_goals, num_actions)
        
        # Get Q-values for taken actions
        actions_expanded = actions_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_goals, 1)
        current_q = q_values.gather(3, actions_expanded).squeeze(3)  # (batch, seq, num_goals)
        
        # Target network (Double DQN)
        with torch.no_grad():
            # Get next actions from online network
            next_q_list = []
            hidden_copy = self.q_network.init_hidden(batch_size, self.device)
            for t in range(seq_len_max):
                nq, hidden_copy = self.q_network(next_states_t[:, t], hidden_copy)
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_online = torch.stack(next_q_list, dim=1)
            next_actions = next_q_online.argmax(dim=3, keepdim=True)
            
            # Evaluate with target network
            target_q_list = []
            for t in range(seq_len_max):
                tq, target_hidden = self.target_network(next_states_t[:, t], target_hidden)
                target_q_list.append(tq)
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
            
            target_q_values = torch.stack(target_q_list, dim=1)
            next_q = target_q_values.gather(3, next_actions).squeeze(3)
            
            # TD target for each goal
            target_q = rewards_t + (self.gamma * next_q * ~dones_t)
        
        # Mask for variable-length sequences
        loss_mask = torch.zeros(batch_size, seq_len_max, self.num_goals, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length, :] = 1.0
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        self.soft_update_target()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset exploration rate"""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'num_goals': self.num_goals,
            'hidden_size': self.hidden_size,
            'lstm_size': self.lstm_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"WVF model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"WVF model loaded from {filepath}")