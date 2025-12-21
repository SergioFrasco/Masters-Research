"""
Unified World Value Functions (WVF) Agent for Compositional RL

REVISED IMPLEMENTATION - Version 2

Fixes from previous version:
1. Softer R̄_MIN penalty (was too harsh, destabilized learning)
2. Hindsight-style goal learning: when reaching a goal, give POSITIVE 
   signal for that goal's Q̄, don't harshly penalize others
3. More stable training with separate positive/negative learning rates
4. Better exploration during training

Based on Nangue Tasse et al.'s Extended Value Functions, but adapted
for neural network function approximation with limited samples.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class FrameStack:
    """Stack the last k frames for short-term memory."""
    
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame):
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stacked()
    
    def step(self, frame):
        self.frames.append(frame.copy())
        return self._get_stacked()
    
    def _get_stacked(self):
        return np.concatenate(list(self.frames), axis=0)


class GoalConditionedEVFNetwork(nn.Module):
    """
    Extended Value Function Network - learns Q̄(s, g, a) for ALL goals.
    
    Architecture: Shared CNN/LSTM backbone with separate heads per goal.
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_goals=4, action_size=3,
                 hidden_size=256, lstm_size=128):
        super(GoalConditionedEVFNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_goals = num_goals
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN backbone (shared across all goals)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape[0], 
                                                             input_shape[1], 
                                                             input_shape[2])
        
        # LSTM for temporal reasoning (shared)
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Dueling architecture per goal for more stable learning
        self.goal_value_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(num_goals)
        ])
        
        self.goal_advantage_streams = nn.ModuleList([
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
        Forward pass - outputs Q̄(s, g, a) for ALL goals.
        
        Returns:
            q_values: (batch, num_goals, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # CNN features (shared)
        conv_features = self.conv(state)
        conv_features = conv_features.view(batch_size, -1)
        conv_features = conv_features.unsqueeze(1)
        
        if hidden is not None:
            lstm_out, hidden = self.lstm(conv_features, hidden)
        else:
            lstm_out, hidden = self.lstm(conv_features)
        
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Dueling Q-values for each goal
        q_per_goal = []
        for i in range(self.num_goals):
            value = self.goal_value_streams[i](lstm_out)  # (batch, 1)
            advantage = self.goal_advantage_streams[i](lstm_out)  # (batch, action_size)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            q_per_goal.append(q)
        
        q_values = torch.stack(q_per_goal, dim=1)  # (batch, num_goals, action_size)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class HindsightReplayBuffer:
    """
    Replay buffer with hindsight-style goal relabeling.
    
    Key insight: When we reach goal G, that's a POSITIVE example for Q̄(s, G, a).
    We don't need to harshly penalize other goals - just don't give them positive signal.
    """
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reached_goal_idx, task_goal_indices, done):
        """
        Store transition with goal information.
        
        Args:
            state: Current observation
            action: Action taken
            next_state: Next observation
            reached_goal_idx: Index of goal reached (or -1 if none)
            task_goal_indices: List of goal indices that are valid for current task
            done: Whether episode ended
        """
        self.buffer.append((state, action, next_state, reached_goal_idx, task_goal_indices, done))
    
    def sample(self, batch_size, num_goals):
        """
        Sample batch with hindsight goal relabeling.
        
        For each transition, we create training targets for ALL goals:
        - If this transition reached goal G: reward = +1 for G, 0 for others (if not done) or small negative
        - If no goal reached: step penalty for all
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = []
        actions = []
        next_states = []
        rewards_per_goal = []  # (batch, num_goals)
        dones_per_goal = []    # (batch, num_goals)
        
        for state, action, next_state, reached_goal_idx, task_goal_indices, done in batch:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            
            # Compute rewards for each goal
            rewards = []
            dones = []
            
            for g in range(num_goals):
                if reached_goal_idx == -1:
                    # No goal reached - small step penalty
                    rewards.append(-0.01)
                    dones.append(False)
                elif reached_goal_idx == g:
                    # We reached THIS goal - positive reward!
                    rewards.append(1.0)
                    dones.append(True)
                else:
                    # We reached a DIFFERENT goal
                    # Soft penalty: this is useful info (we know how to get here, 
                    # but it's not the goal we want)
                    rewards.append(-0.5)  # Much softer than R̄_MIN = -10
                    dones.append(True)
            
            rewards_per_goal.append(rewards)
            dones_per_goal.append(dones)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards_per_goal),
            np.array(dones_per_goal)
        )
    
    def __len__(self):
        return len(self.buffer)


class UnifiedWorldValueFunctionAgent:
    """
    REVISED World Value Function (WVF) Agent - Version 2
    
    Key changes from v1:
    1. Softer penalties for wrong goals (-0.5 instead of -10)
    2. Hindsight-style replay: reaching any goal is informative
    3. Dueling architecture for more stable learning
    4. Simpler replay buffer (no sequence-based, just transitions)
    5. More aggressive target network updates
    
    The composition formula remains:
    Q̄*_{B AND S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}
    π(s) = argmax_a max_g Q̄*_{B AND S}(s, g, a)
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
    
    def __init__(self, env, k_frames=4, learning_rate=0.0003, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=64,
                 hidden_size=256, lstm_size=128,
                 tau=0.01, grad_clip=10.0,
                 # These parameters kept for API compatibility
                 seq_len=4, r_correct=1.0, r_wrong=-0.1, step_penalty=-0.005):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.num_goals = len(self.GOALS)
        
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
        print(f"UnifiedWorldValueFunctionAgent (REVISED v2) using device: {self.device}")
        
        # Get observation shape
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        if sample_img.shape[0] in [3, 4]:
            self.single_frame_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.single_frame_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        self.obs_shape = (k_frames * 3, self.single_frame_shape[1], self.single_frame_shape[2])
        
        print(f"Observation shape: {self.obs_shape}")
        print(f"Number of goals: {self.num_goals}")
        print(f"Goals: {self.GOALS}")
        print(f"Using SOFT penalties and hindsight-style learning")
        
        self.frame_stack = FrameStack(k=k_frames)
        self._create_networks()
        
        self.current_hidden = None
        self.current_task = None
        
        # Track training progress
        self.training_steps = 0
    
    def _create_networks(self):
        self.q_network = GoalConditionedEVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        
        self.target_network = GoalConditionedEVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = HindsightReplayBuffer(capacity=self.memory_size)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"  Network parameters: {total_params:,}")
    
    def sample_task(self):
        return random.choice(self.PRIMITIVES)
    
    def compute_reward(self, info, current_task):
        """Backward-compatible reward computation."""
        contacted = info.get('contacted_object', None)
        if contacted is None:
            return -0.01, False
        
        task_goals = self.TASK_GOALS[current_task]
        if contacted in task_goals:
            return 1.0, True
        else:
            return -0.1, True
    
    def get_reached_goal_idx(self, info):
        """Get index of reached goal, or -1 if none."""
        contacted = info.get('contacted_object', None)
        if contacted is None or contacted not in self.GOAL_TO_IDX:
            return -1
        return self.GOAL_TO_IDX[contacted]
    
    def preprocess_frame(self, obs):
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
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        if task_name is not None:
            self.current_task = task_name
        
        return stacked
    
    def step_episode(self, obs):
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, task_idx=None, epsilon=None):
        """Select action during training based on current task's goals."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_all_goals, self.current_hidden = self.q_network(state, self.current_hidden)
            self.current_hidden = (self.current_hidden[0].detach(),
                                   self.current_hidden[1].detach())
            
            q_all_goals = q_all_goals[0]  # (num_goals, num_actions)
            
            if self.current_task is not None:
                # Get Q-values for task's goals and take max
                task_goal_indices = [self.GOAL_TO_IDX[g] for g in self.TASK_GOALS[self.current_task]]
                q_task_goals = q_all_goals[task_goal_indices]  # (2, num_actions)
                q_max = q_task_goals.max(dim=0)[0]  # (num_actions,)
                return q_max.argmax().item()
            else:
                q_max = q_all_goals.max(dim=0)[0]
                return q_max.argmax().item()
    
    def select_action_composed(self, stacked_obs, features):
        """
        Zero-shot composition using min over tasks, max over goals.
        
        Q̄*_{B AND S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}
        π(s) = argmax_a max_g Q̄*_{B AND S}(s, g, a)
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_all_goals, new_hidden = self.q_network(state, self.current_hidden)
            q_all_goals = q_all_goals[0]  # (num_goals, num_actions)
            self.current_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            # For composition, we need to find goals that are in ALL task goal sets
            # and take min of their Q-values across tasks
            
            # Get goal sets for each feature/task
            goal_sets = [set(self.TASK_GOALS[f]) for f in features]
            
            # Find intersection (goals that satisfy ALL features)
            valid_goals = goal_sets[0]
            for gs in goal_sets[1:]:
                valid_goals = valid_goals.intersection(gs)
            
            if len(valid_goals) == 0:
                # No valid goals - fall back to max over all
                q_max = q_all_goals.max(dim=0)[0]
                return q_max.argmax().item()
            
            # For each valid goal, compute min Q-value across task perspectives
            best_q = float('-inf')
            best_action = 0
            
            for goal in valid_goals:
                goal_idx = self.GOAL_TO_IDX[goal]
                q_goal = q_all_goals[goal_idx]  # (num_actions,)
                
                # This goal is valid for ALL tasks in features, so its Q̄ should be high
                # for all of them. We take the Q-value directly since we're looking at
                # the intersection.
                q_max_for_goal = q_goal.max().item()
                
                if q_max_for_goal > best_q:
                    best_q = q_max_for_goal
                    best_action = q_goal.argmax().item()
            
            return best_action
    
    def remember(self, state, task_idx, action, reward, next_state, done):
        """Backward-compatible remember - but we need more info."""
        # This is called from experiment_utils but doesn't have goal info
        # We'll use remember_with_goal instead in the training loop
        pass
    
    def remember_with_goal(self, state, action, next_state, reached_goal_idx, done):
        """Store transition with goal information."""
        task_goal_indices = []
        if self.current_task:
            task_goal_indices = [self.GOAL_TO_IDX[g] for g in self.TASK_GOALS[self.current_task]]
        
        self.memory.push(state, action, next_state, reached_goal_idx, task_goal_indices, done)
    
    def soft_update_target(self):
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Train on a batch of transitions."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size, self.num_goals
        )
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)  # (batch, num_goals)
        dones_t = torch.BoolTensor(dones).to(self.device)  # (batch, num_goals)
        
        batch_size = states_t.shape[0]
        
        # Get current Q-values
        q_values, _ = self.q_network(states_t)  # (batch, num_goals, num_actions)
        
        # Get Q-values for taken actions
        actions_expanded = actions_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_goals, 1)
        current_q = q_values.gather(2, actions_expanded).squeeze(2)  # (batch, num_goals)
        
        # Compute targets using Double DQN
        with torch.no_grad():
            # Get next actions from online network
            next_q_online, _ = self.q_network(next_states_t)
            next_actions = next_q_online.argmax(dim=2, keepdim=True)  # (batch, num_goals, 1)
            
            # Evaluate with target network
            next_q_target, _ = self.target_network(next_states_t)
            next_q = next_q_target.gather(2, next_actions).squeeze(2)  # (batch, num_goals)
            
            # TD target
            target_q = rewards_t + self.gamma * next_q * (~dones_t).float()
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        # Soft update target network
        self.training_steps += 1
        if self.training_steps % 4 == 0:
            self.soft_update_target()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_goals': self.num_goals,
            'hidden_size': self.hidden_size,
            'lstm_size': self.lstm_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"WVF model (REVISED v2) saved to {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hidden_size = checkpoint.get('hidden_size', 256)
        self.lstm_size = checkpoint.get('lstm_size', 128)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"WVF model (REVISED v2) loaded from {filepath}")