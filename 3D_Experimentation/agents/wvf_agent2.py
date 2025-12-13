"""
World Value Functions (WVF) Agent for Compositional RL

Based on Nangue Tasse et al.'s Boolean Task Algebra:
- Single goal-conditioned Q-network: Q(s, g, a)
- Train separate "slices" for each primitive task with extended rewards
- Zero-shot composition via min/max over Q-values at evaluation time

Key insight: The network learns to reach ALL goals, but values them
differently depending on which task it's trained on. This enables
zero-shot generalization to compositional tasks never seen during training.
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


class GoalConditionedQNetwork(nn.Module):
    """
    Goal-Conditioned Q-Network: Q(s, g, a)
    
    The goal g is provided as a one-hot vector and concatenated
    with the CNN features before the LSTM/FC layers.
    
    This allows the network to learn different values for the same
    state depending on which goal we're trying to reach.
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_goals=4, action_size=3,
                 hidden_size=128, lstm_size=64):
        super(GoalConditionedQNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_goals = num_goals
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN backbone for visual features
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # LSTM input: CNN features + goal one-hot
        lstm_input_size = self._conv_output_size + num_goals
        
        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
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
    
    def forward(self, state, goal, hidden=None):
        """
        Forward pass.
        
        Args:
            state: (batch, channels, height, width)
            goal: (batch, num_goals) one-hot encoding of target goal
            hidden: Optional LSTM hidden state
        
        Returns:
            q_values: (batch, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # CNN features
        conv_features = self.conv(state)
        conv_features = conv_features.view(batch_size, -1)
        
        # Concatenate with goal
        combined = torch.cat([conv_features, goal], dim=1)
        combined = combined.unsqueeze(1)  # (batch, 1, features+goals)
        
        # LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(combined, hidden)
        else:
            lstm_out, hidden = self.lstm(combined)
        
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Dueling
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def forward_all_goals(self, state, hidden=None):
        """
        Compute Q-values for ALL goals at once.
        
        Returns:
            q_values: (batch, num_goals, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        device = state.device
        
        all_q_values = []
        
        for g in range(self.num_goals):
            # Create one-hot goal vector
            goal = torch.zeros(batch_size, self.num_goals, device=device)
            goal[:, g] = 1.0
            
            q_vals, hidden = self.forward(state, goal, hidden)
            all_q_values.append(q_vals)
        
        # Stack: (batch, num_goals, action_size)
        return torch.stack(all_q_values, dim=1), hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """Episode-based replay buffer storing (state, goal, action, reward, next_state, done)."""
    
    def __init__(self, capacity=2000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push(self, state, goal, action, reward, next_state, done):
        self.current_episode.append((state, goal, action, reward, next_state, done))
        if done:
            self.end_episode()
    
    def end_episode(self):
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=4):
        batch = []
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            if len(episode) > seq_len:
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                sequence = episode
            batch.append(sequence)
        return batch
    
    def __len__(self):
        return len(self.episodes)
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = []


class WorldValueFunctionAgent:
    """
    World Value Function (WVF) Agent
    
    Implements Nangue Tasse et al.'s approach for zero-shot compositional generalization:
    
    1. Goal-conditioned Q-network: Q(s, g, a) - knows how to reach any goal
    2. Extended reward: penalty for reaching wrong goal, teaches goal discrimination
    3. Separate training for each primitive task (red, blue, box, sphere)
    4. Zero-shot composition via min (conjunction) at evaluation
    
    Goals: [red_box, blue_box, red_sphere, blue_sphere]
    Primitives: red, blue, box, sphere
    
    At evaluation, compositional tasks like "red_box" are solved zero-shot by:
        Q_composed(s, g, a) = min(Q_red(s, g, a), Q_box(s, g, a))
        action = argmax_a max_g Q_composed(s, g, a)
    """
    
    # Goal space - the 4 objects in the environment
    GOALS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}
    
    # Primitive tasks
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    
    # Which goals satisfy which primitive
    PRIMITIVE_GOALS = {
        'red': ['red_box', 'red_sphere'],
        'blue': ['blue_box', 'blue_sphere'],
        'box': ['red_box', 'blue_box'],
        'sphere': ['red_sphere', 'blue_sphere'],
    }
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=128, lstm_size=64, tau=0.005, grad_clip=10.0,
                 r_min=-10.0):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.seq_len = seq_len
        self.num_goals = len(self.GOALS)
        
        # Extended reward parameters
        self.r_min = r_min  # Penalty for reaching wrong goal
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.grad_clip = grad_clip
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WVF Agent using device: {self.device}")
        
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
        print(f"Goal space: {self.GOALS}")
        print(f"Extended reward r_min: {self.r_min}")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # One Q-network per primitive task (all are goal-conditioned)
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.memories = {}
        
        for primitive in self.PRIMITIVES:
            # Online network
            self.q_networks[primitive] = GoalConditionedQNetwork(
                input_shape=self.obs_shape,
                num_goals=self.num_goals,
                action_size=self.action_dim,
                hidden_size=hidden_size,
                lstm_size=lstm_size
            ).to(self.device)
            
            # Target network
            self.target_networks[primitive] = GoalConditionedQNetwork(
                input_shape=self.obs_shape,
                num_goals=self.num_goals,
                action_size=self.action_dim,
                hidden_size=hidden_size,
                lstm_size=lstm_size
            ).to(self.device)
            self.target_networks[primitive].load_state_dict(
                self.q_networks[primitive].state_dict()
            )
            self.target_networks[primitive].eval()
            
            # Optimizer
            self.optimizers[primitive] = optim.Adam(
                self.q_networks[primitive].parameters(),
                lr=learning_rate
            )
            
            # Replay buffer
            self.memories[primitive] = EpisodeReplayBuffer(capacity=memory_size)
        
        # Current state
        self.current_primitive = None
        self.epsilon = epsilon_start
        self.current_hidden = None
        self.current_target_goal = None
        
        total_params = sum(p.numel() for p in self.q_networks['red'].parameters())
        print(f"Parameters per primitive network: {total_params:,}")
        print(f"Total parameters (4 networks): {total_params * 4:,}")
    
    def set_training_primitive(self, primitive):
        """Set which primitive we're currently training."""
        assert primitive in self.PRIMITIVES, f"Unknown primitive: {primitive}"
        self.current_primitive = primitive
        self.epsilon = self.epsilon_start
        print(f"Training primitive: {primitive}")
        print(f"  Valid goals for this task: {self.PRIMITIVE_GOALS[primitive]}")
    
    def sample_target_goal(self):
        """
        Sample a target goal for this episode.
        
        We sample uniformly from ALL goals, not just valid ones for current task.
        This ensures the network learns Q(s, g, a) for ALL goals g.
        """
        goal_name = random.choice(self.GOALS)
        self.current_target_goal = goal_name
        return self.GOAL_TO_IDX[goal_name]
    
    def get_goal_one_hot(self, goal_idx):
        """Convert goal index to one-hot tensor."""
        one_hot = torch.zeros(self.num_goals, device=self.device)
        one_hot[goal_idx] = 1.0
        return one_hot
    
    def compute_extended_reward(self, info, target_goal_idx, step_penalty=-0.005):
        """
        Compute extended reward based on Nangue Tasse's formulation.
        
        Extended reward structure:
        - Reach target goal AND it satisfies primitive: +1
        - Reach target goal but doesn't satisfy primitive: r_min
        - Reach different goal than target: r_min
        - No goal reached: step_penalty
        """
        contacted = info.get('contacted_object', None)
        
        if contacted is None:
            return step_penalty, False
        
        contacted_goal_idx = self.GOAL_TO_IDX.get(contacted, None)
        
        if contacted_goal_idx is None:
            return step_penalty, False
        
        # Check if this goal satisfies the current primitive task
        primitive = self.current_primitive
        valid_goals = self.PRIMITIVE_GOALS[primitive]
        goal_satisfies_task = contacted in valid_goals
        
        # Extended reward logic
        if contacted_goal_idx == target_goal_idx:
            # Reached the target goal
            if goal_satisfies_task:
                reward = 1.0
            else:
                reward = self.r_min
        else:
            # Reached a different goal
            reward = self.r_min
        
        return reward, True  # True = episode done
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame tensor."""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if img.shape[0] in [3, 4]:
            pass
        else:
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
    
    def reset_episode(self, obs):
        """Reset for new episode and sample a target goal."""
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        # Sample target goal for this episode
        target_goal_idx = self.sample_target_goal()
        
        # Reset hidden state
        if self.current_primitive:
            self.current_hidden = self.q_networks[self.current_primitive].init_hidden(
                batch_size=1, device=self.device
            )
        
        return stacked, target_goal_idx
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, target_goal_idx, epsilon=None):
        """Select action using epsilon-greedy, conditioned on target goal."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        goal = self.get_goal_one_hot(target_goal_idx).unsqueeze(0)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_networks[self.current_primitive](
                state, goal, self.current_hidden
            )
            self.current_hidden = (
                self.current_hidden[0].detach(),
                self.current_hidden[1].detach()
            )
            return q_values.argmax().item()
    
    def select_action_composed(self, stacked_obs, features):
        """
        Select action using composed Q-values for zero-shot evaluation.
        
        Boolean composition for conjunction (AND):
            Q_composed(s, g, a) = min over features of Q_feature(s, g, a)
        
        Then: action = argmax_a max_g Q_composed(s, g, a)
        
        Args:
            stacked_obs: Frame-stacked observation
            features: List of primitive features to compose, e.g. ['red', 'box']
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        # Get Q-values for all goals from each feature network
        q_all_features = []
        
        with torch.no_grad():
            for feature in features:
                hidden = self.q_networks[feature].init_hidden(1, self.device)
                q_all_goals, _ = self.q_networks[feature].forward_all_goals(state, hidden)
                q_all_features.append(q_all_goals)
            
            # Stack: (num_features, 1, num_goals, action_size)
            q_stacked = torch.stack(q_all_features, dim=0)
            
            # Conjunction: min over features
            q_composed = q_stacked.min(dim=0)[0]  # (1, num_goals, action_size)
            
            # Max over goals, then argmax over actions
            q_max_goal = q_composed.max(dim=1)[0]  # (1, action_size)
            
            return q_max_goal.argmax().item()
    
    def remember(self, state, goal_idx, action, reward, next_state, done):
        """Store transition with goal information."""
        if self.current_primitive:
            goal_one_hot = np.zeros(self.num_goals, dtype=np.float32)
            goal_one_hot[goal_idx] = 1.0
            
            self.memories[self.current_primitive].push(
                state, goal_one_hot, action, reward, next_state, done
            )
    
    def soft_update_target(self, primitive):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_networks[primitive].parameters(),
            self.q_networks[primitive].parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Training step for current primitive."""
        primitive = self.current_primitive
        memory = self.memories[primitive]
        
        if len(memory) < self.batch_size:
            return 0.0
        
        sequences = memory.sample(self.batch_size, self.seq_len)
        
        # Pad and convert to tensors
        max_len = max(len(seq) for seq in sequences)
        
        states_batch, goals_batch, actions_batch = [], [], []
        rewards_batch, next_states_batch, dones_batch, lengths = [], [], [], []
        
        for seq in sequences:
            seq_len = len(seq)
            lengths.append(seq_len)
            
            states = [s[0] for s in seq]
            goals = [s[1] for s in seq]
            actions = [s[2] for s in seq]
            rewards = [s[3] for s in seq]
            next_states = [s[4] for s in seq]
            dones = [s[5] for s in seq]
            
            if seq_len < max_len:
                pad_len = max_len - seq_len
                states.extend([states[-1]] * pad_len)
                goals.extend([goals[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([0.0] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([True] * pad_len)
            
            states_batch.append(states)
            goals_batch.append(goals)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        goals_t = torch.FloatTensor(np.array(goals_batch)).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(dones_batch).to(self.device)
        
        batch_size, seq_len = states_t.shape[:2]
        
        # Forward pass
        hidden = self.q_networks[primitive].init_hidden(batch_size, self.device)
        target_hidden = self.target_networks[primitive].init_hidden(batch_size, self.device)
        
        q_values_list = []
        for t in range(seq_len):
            q_vals, hidden = self.q_networks[primitive](
                states_t[:, t], goals_t[:, t], hidden
            )
            q_values_list.append(q_vals)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        q_values = torch.stack(q_values_list, dim=1)
        current_q = q_values.gather(2, actions_t.unsqueeze(2)).squeeze(2)
        
        # Double DQN target
        with torch.no_grad():
            next_q_list = []
            hidden_copy = self.q_networks[primitive].init_hidden(batch_size, self.device)
            for t in range(seq_len):
                nq, hidden_copy = self.q_networks[primitive](
                    next_states_t[:, t], goals_t[:, t], hidden_copy
                )
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_values = torch.stack(next_q_list, dim=1)
            next_actions = next_q_values.argmax(2, keepdim=True)
            
            target_q_list = []
            for t in range(seq_len):
                tq, target_hidden = self.target_networks[primitive](
                    next_states_t[:, t], goals_t[:, t], target_hidden
                )
                target_q_list.append(tq)
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
            
            target_q_values = torch.stack(target_q_list, dim=1)
            next_q = target_q_values.gather(2, next_actions).squeeze(2)
            
            target_q = rewards_t + (self.gamma * next_q * ~dones_t)
        
        # Masked loss
        loss_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length] = 1.0
        
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        # Backward
        self.optimizers[primitive].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_networks[primitive].parameters(),
            max_norm=self.grad_clip
        )
        self.optimizers[primitive].step()
        
        # Soft update
        self.soft_update_target(primitive)
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save all networks."""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_goals': self.num_goals,
        }
        
        for primitive in self.PRIMITIVES:
            checkpoint[f'q_network_{primitive}'] = self.q_networks[primitive].state_dict()
            checkpoint[f'target_network_{primitive}'] = self.target_networks[primitive].state_dict()
            checkpoint[f'optimizer_{primitive}'] = self.optimizers[primitive].state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"WVF model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load all networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for primitive in self.PRIMITIVES:
            self.q_networks[primitive].load_state_dict(
                checkpoint[f'q_network_{primitive}']
            )
            self.target_networks[primitive].load_state_dict(
                checkpoint[f'target_network_{primitive}']
            )
            self.optimizers[primitive].load_state_dict(
                checkpoint[f'optimizer_{primitive}']
            )
        
        print(f"WVF model loaded from {filepath}")