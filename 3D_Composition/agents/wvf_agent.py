"""
Unified World Value Functions (WVF) Agent for Compositional RL

CRITICAL CHANGE: Single unified model instead of 4 separate models
- Task conditioning: primitive task encoded as one-hot and tiled like goal
- Random task sampling each episode to prevent catastrophic forgetting
- Same composition logic at evaluation time

Based on Nangue Tasse et al.'s Boolean Task Algebra (NeurIPS 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy


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


class UnifiedGoalTaskLSTMNetwork(nn.Module):
    """
    UNIFIED Goal-Conditioned Q-Network with BOTH Goal and Task Tiling + LSTM.
    
    Key change: Network now sees BOTH the goal AND the primitive task
    
    Architecture:
        [Image (12 ch) | Goal Tiled (4 ch) | Task Tiled (4 ch)] -> CNN -> LSTM -> Dueling Q
        
    Input: (batch, 12 + 4 + 4, H, W) = (batch, 20, 60, 80)
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_goals=4, num_tasks=4, action_size=3,
                 hidden_size=128, lstm_size=64):
        super(UnifiedGoalTaskLSTMNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_goals = num_goals
        self.num_tasks = num_tasks
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN input channels = image channels + goal channels + task channels
        cnn_input_channels = input_shape[0] + num_goals + num_tasks  # 12 + 4 + 4 = 20
        
        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        self._conv_output_size = self._get_conv_output_size(cnn_input_channels, 
                                                             input_shape[1], 
                                                             input_shape[2])
        
        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
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
    
    def tile_conditioning(self, state, goal, task):
        """
        Tile BOTH goal and task across spatial dimensions and concatenate with state.
        
        Args:
            state: (batch, C, H, W) image tensor
            goal: (batch, num_goals) one-hot goal vector
            task: (batch, num_tasks) one-hot task vector
        
        Returns:
            combined: (batch, C + num_goals + num_tasks, H, W)
        """
        batch_size = state.size(0)
        H, W = state.size(2), state.size(3)
        
        # Tile goal: (batch, num_goals) -> (batch, num_goals, H, W)
        goal_expanded = goal.view(batch_size, self.num_goals, 1, 1)
        goal_tiled = goal_expanded.expand(batch_size, self.num_goals, H, W)
        
        # Tile task: (batch, num_tasks) -> (batch, num_tasks, H, W)
        task_expanded = task.view(batch_size, self.num_tasks, 1, 1)
        task_tiled = task_expanded.expand(batch_size, self.num_tasks, H, W)
        
        # Concatenate: state + goal + task
        combined = torch.cat([state, goal_tiled, task_tiled], dim=1)
        
        return combined
    
    def forward(self, state, goal, task, hidden=None):
        """
        Forward pass with goal and task tiling.
        
        Args:
            state: (batch, C, H, W) image tensor
            goal: (batch, num_goals) one-hot goal vector
            task: (batch, num_tasks) one-hot task vector
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: (batch, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # Tile goal and task, concatenate with state
        combined = self.tile_conditioning(state, goal, task)  # (batch, C+4+4, H, W)
        
        # CNN features
        conv_features = self.conv(combined)
        conv_features = conv_features.view(batch_size, -1)  # (batch, conv_output)
        
        # LSTM expects (batch, seq_len, features)
        conv_features = conv_features.unsqueeze(1)  # (batch, 1, conv_output)
        
        if hidden is not None:
            lstm_out, hidden = self.lstm(conv_features, hidden)
        else:
            lstm_out, hidden = self.lstm(conv_features)
        
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Dueling Q-values
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for LSTM training.
    Stores complete episodes and samples sequences for training.
    """
    
    def __init__(self, capacity=2000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push(self, state, goal_idx, task_idx, action, reward, next_state, done):
        """Add a transition to the current episode."""
        self.current_episode.append((state, goal_idx, task_idx, action, reward, next_state, done))
        if done:
            self.end_episode()
    
    def end_episode(self):
        """Finalize and store the current episode."""
        if len(self.current_episode) > 0:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=4):
        """Sample sequences from stored episodes."""
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
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = []


class UnifiedWorldValueFunctionAgent:
    """
    UNIFIED World Value Function (WVF) Agent
    
    KEY CHANGE: Single model for all primitive tasks
    - Task conditioning via one-hot tiled across image
    - Random task sampling each episode
    - Same composition at evaluation
    
    Goals: [red_box, blue_box, red_sphere, blue_sphere]
    Tasks: [red, blue, box, sphere]
    """
    
    # Goal space - the 4 objects in the environment
    GOALS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}
    IDX_TO_GOAL = {i: g for i, g in enumerate(GOALS)}
    
    # Primitive tasks and which goals are valid for each
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    TASK_TO_IDX = {t: i for i, t in enumerate(PRIMITIVES)}
    IDX_TO_TASK = {i: t for i, t in enumerate(PRIMITIVES)}
    
    VALID_GOALS = {
        'red': ['red_box', 'red_sphere'],
        'blue': ['blue_box', 'blue_sphere'],
        'box': ['red_box', 'blue_box'],
        'sphere': ['red_sphere', 'blue_sphere'],
    }
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=128, lstm_size=64,
                 tau=0.005, grad_clip=10.0,
                 r_min=-1.0, r_correct=1.0, r_wrong=-1.0, step_penalty=-0.01):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.num_goals = len(self.GOALS)
        self.num_tasks = len(self.PRIMITIVES)
        self.seq_len = seq_len
        
        # Extended reward parameters
        self.r_min = r_min
        self.r_correct = r_correct
        self.r_wrong = r_wrong
        self.step_penalty = step_penalty
        
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
        print(f"UnifiedWorldValueFunctionAgent using device: {self.device}")
        
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
        print(f"Goal conditioning: Tiled as {self.num_goals} extra channels")
        print(f"Task conditioning: Tiled as {self.num_tasks} extra channels")
        print(f"CNN input shape: ({self.obs_shape[0] + self.num_goals + self.num_tasks}, {self.obs_shape[1]}, {self.obs_shape[2]})")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Create UNIFIED networks
        self._create_networks()
        
        # Current episode state
        self.current_hidden = None
    
    def _create_networks(self):
        """Create unified networks."""
        self.q_network = UnifiedGoalTaskLSTMNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            num_tasks=self.num_tasks,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        
        self.target_network = UnifiedGoalTaskLSTMNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            num_tasks=self.num_tasks,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = EpisodeReplayBuffer(capacity=self.memory_size)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"  Unified network parameters: {total_params:,}")
    
    def sample_task(self):
        """Randomly sample a primitive task for this episode."""
        return random.choice(self.PRIMITIVES)
    
    def get_valid_goal_indices(self, primitive):
        """Get indices of valid goals for a primitive task."""
        valid_goals = self.VALID_GOALS[primitive]
        return [self.GOAL_TO_IDX[g] for g in valid_goals]
    
    def sample_target_goal(self, current_task, valid_bias=0.8):
        """
        Sample a target goal with bias toward valid goals for current task.
        
        80% valid goals (strong learning signal)
        20% invalid goals (learn they're bad)
        """
        valid_goals = self.VALID_GOALS[current_task]
        
        if random.random() < valid_bias:
            goal_name = random.choice(valid_goals)
        else:
            invalid_goals = [g for g in self.GOALS if g not in valid_goals]
            goal_name = random.choice(invalid_goals)
        
        return self.GOAL_TO_IDX[goal_name]
    
    def get_goal_one_hot(self, goal_idx, batch_size=1):
        """Convert goal index to one-hot tensor."""
        if isinstance(goal_idx, (list, np.ndarray)):
            one_hot = torch.zeros(len(goal_idx), self.num_goals, device=self.device)
            for i, idx in enumerate(goal_idx):
                one_hot[i, int(idx)] = 1.0
        else:
            one_hot = torch.zeros(batch_size, self.num_goals, device=self.device)
            one_hot[:, goal_idx] = 1.0
        return one_hot
    
    def get_task_one_hot(self, task_idx, batch_size=1):
        """Convert task index to one-hot tensor."""
        if isinstance(task_idx, (list, np.ndarray)):
            one_hot = torch.zeros(len(task_idx), self.num_tasks, device=self.device)
            for i, idx in enumerate(task_idx):
                one_hot[i, int(idx)] = 1.0
        else:
            one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
            one_hot[:, task_idx] = 1.0
        return one_hot
    
    def compute_rewards(self, info, target_goal_idx, current_task):
        """
        Compute TRUE reward and EXTENDED reward.
        
        Extended reward: +1 for ANY valid goal that satisfies the current task.
        """
        contacted = info.get('contacted_object', None)
        
        if contacted is None:
            return 0.0, self.step_penalty, False
        
        target_goal_name = self.IDX_TO_GOAL[target_goal_idx]
        contacted_goal_idx = self.GOAL_TO_IDX.get(contacted, None)
        
        if contacted_goal_idx is None:
            return 0.0, self.step_penalty, False
        
        # TRUE reward: did we satisfy the current task?
        valid_goals_for_task = self.VALID_GOALS[current_task]
        task_satisfied = contacted in valid_goals_for_task
        true_reward = 1.0 if task_satisfied else 0.0
        
        # EXTENDED reward: +1 for ANY object that satisfies the task
        if task_satisfied:
            extended_reward = self.r_correct
        else:
            extended_reward = self.r_wrong
        
        return true_reward, extended_reward, True
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame."""
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
    
    def reset_episode(self, obs):
        """Reset for new episode."""
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        return stacked
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, target_goal_idx, current_task_idx, epsilon=None):
        """Select action using epsilon-greedy, conditioned on goal and task."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        goal = self.get_goal_one_hot(target_goal_idx)
        task = self.get_task_one_hot(current_task_idx)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_network(state, goal, task, self.current_hidden)
            if self.current_hidden is not None:
                self.current_hidden = (self.current_hidden[0].detach(),
                                       self.current_hidden[1].detach())
            return q_values.argmax().item()
    
    def select_action_composed(self, stacked_obs, features, target_goal_idx):
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        goal_one_hot = self.get_goal_one_hot(target_goal_idx)
        
        q_values_per_feature = []
        with torch.no_grad():
            for feature in features:
                task_idx = self.TASK_TO_IDX[feature]
                task_one_hot = self.get_task_one_hot(task_idx)
                
                # Initialize fresh hidden state for each feature evaluation
                hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
                q_vals, _ = self.q_network(state, goal_one_hot, task_one_hot, hidden)
                q_values_per_feature.append(q_vals)
            
            q_composed = torch.stack(q_values_per_feature, dim=0).min(dim=0)[0]
            return q_composed.argmax().item()
    
    def remember(self, state, goal_idx, task_idx, action, reward, next_state, done):
        """Store transition."""
        self.memory.push(state, goal_idx, task_idx, action, reward, next_state, done)
    
    def soft_update_target(self):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        sequences = self.memory.sample(self.batch_size, self.seq_len)
        max_len = max(len(seq) for seq in sequences)
        
        states_batch, goals_batch, tasks_batch, actions_batch = [], [], [], []
        rewards_batch, next_states_batch, dones_batch, lengths = [], [], [], []
        
        for seq in sequences:
            seq_len_actual = len(seq)
            lengths.append(seq_len_actual)
            
            states = [s[0] for s in seq]
            goals = [s[1] for s in seq]
            tasks = [s[2] for s in seq]
            actions = [s[3] for s in seq]
            rewards = [s[4] for s in seq]
            next_states = [s[5] for s in seq]
            dones = [s[6] for s in seq]
            
            if seq_len_actual < max_len:
                pad_len = max_len - seq_len_actual
                states.extend([states[-1]] * pad_len)
                goals.extend([goals[-1]] * pad_len)
                tasks.extend([tasks[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([0.0] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([True] * pad_len)
            
            states_batch.append(states)
            goals_batch.append(goals)
            tasks_batch.append(tasks)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        goals_t = torch.LongTensor(goals_batch).to(self.device)
        tasks_t = torch.LongTensor(tasks_batch).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(dones_batch).to(self.device)
        
        batch_size, seq_len = states_t.shape[:2]
        
        hidden = self.q_network.init_hidden(batch_size, self.device)
        target_hidden = self.target_network.init_hidden(batch_size, self.device)
        
        # Forward pass through online network
        q_values_list = []
        for t in range(seq_len):
            goal_one_hot = torch.zeros(batch_size, self.num_goals, device=self.device)
            task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
            for b in range(batch_size):
                goal_one_hot[b, goals_t[b, t]] = 1.0
                task_one_hot[b, tasks_t[b, t]] = 1.0
            
            q_vals, hidden = self.q_network(states_t[:, t], goal_one_hot, task_one_hot, hidden)
            q_values_list.append(q_vals)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        q_values = torch.stack(q_values_list, dim=1)
        current_q = q_values.gather(2, actions_t.unsqueeze(2)).squeeze(2)
        
        # Target Network (Double DQN)
        with torch.no_grad():
            # Get next actions from online network
            next_q_list = []
            hidden_copy = self.q_network.init_hidden(batch_size, self.device)
            for t in range(seq_len):
                goal_one_hot = torch.zeros(batch_size, self.num_goals, device=self.device)
                task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
                for b in range(batch_size):
                    goal_one_hot[b, goals_t[b, t]] = 1.0
                    task_one_hot[b, tasks_t[b, t]] = 1.0
                
                nq, hidden_copy = self.q_network(next_states_t[:, t], goal_one_hot, task_one_hot, hidden_copy)
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_values = torch.stack(next_q_list, dim=1)
            next_actions = next_q_values.argmax(2, keepdim=True)
            
            # Evaluate actions with target network
            target_q_list = []
            for t in range(seq_len):
                goal_one_hot = torch.zeros(batch_size, self.num_goals, device=self.device)
                task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
                for b in range(batch_size):
                    goal_one_hot[b, goals_t[b, t]] = 1.0
                    task_one_hot[b, tasks_t[b, t]] = 1.0
                
                tq, target_hidden = self.target_network(next_states_t[:, t], goal_one_hot, task_one_hot, target_hidden)
                target_q_list.append(tq)
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
            
            target_q_values = torch.stack(target_q_list, dim=1)
            next_q = target_q_values.gather(2, next_actions).squeeze(2)
            
            target_q = rewards_t + (self.gamma * next_q * ~dones_t)
        
        # Masking for variable-length sequences
        loss_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length] = 1.0
        
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        self.soft_update_target()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save unified model."""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_goals': self.num_goals,
            'num_tasks': self.num_tasks,
            'hidden_size': self.hidden_size,
            'lstm_size': self.lstm_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Unified WVF model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load unified model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hidden_size = checkpoint.get('hidden_size', 128)
        self.lstm_size = checkpoint.get('lstm_size', 64)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"Unified WVF model loaded from {filepath}")