"""
Unified World Value Functions (WVF) Agent for Compositional RL

OPTION A REWRITE - Pure Task Conditioning (No Goal Conditioning)

Key insight from WVF theory (Nangue Tasse et al.):
- Learn Q(s, a, task) for each primitive task
- Primitive tasks: red, blue, box, sphere
- Composition: Q_composed(s, a) = min(Q(s, a, task1), Q(s, a, task2))
  - min = AND (pessimistic: action must be good for BOTH tasks)

This is simpler and matches the WVF theory more closely:
- No goal conditioning needed
- Task tells you WHAT to look for (color or shape)
- Reward is +1 for reaching ANY object satisfying the task
- Composition naturally emerges from min over task Q-values

Based on Nangue Tasse et al.'s Boolean Task Algebra (NeurIPS 2020)
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


class TaskConditionedLSTMNetwork(nn.Module):
    """
    Task-Conditioned Q-Network with LSTM.
    
    NO goal conditioning - only task conditioning.
    This matches WVF theory: learn Q(s, a, task) for primitive tasks,
    then compose via min for AND.
    
    Architecture:
        [Image (12 ch) | Task Tiled (4 ch)] -> CNN -> LSTM -> Dueling Q
        
    Input: (batch, 12 + 4, H, W) = (batch, 16, 60, 80)
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_tasks=4, action_size=3,
                 hidden_size=128, lstm_size=64):
        super(TaskConditionedLSTMNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_tasks = num_tasks
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN input channels = image channels + task channels (NO goal channels)
        cnn_input_channels = input_shape[0] + num_tasks  # 12 + 4 = 16
        
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
    
    def tile_task(self, state, task):
        """
        Tile task across spatial dimensions and concatenate with state.
        
        Args:
            state: (batch, C, H, W) image tensor
            task: (batch, num_tasks) one-hot task vector
        
        Returns:
            combined: (batch, C + num_tasks, H, W)
        """
        batch_size = state.size(0)
        H, W = state.size(2), state.size(3)
        
        # Tile task: (batch, num_tasks) -> (batch, num_tasks, H, W)
        task_expanded = task.view(batch_size, self.num_tasks, 1, 1)
        task_tiled = task_expanded.expand(batch_size, self.num_tasks, H, W)
        
        # Concatenate: state + task
        combined = torch.cat([state, task_tiled], dim=1)
        
        return combined
    
    def forward(self, state, task, hidden=None):
        """
        Forward pass with task tiling.
        
        Args:
            state: (batch, C, H, W) image tensor
            task: (batch, num_tasks) one-hot task vector
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: (batch, action_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # Tile task and concatenate with state
        combined = self.tile_task(state, task)  # (batch, C+4, H, W)
        
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
    
    def push(self, state, task_idx, action, reward, next_state, done):
        """Add a transition to the current episode."""
        self.current_episode.append((state, task_idx, action, reward, next_state, done))
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
    UNIFIED World Value Function (WVF) Agent - OPTION A
    
    Pure Task Conditioning (No Goal Conditioning)
    
    Theory:
    - Learn Q(s, a, task) for primitive tasks: red, blue, box, sphere
    - Each task defines a reward function: +1 for ANY object satisfying the task
    - Composition via min: Q_AND(s, a) = min(Q(s,a,task1), Q(s,a,task2))
    
    Example:
    - Q(s, a, task=blue) = expected return for reaching any blue object
    - Q(s, a, task=sphere) = expected return for reaching any sphere
    - Q_blue_sphere(s, a) = min(Q_blue, Q_sphere) = value for reaching blue AND sphere
    
    This matches WVF theory exactly.
    
    Tasks: [red, blue, box, sphere]
    Valid objects per task:
        red -> {red_box, red_sphere}
        blue -> {blue_box, blue_sphere}  
        box -> {red_box, blue_box}
        sphere -> {red_sphere, blue_sphere}
    """
    
    # Primitive tasks
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    TASK_TO_IDX = {t: i for i, t in enumerate(PRIMITIVES)}
    IDX_TO_TASK = {i: t for i, t in enumerate(PRIMITIVES)}
    
    # Which objects satisfy each task
    VALID_OBJECTS = {
        'red': ['red_box', 'red_sphere'],
        'blue': ['blue_box', 'blue_sphere'],
        'box': ['red_box', 'blue_box'],
        'sphere': ['red_sphere', 'blue_sphere'],
    }
    
    # All possible objects
    ALL_OBJECTS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=128, lstm_size=64,
                 tau=0.005, grad_clip=10.0,
                 r_correct=1.0, r_wrong=-0.1, step_penalty=-0.005):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.num_tasks = len(self.PRIMITIVES)
        self.seq_len = seq_len
        
        # Reward parameters
        self.r_correct = r_correct    # Reward for reaching valid object
        self.r_wrong = r_wrong        # Penalty for reaching wrong object
        self.step_penalty = step_penalty  # Small step penalty
        
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
        print(f"UnifiedWorldValueFunctionAgent (Option A) using device: {self.device}")
        
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
        print(f"Task conditioning: Tiled as {self.num_tasks} extra channels")
        print(f"CNN input shape: ({self.obs_shape[0] + self.num_tasks}, {self.obs_shape[1]}, {self.obs_shape[2]})")
        print(f"NO goal conditioning - pure WVF approach")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Create networks
        self._create_networks()
        
        # Current episode state
        self.current_hidden = None
        self.current_task_idx = None
    
    def _create_networks(self):
        """Create task-conditioned networks."""
        self.q_network = TaskConditionedLSTMNetwork(
            input_shape=self.obs_shape,
            num_tasks=self.num_tasks,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        
        self.target_network = TaskConditionedLSTMNetwork(
            input_shape=self.obs_shape,
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
        print(f"  Task-conditioned network parameters: {total_params:,}")
    
    def sample_task(self):
        """Randomly sample a primitive task for this episode."""
        return random.choice(self.PRIMITIVES)
    
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
    
    def compute_reward(self, info, current_task):
        """
        Compute reward based on task satisfaction.
        
        Simple WVF reward:
        - +1 for reaching ANY object that satisfies the current task
        - -0.1 for reaching wrong object
        - small step penalty otherwise
        
        Args:
            info: Environment info dict with 'contacted_object'
            current_task: Current primitive task name (e.g., 'blue')
            
        Returns:
            reward: float
            done: bool (True if any object was contacted)
        """
        contacted = info.get('contacted_object', None)
        
        if contacted is None:
            return self.step_penalty, False
        
        # Check if contacted object satisfies the task
        valid_objects = self.VALID_OBJECTS[current_task]
        task_satisfied = contacted in valid_objects
        
        if task_satisfied:
            return self.r_correct, True
        else:
            return self.r_wrong, True
    
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
    
    def reset_episode(self, obs, task_name=None):
        """
        Reset for new episode.
        
        Args:
            obs: Initial observation
            task_name: Optional task name (if None, will be set later)
        """
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        # Initialize fresh hidden state for new episode
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        # Set current task if provided
        if task_name is not None:
            self.current_task_idx = self.TASK_TO_IDX[task_name]
        
        return stacked
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, task_idx=None, epsilon=None):
        """
        Select action using epsilon-greedy, conditioned on task.
        
        Args:
            stacked_obs: Stacked observation frames
            task_idx: Task index (uses self.current_task_idx if None)
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            action: Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if task_idx is None:
            task_idx = self.current_task_idx
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        task = self.get_task_one_hot(task_idx)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_network(state, task, self.current_hidden)
            # Detach hidden state to prevent gradient accumulation
            self.current_hidden = (self.current_hidden[0].detach(),
                                   self.current_hidden[1].detach())
            return q_values.argmax().item()
    
    def select_action_composed(self, stacked_obs, features):
        """
        Select action using Boolean composition (min over task Q-values).
        
        This is the core WVF composition:
        Q_AND(s, a) = min(Q(s, a, task1), Q(s, a, task2))
        
        Args:
            stacked_obs: Current stacked observation
            features: List of primitive tasks to compose (e.g., ['blue', 'sphere'])
            
        Returns:
            action: The action that maximizes the composed Q-value
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        q_values_per_task = []
        
        with torch.no_grad():
            # Evaluate Q-values for each primitive task
            for task_name in features:
                task_idx = self.TASK_TO_IDX[task_name]
                task_one_hot = self.get_task_one_hot(task_idx)
                
                # Use current hidden state for temporal context
                q_vals, _ = self.q_network(state, task_one_hot, self.current_hidden)
                q_values_per_task.append(q_vals)
            
            # Boolean AND = min over Q-values (pessimistic composition)
            # Shape: (num_tasks, batch=1, num_actions) -> min over dim 0
            q_composed = torch.stack(q_values_per_task, dim=0).min(dim=0)[0]
            best_action = q_composed.argmax().item()
            
            # Update hidden state using the first task (arbitrary but consistent)
            # This maintains temporal context across steps
            first_task_idx = self.TASK_TO_IDX[features[0]]
            first_task_one_hot = self.get_task_one_hot(first_task_idx)
            _, new_hidden = self.q_network(state, first_task_one_hot, self.current_hidden)
            self.current_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            return best_action
    
    def remember(self, state, task_idx, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, task_idx, action, reward, next_state, done)
    
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
        
        states_batch, tasks_batch, actions_batch = [], [], []
        rewards_batch, next_states_batch, dones_batch, lengths = [], [], [], []
        
        for seq in sequences:
            seq_len_actual = len(seq)
            lengths.append(seq_len_actual)
            
            states = [s[0] for s in seq]
            tasks = [s[1] for s in seq]
            actions = [s[2] for s in seq]
            rewards = [s[3] for s in seq]
            next_states = [s[4] for s in seq]
            dones = [s[5] for s in seq]
            
            # Pad sequences to max length
            if seq_len_actual < max_len:
                pad_len = max_len - seq_len_actual
                states.extend([states[-1]] * pad_len)
                tasks.extend([tasks[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([0.0] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([True] * pad_len)
            
            states_batch.append(states)
            tasks_batch.append(tasks)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        tasks_t = torch.LongTensor(tasks_batch).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(dones_batch).to(self.device)
        
        batch_size, seq_len = states_t.shape[:2]
        
        # Initialize hidden states
        hidden = self.q_network.init_hidden(batch_size, self.device)
        target_hidden = self.target_network.init_hidden(batch_size, self.device)
        
        # Forward pass through online network
        q_values_list = []
        for t in range(seq_len):
            task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
            for b in range(batch_size):
                task_one_hot[b, tasks_t[b, t]] = 1.0
            
            q_vals, hidden = self.q_network(states_t[:, t], task_one_hot, hidden)
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
                task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
                for b in range(batch_size):
                    task_one_hot[b, tasks_t[b, t]] = 1.0
                
                nq, hidden_copy = self.q_network(next_states_t[:, t], task_one_hot, hidden_copy)
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_values = torch.stack(next_q_list, dim=1)
            next_actions = next_q_values.argmax(2, keepdim=True)
            
            # Evaluate actions with target network
            target_q_list = []
            for t in range(seq_len):
                task_one_hot = torch.zeros(batch_size, self.num_tasks, device=self.device)
                for b in range(batch_size):
                    task_one_hot[b, tasks_t[b, t]] = 1.0
                
                tq, target_hidden = self.target_network(next_states_t[:, t], task_one_hot, target_hidden)
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
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset exploration rate to initial value."""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_tasks': self.num_tasks,
            'hidden_size': self.hidden_size,
            'lstm_size': self.lstm_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"WVF model (Option A) saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hidden_size = checkpoint.get('hidden_size', 128)
        self.lstm_size = checkpoint.get('lstm_size', 64)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"WVF model (Option A) loaded from {filepath}")