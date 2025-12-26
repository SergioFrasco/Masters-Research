"""
Unified LSTM-DQN Agent with Extended Task Conditioning

Key Changes from Original:
1. Task space now includes green: [red, blue, green, box, sphere]
2. Input shape: (k*3 + 5, H, W) - stacked RGB frames + 5 task channels
3. Green never appears during training, only evaluation
4. Compositional task encoding for zero-shot generalization testing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class FrameStack:
    """
    Stack the last k frames to give the agent short-term memory.
    Now also handles task channel appending with 5-dim task space.
    """
    
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
        self.current_task_channels = None
    
    def reset(self, frame, task_channels):
        """Reset with a new frame and task, filling the stack with copies"""
        self.current_task_channels = task_channels
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stacked()
    
    def step(self, frame):
        """Add a new frame and return stacked observation with task"""
        self.frames.append(frame.copy())
        return self._get_stacked()
    
    def _get_stacked(self):
        """
        Stack frames along channel dimension: (k*C, H, W)
        Then append task channels: (k*C + num_tasks, H, W)
        """
        stacked_frames = np.concatenate(list(self.frames), axis=0)
        
        if self.current_task_channels is not None:
            # Append task channels
            return np.concatenate([stacked_frames, self.current_task_channels], axis=0)
        
        return stacked_frames
    
    def __len__(self):
        return len(self.frames)


class UnifiedHybridLSTM_DQN3D(nn.Module):
    """
    Unified Hybrid architecture with Extended Goal Tiling (5-dim task space)
    
    Flow:
    1. Stacked frames + task channels (12+5, 60, 80) → CNN → features
    2. Features → LSTM (128 hidden) → temporal embedding
    3. Temporal embedding → FC layers → Q-values
    
    The task is encoded as 5 additional binary channels (one-hot).
    Task space: [red, blue, green, box, sphere]
    NOTE: Green is reserved for zero-shot evaluation only!
    """
    
    def __init__(self, input_shape=(17, 60, 80), action_size=3, 
                 hidden_size=256, lstm_size=128):
        super(UnifiedHybridLSTM_DQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN processes stacked frames + task channels
        # Input: (17, 60, 80) if k=4 stacking + 5 task channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # Small LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
            return int(np.prod(output.shape[1:]))
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
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
    
    def forward(self, x, hidden=None):
        """
        Forward pass with optional hidden state.
        
        Args:
            x: Input tensor (batch, channels+task, height, width)
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: Q-values for each action
            hidden: Updated LSTM hidden state
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = self.conv(x)
        features = features.view(batch_size, -1)
        
        # Reshape for LSTM: (batch, seq_len=1, features)
        features = features.unsqueeze(1)
        
        # LSTM processing
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            lstm_out, hidden = self.lstm(features)
        
        # Take the last output from the sequence
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        q_values = self.fc(lstm_out)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class DuelingUnifiedHybridLSTM_DQN3D(nn.Module):
    """
    Dueling version of the Unified Hybrid LSTM-DQN with extended goal tiling.
    Task space: [red, blue, green, box, sphere] (5 dimensions)
    """
    
    def __init__(self, input_shape=(17, 60, 80), action_size=3, 
                 hidden_size=256, lstm_size=128):
        super(DuelingUnifiedHybridLSTM_DQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # Shared CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
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
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # CNN features
        features = self.conv(x)
        features = features.view(batch_size, -1)
        features = features.unsqueeze(1)
        
        # LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            lstm_out, hidden = self.lstm(features)
        
        lstm_out = lstm_out[:, -1, :]
        
        # Dueling architecture
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        
        # Combine with mean subtraction
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class TaskAwareEpisodeReplayBuffer:
    """
    Episode-based replay buffer that tracks tasks and ensures balanced sampling.
    
    Each episode is stored with its task label, allowing us to:
    1. Track task distribution in the buffer
    2. Sample balanced batches across tasks
    3. Monitor per-task replay frequency
    
    NOTE: Only tracks training tasks (red, blue, box, sphere) - NOT green!
    """
    
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
        self.current_task = None
        
        # Track task distribution (training tasks only)
        self.task_counts = {'red': 0, 'blue': 0, 'box': 0, 'sphere': 0}
    
    def start_episode(self, task_name):
        """Start a new episode with the given task"""
        self.current_episode = []
        self.current_task = task_name
    
    def push_transition(self, state, action, reward, next_state, done):
        """Add a transition to the current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        """Finalize the current episode and store it with task label"""
        if len(self.current_episode) > 0 and self.current_task is not None:
            self.episodes.append({
                'task': self.current_task,
                'transitions': self.current_episode
            })
            if self.current_task in self.task_counts:
                self.task_counts[self.current_task] += 1
            self.current_episode = []
            self.current_task = None
    
    def sample(self, batch_size, seq_len=8):
        """
        Sample a batch of sequences with balanced task representation.
        
        Strategy: Sample roughly equal number of sequences from each task.
        """
        if len(self.episodes) < batch_size:
            return []
        
        # Group episodes by task
        task_episodes = {'red': [], 'blue': [], 'box': [], 'sphere': []}
        for ep_data in self.episodes:
            task = ep_data['task']
            if task in task_episodes:
                task_episodes[task].append(ep_data['transitions'])
        
        # Calculate samples per task (roughly equal)
        sequences_per_task = batch_size // 4
        remainder = batch_size % 4
        
        batch = []
        tasks_with_episodes = [t for t in task_episodes if len(task_episodes[t]) > 0]
        
        if len(tasks_with_episodes) == 0:
            return []
        
        # Sample from each task
        for task in tasks_with_episodes:
            n_samples = sequences_per_task
            if remainder > 0:
                n_samples += 1
                remainder -= 1
            
            episodes_for_task = task_episodes[task]
            for _ in range(min(n_samples, len(episodes_for_task))):
                episode = random.choice(episodes_for_task)
                
                if len(episode) > seq_len:
                    start = random.randint(0, len(episode) - seq_len)
                    sequence = episode[start:start + seq_len]
                else:
                    sequence = episode
                
                batch.append(sequence)
        
        # If we didn't get enough sequences, fill from any available task
        while len(batch) < batch_size and len(self.episodes) > 0:
            ep_data = random.choice(self.episodes)
            episode = ep_data['transitions']
            
            if len(episode) > seq_len:
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                sequence = episode
            
            batch.append(sequence)
        
        return batch
    
    def get_task_distribution(self):
        """Return the distribution of tasks in the buffer"""
        total = sum(self.task_counts.values())
        if total == 0:
            return {t: 0.0 for t in self.task_counts}
        return {t: count / total for t, count in self.task_counts.items()}
    
    def __len__(self):
        return len(self.episodes)


class UnifiedLSTMDQNAgent3D:
    """
    Unified LSTM-DQN Agent with Extended Task Conditioning.
    
    Key features:
    1. Goal tiling with 5-dim task space: [red, blue, green, box, sphere]
    2. Green is NEVER used during training - reserved for zero-shot evaluation
    3. Compositional task encoding for testing generalization
    4. Single model trained on all tasks with uniform sampling
    5. Task-aware replay buffer with balanced sampling
    6. Per-task performance tracking
    
    IMPORTANT: This allows fair comparison with UnifiedDQNAgent for:
    - Compositional generalization (combining learned primitives)
    - Zero-shot generalization (unseen green objects)
    """
    
    # Task encoding: one-hot vectors for [red, blue, green, box, sphere]
    # NOTE: Green (index 2) is NEVER used during training!
    TASK_ENCODINGS = {
        'red': [1.0, 0.0, 0.0, 0.0, 0.0],
        'blue': [0.0, 1.0, 0.0, 0.0, 0.0],
        'green': [0.0, 0.0, 1.0, 0.0, 0.0],  # NEVER used in training!
        'box': [0.0, 0.0, 0.0, 1.0, 0.0],
        'sphere': [0.0, 0.0, 0.0, 0.0, 1.0],
    }
    
    NUM_TASKS = 5  # Extended to include green
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
                 memory_size=5000, batch_size=32, seq_len=8,
                 hidden_size=256, lstm_size=128, use_dueling=True,
                 tau=0.005, use_double_dqn=True, grad_clip=10.0):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.seq_len = seq_len
        self.num_tasks = self.NUM_TASKS
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Stability parameters
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.grad_clip = grad_clip
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Unified LSTM-DQN Agent using device: {self.device}")
        print(f"Frame stacking: k={k_frames}, LSTM size: {lstm_size}")
        print(f"Task space: [red, blue, green, box, sphere] (5 dimensions)")
        print(f"NOTE: Green reserved for zero-shot evaluation only!")
        
        # Get observation shape
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        # Determine single frame shape (C, H, W)
        if sample_img.shape[0] in [3, 4]:
            self.single_frame_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.single_frame_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        # Stacked observation shape WITH task channels
        # k frames * 3 channels + 5 task channels
        self.obs_shape = (
            k_frames * 3 + self.num_tasks,
            self.single_frame_shape[1],
            self.single_frame_shape[2]
        )
        
        print(f"Single frame shape: {self.single_frame_shape}")
        print(f"Stacked observation shape (with task): {self.obs_shape}")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Current task tracking
        self.current_task = None
        self.current_task_channels = None
        
        # Initialize networks
        NetworkClass = DuelingUnifiedHybridLSTM_DQN3D if use_dueling else UnifiedHybridLSTM_DQN3D
        self.q_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            lstm_size=lstm_size
        ).to(self.device)
        
        self.target_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            lstm_size=lstm_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Task-aware replay buffer
        self.memory = TaskAwareEpisodeReplayBuffer(capacity=memory_size)
        
        # Hidden state tracking
        self.current_hidden = None
        
        # Training tracking
        self.update_counter = 0
        self.training_steps = 0
        
        # Per-task performance tracking (training tasks only)
        self.task_success_rates = {t: deque(maxlen=100) for t in ['red', 'blue', 'box', 'sphere']}
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total network parameters: {total_params:,}")
    
    def encode_task(self, task_name):
        """Convert task name to one-hot encoding."""
        if task_name not in self.TASK_ENCODINGS:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(self.TASK_ENCODINGS.keys())}")
        return self.TASK_ENCODINGS[task_name]
    
    def encode_compositional_task(self, features, method='superposition'):
        """
        Encode compositional task with multiple features.
        
        This is the KEY to zero-shot generalization:
        - Trained on: ['red', 'box'], ['blue', 'sphere'], etc.
        - Can compose: ['green', 'box'] without ever seeing it!
        
        Args:
            features: list of feature names (e.g., ['green', 'box'])
            method: 'superposition' (average) or 'max'
        
        Returns:
            encoding: list of floats (5-dim)
        """
        encoding = np.zeros(self.num_tasks, dtype=np.float32)
        
        for feature in features:
            if feature not in self.TASK_ENCODINGS:
                raise ValueError(f"Unknown feature: {feature}")
            feature_encoding = np.array(self.TASK_ENCODINGS[feature], dtype=np.float32)
            
            if method == 'superposition':
                encoding += feature_encoding
            elif method == 'max':
                encoding = np.maximum(encoding, feature_encoding)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        if method == 'superposition' and len(features) > 1:
            encoding = encoding / len(features)
        
        return encoding.tolist()
    
    def create_task_channels(self, task_identifier):
        """
        Create task encoding channels to append to frames.
        
        Args:
            task_identifier: str (like 'red') OR list of features (['green', 'box'])
        
        Returns: (num_tasks, H, W) array of task channels
        """
        H, W = self.single_frame_shape[1], self.single_frame_shape[2]
        
        # Get encoding based on task type
        if isinstance(task_identifier, list):
            # Compositional task
            encoding = self.encode_compositional_task(task_identifier, method='superposition')
        else:
            # Simple task
            encoding = self.encode_task(task_identifier)
        
        # Tile encoding across spatial dimensions
        task_channels = np.zeros((self.num_tasks, H, W), dtype=np.float32)
        for i, val in enumerate(encoding):
            task_channels[i, :, :] = val
        
        return task_channels
    
    def set_task(self, task_identifier):
        """
        Set the current task for the episode.
        
        Args:
            task_identifier: str (like 'red') OR list of features (['green', 'box'])
        """
        self.current_task = task_identifier
        self.current_task_channels = self.create_task_channels(task_identifier)
        
        # Only start episode in replay buffer for training tasks
        if isinstance(task_identifier, str) and task_identifier in ['red', 'blue', 'box', 'sphere']:
            self.memory.start_episode(task_identifier)
    
    def soft_update_target_network(self):
        """Soft update target network using Polyak averaging"""
        for target_param, online_param in zip(self.target_network.parameters(), 
                                               self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame tensor (C, H, W)"""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if isinstance(img, np.ndarray):
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
    
    def reset_episode(self, obs, task_identifier):
        """
        Reset for a new episode with specified task.
        
        Args:
            obs: initial observation
            task_identifier: str (like 'red') OR list of features (['green', 'box'])
        
        Returns:
            stacked_obs: Frame-stacked observation with task channels
        """
        # Set task
        self.set_task(task_identifier)
        
        # Preprocess and stack frames
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame, self.current_task_channels)
        
        # Reset LSTM hidden state
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        return stacked
    
    def step_episode(self, obs):
        """
        Process a new observation during an episode.
        Task channels are maintained from reset_episode.
        
        Returns:
            stacked_obs: Frame-stacked observation with task channels
        """
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.step(frame)
        return stacked
    
    def select_action(self, stacked_obs, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        Maintains hidden state across the episode.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_network(state, self.current_hidden)
            self.current_hidden = (self.current_hidden[0].detach(), 
                                  self.current_hidden[1].detach())
            return q_values.argmax().item()
    
    def remember(self, stacked_obs, action, reward, next_stacked_obs, done):
        """Store transition in episode buffer"""
        self.memory.push_transition(stacked_obs, action, reward, next_stacked_obs, done)
    
    def train_step(self):
        """
        Perform one BATCHED training step with sequence sampling.
        Samples balanced across tasks.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample sequences from episodes (balanced across tasks)
        sequences = self.memory.sample(self.batch_size, self.seq_len)
        
        if len(sequences) == 0:
            return 0.0
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        lengths = []
        
        for seq in sequences:
            seq_len = len(seq)
            lengths.append(seq_len)
            
            states = [s[0] for s in seq]
            actions = [s[1] for s in seq]
            rewards = [s[2] for s in seq]
            next_states = [s[3] for s in seq]
            dones = [s[4] for s in seq]
            
            # Pad if needed
            if seq_len < max_len:
                pad_len = max_len - seq_len
                states.extend([states[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([0.0] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([True] * pad_len)
            
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_tensor = torch.LongTensor(actions_batch).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_tensor = torch.BoolTensor(dones_batch).to(self.device)
        
        batch_size, seq_len = states_tensor.shape[:2]
        
        # Initialize hidden states
        hidden = self.q_network.init_hidden(batch_size=batch_size, device=self.device)
        target_hidden = self.target_network.init_hidden(batch_size=batch_size, device=self.device)
        
        # Process sequences through LSTM
        q_values_list = []
        for t in range(seq_len):
            q_vals, hidden = self.q_network(states_tensor[:, t], hidden)
            q_values_list.append(q_vals)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        q_values = torch.stack(q_values_list, dim=1)
        current_q = q_values.gather(2, actions_tensor.unsqueeze(2)).squeeze(2)
        
        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_q_list = []
                hidden_copy = self.q_network.init_hidden(batch_size=batch_size, device=self.device)
                for t in range(seq_len):
                    nq, hidden_copy = self.q_network(next_states_tensor[:, t], hidden_copy)
                    next_q_list.append(nq)
                    hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
                
                next_q_values = torch.stack(next_q_list, dim=1)
                next_actions = next_q_values.argmax(2, keepdim=True)
                
                target_q_list = []
                for t in range(seq_len):
                    tq, target_hidden = self.target_network(next_states_tensor[:, t], target_hidden)
                    target_q_list.append(tq)
                    target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
                
                target_q_values = torch.stack(target_q_list, dim=1)
                next_q = target_q_values.gather(2, next_actions).squeeze(2)
            else:
                target_q_list = []
                for t in range(seq_len):
                    tq, target_hidden = self.target_network(next_states_tensor[:, t], target_hidden)
                    target_q_list.append(tq)
                    target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
                
                target_q_values = torch.stack(target_q_list, dim=1)
                next_q = target_q_values.max(2)[0]
            
            target_q = rewards_tensor + (self.gamma * next_q * ~dones_tensor)
        
        # Compute loss only on valid timesteps
        loss_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length] = 1.0
        
        # Masked loss
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update_target_network()
        
        self.update_counter += 1
        self.training_steps += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to starting value"""
        self.epsilon = self.epsilon_start
    
    def update_task_success(self, task_name, success):
        """Track per-task success rates (training tasks only)"""
        if task_name in self.task_success_rates:
            self.task_success_rates[task_name].append(1 if success else 0)
    
    def get_task_success_rates(self):
        """Get current success rates for each task"""
        rates = {}
        for task, successes in self.task_success_rates.items():
            if len(successes) > 0:
                rates[task] = np.mean(successes)
            else:
                rates[task] = 0.0
        return rates
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'task_success_rates': dict(self.task_success_rates),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {filepath}")
    
    def get_q_values(self, stacked_obs):
        """Get Q-values for debugging."""
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.q_network(state, self.current_hidden)
            return q_values.squeeze().cpu().numpy()