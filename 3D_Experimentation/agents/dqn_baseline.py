"""
DQN Agent for MiniWorld 3D Environment

This agent learns from raw RGB observations and is trained ONLY on simple tasks.
It will struggle with compositional tasks because it has no factored representation
of object features (color vs shape).

Key Design Decisions:
1. Uses CNN to process raw RGB images from MiniWorld
2. Trained ONLY on simple tasks (red, blue, box, sphere)
3. Evaluated on compositional tasks (red_box, blue_box, etc.)
4. No task encoding - the agent must learn purely from visual observations and rewards
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class DQN3D(nn.Module):
    """
    Deep Q-Network for 3D MiniWorld environment using CNN on RGB images.
    
    Architecture:
    - 3 convolutional layers for feature extraction
    - 2 fully connected layers for Q-value estimation
    
    Input: RGB image (3, H, W)
    Output: Q-values for each action (3 actions: turn_left, turn_right, move_forward)
    """
    
    def __init__(self, input_shape=(3, 60, 80), action_size=3, hidden_size=256):
        super(DQN3D, self).__init__()
        
        self.input_shape = input_shape
        
        # CNN for processing RGB images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate conv output size dynamically
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Initialize weights
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
    
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch, channels, height, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class DuelingDQN3D(nn.Module):
    """
    Dueling DQN architecture - separates value and advantage streams.
    Often performs better than vanilla DQN.
    """
    
    def __init__(self, input_shape=(3, 60, 80), action_size=3, hidden_size=256):
        super(DuelingDQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        
        # Shared CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self._conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self._conv_output_size, hidden_size),
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
    
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (with mean subtraction for stability)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent3D:
    """
    DQN Agent for MiniWorld 3D environment.
    
    This agent:
    1. Processes raw RGB observations through a CNN
    2. Uses epsilon-greedy exploration
    3. Learns via experience replay and target network
    
    Key limitation: It learns a single entangled representation,
    so it cannot compose features for compositional tasks.
    """
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=32, target_update_freq=1000,
                 hidden_size=256, use_dueling=True):
        
        self.env = env
        self.action_dim = 3  # turn_left, turn_right, move_forward
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        
        # Get observation shape from environment
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        # Determine input shape (C, H, W)
        if sample_img.shape[0] in [3, 4]:  # Already (C, H, W)
            self.obs_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:  # (H, W, C)
            self.obs_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        print(f"Observation shape: {self.obs_shape}")
        
        # Initialize networks
        NetworkClass = DuelingDQN3D if use_dueling else DQN3D
        self.q_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.target_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Training tracking
        self.update_counter = 0
        self.training_steps = 0
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total network parameters: {total_params:,}")
    
    def preprocess_obs(self, obs):
        """
        Convert observation to tensor suitable for CNN.
        
        Input: Raw observation from environment (dict with 'image' or raw array)
        Output: Tensor of shape (C, H, W) normalized to [0, 1]
        """
        # Extract image from observation
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # Ensure correct format (H, W, C) -> (C, H, W)
        if isinstance(img, np.ndarray):
            # Check if already (C, H, W)
            if img.shape[0] in [3, 4]:
                pass  # Already correct
            else:  # (H, W, C)
                img = np.transpose(img, (2, 0, 1))
            
            # Normalize to [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            
            # Take only RGB if RGBA
            if img.shape[0] == 4:
                img = img[:3]
        
        return torch.FloatTensor(img).to(self.device)
    
    def select_action(self, obs, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Current observation
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            action: Integer action (0=turn_left, 1=turn_right, 2=move_forward)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation
        state = self.preprocess_obs(obs)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()
    
    def remember(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer"""
        state = self.preprocess_obs(obs)
        next_state = self.preprocess_obs(next_obs)
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step using experience replay.
        
        Returns:
            loss: Training loss (0.0 if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)  # Huber loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        self.training_steps += 1
        
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to starting value"""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'obs_shape': self.obs_shape,
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
    
    def get_q_values(self, obs):
        """Get Q-values for debugging/visualization"""
        state = self.preprocess_obs(obs)
        with torch.no_grad():
            return self.q_network(state.unsqueeze(0)).squeeze().cpu().numpy()


class DQNAgentWithFrameStack(DQNAgent3D):
    """
    DQN Agent with frame stacking for temporal information.
    Stacks last N frames to give agent sense of motion/direction.
    """
    
    def __init__(self, env, num_frames=4, **kwargs):
        self.num_frames = num_frames
        self.frame_buffer = deque(maxlen=num_frames)
        
        # Initialize parent
        super().__init__(env, **kwargs)
        
        # Update observation shape for stacked frames
        self.obs_shape = (3 * num_frames, self.obs_shape[1], self.obs_shape[2])
        
        # Reinitialize networks with new input shape
        NetworkClass = DuelingDQN3D if kwargs.get('use_dueling', False) else DQN3D
        self.q_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=kwargs.get('hidden_size', 256)
        ).to(self.device)
        
        self.target_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=kwargs.get('hidden_size', 256)
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        print(f"Frame stacking enabled: {num_frames} frames")
        print(f"Stacked observation shape: {self.obs_shape}")
    
    def reset_frame_buffer(self, obs):
        """Reset frame buffer with initial observation"""
        frame = self._preprocess_single_frame(obs)
        for _ in range(self.num_frames):
            self.frame_buffer.append(frame)
    
    def _preprocess_single_frame(self, obs):
        """Preprocess single frame (without stacking)"""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, np.ndarray):
            if img.shape[0] not in [3, 4]:
                img = np.transpose(img, (2, 0, 1))
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            if img.shape[0] == 4:
                img = img[:3]
        
        return torch.FloatTensor(img)
    
    def preprocess_obs(self, obs):
        """Preprocess observation with frame stacking"""
        frame = self._preprocess_single_frame(obs)
        self.frame_buffer.append(frame)
        
        # Stack frames along channel dimension
        stacked = torch.cat(list(self.frame_buffer), dim=0)
        return stacked.to(self.device)