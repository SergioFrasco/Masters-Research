"""
DQN Agent for MiniWorld 3D Environment - STABLE VERSION

Key improvements for training stability:
1. Soft target updates (Polyak averaging) instead of hard updates
2. Gradient clipping
3. Better weight initialization
4. Optional double DQN to reduce overestimation
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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DuelingDQN3D(nn.Module):
    """
    Dueling DQN architecture - separates value and advantage streams.
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
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity=100000):
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
    STABLE DQN Agent for MiniWorld 3D environment.
    
    Key stability features:
    1. Soft target updates (tau parameter) - gradual target network updates
    2. Double DQN (optional) - reduces Q-value overestimation
    3. Gradient clipping - prevents exploding gradients
    4. Huber loss - more robust to outliers than MSE
    """
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
                 memory_size=100000, batch_size=64, target_update_freq=1,
                 hidden_size=256, use_dueling=True,
                 tau=0.005,           # Soft update coefficient (0.005 = slow, stable updates)
                 use_double_dqn=True, # Use Double DQN to reduce overestimation
                 grad_clip=10.0):     # Gradient clipping threshold
        
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
        
        # STABILITY PARAMETERS
        self.tau = tau                      # Soft update coefficient
        self.use_double_dqn = use_double_dqn
        self.grad_clip = grad_clip
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        print(f"Stability settings: tau={tau}, double_dqn={use_double_dqn}, grad_clip={grad_clip}")
        
        # Get observation shape from environment
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        # Determine input shape (C, H, W)
        if sample_img.shape[0] in [3, 4]:
            self.obs_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
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
        
        # Optimizer with slightly lower learning rate for stability
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Training tracking
        self.update_counter = 0
        self.training_steps = 0
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total network parameters: {total_params:,}")
    
    def soft_update_target_network(self):
        """
        Soft update target network using Polyak averaging:
        θ_target = τ * θ_online + (1 - τ) * θ_target
        
        This is much more stable than hard updates!
        """
        for target_param, online_param in zip(self.target_network.parameters(), 
                                               self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def preprocess_obs(self, obs):
        """Convert observation to tensor suitable for CNN."""
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
        
        return torch.FloatTensor(img).to(self.device)
    
    def select_action(self, obs, epsilon=None):
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
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
        Perform one training step with stability improvements.
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
        
        # Target Q values with Double DQN (optional)
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to SELECT action, target network to EVALUATE
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute Huber loss (more robust than MSE)
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        # Soft update target network EVERY step (with small tau)
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