"""
Unified DQN Agent with Extended Task Conditioning

Key changes:
1. Task space now includes green: [red, blue, green, box, sphere]
2. Input shape: (8, H, W) - 3 RGB + 5 task channels
3. Green never appears during training, only evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class TaskConditionedDQN3D(nn.Module):
    """Task-conditioned DQN with 5-dimensional task space."""
    
    def __init__(self, input_shape=(8, 60, 80), action_size=3, hidden_size=256):
        super(TaskConditionedDQN3D, self).__init__()
        
        self.input_shape = input_shape
        
        # CNN for processing RGB + task channels (3 RGB + 5 task = 8 total)
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self._conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TaskConditionedDuelingDQN3D(nn.Module):
    """Task-conditioned Dueling DQN with 5-dimensional task space."""
    
    def __init__(self, input_shape=(8, 60, 80), action_size=3, hidden_size=256):
        super(TaskConditionedDuelingDQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        
        # Shared CNN backbone (8 channels: 3 RGB + 5 task)
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        self.value_stream = nn.Sequential(
            nn.Linear(self._conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
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
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class UnifiedDQNAgent:
    """
    Unified DQN Agent with extended task conditioning including green.
    
    IMPORTANT: Green is in the encoding space but never used during training!
    This allows zero-shot evaluation on unseen green objects.
    """
    
    # Task encoding: one-hot vectors for [red, blue, green, box, sphere]
    TASK_ENCODINGS = {
        'red': [1.0, 0.0, 0.0, 0.0, 0.0],
        'blue': [0.0, 1.0, 0.0, 0.0, 0.0],
        'green': [0.0, 0.0, 1.0, 0.0, 0.0],  # NEVER used in training!
        'box': [0.0, 0.0, 0.0, 1.0, 0.0],
        'sphere': [0.0, 0.0, 0.0, 0.0, 1.0],
    }
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
                 memory_size=100000, batch_size=64, target_update_freq=1,
                 hidden_size=256, use_dueling=True,
                 tau=0.005, use_double_dqn=True, grad_clip=10.0):
        
        self.env = env
        self.action_dim = 3
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.grad_clip = grad_clip
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Unified DQN Agent using device: {self.device}")
        print(f"Task space: [red, blue, green, box, sphere]")
        print(f"NOTE: Green reserved for zero-shot evaluation only!")
        
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        if sample_img.shape[0] in [3, 4]:
            self.img_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.img_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        # Task-conditioned input: 3 RGB + 5 task = 8 total
        self.obs_shape = (8, self.img_shape[1], self.img_shape[2])
        
        print(f"Image shape: {self.img_shape}")
        print(f"Task-conditioned input: {self.obs_shape} (3 RGB + 5 task channels)")
        
        NetworkClass = TaskConditionedDuelingDQN3D if use_dueling else TaskConditionedDQN3D
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
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=memory_size)
        
        self.update_counter = 0
        self.training_steps = 0
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total network parameters: {total_params:,}")
    
    def encode_task(self, task_name):
        """Convert task name to one-hot encoding."""
        if task_name not in self.TASK_ENCODINGS:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(self.TASK_ENCODINGS.keys())}")
        
        encoding = self.TASK_ENCODINGS[task_name]
        return torch.FloatTensor(encoding).to(self.device)
    
    def encode_compositional_task(self, features, method='superposition'):
        """
        Encode compositional task with multiple features.
        
        This is the KEY to zero-shot generalization:
        - Trained on: ['red', 'box'], ['blue', 'sphere'], etc.
        - Can compose: ['green', 'box'] without ever seeing it!
        """
        encoding = torch.zeros(5, device=self.device)  # 5-dim now
        
        for feature in features:
            if feature not in self.TASK_ENCODINGS:
                raise ValueError(f"Unknown feature: {feature}")
            feature_encoding = torch.FloatTensor(self.TASK_ENCODINGS[feature]).to(self.device)
            
            if method == 'superposition':
                encoding += feature_encoding
            elif method == 'max':
                encoding = torch.maximum(encoding, feature_encoding)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        if method == 'superposition' and len(features) > 1:
            encoding = encoding / len(features)
        
        return encoding
    
    def tile_task_encoding(self, task_encoding, height, width):
        """Tile task encoding across spatial dimensions."""
        return task_encoding.view(-1, 1, 1).expand(-1, height, width)
    
    def soft_update_target_network(self):
        """Soft update target network using Polyak averaging."""
        for target_param, online_param in zip(self.target_network.parameters(), 
                                               self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def preprocess_obs(self, obs, task_identifier):
        """
        Convert observation to task-conditioned tensor.
        
        Args:
            obs: raw observation
            task_identifier: str (like 'red') OR list of features (['green', 'box'])
        
        Returns:
            torch.Tensor of shape (8, H, W)
        """
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if isinstance(img, np.ndarray):
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
        
        img_tensor = torch.FloatTensor(img).to(self.device)
        
        # Handle both single and compositional tasks
        if isinstance(task_identifier, list):
            task_encoding = self.encode_compositional_task(task_identifier, method='superposition')
        else:
            task_encoding = self.encode_task(task_identifier)
        
        height, width = img_tensor.shape[1], img_tensor.shape[2]
        task_tiled = self.tile_task_encoding(task_encoding, height, width)
        
        # Concatenate: (3, H, W) + (5, H, W) = (8, H, W)
        conditioned_obs = torch.cat([img_tensor, task_tiled], dim=0)
        
        return conditioned_obs
    
    def select_action(self, obs, task_name, epsilon=None):
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self.preprocess_obs(obs, task_name)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()
    
    def remember(self, obs, task_name, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        state = self.preprocess_obs(obs, task_name).cpu()
        next_state = self.preprocess_obs(next_obs, task_name).cpu()
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        self.soft_update_target_network()
        
        self.update_counter += 1
        self.training_steps += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to starting value."""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {filepath}")
    
    def get_q_values(self, obs, task_name):
        """Get Q-values for debugging."""
        state = self.preprocess_obs(obs, task_name)
        with torch.no_grad():
            return self.q_network(state.unsqueeze(0)).squeeze().cpu().numpy()