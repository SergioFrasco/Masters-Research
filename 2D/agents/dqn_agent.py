import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedVisionQNetwork(nn.Module):
    def __init__(self, grid_size, action_size):
        super(ImprovedVisionQNetwork, self).__init__()
        
        # More sophisticated CNN architecture
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Adaptive pooling to handle different grid sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size after adaptive pooling
        self.flattened_size = 32 * 4 * 4  # 512
        
        # Separate processing for direction
        self.dir_fc = nn.Linear(1, 16)
        
        # Combined feature processing
        self.fc1 = nn.Linear(self.flattened_size + 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, grid, direction):
        # Process grid through CNN
        x = F.relu(self.bn1(self.conv1(grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Process direction
        dir_features = F.relu(self.dir_fc(direction))
        
        # Combine features
        combined = torch.cat((x, dir_features), dim=1)
        
        # Forward through fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        return self.out(x)

class VisionDQNAgent:
    """
    Improved Vision-based Deep Q-Network agent with better architecture and training strategies.
    """
    
    def __init__(self, env, action_size=3, learning_rate=0.0005, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=32, target_update_freq=1000,
                 learning_starts=1000, train_freq=4):
        
        self.env = env
        self.grid_size = env.size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Improved hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        
        # Experience replay with prioritization support
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = ImprovedVisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.target_network = ImprovedVisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.update_target_network()
        
        # Improved optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10000, 
            gamma=0.9
        )
        
        # Huber loss for more stable training
        self.loss_fn = nn.SmoothL1Loss()
        
        # Training tracking
        self.training_step = 0
        self.episode_step = 0
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
    
    def get_vision_state(self, obs=None):
        """
        Enhanced vision-based state representation with better feature engineering.
        """
        # Get the raw grid encoding from environment
        grid = self.env.grid.encode()
        
        # Extract object layer
        object_layer = grid[..., 0].astype(np.float32)
        
        # Create enhanced two-channel representation
        env_channel = np.zeros_like(object_layer, dtype=np.float32)
        agent_channel = np.zeros_like(object_layer, dtype=np.float32)
        
        # Environment channel with better feature encoding
        env_channel[object_layer == 2] = -1.0  # Wall → -1 (obstacle)
        env_channel[object_layer == 1] = 0.0   # Open space → 0 (neutral)
        env_channel[object_layer == 8] = 1.0   # Goal → 1 (reward)
        
        # Agent channel with directional information
        agent_x, agent_y = self.env.agent_pos
        agent_channel[agent_y, agent_x] = 1.0
        
        # Add agent's "vision cone" - mark cells in front of agent
        agent_dir = self.env.agent_dir
        front_positions = self._get_front_positions(agent_x, agent_y, agent_dir)
        for fx, fy in front_positions:
            if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                agent_channel[fy, fx] = max(agent_channel[fy, fx], 0.3)
        
        # Stack channels and add batch dimension implicitly handled later
        grid_input = np.stack([env_channel, agent_channel], axis=0)  # Changed to channel-first
        
        # Normalize agent direction
        agent_dir_norm = self.env.agent_dir / 3.0
        
        return {
            'grid': grid_input,
            'agent_dir': agent_dir_norm
        }
    
    def _get_front_positions(self, x, y, direction, distance=2):
        """Get positions in front of the agent up to a certain distance."""
        positions = []
        
        # Direction vectors: 0=right, 1=down, 2=left, 3=up
        dx_dy = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dx_dy[direction]
        
        for i in range(1, distance + 1):
            positions.append((x + i * dx, y + i * dy))
        
        return positions
    
    def get_action(self, state_dict, epsilon=None):
        """
        Enhanced action selection with better exploration strategies.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Implement epsilon decay schedule
        if self.episode_step < self.learning_starts:
            # Pure exploration during initial phase
            return random.randrange(self.action_size)
        
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensors
        grid = torch.tensor(state_dict['grid'], dtype=torch.float32).unsqueeze(0).to(self.device)
        direction = torch.tensor([[state_dict['agent_dir']]], dtype=torch.float32).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(grid, direction)
            
        # Track Q-values for analysis
        self.recent_q_values.append(torch.max(q_values).item())
        
        return torch.argmax(q_values).item()
    
    def remember(self, state_dict, action, reward, next_state_dict, done):
        """Enhanced experience storage with reward clipping."""
        # Clip rewards for stability
        clipped_reward = np.clip(reward, -1, 1)
        self.memory.append((state_dict, action, clipped_reward, next_state_dict, done))
    
    def replay(self):
        """
        Improved training with gradient clipping and better target computation.
        """
        if len(self.memory) < self.learning_starts:
            return None, None
        
        if self.episode_step % self.train_freq != 0:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        state_dicts = [b[0] for b in batch]
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_state_dicts = [b[3] for b in batch]
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        # Convert states to tensors
        grid_batch = torch.stack([
            torch.tensor(s['grid'], dtype=torch.float32) for s in state_dicts
        ]).to(self.device)

        dir_batch = torch.tensor(
            [[s['agent_dir']] for s in state_dicts], 
            dtype=torch.float32
        ).to(self.device)

        next_grid_batch = torch.stack([
            torch.tensor(s['grid'], dtype=torch.float32) for s in next_state_dicts
        ]).to(self.device)

        next_dir_batch = torch.tensor(
            [[s['agent_dir']] for s in next_state_dicts], 
            dtype=torch.float32
        ).to(self.device)

        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(grid_batch, dir_batch)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use main network to select actions, target network to evaluate
        self.q_network.eval()
        with torch.no_grad():
            next_q_values = self.q_network(next_grid_batch, next_dir_batch)
            next_actions = torch.argmax(next_q_values, dim=1)
            
            target_next_q_values = self.target_network(next_grid_batch, next_dir_batch)
            next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        # Track loss
        self.recent_losses.append(loss.item())
        
        return loss.item(), torch.mean(current_q_values).item()
    
    def update_target_network(self):
        """Soft update of target network for more stable training."""
        tau = 0.005  # Soft update parameter
        
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Improved epsilon decay with minimum threshold."""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def step(self):
        """Call this at each environment step."""
        self.episode_step += 1
    
    def reset_episode(self):
        """Call this at the start of each episode."""
        self.episode_step = 0
    
    def get_stats(self):
        """Get training statistics for monitoring."""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_q_value': np.mean(self.recent_q_values) if self.recent_q_values else 0,
            'memory_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save_model(self, filepath):
        """Save the trained model with optimizer state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model with optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
    
    def get_q_values(self, state_dict):
        """Get Q-values for all actions in given state."""
        grid = torch.tensor(state_dict['grid'], dtype=torch.float32).unsqueeze(0).to(self.device)
        direction = torch.tensor([[state_dict['agent_dir']]], dtype=torch.float32).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(grid, direction)
        return q_values[0].cpu().numpy()