import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class VisualDQN(nn.Module):
    """CNN-based DQN that processes egocentric grid observations"""
    
    def __init__(self, input_channels=2, view_size=10, action_size=4, hidden_size=512):
        super(VisualDQN, self).__init__()

        self.view_size = view_size
        self.action_size = action_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, view_size, view_size)
            conv_out = self.conv_layers(dummy_input)
            conv_output_size = conv_out.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, action_size)
        )

        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        conv_out = self.conv_layers(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        q_values = self.fc_layers(flattened)
        return q_values

class VisualDQNAgent:
    """
    DQN Agent that uses egocentric visual observations
    Agent is always centered in the observation window
    """
    
    def __init__(self, env, view_size=10, action_size=4, learning_rate=0.0001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=1000):
        
        self.env = env
        self.grid_size = env.size
        self.view_size = view_size  # Size of the egocentric view window
        self.action_size = action_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Networks - input channels: walls/obstacles, goals/rewards (agent is implicitly at center)
        self.q_network = VisualDQN(
            input_channels=2,  # walls + goals (agent position is implicit)
            view_size=self.view_size,
            action_size=self.action_size
        ).to(self.device)
        
        self.target_network = VisualDQN(
            input_channels=2,
            view_size=self.view_size,
            action_size=self.action_size
        ).to(self.device)
        
        self.update_target_network()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training tracking
        self.training_step = 0
        self.episode_step = 0
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
    
    def get_egocentric_view(self, obs):
        """
        Extract an egocentric view centered on the agent
        Returns a window of size view_size x view_size with agent at center
        """
        # Get the encoded grid from environment
        grid = self.env.grid.encode()  # Shape: (H, W, 3)
        agent_pos = self.env.agent_pos
        
        # Calculate the bounds of the egocentric window
        half_view = self.view_size // 2
        
        # Create padded version of the grid to handle edge cases
        pad_size = half_view + 1
        
        # Initialize channels for the full padded grid
        walls_channel_full = np.zeros((self.grid_size + 2*pad_size, self.grid_size + 2*pad_size), dtype=np.float32)
        goals_channel_full = np.zeros((self.grid_size + 2*pad_size, self.grid_size + 2*pad_size), dtype=np.float32)
        
        # Fill the center with actual grid data
        object_layer = grid[..., 0]  # Object types
        
        # Walls channel (including boundaries as walls)
        walls_channel_full[:pad_size, :] = 1.0  # Top boundary
        walls_channel_full[-pad_size:, :] = 1.0  # Bottom boundary  
        walls_channel_full[:, :pad_size] = 1.0  # Left boundary
        walls_channel_full[:, -pad_size:] = 1.0  # Right boundary
        
        # Fill actual walls in the center region
        walls_channel_full[pad_size:pad_size+self.grid_size, pad_size:pad_size+self.grid_size][object_layer == 2] = 1.0
        
        # Goals channel (only in the valid grid area)
        goals_channel_full[pad_size:pad_size+self.grid_size, pad_size:pad_size+self.grid_size][object_layer == 8] = 1.0
        
        # Extract egocentric window centered on agent
        agent_x_padded = agent_pos[0] + pad_size
        agent_y_padded = agent_pos[1] + pad_size
        
        x_start = agent_x_padded - half_view
        x_end = agent_x_padded + half_view + 1
        y_start = agent_y_padded - half_view  
        y_end = agent_y_padded + half_view + 1
        
        walls_view = walls_channel_full[x_start:x_end, y_start:y_end]
        goals_view = goals_channel_full[x_start:x_end, y_start:y_end]
        
        # Stack channels: (2, H, W) - No agent channel since agent is implicitly at center
        egocentric_view = np.stack([walls_view, goals_view], axis=0)
        
        # Add batch dimension and convert to tensor: (1, 2, H, W)
        view_tensor = torch.tensor(egocentric_view[np.newaxis, ...], 
                                 dtype=torch.float32).to(self.device)
        
        return view_tensor
    
    def get_action(self, obs, epsilon=None):
        """
        Choose action based on egocentric visual observation
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        # Get egocentric view
        egocentric_input = self.get_egocentric_view(obs)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(egocentric_input)
            
        # Track Q-values for analysis
        self.recent_q_values.append(torch.max(q_values).item())
        
        return torch.argmax(q_values).item()
    
    def remember(self, obs, action, reward, next_obs, done):
        """Store experience in replay buffer"""
        # Store raw observations - egocentric processing happens during training
        self.memory.append((obs, action, reward, next_obs, done))
    
    def train(self):
        """
        Train the DQN using egocentric visual observations
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        current_obs_batch = []
        next_obs_batch = []
        actions = []
        rewards = []
        dones = []
        
        for obs, action, reward, next_obs, done in batch:
            current_obs_batch.append(obs)
            next_obs_batch.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Process egocentric views for entire batch
        current_states = torch.cat([self.get_egocentric_view(obs) for obs in current_obs_batch], dim=0)
        next_states = torch.cat([self.get_egocentric_view(obs) for obs in next_obs_batch], dim=0)
        
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(current_states)
        current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            max_next_q = torch.max(target_q_values, dim=1)[0]
            target_q = rewards_tensor + (self.gamma * max_next_q * (1 - dones_tensor))
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        self.recent_losses.append(loss.item())
        return loss.item(), torch.mean(current_q_values).item()
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_step = 0
    
    def step(self):
        """Call this at each environment step"""
        self.episode_step += 1
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_q_value': np.mean(self.recent_q_values) if self.recent_q_values else 0,
            'memory_size': len(self.memory)
        }