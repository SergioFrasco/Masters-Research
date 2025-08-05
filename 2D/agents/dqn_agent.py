import numpy as np
# import tensorflow as tf
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionQNetwork(nn.Module):
    def __init__(self, grid_size, action_size):
        super(VisionQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.flattened_size = 16 * grid_size * grid_size
        self.fc1 = nn.Linear(self.flattened_size + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, grid, direction):
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, direction), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class VisionDQNAgent:
    """
    Vision-based Deep Q-Network agent for fair comparison with Successor agent.
    Processes raw grid observations instead of getting privileged goal position info.
    """
    
    def __init__(self, env, action_size=3, learning_rate=0.001, 
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=100):
        
        self.env = env
        self.grid_size = env.size
        self.action_size = action_size  # [turn_left, turn_right, move_forward]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
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
        
        # Neural networks
        self.q_network = VisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.target_network = VisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.update_target_network()
        
        # Training tracking
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.training_step = 0
    
    # Old network building method, worked when given explicit goal position
    # def _build_network(self):
    #     """Build the neural network for Q-value approximation."""
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dense(self.action_size, activation='linear')
    #     ])
        
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
    #                  loss='mse')
    #     return model
    
    def get_state_vector(self, obs=None):
        """
        Convert environment state to state vector.
        Returns: [agent_x, agent_y, agent_direction, goal_x, goal_y]
        """
        # Get agent position and direction
        agent_x, agent_y = self.env.agent_pos
        agent_dir = self.env.agent_dir
        
        # Find goal position
        goal_x, goal_y = self._find_goal_position()
        
        # Normalize positions to [0, 1] range
        state = np.array([
            agent_x / (self.grid_size - 1),
            agent_y / (self.grid_size - 1),
            agent_dir / 3.0,  # Direction is 0-3, normalize to [0, 1]
            goal_x / (self.grid_size - 1),
            goal_y / (self.grid_size - 1)
        ], dtype=np.float32)
        
        return state
    
    def get_vision_state(self, obs=None):
        """
        Get vision-based state representation for fair comparison with Successor agent.
        
        Returns:
            dict with:
                'grid': normalized grid observation (height, width, 2) - includes agent position
                'agent_dir': normalized agent direction (scalar)
        """
        # Get the raw grid encoding from environment
        grid = self.env.grid.encode()
        
        # Extract object layer (first channel contains object types)
        object_layer = grid[..., 0].astype(np.float32)
        
        # Create two-channel representation
        # Channel 0: Environment objects (walls, goals)
        # Channel 1: Agent position
        
        env_channel = np.zeros_like(object_layer, dtype=np.float32)
        agent_channel = np.zeros_like(object_layer, dtype=np.float32)
        
        # Environment channel - same normalization as Successor agent
        env_channel[object_layer == 2] = 0.0  # Wall → 0
        env_channel[object_layer == 1] = 0.0  # Open space → 0  
        env_channel[object_layer == 8] = 1.0  # Goal → 1
        
        # Agent channel - mark agent position
        agent_x, agent_y = self.env.agent_pos
        agent_channel[agent_y, agent_x] = 1.0  # Agent → 1
        
        # Stack channels: (height, width, 2)
        grid_input = np.stack([env_channel, agent_channel], axis=-1)
        
        # Get agent direction (still useful non-visual info)
        agent_dir = self.env.agent_dir / 3.0  # Normalize to [0, 1]
        
        return {
            'grid': grid_input,
            'agent_dir': agent_dir
        }

    def _find_goal_position(self):
        """
        Find goal position - keep this method but only using it for debugging/analysis.
        """
        grid = self.env.grid.encode()
        object_layer = grid[..., 0]
        goal_positions = np.where(object_layer == 8)
        if len(goal_positions[0]) > 0:
            return goal_positions[1][0], goal_positions[0][0]
        else:
            return self.grid_size // 2, self.grid_size // 2
        
    def get_action(self, state_dict, epsilon=None):
        """
        Choose action using epsilon-greedy policy with vision-based state.
        
        Args:
            state_dict: Dictionary with 'grid' and 'agent_dir' keys
            epsilon: Exploration rate (uses self.epsilon if None)
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        grid = torch.tensor(state_dict['grid'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        direction = torch.tensor([[state_dict['agent_dir']]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(grid, direction)
        return torch.argmax(q_values).item()

    
    def remember(self, state_dict, action, reward, next_state_dict, done):
        """
        Store experience in replay buffer.
        Now stores state dictionaries instead of simple vectors.
        """
        self.memory.append((state_dict, action, reward, next_state_dict, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        state_dicts = [b[0] for b in batch]
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_state_dicts = [b[3] for b in batch]
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        grid_batch = torch.stack([
            torch.tensor(s['grid'], dtype=torch.float32).permute(2, 0, 1) for s in state_dicts
        ]).to(self.device)

        dir_batch = torch.tensor([[s['agent_dir']] for s in state_dicts], dtype=torch.float32).to(self.device)

        next_grid_batch = torch.stack([
            torch.tensor(s['grid'], dtype=torch.float32).permute(2, 0, 1) for s in next_state_dicts
        ]).to(self.device)

        next_dir_batch = torch.tensor([[s['agent_dir']] for s in next_state_dicts], dtype=torch.float32).to(self.device)

        # Predict Q-values
        current_q = self.q_network(grid_batch, dir_batch)
        next_q = self.target_network(next_grid_batch, next_dir_batch)

        # Compute target
        target_q = current_q.clone().detach()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q[i])

        # Loss and optimization
        self.q_network.train()
        self.optimizer.zero_grad()
        predicted_q = self.q_network(grid_batch, dir_batch)
        loss = self.loss_fn(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item(), None

    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save(self.q_network.state_dict(), filepath)
    

    def load_model(self, filepath):
        """Load a trained model."""
        self.q_network.load_state_dict(torch.load(filepath))
        self.q_network.to(self.device)
        self.update_target_network()
    
    def get_q_values(self, state_dict):
        """Get Q-values for all actions in given state."""
        grid = torch.tensor(state_dict['grid'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        direction = torch.tensor([[state_dict['agent_dir']]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(grid, direction)
        return q_values[0].cpu().numpy()

