import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network for partial observability with vision integration"""
    
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgentPartial:
    """DQN Agent for partially observable gridworld with path integration and vision"""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.05, epsilon_decay=0.9995, memory_size=10000, 
                 batch_size=32, target_update_freq=100, hidden_dim=128):
        
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3  # action space
        
        # Path integration components
        self.internal_pos = None
        self.internal_dir = None
        self.initial_pos = None
        self.initial_dir = None
        
        # Vision and reward mapping
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # DQN hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State representation: 13x13 view + position (2) + direction (4)
        self.state_dim = 13 * 13 + 2 + 4  # 175 features
        
        # Neural networks
        self.q_network = DQN(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        print(f"DQN Agent initialized with state dim: {self.state_dim}")
        print(f"Using device: {self.device}")

    def reset_path_integration(self):
        """Reset path integration state for new episode"""
        self.internal_pos = None
        self.internal_dir = None
        self.initial_pos = None
        self.initial_dir = None

    def initialize_path_integration(self, obs):
        """Initialize path integration from first observation"""
        # Extract agent position and direction from observation
        # In MiniGrid, agent position and direction are in obs dict
        if 'agent_pos' in obs:
            self.internal_pos = list(obs['agent_pos'])
            self.initial_pos = list(obs['agent_pos'])
        else:
            # Fallback: try to extract from environment
            # print("Warning: 'agent_pos' not in obs, using env.agent_pos")
            self.internal_pos = list(self.env.agent_pos)
            self.initial_pos = list(self.env.agent_pos)
        
        if 'agent_dir' in obs:
            self.internal_dir = obs['agent_dir']
            self.initial_dir = obs['agent_dir']
        else:
            # Fallback: try to extract from environment
            # print("Warning: 'agent_dir' not in obs, using env.agent_dir")
            self.internal_dir = self.env.agent_dir
            self.initial_dir = self.env.agent_dir
            
        # print(f"Path integration initialized: pos={self.internal_pos}, dir={self.internal_dir}")

    def update_internal_state(self, action):
        """Update internal position and direction based on action taken"""
        if self.internal_pos is None or self.internal_dir is None:
            return
        
        if action == 0:  # Turn left
            self.internal_dir = (self.internal_dir - 1) % 4
        elif action == 1:  # Turn right
            self.internal_dir = (self.internal_dir + 1) % 4
        elif action == 2:  # Move forward
            # Direction: 0=right, 1=down, 2=left, 3=up
            if self.internal_dir == 0:  # Right
                self.internal_pos[0] = min(self.internal_pos[0] + 1, self.grid_size - 1)
            elif self.internal_dir == 1:  # Down
                self.internal_pos[1] = min(self.internal_pos[1] + 1, self.grid_size - 1)
            elif self.internal_dir == 2:  # Left
                self.internal_pos[0] = max(self.internal_pos[0] - 1, 0)
            elif self.internal_dir == 3:  # Up
                self.internal_pos[1] = max(self.internal_pos[1] - 1, 0)
        # Actions 3, 4, 5 (pickup, drop, toggle) don't change position

    def _get_agent_pos_from_env(self):
        """Get agent position directly from environment"""
        # Use the SAME conversion as you use for boxes
        x = int(round(self.env.agent.pos[0] /  self.env.grid_size))
        z = int(round(self.env.agent.pos[2] / self.env.grid_size))
        return (x, z)
    
    def _get_agent_dir_from_env(self):
        """Get agent direction directly from environment"""
        angle = self.env.agent.dir
        # Convert angle to cardinal direction: 0=East, 1=South, 2=West, 3=North
        # MiniWorld uses CLOCKWISE rotation: 0°=East, 90°=North, 180°=West, 270°=South
        degrees = (np.degrees(angle) % 360)
        if degrees < 45 or degrees >= 315:
            return 0  # East (+X)
        elif 45 <= degrees < 135:
            return 3  # North (-Z) at 90°
        elif 135 <= degrees < 225:
            return 2  # West (-X) at 180°
        else:  # 225 <= degrees < 315
            return 1  # South (+Z) at 270°
    
    def create_egocentric_observation(self, goal_pos_red=None, goal_pos_blue=None, matrix_size=13):
        """
        Create an egocentric observation matrix where:
        - Agent is always at the bottom-middle cell, facing upward.
        - Goal positions (red, blue) are given in the agent's egocentric coordinates.
            (x = right, z = forward)
        
        Args:
            goal_pos_red  : Tuple (x_right, z_forward) or None
            goal_pos_blue : Tuple (x_right, z_forward) or None
            matrix_size   : Size of the square matrix (default 13x13)

        Returns:
            ego_matrix: numpy array of shape (matrix_size, matrix_size)
                        Red goal marked as 1, Blue goal as 1
        """
        import numpy as np

        # Initialize empty egocentric matrix
        ego_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)

        # Agent position (bottom-center)
        agent_row = matrix_size - 1
        agent_col = matrix_size // 2

        def place_goal(pos, value):
            if pos is None:
                return
            gx, gz = pos  # (right, forward)
            # Convert to matrix coordinates
            ego_row = agent_row - gz  # forward is upward (smaller row)
            ego_col = agent_col - gx  # right is right (larger col)

            # Check bounds and place marker
            if 0 <= ego_row < matrix_size and 0 <= ego_col < matrix_size:
                ego_matrix[int(ego_row), int(ego_col)] = value

        # Place red and blue goals
        place_goal(goal_pos_red, 1.0)
        place_goal(goal_pos_blue, 1.0)

        return ego_matrix


    def get_dqn_state(self, obs):
        """Convert partial observation to DQN input state vector"""
        view_flat = obs.flatten().astype(np.float32)  # 49 values
        
        # Normalize view values to [0, 1] range
        view_flat = view_flat / 1.0  # Assuming max object type is ~1
        
        # Get normalized position 
        pos_x = self._get_agent_pos_from_env()[0] / (self.grid_size - 1)
        pos_z = self._get_agent_pos_from_env()[1] / (self.grid_size - 1)

        
        position = np.array([pos_x, pos_z], dtype=np.float32)
        
        # Get one-hot direction (using path integration)
        direction_onehot = np.zeros(4, dtype=np.float32)
        if self.internal_dir is not None:
            direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate all features
        state = np.concatenate([view_flat, position, direction_onehot])
        return torch.FloatTensor(state).to(self.device)

    def select_action_dqn(self, obs, epsilon):
        """Select action using DQN with epsilon-greedy exploration"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self.get_dqn_state(obs)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()

    def sample_random_action(self, obs, epsilon=1.0):
        """Sample random action (for compatibility with existing code)"""
        return random.randint(0, self.action_dim - 1)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train_dqn(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract components
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in batch], device=self.device, dtype=torch.float32)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], device=self.device, dtype=torch.bool)
        
        # Current Q values: Q(s,a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def verify_path_integration(self, obs):
        """Verify path integration accuracy against ground truth"""
        if 'agent_pos' in obs:
            true_pos = obs['agent_pos']
        else:
            true_pos = self.env.agent_pos
            
        if 'agent_dir' in obs:
            true_dir = obs['agent_dir']
        else:
            true_dir = self.env.agent_dir
        
        pos_error = abs(self.internal_pos[0] - true_pos[0]) + abs(self.internal_pos[1] - true_pos[1])
        dir_error = self.internal_dir != true_dir
        
        if pos_error > 0 or dir_error:
            error_msg = f"Path integration error: internal_pos={self.internal_pos}, true_pos={true_pos}, "
            error_msg += f"internal_dir={self.internal_dir}, true_dir={true_dir}"
            return False, error_msg
        
        return True, "Accurate"

    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']