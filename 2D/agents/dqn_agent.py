import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, state_size=5, action_size=3, learning_rate=0.001, 
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=100,
                 device=None):
        self.env = env
        self.grid_size = env.size
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.memory = deque(maxlen=memory_size)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build networks
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Extra ones added to train the vision model
        # Initialize the true map to track discovered reward locations and predictions
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.true_reward_map_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Individual Reward maps that are composed with the SR
        # Initialize individual reward maps: one per state
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Track states we have visited to inform our map updates correctly
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.training_step = 0

    import numpy as np

    def predicted_goal_from_reward_map(self, predicted_reward_map):
        """
        Given a 2D reward map (numpy array), returns the (goal_x, goal_y) coordinates
        of the maximum predicted reward location.
        """
        max_idx = np.unravel_index(np.argmax(predicted_reward_map), predicted_reward_map.shape)
        goal_y, goal_x = max_idx  # Usually numpy is (row, col) = (y, x)
        return goal_x, goal_y

    # Older, perfect knowledge
    # def get_state_vector(self, obs=None):
    #     """
    #     Convert environment state to state vector.
    #     Returns: [agent_x, agent_y, agent_direction, goal_x, goal_y]
    #     """
    #     # Get agent position and direction
    #     agent_x, agent_y = self.env.agent_pos
    #     agent_dir = self.env.agent_dir
        
    #     # Find goal position
    #     goal_x, goal_y = self._find_goal_position()
        
    #     # Normalize positions to [0, 1] range
    #     state = np.array([
    #         agent_x / (self.grid_size - 1),
    #         agent_y / (self.grid_size - 1),
    #         agent_dir / 3.0,  # Direction is 0-3, normalize to [0, 1]
    #         goal_x / (self.grid_size - 1),
    #         goal_y / (self.grid_size - 1)
    #     ], dtype=np.float32)
        
    #     return state
    
    def get_state_vector(self, obs=None, predicted_reward_map=None):
        """
        Convert environment state to state vector.
        If predicted_reward_map is given, extract goal from it.
        """
        agent_x, agent_y = self.env.agent_pos
        agent_dir = self.env.agent_dir

        if predicted_reward_map is not None:
            goal_x, goal_y = self.predicted_goal_from_reward_map(predicted_reward_map)
        else:
            goal_x, goal_y = self.grid_size // 2, self.grid_size // 2 #return center as fallback

        state = np.array([
            agent_x / (self.grid_size - 1),
            agent_y / (self.grid_size - 1),
            agent_dir / 3.0,
            goal_x / (self.grid_size - 1),
            goal_y / (self.grid_size - 1)
        ], dtype=np.float32)

        return state

    
    def _find_goal_position(self):
        """Find the goal position in the current environment."""
        grid = self.env.grid.encode()
        object_layer = grid[..., 0]
        
        # Find where goal (object type 8) is located
        goal_positions = np.where(object_layer == 8)
        if len(goal_positions[0]) > 0:
            return goal_positions[1][0], goal_positions[0][0]  # x, y
        else:
            # If no goal found, return center as fallback
            return self.grid_size // 2, self.grid_size // 2

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # batch dim
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e[1] for e in batch]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array([e[2] for e in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([e[4] for e in batch]), dtype=torch.float32, device=self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item(), None

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load_model(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_network()

    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return q_values.cpu().numpy().flatten()
