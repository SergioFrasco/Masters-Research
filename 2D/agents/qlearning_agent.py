import numpy as np
from gym import spaces

class QLearningAgent:
    """Simple Q-learning baseline agent for comparison"""
    
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Calculate state and action dimensions
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        self.action_size = 3  # MiniGrid: left, right, forward
        
        # Initialize Q-table
        self.q_table = np.zeros((self.state_size, self.action_size))
        
    def get_state_index(self, obs):
        """Convert environment observation to state index"""
        agent_pos = self.env.agent_pos
        x, y = agent_pos
        return y * self.grid_size + x
    
    def choose_action(self, state_idx):
        """Choose action using epsilon-greedy policy"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state_idx, action, reward, next_state_idx, done):
        """Update Q-table using Q-learning update rule"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)