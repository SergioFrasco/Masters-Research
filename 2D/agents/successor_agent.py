import numpy as np
from minigrid.core.world_object import Goal
from gym import spaces

class SuccessorAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Calculate state size based on grid dimensions
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        
        # MiniGrid default actions: left, right, forward
        self.action_size = 3
        
        # Initialize successor features matrix
        self.M = np.stack([np.identity(self.state_size) for _ in range(self.action_size)])
        self.w = np.zeros([self.state_size])
        
    def get_state_index(self, obs):
        """Convert MiniGrid observation to state index"""
        agent_pos = self.env.agent_pos
        return agent_pos[0] + agent_pos[1] * self.grid_size
    
    def Q_estimates(self, state_idx, goal=None):
        """Generate Q values for all actions"""
        if goal is None:
            goal = self.w
        else:
            goal = self._onehot(goal, self.state_size)
        return np.matmul(self.M[:, state_idx, :], goal)
    
    def sample_action(self, obs, goal=None, epsilon=0.0):
        """Sample action using epsilon-greedy approach"""
        state_idx = self.get_state_index(obs)
        
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state_idx, goal)
            action = np.argmax(Qs)
        return action
    
    def update(self, current_exp, next_exp=None):
        """Update both reward weights and successor features"""
        error_w = self.update_w(current_exp)
        error_sr = 0
        if next_exp is not None:
            error_sr = self.update_sr(current_exp, next_exp)
        return error_w, error_sr
    
    def update_w(self, current_exp):
        """Update reward weights"""
        s_1 = current_exp[2]  # next state index
        r = current_exp[3]    # reward
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        return error
    
    def update_sr(self, current_exp, next_exp):
        """Update successor features using SARSA TD learning"""
        s = current_exp[0]    # current state index
        s_a = current_exp[1]  # current action
        s_1 = current_exp[2]  # next state index
        s_a_1 = next_exp[1]   # next action
        d = current_exp[4]    # done flag
        
        I = self._onehot(s, self.state_size)
        
        if d:
            td_error = (I + self.gamma * self._onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
            
        self.M[s_a, s, :] += self.learning_rate * td_error
        return np.mean(np.abs(td_error))
    
    def _onehot(self, index, size):
        """Create one-hot encoded vector"""
        vec = np.zeros(size)
        vec[index] = 1
        return vec

    def is_goal_state(self, obs):
        """Check if current state contains a goal"""
        agent_pos = self.env.agent_pos
        cell = self.env.grid.get(*agent_pos)
        return isinstance(cell, Goal)