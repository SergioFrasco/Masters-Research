import numpy as np
from gym import spaces

class SARSASRAgent:
    """
    SARSA-based Successor Representation agent following Arthur Juliani's approach.
    Uses on-policy SARSA updates for both SR and reward prediction.
    """
    
    def __init__(self, env, learning_rate=0.01, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State and action dimensions
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        self.action_size = 3  # MiniGrid: left, right, forward
        
        # Initialize Successor Representation matrix
        # M[s,a,s'] = expected discounted future occupancy of state s' when taking action a in state s
        self.M = np.zeros((self.state_size, self.action_size, self.state_size))
        
        for s in range(self.state_size):
            for a in range(self.action_size):
                self.M[s, a, s] = 1.0 
                
        # Reward prediction weights - learns to predict immediate rewards
        self.w = np.zeros(self.state_size)
        
        # Track last state-action for SARSA updates
        self.last_state = None
        self.last_action = None
        
    def get_state_index(self, obs):
        """Convert environment observation to state index"""
        agent_pos = self.env.agent_pos
        x, y = agent_pos
        return y * self.grid_size + x
    
    def choose_action(self, state_idx):
        """Choose action using epsilon-greedy policy based on value function"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            # Compute Q-values by combining SR with reward predictions
            q_values = self.compute_q_values(state_idx)
            return np.argmax(q_values)
    
    def compute_q_values(self, state_idx):
        """Compute Q-values using SR and reward predictions: Q(s,a) = M(s,a) · w"""
        q_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            # Q(s,a) = sum over s' of M(s,a,s') * w(s')
            q_values[a] = np.dot(self.M[state_idx, a, :], self.w)
        return q_values
    
    def update_sr(self, state, action, reward, next_state, next_action, done):
        """
        Update Successor Representation using SARSA:
        M(s,a) = M(s,a) + lr * [I(s) + γ * M(s',a') - M(s,a)]
        """
        I_s = np.zeros(self.state_size)
        I_s[state] = 1.0
        
        if done:
            td_target = I_s
        else:
            # SARSA SR: bootstrap from the next state-action pair's SR
            td_target = I_s + self.gamma * self.M[next_state, next_action, :]
        
        td_error = td_target - self.M[state, action, :]
        self.M[state, action, :] += self.learning_rate * td_error


    def update_reward_prediction(self, state, action, reward, next_state, next_action, done):
        """
        Update reward prediction weights using SARSA:
        w(s) = w(s) + lr * [r + γ * w(s') - w(s)]
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.w[next_state]
        
        # Update reward prediction for current state
        td_error = td_target - self.w[state]
        self.w[state] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Full SARSA update for both SR and reward prediction.
        This is called after we know both current and next actions.
        """
        # Update SR matrix
        sr_error = self.update_sr(state, action, reward, next_state, next_action, done)
        
        # Update reward prediction
        w_error = self.update_reward_prediction(state, action, reward, next_state, next_action, done)
        
        return w_error, sr_error
    
    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_value_function(self, state_idx):
        """Get state value by taking max over actions"""
        q_values = self.compute_q_values(state_idx)
        return np.max(q_values)
    
    def get_policy(self, state_idx):
        """Get action probabilities under current policy"""
        q_values = self.compute_q_values(state_idx)
        
        # Epsilon-greedy policy
        policy = np.ones(self.action_size) * (self.epsilon / self.action_size)
        best_action = np.argmax(q_values)
        policy[best_action] += (1.0 - self.epsilon)
        
        return policy
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        self.last_state = None
        self.last_action = None