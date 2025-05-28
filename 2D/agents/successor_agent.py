import numpy as np

from utils.matrices import onehot
from minigrid.core.world_object import Goal
from gym import spaces

class SuccessorAgent:
    # learning rate = 0.1
    # Tried 0.01
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
        # CHANGED initialize to zeros
        # self.M = np.zeros((self.action_size, self.state_size, self.state_size))
        self.M = np.stack([np.identity(self.state_size) for _ in range(self.action_size)])

        # self.M = np.stack([np.identity(self.state_size) for _ in range(self.action_size)])
        self.w = np.zeros([self.state_size])

        # Initialize the true map to track discovered reward locations and predictions
        # Initially filled with zeros, shape: (grid_size, grid_size)
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.true_reward_map_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Individual Reward maps that are composed with the SR
        # Initialize individual reward maps: one per state
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # World Value Function - Mappings of values to each state goal pair
        self.wvf = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)
        # self.wvf = np.zeros((self.state_size, self.state_size)) 

        # Track visit counts for exploration bonus
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.exploration_weight = 1.0  # Weight for exploration bonus

    def update_visit_counts(self, agent_pos):
        """Update visit counts for exploration bonus"""
        x, y = agent_pos
        self.visit_counts[y, x] += 1

    # older non-trandsformed version
    def get_state_index(self, obs):
        """Convert MiniGrid observation to state index"""
        agent_pos = self.env.agent_pos
        return agent_pos[0] + agent_pos[1] * self.grid_size
    
    # Trying to transform it to fit human render
    # def get_state_index(self, obs):
    #     # raw MiniGrid agent_pos is (x,y) with (0,0) bottom-left
    #     x, y = self.env.agent_pos

    #     # AE saw:
    #     #   normalized_grid = flipud  →  row’ = H-1 - row
    #     #   then rot90(k=-1) → 90° CW: (r’,c’) → (c',  H-1-r')
    #     H = self.grid_size

    #     # apply same sequence to (x,y):
    #     # 1) flipud on y: y1 = H-1 - y
    #     # 2) rot90 CW: new_x =  y1,  new_y = H-1 - x
    #     y1 = H - 1 - y
    #     x2 =  y1
    #     y2 =  H - 1 - x

    #     # now flatten in row-major order (row = y2, col = x2)
    #     return int(x2 + y2 * H)

    
    # def Q_estimates(self, state_idx, goal=None):
    #     """Generate Q values for all actions"""
    #     if goal is None:
    #         goal = self.w
    #     else:
    #         goal = self._onehot(goal, self.state_size)
    #     return np.matmul(self.M[:, state_idx, :], goal)
    
    # def sample_action(self, obs, goal=None, epsilon=0.0):
    #     """Sample action using epsilon-greedy approach"""
    #     state_idx = self.get_state_index(obs)
        
    #     if np.random.uniform(0, 1) < epsilon:
    #         action = np.random.randint(self.action_size)
    #     else:
    #         Qs = self.Q_estimates(state_idx, goal)
    #         action = np.argmax(Qs)
    #     return action
    
    def sample_random_action(self, obs, goal=None, epsilon=0.0):
        """Sample an action uniformly at random"""
        return np.random.randint(self.action_size)


    def sample_action_with_wvf(self, obs, epsilon=0.0, chosen_reward_map=None):
        """
        Sample an action using epsilon-greedy selection with improved target selection
        """
        state_idx = self.get_state_index(obs)
        
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        else:
            # If no specific reward map provided, select one using improved method
            if chosen_reward_map is None:
                chosen_reward_map = self.select_target_with_exploration()
            
            # Compute Q-values using the chosen reward map
            values = self.value_estimates_with_wvf(state_idx, chosen_reward_map)
            action = np.argmax(values)
            return action

    def select_target_with_exploration(self):
        """
        Select target using exploration bonus and probabilistic selection
        """
        agent_y, agent_x = self.env.agent_pos
        reward_threshold = 0.5
        
        # Step 1: Add exploration bonus to WVF
        exploration_bonus = self.exploration_weight / (self.visit_counts + 1)
        modified_wvf = self.wvf + exploration_bonus[np.newaxis, :, :]
        
        # Step 2: Find maps with values exceeding threshold (using modified WVF)
        exceeds_threshold = modified_wvf > reward_threshold
        
        # Step 3: Compute distances
        H, W = modified_wvf.shape[1:]
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        distances = np.sqrt((y_coords - agent_y)**2 + (x_coords - agent_x)**2)
        distances_broadcasted = np.broadcast_to(distances, modified_wvf.shape)
        
        # Step 4: Mask distances and find valid maps
        masked_distances = np.where(exceeds_threshold, distances_broadcasted, np.inf)
        min_dist_per_map = masked_distances.min(axis=(1, 2))
        valid_maps = np.any(exceeds_threshold, axis=(1, 2))
        
        if not np.any(valid_maps):
            # No valid maps, return the first one as fallback
            return self.wvf[0]
        
        # Step 5: Probabilistic selection instead of always picking closest
        valid_indices = np.where(valid_maps)[0]
        
        if len(valid_indices) == 1:
            # Only one valid map
            best_map_index = valid_indices[0]
        else:
            # Multiple valid maps - use probabilistic selection
            max_vals = modified_wvf.max(axis=(1, 2))
            valid_max_vals = max_vals[valid_indices]
            valid_min_dists = min_dist_per_map[valid_indices]
            
            # Compute weights: higher value, lower distance = higher weight
            # Add small epsilon to avoid division by zero
            value_weights = valid_max_vals + 1e-6
            distance_weights = 1.0 / (valid_min_dists + 1e-6)
            combined_weights = value_weights * distance_weights
            
            # Normalize to probabilities
            probabilities = combined_weights / combined_weights.sum()
            
            # Sample based on probabilities
            chosen_idx = np.random.choice(len(valid_indices), p=probabilities)
            best_map_index = valid_indices[chosen_idx]
        
        return modified_wvf[best_map_index]  
    # def sample_action_with_wvf(self, obs, chosen_reward_map, epsilon=0.0):
    #     """
    #     Sample an action using epsilon-greedy selection where the values are computed
    #     using the successor representation and a given reward map.
        
    #     Parameters:
    #     obs: observation from which to determine the current state.
    #     chosen_reward_map: a 2D reward map (grid_size x grid_size) to be used as the goal.
    #     epsilon: probability of choosing a random action.
        
    #     Returns:
    #     The selected action (an integer).
    #     """
    #     # Convert observation into a state index (or coordinate) used by SR.
    #     state_idx = self.get_state_index(obs)
        
    #     if np.random.uniform(0, 1) < epsilon:
    #         return np.random.randint(self.action_size)
    #     else:
    #         # print("non random action taken")
    #         # Compute Q-values using the chosen reward map.
    #         values = self.value_estimates_with_wvf(state_idx, chosen_reward_map)
    #         action = np.argmax(values)
    #         # Break Ties
    #         return action


    def value_estimates_with_wvf(self, state_idx, reward_map):
        """
        Generate values for all actions given a reward map.
        
        Parameters:
        state_idx: index (or indices) corresponding to the current state.
        reward_map: a 2D reward map, e.g. shape (grid_size, grid_size)
                    which must be flattened to match the state representation.
        
        Returns:
        Q-values for each action: shape (action_size,)
        """
        # Flatten the reward map to create a reward vector compatible with SR
        goal_vector = reward_map.flatten()  # shape: (state_size,)
        
        # Compute Q-values as the dot product of the SR for each action and the goal vector.
        # self.M has shape (action_size, state_size, state_size),
        # and indexing with state_idx gives a slice of shape (action_size, state_size).
        Qs = np.matmul(self.M[:, state_idx, :], goal_vector)
        return Qs
    
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
        """
        Update successor features using policy-independent learning.
        The SR should capture state transition dynamics regardless of reward structure.
        """
        s = current_exp[0]    # current state index
        s_a = current_exp[1]  # current action
        s_1 = current_exp[2]  # next state index
        done = current_exp[4] # terminal flag

        I = self._onehot(s, self.state_size)

        if done:
            # Terminal state: no future state occupancy expected
            td_target = I
        else:
            # For continuing states, we need to be careful about action selection
            # Option 1: Use the actual next action that was taken (SARSA-style)
            if next_exp is not None:
                s_a_1 = next_exp[1]  # actual next action
                td_target = I + self.gamma * self.M[s_a_1, s_1, :]
            else:
                # Option 2: Use uniform policy for SR learning (policy-independent)
                # Average over all possible next actions
                uniform_next_sr = np.mean(self.M[:, s_1, :], axis=0)
                td_target = I + self.gamma * uniform_next_sr
                
                # Option 3: Use epsilon-greedy with high epsilon for exploration
                # if np.random.random() < 0.8:  # High exploration
                #     random_action = np.random.randint(self.action_size)
                #     td_target = I + self.gamma * self.M[random_action, s_1, :]
                # else:
                #     # Occasionally use current policy
                #     best_a_prime = np.argmax([
                #         np.dot(self.M[a, s_1, :], self.w) for a in range(self.action_size)
                #     ])
                #     td_target = I + self.gamma * self.M[best_a_prime, s_1, :]

        td_error = td_target - self.M[s_a, s, :]
        self.M[s_a, s, :] += self.learning_rate * td_error

        # Apply small L2 shrinkage for stability
        lambda_reg = 1e-4
        self.M[s_a, s, :] *= (1.0 - lambda_reg)

        # # Regularization
        entropy_reg = 0.01
        self.M[s_a, s, :] += entropy_reg * (1.0/self.state_size - self.M[s_a, s, :])

        return np.mean(np.abs(td_error))
    
    # def update_sr(self, current_exp, next_exp):
    #     """
    #     Update successor features using off-policy max (Q-learning style) update.
    #     Also applies a small L2 shrinkage to stabilize learning.
    #     """
    #     s = current_exp[0]    # current state index
    #     s_a = current_exp[1]  # current action
    #     s_1 = current_exp[2]  # next state index
    #     done = current_exp[4] # terminal flag

    #     I = self._onehot(s, self.state_size)

    #     if done:
    #         td_target = I
    #     else:
    #         # Use greedy (off-policy) next action for SR update
    #         best_a_prime = np.argmax([
    #             np.dot(self.M[a, s_1, :], self.w) for a in range(self.action_size)
    #         ])
    #         td_target = I + self.gamma * self.M[best_a_prime, s_1, :]

    #     td_error = td_target - self.M[s_a, s, :]
    #     self.M[s_a, s, :] += self.learning_rate * td_error

    #     # Apply small L2 shrinkage
    #     lambda_reg = 1e-4
    #     self.M[s_a, s, :] *= (1.0 - lambda_reg)

    #     return np.mean(np.abs(td_error))

    # def update_sr(self, current_exp, next_exp):
    #     """Update successor features using SARSA TD learning"""
    #     s = current_exp[0]    # current state index
    #     s_a = current_exp[1]  # current action
    #     s_1 = current_exp[2]  # next state index
    #     s_a_1 = next_exp[1]   # next action
    #     d = current_exp[4]    # done flag
        
    #     I = self._onehot(s, self.state_size)
        
    #     if d:
    #         td_error = (I + self.gamma * self._onehot(s_1, self.state_size) - self.M[s_a, s, :])
    #     else:
    #         td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
            
    #     self.M[s_a, s, :] += self.learning_rate * td_error
    #     return np.mean(np.abs(td_error))
    
    # def update_sr(self, current_exp, next_exp):
    #     """Update successor features using SARSA TD learning"""
    #     s = current_exp[0]    # current state index
    #     s_a = current_exp[1]  # current action
    #     s_1 = current_exp[2]  # next state index
    #     s_a_1 = next_exp[1]   # next action
    #     done = current_exp[4] # terminal flag

    #     I = self._onehot(s, self.state_size)

    #     if done:
    #         td_target = I
    #     else:
    #         td_target = I + self.gamma * self.M[s_a_1, s_1, :]
        
    #     td_error = td_target - self.M[s_a, s, :]
    #     self.M[s_a, s, :] += self.learning_rate * td_error

    #     return np.mean(np.abs(td_error))

    
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
