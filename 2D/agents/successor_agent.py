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
        """Convert MiniGrid observation to state index - make consistent"""
        agent_pos = self.env.agent_pos
        x, y = agent_pos
        return y * self.grid_size + x  # Use (y,x) consistently
    
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


    # Helper funcitons for local action selection
    def _get_next_position(self, current_pos, action):
        """
        Simulate the next position given current position and action
        MiniGrid action space:
        - 0: turn left
        - 1: turn right  
        - 2: move forward
        """
        x, y = current_pos
        agent_dir = self.env.agent_dir
        
        if action == 0:  # turn left
            # Position doesn't change, only direction changes
            return (x, y)
        elif action == 1:  # turn right
            # Position doesn't change, only direction changes
            return (x, y)
        elif action == 2:  # move forward
            # Move in current direction
            if agent_dir == 0:  # right
                return (x + 1, y)
            elif agent_dir == 1:  # down
                return (x, y + 1)
            elif agent_dir == 2:  # left
                return (x - 1, y)
            elif agent_dir == 3:  # up
                return (x, y - 1)
        
        return current_pos

    def _position_to_state_index(self, pos):
        """Convert position to state index"""
        x, y = pos
        return x + y * self.grid_size

    def _is_valid_position(self, pos):
        """Check if position is valid (within bounds and not a wall)"""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        # Check if it's a wall
        cell = self.env.grid.get(x, y)
        from minigrid.core.world_object import Wall
        return cell is None or not isinstance(cell, Wall)

    # the previous one had to have a higher agent pos value than moving forward, this one intelligently makes the move to maximize the next step too
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """
        Sample action by evaluating all 4 neighboring positions and choosing
        the action sequence that gets us to the highest-value neighbor
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        current_pos = self.env.agent_pos
        x, y = current_pos
        current_dir = self.env.agent_dir
        
        # Define the 4 neighboring positions
        neighbors = [
            ((x + 1, y), 0),  # right, direction 0
            ((x, y + 1), 1),  # down, direction 1  
            ((x - 1, y), 2),  # left, direction 2
            ((x, y - 1), 3),  # up, direction 3
        ]
        
        best_value = -np.inf
        best_action = np.random.randint(self.action_size)
        
        for neighbor_pos, target_dir in neighbors:
            if self._is_valid_position(neighbor_pos):
                next_y, next_x = neighbor_pos
                
                # Get max WVF value at this neighbor
                max_value_across_maps = np.max(self.wvf[:, next_y, next_x])
                
                if max_value_across_maps > best_value:
                    best_value = max_value_across_maps
                    # Determine what action to take to move toward this neighbor
                    best_action = self._get_action_toward_direction(current_dir, target_dir)
        
        return best_action

    def _get_action_toward_direction(self, current_dir, target_dir):
        """
        Get the action needed to face the target direction from current direction
        """
        if current_dir == target_dir:
            return 2  # move forward
        
        # Calculate the shortest rotation
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:  # need to turn right once
            return 1  # turn right
        elif diff == 3:  # need to turn left once (or right 3 times)
            return 0  # turn left  
        elif diff == 2:  # need to turn around (180 degrees)
            # Choose randomly between left and right (both take 2 steps)
            return np.random.choice([0, 1])
        
        return 2  # fallback: move forward


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
    
    # TODO change this to a sarsa update that only occurs at the end of an episode
    # TODO pick by best action, if 2 action shave the same value divide by those 2
    #  Multiply by probability of tkzing that action
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
                # For terminal states only
                td_target = I

        td_error = td_target - self.M[s_a, s, :]
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
