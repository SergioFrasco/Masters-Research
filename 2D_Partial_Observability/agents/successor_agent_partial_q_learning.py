import numpy as np

from utils.matrices import onehot
from minigrid.core.world_object import Goal
from gym import spaces

class SuccessorAgentPartialQLearning:
    def __init__(self, env, learning_rate=0.05, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Calculate state size based on grid dimensions
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        
        # MiniGrid default actions: left, right, forward
        self.action_size = 3
        
        # initialization of the SR
        self.M = np.zeros((self.action_size, self.state_size, self.state_size))

        self.w = np.zeros([self.state_size])

        # Initialize the true map to track discovered reward locations and predictions
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.true_reward_map_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Individual Reward maps that are composed with the SR
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Track states we have visited to inform our map updates correctly
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # World Value Function - Mappings of values to each state goal pair
        self.wvf = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Path integration variables
        self.internal_pos = None
        self.internal_dir = None
        self.initialized = False
        
        # Keep track of movement history for debugging
        self.movement_history = []

    def initialize_path_integration(self, obs):
        """Initialize internal position and direction from environment on first call"""
        if not self.initialized:
            self.internal_pos = tuple(self.env.agent_pos)
            self.internal_dir = self.env.agent_dir
            self.initialized = True

    def get_state_index(self, obs):
        """Convert position to state index using internal path integration"""
        if not self.initialized:
            self.initialize_path_integration(obs)
        
        x, y = self.internal_pos
        return y * self.grid_size + x

    def update_internal_state(self, action):
        """Update internal position and direction based on action taken"""
        if not self.initialized:
            print("Warning: Path integration not initialized!")
            return

        # MiniGrid action constants
        TURN_LEFT = 0
        TURN_RIGHT = 1
        MOVE_FORWARD = 2

        x, y = self.internal_pos
        direction = self.internal_dir

        if action == TURN_LEFT:
            self.internal_dir = (direction - 1) % 4
        elif action == TURN_RIGHT:
            self.internal_dir = (direction + 1) % 4
        elif action == MOVE_FORWARD:
            if direction == 0:  # Right
                new_x, new_y = x + 1, y
            elif direction == 1:  # Down
                new_x, new_y = x, y + 1
            elif direction == 2:  # Left
                new_x, new_y = x - 1, y
            elif direction == 3:  # Up
                new_x, new_y = x, y - 1
            else:
                new_x, new_y = x, y

            if self._is_valid_position((new_x, new_y)):
                self.internal_pos = (new_x, new_y)
                self.movement_history.append(f"Moved from ({x},{y}) to ({new_x},{new_y})")
            else:
                self.movement_history.append(f"Hit wall/boundary at ({new_x},{new_y}), stayed at ({x},{y})")

    def reset_path_integration(self):
        """Reset path integration for new episode"""
        self.initialized = False
        self.internal_pos = None
        self.internal_dir = None
        self.movement_history = []

    def sample_random_action(self, obs, goal=None, epsilon=0.0):
        """Sample an action uniformly at random"""
        if not self.initialized:
            self.initialize_path_integration(obs)
        return np.random.randint(self.action_size)

    def _is_valid_position(self, pos):
        """Check if position is valid (within bounds and not a wall)"""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        cell = self.env.grid.get(x, y)
        from minigrid.core.world_object import Wall
        return cell is None or not isinstance(cell, Wall)

    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """Sample action using WVF with path integration"""
        if not self.initialized:
            self.initialize_path_integration(obs)
            
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        x, y = self.internal_pos
        current_dir = self.internal_dir
        
        neighbors = [
            ((y, x + 1), 0), ((y + 1, x), 1), 
            ((y, x - 1), 2), ((y - 1, x), 3)
        ]
        
        valid_actions = []
        valid_values = []
        
        for neighbor_pos, target_dir in neighbors:
            if self._is_valid_position(neighbor_pos):
                next_y, next_x = neighbor_pos
                max_value_across_maps = np.max(self.wvf[:, next_y, next_x])
                action_to_take = self._get_action_toward_direction(current_dir, target_dir)
                
                valid_actions.append(action_to_take)
                valid_values.append(max_value_across_maps)
        
        if not valid_actions:
            return np.random.randint(self.action_size)
        
        best_value = max(valid_values)
        best_indices = [i for i, v in enumerate(valid_values) if v == best_value]
        
        chosen_index = np.random.choice(best_indices)
        return valid_actions[chosen_index]

    def _get_action_toward_direction(self, current_dir, target_dir):
        """Get the action needed to face the target direction from current direction"""
        if current_dir == target_dir:
            return 2  # move forward
        
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            return 1  # turn right
        elif diff == 3:
            return 0  # turn left  
        elif diff == 2:
            return np.random.choice([0, 1])
        
        return 2

    def update_sr(self, current_exp, next_exp):
        """
        Q-LEARNING UPDATE for successor features.
        Key difference: uses MAX over actions for next state, not the actual next action.
        """
        s = current_exp[0]    # current state index
        s_a = current_exp[1]  # current action
        s_1 = current_exp[2]  # next state index
        done = current_exp[4] # terminal flag
        
        # MiniGrid action constants
        TURN_LEFT = 0
        TURN_RIGHT = 1
        MOVE_FORWARD = 2
        
        # Only update SR for move forward actions (actual state transitions)
        if s_a != MOVE_FORWARD:
            return 0.0
        
        # Additional safety check: ensure we actually transitioned states
        if s == s_1 and not done:
            return 0.0
        
        I = self._onehot(s, self.state_size)
        
        if done:
            # Terminal state: no future state occupancy expected
            td_target = I
        else:
            # Q-LEARNING: Take MAX over all forward actions from next state
            # We consider only MOVE_FORWARD action since turns don't change SR
            td_target = I + self.gamma * self.M[MOVE_FORWARD, s_1, :]
        
        td_error = td_target - self.M[s_a, s, :]
        self.M[s_a, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))

    def update(self, current_exp, next_exp=None):
        """Update both reward weights and successor features."""
        error_w = self.update_w(current_exp)
        
        error_sr = 0
        if next_exp is not None:
            error_sr = self.update_sr(current_exp, next_exp)
        
        return error_w, error_sr

    def update_w(self, current_exp):
        """Update reward weights - only when we actually observe a reward."""
        s_1 = current_exp[2]
        r = current_exp[3]
        
        if r != 0:
            error = r - self.w[s_1]
            self.w[s_1] += self.learning_rate * error
            return error
        
        return 0.0

    def _onehot(self, index, size):
        """Create one-hot encoded vector"""
        vec = np.zeros(size)
        vec[index] = 1
        return vec

    def is_goal_state(self, obs):
        """Check if current internal position contains a goal"""
        if not self.initialized:
            self.initialize_path_integration(obs)
        
        x, y = self.internal_pos
        cell = self.env.grid.get(x, y)
        return isinstance(cell, Goal)

    def verify_path_integration(self, obs):
        """Debug function to verify path integration accuracy"""
        if not self.initialized:
            return True, "Not initialized yet"
        
        actual_pos = tuple(self.env.agent_pos)
        actual_dir = self.env.agent_dir
        
        pos_match = self.internal_pos == actual_pos
        dir_match = self.internal_dir == actual_dir
        
        if not pos_match or not dir_match:
            return False, f"Mismatch - Internal: pos={self.internal_pos}, dir={self.internal_dir} | Actual: pos={actual_pos}, dir={actual_dir}"
        
        return True, "Path integration accurate"