import numpy as np

from utils.matrices import onehot
from minigrid.core.world_object import Goal
from gym import spaces

class SuccessorAgentPartialSARSA:
    # Changed gamma from 0.99
    def __init__(self, env, learning_rate=0.05, gamma=0.99):
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
        # self.M += np.random.normal(0, 0.01, self.M.shape) # Add small random noise

        self.w = np.zeros([self.state_size])

        # Initialize the true map to track discovered reward locations and predictions
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.true_reward_map_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Individual Reward maps that are composed with the SR
        # Initialize individual reward maps: one per state
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Track states we have visited to inform our map updates correctly
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # World Value Function - Mappings of values to each state goal pair
        self.wvf = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Path integration variables
        self.internal_pos = None  # Will be initialized on first reset
        self.internal_dir = None  # Will be initialized on first reset
        self.initialized = False
        
        # Keep track of movement history for debugging
        self.movement_history = []

    def initialize_path_integration(self, obs):
        """Initialize internal position and direction from environment on first call"""
        if not self.initialized:
            self.internal_pos = tuple(self.env.agent_pos)  # (x, y)
            self.internal_dir = self.env.agent_dir
            self.initialized = True
            # print(f"Path integration initialized: pos={self.internal_pos}, dir={self.internal_dir}")

    def get_state_index(self, obs):
        """Convert position to state index using internal path integration"""
        if not self.initialized:
            self.initialize_path_integration(obs)
        
        x, y = self.internal_pos
        return y * self.grid_size + x  # Use (y,x) consistently

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
            # Turn left (counterclockwise)
            self.internal_dir = (direction - 1) % 4
        elif action == TURN_RIGHT:
            # Turn right (clockwise)
            self.internal_dir = (direction + 1) % 4
        elif action == MOVE_FORWARD:
            # Move forward in current direction
            # Direction mapping: 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
            if direction == 0:  # Right
                new_x, new_y = x + 1, y
            elif direction == 1:  # Down
                new_x, new_y = x, y + 1
            elif direction == 2:  # Left
                new_x, new_y = x - 1, y
            elif direction == 3:  # Up
                new_x, new_y = x, y - 1
            else:
                new_x, new_y = x, y  # Invalid direction, don't move

            # Check if the new position is valid (within bounds and not a wall)
            if self._is_valid_position((new_x, new_y)):
                self.internal_pos = (new_x, new_y)
                self.movement_history.append(f"Moved from ({x},{y}) to ({new_x},{new_y})")
            else:
                # Hit a wall or boundary, don't update position
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
        
        # Check if it's a wall using environment grid
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
        
        # Find the best value among valid actions
        best_value = max(valid_values)
        best_indices = [i for i, v in enumerate(valid_values) if v == best_value]
        
        # If multiple actions have the same best value, choose randomly among them
        chosen_index = np.random.choice(best_indices)
        return valid_actions[chosen_index]

    def _get_action_toward_direction(self, current_dir, target_dir):
        """Get the action needed to face the target direction from current direction"""
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

    # def update_sr(self, num_forward_steps,current_exp, next_exp):
    #     """Update successor features using policy-independent learning."""
    #     s = current_exp[0]    # current state index
    #     s_a = current_exp[1]  # current action
    #     s_1 = current_exp[2]  # next state index
    #     done = current_exp[4] # terminal flag
        
    #     # MiniGrid action constants
    #     TURN_LEFT = 0
    #     TURN_RIGHT = 1
    #     MOVE_FORWARD = 2
        
    #     # update SR for move forward actions (actual state transitions)
    #     if s_a != MOVE_FORWARD:
    #         return 0.0  # No update, return zero TD error
        
    #     # safety check: ensure we actually transitioned states
    #     if s == s_1 and not done:
    #         return 0.0  # No state transition occurred
        
    #     I = self._onehot(s, self.state_size)
        
    #     if done:
    #         # terminal: no future state occupancy expected
    #         td_target = I
    #     else:
    #         # For continuing states, we need to handle the temporal discount properly
    #         if next_exp is not None:
    #             s_a_1 = next_exp[1]  # actual next action
                
    #             if s_a_1 == MOVE_FORWARD:
    #                 # next action transitions states, use normal discount
    #                 td_target = I + self.gamma * self.M[s_a_1, s_1, :]
    #             else:
    #                 # next action is a turn - it doesn't change state but takes time
    #                 td_target = I + self.M[MOVE_FORWARD, s_1, :]  # No gamma discount for turns
    #         else:
    #             # Fallback
    #             td_target = I
        
    #     td_error = td_target - self.M[s_a, s, :]
    #     self.M[s_a, s, :] += self.learning_rate * td_error
        
    #     return np.mean(np.abs(td_error))
    
    # def update_sr(self, num_forward_steps, current_exp, next_exp):
    #     """Update successor features for forward actions only."""
    #     s = current_exp[0]    # current state index
    #     s_a = current_exp[1]  # current action
    #     s_1 = current_exp[2]  # next state index
    #     done = current_exp[4] # terminal flag
        
    #     MOVE_FORWARD = 2
        
    #     # Only update SR for forward actions
    #     if s_a != MOVE_FORWARD:
    #         return 0.0
        
    #     # Safety check for actual state transition
    #     if s == s_1 and not done:
    #         return 0.0
        
    #     # Only update SR on forward movements
    #     # This models place cell representations
    #     if s_a == MOVE_FORWARD:
    #         I = self._onehot(s, self.state_size)
    #         if done:
    #             td_target = I
    #         else:
    #             # Only bootstrap from forward SR
    #             td_target = I + self.gamma * self.M[MOVE_FORWARD, s_1, :]
            
    #         td_error = td_target - self.M[MOVE_FORWARD, s, :]
    #         self.M[MOVE_FORWARD, s, :] += self.learning_rate * td_error
        
    #     return np.mean(np.abs(td_error))

    def update_sr(self, current_exp, next_exp):
        """Full SARSA SR update (all actions) with safe handling of None states."""
        state = int(current_exp[0])
        action = int(current_exp[1])
        done = current_exp[4]

        I_s = np.zeros(self.state_size)
        I_s[state] = 1.0

        # Defensive guard — if next_exp is malformed or terminal
        if (
            done 
            or next_exp is None 
            or next_exp[1] is None 
            or next_exp[2] is None
        ):
            td_target = I_s
        else:
            next_action = int(next_exp[1])
            next_state = int(next_exp[2])
            td_target = I_s + self.gamma * self.M[next_action, next_state, :]

        td_error = td_target - self.M[action, state, :]
        self.M[action, state, :] += self.learning_rate * td_error

        return np.mean(np.abs(td_error))


    def update(self, current_exp, next_exp=None):
        """Update both reward weights and successor features."""
        # Always update reward weights when we observe a reward
        error_w = self.update_w(current_exp)
        
        # Only update SR for actual state transitions (move forward)
        error_sr = 0
        if next_exp is not None:
            error_sr = self.update_sr(current_exp, next_exp)

        
        return error_w, error_sr

    def update_w(self, current_exp):
        """Update reward weights - only when we actually observe a reward."""
        s_1 = current_exp[2]  # next state index
        r = current_exp[3]    # reward
        
        # Only update if we actually received a reward
        if r != 0:
            error = r - self.w[s_1]
            self.w[s_1] += self.learning_rate * error
            return error
        
        return 0.0  # No reward observed, no update

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