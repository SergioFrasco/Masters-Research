import numpy as np

from utils.matrices import onehot
from minigrid.core.world_object import Goal
from gym import spaces

class SuccessorAgent:
    # learning rate = 0.1
    # Tried 0.01
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
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
        self.M += np.random.normal(0, 0.01, self.M.shape) # Add small random noise

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

        # Path integration for partial observability
        self.estimated_pos = (1,1)  # Start position estimate
        self.estimated_dir = 0  # Start direction estimate
        self.position_confidence = 1.0  # Confidence in position estimate

        # Track movement history for error correction
        self.movement_history = []
        self.max_history = 100

    # Functions added or adapted for partial observability
    def update_position_estimate(self, action):
        """
        Update internal position and direction estimate given the action.
        This is a *path integration* update – assumes actions succeed.
        """
        # Turn left
        if action == 0:  
            self.estimated_dir = (self.estimated_dir - 1) % 4

        # Turn right
        elif action == 1:  
            self.estimated_dir = (self.estimated_dir + 1) % 4

        # Move forward
        elif action == 2:  
            x, y = self.estimated_pos  # Unpack tuple
            
            dx, dy = 0, 0
            if self.estimated_dir == 0:   # facing right
                dx = 1
            elif self.estimated_dir == 1: # facing down
                dy = 1
            elif self.estimated_dir == 2: # facing left
                dx = -1
            elif self.estimated_dir == 3: # facing up
                dy = -1

            new_x = x + dx
            new_y = y + dy

            # Clamp to within grid bounds
            if hasattr(self, "grid_size"):
                new_x = max(0, min(self.grid_size - 1, new_x))
                new_y = max(0, min(self.grid_size - 1, new_y))

            self.estimated_pos = (new_x, new_y)  # Store as tuple
            
    def _obs_to_global_coords(self, obs_x, obs_y, agent_x, agent_y, agent_dir, obs_width, obs_height):
        """
        Version with bounds checking and error handling.
        Recommended for production use.
        """
        
        # Validate input coordinates
        if not (0 <= obs_x < obs_width and 0 <= obs_y < obs_height):
            raise ValueError(f"Observation coordinates ({obs_x}, {obs_y}) out of bounds for observation size ({obs_width}, {obs_height})")
        
        if not (0 <= agent_dir <= 3):
            raise ValueError(f"Invalid agent direction: {agent_dir}. Must be 0-3.")
        
        # Agent position in observation (bottom-center)
        agent_obs_x = obs_width // 2
        agent_obs_y = obs_height - 1
        
        # Relative coordinates in observation frame
        rel_obs_x = obs_x - agent_obs_x
        rel_obs_y = agent_obs_y - obs_y
        
        # Rotation matrix application
        rotation_matrices = {
            0: (rel_obs_y, -rel_obs_x),    # Right: 90° CW
            1: (rel_obs_x, rel_obs_y),     # Down: 180°  
            2: (-rel_obs_y, rel_obs_x),    # Left: 90° CCW
            3: (-rel_obs_x, -rel_obs_y)    # Up: 180°
        }
        
        global_offset_x, global_offset_y = rotation_matrices[agent_dir]
        
        # Final global coordinates
        global_x = agent_x + global_offset_x
        global_y = agent_y + global_offset_y
        
        # Optional bounds checking for global coordinates
        if hasattr(self, 'grid_size'):
            global_x = max(0, min(self.grid_size - 1, global_x))
            global_y = max(0, min(self.grid_size - 1, global_y))
        
        return global_x, global_y


    def test_obs_to_global_coords(self):
        """
        Test function to verify coordinate transformation correctness.
        Call this during debugging to validate your transformations.
        """
        print("Testing observation to global coordinate transformation...")
        
        # Test case 1: Agent facing right, looking at position directly ahead
        agent_x, agent_y, agent_dir = 5, 5, 0  # Facing right
        obs_width, obs_height = 7, 7
        
        # Position directly in front of agent in observation
        obs_x, obs_y = 3, 5  # Center-x, one row above agent
        
        global_x, global_y = self._obs_to_global_coords(
            obs_x, obs_y, agent_x, agent_y, agent_dir, obs_width, obs_height
        )
        
        expected_x, expected_y = 6, 5  # Should be one step right of agent
        print(f"Test 1 - Expected: ({expected_x}, {expected_y}), Got: ({global_x}, {global_y})")
        assert (global_x, global_y) == (expected_x, expected_y), "Test 1 failed!"
        
        # Test case 2: Agent facing up, looking at position to the right in observation
        agent_x, agent_y, agent_dir = 5, 5, 3  # Facing up
        obs_x, obs_y = 4, 6  # One step right of center, same row as agent
        
        global_x, global_y = self._obs_to_global_coords(
            obs_x, obs_y, agent_x, agent_y, agent_dir, obs_width, obs_height
        )
        
        expected_x, expected_y = 4, 5  # Should be one step left of agent (obs right becomes global left)
        print(f"Test 2 - Expected: ({expected_x}, {expected_y}), Got: ({global_x}, {global_y})")
        assert (global_x, global_y) == (expected_x, expected_y), "Test 2 failed!"
        
        print("All coordinate transformation tests passed!")

    def extract_local_observation_info(self, obs):
        """Extract what the agent can see from its partial observation"""
        agent_x, agent_y = self.estimated_pos
        agent_dir = self.estimated_dir
        
        # Get all goal positions in local coordinates
        local_goal_positions = self._extract_all_goals_from_obs(obs)
        
        observed_positions = []
        observed_values = []
        
        # Convert all observed positions to global coordinates
        for local_y in range(obs.shape[0]):
            for local_x in range(obs.shape[1]):
                global_x, global_y = self._local_to_global_coords(
                    local_x, local_y, agent_x, agent_y, agent_dir, obs.shape[0]
                )
                
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    # Check if this position contains a goal
                    obs_value = 1.0 if (local_x, local_y) in local_goal_positions else 0.0
                    observed_positions.append((global_x, global_y))
                    observed_values.append(obs_value)
        
        return observed_positions, observed_values
        
    def _extract_all_goals_from_obs(self, obs):
        """Extract goals using MiniGrid's internal representation"""
        goal_positions = []
        
        # If obs is the wrapped image observation, we need to check the actual grid
        # Get the current grid state from environment
        grid = self.env.grid.encode()
        agent_x, agent_y = self.estimated_pos
        agent_dir = self.estimated_dir
        
        # Check what the agent can actually see in the grid
        view_size = obs.shape[0]
        for local_y in range(view_size):
            for local_x in range(view_size):
                global_x, global_y = self._local_to_global_coords(
                    local_x, local_y, agent_x, agent_y, agent_dir, view_size
                )
                
                # Check if this global position contains a goal in the actual grid
                if (0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size):
                    cell = self.env.grid.get(global_x, global_y)
                    if cell is not None and cell.type == 'goal':
                        goal_positions.append((local_x, local_y))
        
        return goal_positions

    def _local_to_global_coords(self, local_x, local_y, agent_x, agent_y, agent_dir, view_size):
        """Convert local observation coordinates to global coordinates"""
        # This is specific to MiniGrid's observation format
        # Agent is typically at position (view_size//2, view_size-1) looking up in local coords
        
        # Relative position in agent's coordinate frame
        rel_x = local_x - view_size // 2
        rel_y = view_size - 1 - local_y  # Flip Y axis
        
        # Rotate based on agent direction
        if agent_dir == 0:  # facing right
            global_offset_x, global_offset_y = rel_y, -rel_x
        elif agent_dir == 1:  # facing down  
            global_offset_x, global_offset_y = rel_x, rel_y
        elif agent_dir == 2:  # facing left
            global_offset_x, global_offset_y = -rel_y, rel_x
        else:  # facing up
            global_offset_x, global_offset_y = -rel_x, -rel_y
        
        return agent_x + global_offset_x, agent_y + global_offset_y
       # Updated get state index for Partial Observability
    def get_state_index(self, obs):
        """Use estimated position instead of true position"""
        agent_pos = self.env.agent_pos
        x, y = agent_pos
        return y * self.grid_size + x  # Use (y,x) consistently

    # def extract_local_observation_info(self, obs, view_size=7):
    #     """
    #     Extract what the agent can see from its partial observation and convert to global coordinates.
    #     This function bridges between the MiniGrid observation format and our egocentric processing.
        
    #     Args:
    #         obs: MiniGrid observation (RGB image or encoded observation)
    #         view_size: Size of the egocentric view window (default 7x7)
        
    #     Returns:
    #         observed_positions: List of (global_x, global_y) tuples for visible positions
    #         observed_values: List of values (0.0 for empty, 1.0 for goal) corresponding to positions
    #     """
    #     agent_x, agent_y = self.estimated_pos
    #     agent_dir = self.estimated_dir
        
    #     observed_positions = []
    #     observed_values = []
        
    #     # Method 1: If obs is the raw MiniGrid RGB observation
    #     if hasattr(obs, 'shape') and len(obs.shape) == 3:
    #         # obs is RGB image from partial observation
    #         obs_height, obs_width = obs.shape[:2]
            
    #         # Extract goal positions from the RGB observation
    #         goal_positions_in_obs = self._extract_goals_from_rgb_obs(obs)
            
    #         # Convert each position in the observation to global coordinates
    #         for obs_y in range(obs_height):
    #             for obs_x in range(obs_width):
    #                 # Convert observation coordinates to global coordinates
    #                 global_x, global_y = self._obs_to_global_coords(
    #                     obs_x, obs_y, agent_x, agent_y, agent_dir, obs_width, obs_height
    #                 )
                    
    #                 # Check if this global position is within environment bounds
    #                 if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
    #                     # Determine the value at this position
    #                     obs_value = 1.0 if (obs_x, obs_y) in goal_positions_in_obs else 0.0
    #                     observed_positions.append((global_x, global_y))
    #                     observed_values.append(obs_value)
        
    #     # Method 2: Direct environment inspection (more reliable)
    #     # We can also directly check what the agent should be able to see in the environment
    #     else:
    #         # Use egocentric view approach to determine what's visible
    #         center_x = view_size // 2
    #         agent_ego_y = view_size - 1
            
    #         for ego_y in range(view_size):
    #             for ego_x in range(view_size):
    #                 # Convert egocentric coordinates to global coordinates
    #                 global_x, global_y = self._egocentric_to_global_coords(
    #                     ego_x, ego_y, agent_x, agent_y, agent_dir, view_size
    #                 )
                    
    #                 # Check if global position is within environment bounds
    #                 if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
    #                     # Check what's actually at this position in the environment
    #                     cell = self.env.grid.get(global_x, global_y)
                        
    #                     if cell is not None and cell.type == 'goal':
    #                         obs_value = 1.0
    #                     else:
    #                         obs_value = 0.0
                        
    #                     observed_positions.append((global_x, global_y))
    #                     observed_values.append(obs_value)
        
    #     return observed_positions, observed_values

    # def _extract_goals_from_rgb_obs(self, obs):
    #     """
    #     Extract goal positions from RGB observation image.
    #     This is tricky because we need to identify goals from pixel colors.
        
    #     Args:
    #         obs: RGB observation array of shape (height, width, 3)
        
    #     Returns:
    #         goal_positions: List of (x, y) tuples in observation coordinates
    #     """
    #     goal_positions = []
        
    #     # MiniGrid goal objects typically have a specific color
    #     # Goals are usually green: RGB approximately (0, 255, 0) or similar
    #     # You might need to adjust these values based on your specific MiniGrid setup
        
    #     height, width = obs.shape[:2]
        
    #     for y in range(height):
    #         for x in range(width):
    #             pixel = obs[y, x]
                
    #             # Check if this pixel represents a goal
    #             # Goals in MiniGrid are typically bright green
    #             if self._is_goal_pixel(pixel):
    #                 goal_positions.append((x, y))
        
    #     return goal_positions

    # def _is_goal_pixel(self, pixel):
    #     """
    #     Determine if a pixel represents a goal object.
        
    #     Args:
    #         pixel: RGB pixel values [r, g, b]
        
    #     Returns:
    #         bool: True if pixel represents a goal
    #     """
    #     r, g, b = pixel
        
    #     # Goals are typically bright green in MiniGrid
    #     # You may need to adjust these thresholds based on your environment
    #     if g > 200 and r < 100 and b < 100:  # Bright green
    #         return True
        
    #     # Alternative: Check for specific goal colors
    #     # MiniGrid goals
 
    
    def sample_random_action(self, obs, goal=None, epsilon=0.0):
        """Sample an action uniformly at random"""
        return np.random.randint(self.action_size)

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
    # def sample_action_with_wvf(self, obs, epsilon=0.0):
    #     """
    #     Sample action by evaluating all 4 neighboring positions and choosing
    #     the action sequence that gets us to the highest-value neighbor
    #     """
    #     if np.random.uniform(0, 1) < epsilon:
    #         return np.random.randint(self.action_size)
        
    #     current_pos = self.env.agent_pos
    #     x, y = current_pos
    #     current_dir = self.env.agent_dir
        
    #     # Define the 4 neighboring positions
    #     neighbors = [
    #         ((x + 1, y), 0),  # right, direction 0
    #         ((x, y + 1), 1),  # down, direction 1  
    #         ((x - 1, y), 2),  # left, direction 2
    #         ((x, y - 1), 3),  # up, direction 3
    #     ]
        
    #     best_value = -np.inf
    #     best_action = np.random.randint(self.action_size)
        
    #     for neighbor_pos, target_dir in neighbors:
    #         if self._is_valid_position(neighbor_pos):
    #             next_y, next_x = neighbor_pos
                
    #             # Get max WVF value at this neighbor
    #             max_value_across_maps = np.max(self.wvf[:, next_y, next_x])
                
    #             if max_value_across_maps > best_value:
    #                 best_value = max_value_across_maps
    #                 # Determine what action to take to move toward this neighbor
    #                 best_action = self._get_action_toward_direction(current_dir, target_dir)
        
    #     return best_action
    
    # New one which only chooses between valid actions
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        current_pos = self.estimated_pos
        x, y = current_pos
        current_dir = self.estimated_dir
        
        neighbors = [
            ((x + 1, y), 0), ((x, y + 1), 1), 
            ((x - 1, y), 2), ((x, y - 1), 3)
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
    
    def update_sr(self, current_exp, next_exp):
        """
        Update successor features using policy-independent learning.
        The SR should capture state transition dynamics regardless of reward structure.
        Only updates when action is 'move forward' (actual state transition).
        """
        # agent_pos = tuple(self.env.unwrapped.agent_pos)
        # agent_dir = self.env.unwrapped.agent_dir
        # exact_pose = (agent_pos, agent_dir)

        # estimated_pose = (tuple(self.estimated_pos), self.estimated_dir)

        # if exact_pose != estimated_pose:
        #     print("Mismatch detected!")
        #     print("Exact Pose:     ", exact_pose)
        #     print("Estimated Pose: ", estimated_pose)

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
            return 0.0  # No update, return zero TD error
        
        # Additional safety check: ensure we actually transitioned states
        # This handles edge cases where move forward might fail (e.g., hitting wall)
        if s == s_1 and not done:
            return 0.0  # No actual state transition occurred
        
        I = self._onehot(s, self.state_size)
        
        if done:
            # Terminal state: no future state occupancy expected
            td_target = I
        else:
            # For continuing states, we need to handle the temporal discount properly
            if next_exp is not None:
                s_a_1 = next_exp[1]  # actual next action
                
                # Key insight: If next action is also move forward, we discount normally
                # If next action is a turn, we need to look ahead to the next move forward
                if s_a_1 == MOVE_FORWARD:
                    # Next action transitions states, use normal discount
                    td_target = I + self.gamma * self.M[s_a_1, s_1, :]
                else:
                    # Next action is a turn - it doesn't change state but takes time
                    # We need to account for the temporal cost of turning
                    # Option 1: Use undiscounted SR since no spatial transition
                    td_target = I + self.M[MOVE_FORWARD, s_1, :]  # No gamma discount for turns
                    
                    # Option 2: Use discounted but with move forward SR
                    # td_target = I + self.gamma * self.M[MOVE_FORWARD, s_1, :]
            else:
                # Fallback case - shouldn't happen in normal operation
                td_target = I
        
        td_error = td_target - self.M[s_a, s, :]
        self.M[s_a, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))

    def update(self, current_exp, next_exp=None):
        """
        Update both reward weights and successor features.
        Modified to handle temporal consistency.
        """
        # Always update reward weights when we observe a reward
        error_w = self.update_w(current_exp)
        
        # Only update SR for actual state transitions (move forward)
        error_sr = 0
        if next_exp is not None:
            error_sr = self.update_sr(current_exp, next_exp)
        
        return error_w, error_sr

    def update_w(self, current_exp):
        """
        Update reward weights - now more careful about when to update.
        Only update when we actually observe a reward.
        """
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
        """Check if current state contains a goal"""
        agent_pos = self.env.agent_pos
        cell = self.env.grid.get(*agent_pos)
        return isinstance(cell, Goal)
