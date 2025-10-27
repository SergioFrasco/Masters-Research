# ============================================================================
# UPDATED successor_agent.py - Direct environment access 
# ============================================================================

import numpy as np

class RandomAgentWithSR:
    """Random agent that learns Successor Representation"""
    
    def __init__(self, env, learning_rate=0.05, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # Grid setup - MiniWorld uses continuous coordinates
        self.grid_size = env.size  # e.g., 10 for 10x10 room
        self.state_size = self.grid_size * self.grid_size
        
        # Initialize SR matrix: M[action, from_state, to_state]
        self.M = np.zeros((self.action_size, self.state_size, self.state_size))
        
        # Store previous experience for TD update
        self.prev_state = None
        self.prev_action = None

        # Initialize the true map to track discovered reward locations and predictions
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))

        # World Value Function - Mappings of values to each state goal pair
        self.wvf = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Initialize individual reward maps: one per state
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)
    
    def sample_action_with_wvf(self, obs, epsilon=0.0):

        """Sample action using WVF"""
        # if not self.initialized:
        #     self.initialize_path_integration(obs)
            
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        x, z = self._get_agent_pos_from_env()
        current_dir = self._get_agent_dir_from_env()
        
        # Fixed: neighbors in (x, y) format
        neighbors = [
            ((x + 1, z), 0),  # Right
            ((x, z + 1), 1),  # Down
            ((x - 1, z), 2),  # Left
            ((x, z - 1), 3)   # Up
        ]
        
        valid_actions = []
        valid_values = []
        
        for neighbor_pos, target_dir in neighbors:
            if self._is_valid_position(neighbor_pos):  # Now passing (x, y)
                next_x, next_y = neighbor_pos  # Unpack as (x, y)
                max_value_across_maps = np.max(self.wvf[:, next_y, next_x])  # Index as [y, x]
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
    
    def _is_valid_position(self, pos):
        """Check if position is valid by simulating movement"""
        import numpy as np
        
        x, z = pos[0], pos[1]
        
        # Store original agent position
        original_pos = self._get_agent_pos_from_env()
        original_dir = self._get_agent_dir_from_env()
        
        # Calculate direction vector to new position
        dx = x - original_pos[0]
        dz = z - original_pos[1]
        
        # Try to move the agent to the new position
        # Set a temporary position
        new_pos = np.array([x, original_pos[1], z])
        
        # Check boundaries first
        boundary = 9  # Adjust based on your environment
        if abs(x) > boundary or abs(z) > boundary:
            return False
        
        # Simple distance-based collision check
        agent_radius = 0.18
        for entity in self.env.entities:
            if hasattr(entity, 'pos') and hasattr(entity, 'radius'):
                dist = np.linalg.norm(new_pos - entity.pos)
                if dist < agent_radius + entity.radius:
                    return False
        
        return True
    
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
    
    def get_state_index(self):
        """Convert current grid position to flat state index"""
        x, z = self._get_agent_pos_from_env()
        # Clamp to valid range
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def update_sr(self, s, action, s_next, next_action, done):
        """Update SR matrix using TD learning with next action (SARSA-style)"""
        # Skip turns - they don't change state
        if action != self.MOVE_FORWARD:
            return 0.0
        
        # Skip if we didn't actually move
        if s == s_next and not done:
            return 0.0
        
        # One-hot encoding of current state
        I = np.zeros(self.state_size)
        I[s] = 1.0
        
        if done:
            td_target = I
        else:
            if next_action == self.MOVE_FORWARD:
                td_target = I + self.gamma * self.M[next_action, s_next, :]
            else:
                td_target = I + self.M[self.MOVE_FORWARD, s_next, :]
        
        # TD error and update
        td_error = td_target - self.M[action, s, :]
        self.M[action, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))
    
    def select_action(self):
        """Select a random action"""
        return np.random.randint(self.action_size)
    
    def reset(self):
        """Reset for new episode"""
        self.prev_state = None
        self.prev_action = None
    
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
