# ============================================================================
# UPDATED successor_agent.py - Direct environment access (no path integration)
# ============================================================================

import numpy as np

class RandomAgentWithSR:
    """Random agent that learns Successor Representation"""
    
    def __init__(self, env, learning_rate=0.01, gamma=0.95):
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
        x = int(round(self.env.agent.pos[0]))
        z = int(round(self.env.agent.pos[2]))
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
    
    def create_egocentric_observation(self, goal_global_pos=None, matrix_size=11):
        """
        Create a mock egocentric observation matrix where agent is always at 
        bottom-middle facing upward.
        
        Args:
            goal_global_pos: Tuple (goal_x, goal_z) in global coordinates, or None
            matrix_size: Size of the square matrix (default 11x11)
        
        Returns:
            ego_matrix: numpy array of shape (matrix_size, matrix_size)
        """
        # Initialize empty egocentric matrix
        ego_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        
        # If no goal detected, return empty matrix
        if goal_global_pos is None:
            return ego_matrix
        
        # Get agent's current position and direction from environment
        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()
        goal_x, goal_z = goal_global_pos
        
        # Calculate offset from agent to goal in GLOBAL coordinates
        dx = goal_x - agent_x  # Positive = goal is east of agent
        dz = goal_z - agent_z  # Positive = goal is south of agent
        
        # Agent is always at bottom-middle in egocentric view
        agent_ego_row = matrix_size - 1  # Bottom row (index 10 for 11x11)
        agent_ego_col = matrix_size // 2  # Middle column (index 5 for 11x11)
        
        # Transform global offset to egocentric coordinates based on agent direction
        if agent_dir == 3:  # Agent facing North in global frame
            ego_row = agent_ego_row + dz
            ego_col = agent_ego_col + dx
            
        elif agent_dir == 0:  # Agent facing East in global frame
            ego_row = agent_ego_row - dx
            ego_col = agent_ego_col + dz
            
        elif agent_dir == 1:  # Agent facing South in global frame
            ego_row = agent_ego_row - dz
            ego_col = agent_ego_col - dx
            
        elif agent_dir == 2:  # Agent facing West in global frame
            ego_row = agent_ego_row + dx
            ego_col = agent_ego_col - dz
        
        # Check if goal position is within the egocentric matrix bounds
        if 0 <= ego_row < matrix_size and 0 <= ego_col < matrix_size:
            ego_matrix[int(ego_row), int(ego_col)] = 1.0
        else:
            print("Error: Goal out of egocentric view bounds")
            print(f"  Agent pos: ({agent_x}, {agent_z}), dir: {agent_dir}")
            print(f"  Goal pos: ({goal_x}, {goal_z})")
            print(f"  Offsets: dx={dx}, dz={dz}")
            print(f"  Ego coords: row={ego_row}, col={ego_col}")
        
        return ego_matrix