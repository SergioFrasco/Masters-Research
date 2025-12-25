# ============================================================================
# With Feature Maps and Task Composition + Confidence Accumulation
# ============================================================================

import numpy as np

class SuccessorAgent:
    """A Successor agent with feature-based WVF composition"""
    
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

        # Feature map - stores confidence values (0.0 to 1.0)
        self.feature_map = {
            "red": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "blue": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "box": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "sphere": np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        }
        
        # Confidence parameters
        self.confidence_boost = 0.4  # How much to increase confidence per detection
        self.step_decay_factor = 0.98  # Within-episode decay to filter detector noise
        self.confidence_threshold = 0.3  # Threshold for considering a location valid
        
        # Composed reward map (task-specific)
        self.composed_reward_map = np.zeros((self.grid_size, self.grid_size))
        
        # WVF - result of SR @ composed_reward_map (for action selection)
        self.wvf = np.zeros((self.grid_size, self.grid_size))

        # Keep old reward_maps for backward compatibility if needed
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)
    
    def update_feature_map(self, detected_objects, positions):
        """Update feature map with within-episode decay for noisy detector"""
        
        # FIRST: Decay existing confidence each step to filter out noise
        for feature in self.feature_map:
            self.feature_map[feature] *= self.step_decay_factor
        
        # SECOND: Get agent info
        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()
        
        # THIRD: Boost confidence for newly detected objects
        for obj_name in detected_objects:
            if obj_name in positions and positions[obj_name] is not None:
                dx, dz = positions[obj_name]
                dx, dz = int(round(dx)), int(round(dz))
                
                # Convert ego-centric to global coordinates
                global_x, global_z = self._ego_to_global(dx, dz, agent_x, agent_z, agent_dir)
                
                # Bounds check
                if not (0 <= global_x < self.grid_size and 0 <= global_z < self.grid_size):
                    continue
                
                # Accumulate confidence (cap at 1.0)
                if "red" in obj_name:
                    self.feature_map["red"][global_z, global_x] = min(1.0, 
                        self.feature_map["red"][global_z, global_x] + self.confidence_boost)
                if "blue" in obj_name:
                    self.feature_map["blue"][global_z, global_x] = min(1.0,
                        self.feature_map["blue"][global_z, global_x] + self.confidence_boost)
                if "box" in obj_name:
                    self.feature_map["box"][global_z, global_x] = min(1.0,
                        self.feature_map["box"][global_z, global_x] + self.confidence_boost)
                if "sphere" in obj_name:
                    self.feature_map["sphere"][global_z, global_x] = min(1.0,
                        self.feature_map["sphere"][global_z, global_x] + self.confidence_boost)
    
    def _ego_to_global(self, dx_ego, dz_ego, agent_x, agent_z, agent_dir):
        """Convert egocentric coordinates to global grid coordinates"""
        if agent_dir == 3:  # North
            dx_world, dz_world = dx_ego, dz_ego
        elif agent_dir == 0:  # East
            dx_world, dz_world = -dz_ego, dx_ego
        elif agent_dir == 1:  # South
            dx_world, dz_world = -dx_ego, -dz_ego
        elif agent_dir == 2:  # West
            dx_world, dz_world = dz_ego, -dx_ego
        
        global_x = agent_x + dx_world
        global_z = agent_z + dz_world
        return global_x, global_z
    
    def compose_reward_map(self, task):
        """Compose feature maps based on task requirements with confidence thresholding"""
        features = task["features"]
        
        if len(features) == 1:
            # Simple task - threshold single feature map
            feature_map = self.feature_map[features[0]]
            self.composed_reward_map = (feature_map > self.confidence_threshold).astype(np.float32)
        else:
            # Compositional task - threshold each map, then take minimum (AND logic)
            thresholded_maps = []
            for f in features:
                thresholded = (self.feature_map[f] > self.confidence_threshold).astype(np.float32)
                thresholded_maps.append(thresholded)
            self.composed_reward_map = np.minimum.reduce(thresholded_maps)
    
    def compute_wvf(self):
        """Compute WVF by applying SR to composed reward map"""
        MOVE_FORWARD = 2
        M_forward = self.M[MOVE_FORWARD, :, :]
        R_flat = self.composed_reward_map.flatten()
        V_flat = M_forward @ R_flat
        self.wvf = V_flat.reshape(self.grid_size, self.grid_size)
    
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """Sample action using WVF (computed from feature composition)"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        x, z = self._get_agent_pos_from_env()
        current_dir = self._get_agent_dir_from_env()
        
        # Neighbors in (x, z) format
        neighbors = [
            ((x + 1, z), 0),  # Right
            ((x, z + 1), 1),  # Down
            ((x - 1, z), 2),  # Left
            ((x, z - 1), 3)   # Up
        ]
        
        valid_actions = []
        valid_values = []
        
        for neighbor_pos, target_dir in neighbors:
            if self._is_valid_position(neighbor_pos):
                next_x, next_z = neighbor_pos
                # Use composed WVF for action selection
                value = self.wvf[next_z, next_x]
                action_to_take = self._get_action_toward_direction(current_dir, target_dir)
                
                valid_actions.append(action_to_take)
                valid_values.append(value)
        
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
        x, z = pos[0], pos[1]
        
        # Store original agent position
        original_pos = self._get_agent_pos_from_env()
        
        # Calculate new position
        new_pos = np.array([x, original_pos[1], z])
        
        # Check boundaries first
        boundary = self.grid_size - 1
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
        """Reset for new episode - completely clear feature maps for new environment"""
        self.prev_state = None
        self.prev_action = None
        
        # Zero out feature maps completely - new episode = new environment with new object positions
        for feature in self.feature_map:
            self.feature_map[feature].fill(0)
        
        self.composed_reward_map.fill(0)
        self.wvf.fill(0)