import numpy as np
import torch

class WVFAgent:
    """
    Agent that learns World Value Functions using an MLP.
    FIXED VERSION with stability improvements.
    """
    
    def __init__(self, env, wvf_model, optimizer, learning_rate=0.0001, gamma=0.99, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.device = device
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # Grid setup
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        
        # WVF MLP model
        self.wvf_model = wvf_model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        
        # Track the reward map (from vision model)
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        
        # Track visited positions for path integration
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Cache for current Q-values (updated each step)
        self.current_q_values = None
        
        # NEW: Track statistics for debugging
        self.update_count = 0
        self.loss_history = []
        
    def update_q_values(self):
        """
        Run forward pass through MLP to get current Q-values for all states.
        FIXED: Added input normalization for stability.
        """
        # Normalize reward map to [0, 1] range
        reward_map = self.true_reward_map.copy()
        max_val = reward_map.max()
        if max_val > 0:
            reward_map = reward_map / max_val
        
        # Convert to tensor
        reward_map_tensor = torch.tensor(reward_map, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get agent position
        agent_pos_idx = torch.tensor([self.get_state_index()], dtype=torch.long).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            q_values = self.wvf_model(reward_map_tensor, agent_pos_idx)
            # Clamp Q-values to prevent extreme values
            q_values = torch.clamp(q_values, -10, 10)
            self.current_q_values = q_values.squeeze(0).cpu().numpy()
    
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """
        Sample action using WVF (now from MLP) with epsilon-greedy exploration.
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Update Q-values from MLP
        self.update_q_values()
        
        # If Q-values are all zeros or invalid, explore randomly
        if self.current_q_values is None or np.allclose(self.current_q_values, 0):
            return np.random.randint(self.action_size)
        
        x, z = self._get_agent_pos_from_env()
        current_dir = self._get_agent_dir_from_env()
        
        # Check neighbors in (x, z) format
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
                # Get max Q-value across all actions for this neighbor position
                max_q_value = np.max(self.current_q_values[next_z, next_x, :])
                action_to_take = self._get_action_toward_direction(current_dir, target_dir)
                
                valid_actions.append(action_to_take)
                valid_values.append(max_q_value)
        
        if not valid_actions:
            return np.random.randint(self.action_size)
        
        # Choose best action (break ties randomly)
        best_value = max(valid_values)
        best_indices = [i for i, v in enumerate(valid_values) if abs(v - best_value) < 1e-6]
        chosen_index = np.random.choice(best_indices)
        
        return valid_actions[chosen_index]
    
    def update(self, experience):
        """
        Update WVF MLP using TD learning.
        FIXED: Added gradient clipping, input normalization, and loss bounds.
        
        Experience format: [state, action, next_state, reward, done]
        """
        s = experience[0]      # current state index
        a = experience[1]      # action taken
        s_next = experience[2] # next state index
        r = experience[3]      # reward
        done = experience[4]   # terminal flag
        
        # Skip update if no movement (prevents overfitting to no-op)
        if s == s_next and not done:
            return 0.0
        
        # Normalize reward map to [0, 1] range
        reward_map = self.true_reward_map.copy()
        max_val = reward_map.max()
        if max_val > 0:
            reward_map = reward_map / max_val
        
        # Convert to tensors
        reward_map_tensor = torch.tensor(reward_map, dtype=torch.float32).unsqueeze(0).to(self.device)
        state_idx_tensor = torch.tensor([s], dtype=torch.long).to(self.device)
        next_state_idx_tensor = torch.tensor([s_next], dtype=torch.long).to(self.device)
        
        # Get current Q-values
        self.wvf_model.train()
        current_q_values = self.wvf_model(reward_map_tensor, state_idx_tensor)
        
        # Clamp Q-values to prevent explosion
        current_q_values = torch.clamp(current_q_values, -10, 10)
        
        # Extract Q(s, a) for the action taken
        s_row = s // self.grid_size
        s_col = s % self.grid_size
        current_q = current_q_values[0, s_row, s_col, a]
        
        # Compute TD target
        with torch.no_grad():
            if done:
                td_target = torch.tensor(r, dtype=torch.float32).to(self.device)
            else:
                # Get Q-values for next state
                next_q_values = self.wvf_model(reward_map_tensor, next_state_idx_tensor)
                next_q_values = torch.clamp(next_q_values, -10, 10)
                
                s_next_row = s_next // self.grid_size
                s_next_col = s_next % self.grid_size
                max_next_q = torch.max(next_q_values[0, s_next_row, s_next_col, :])
                td_target = r + self.gamma * max_next_q
            
            # Clamp TD target to reasonable range
            td_target = torch.clamp(td_target, -10, 10)
        
        # Compute loss
        loss = self.loss_fn(current_q, td_target)
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Invalid loss detected: {loss.item()}, skipping update")
            return 0.0
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # CRITICAL FIX: Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.wvf_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track statistics
        self.update_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        # Warn if loss is getting large
        if loss_val > 100:
            print(f"⚠️ Large loss detected at update {self.update_count}: {loss_val:.2f}")
        
        return loss_val
    
    def _is_valid_position(self, pos):
        """Check if position is valid (no collision, within bounds)"""
        x, z = pos[0], pos[1]
        
        # Store original agent position
        original_pos = self._get_agent_pos_from_env()
        
        # Check boundaries
        boundary = self.grid_size - 1
        if x < 0 or x > boundary or z < 0 or z > boundary:
            return False
        
        # Simple distance-based collision check
        new_pos = np.array([x, original_pos[1], z])
        agent_radius = 0.18
        
        for entity in self.env.entities:
            if hasattr(entity, 'pos') and hasattr(entity, 'radius'):
                dist = np.linalg.norm(new_pos - entity.pos)
                if dist < agent_radius + entity.radius:
                    return False
        
        return True
    
    def _get_action_toward_direction(self, current_dir, target_dir):
        """Get the action needed to face the target direction"""
        if current_dir == target_dir:
            return 2  # move forward
        
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            return 1  # turn right
        elif diff == 3:
            return 0  # turn left
        elif diff == 2:
            return np.random.choice([0, 1])  # turn around (random direction)
        
        return 2
    
    def _get_agent_pos_from_env(self):
        """Get agent position from environment"""
        x = int(round(self.env.agent.pos[0] / self.env.grid_size))
        z = int(round(self.env.agent.pos[2] / self.env.grid_size))
        return (x, z)
    
    def _get_agent_dir_from_env(self):
        """Get agent direction from environment (0=East, 1=South, 2=West, 3=North)"""
        angle = self.env.agent.dir
        degrees = (np.degrees(angle) % 360)
        if degrees < 45 or degrees >= 315:
            return 0  # East (+X)
        elif 45 <= degrees < 135:
            return 3  # North (-Z) at 90°
        elif 135 <= degrees < 225:
            return 2  # West (-X) at 180°
        else:
            return 1  # South (+Z) at 270°
    
    def get_state_index(self):
        """Convert current grid position to flat state index"""
        x, z = self._get_agent_pos_from_env()
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def reset(self):
        """Reset for new episode"""
        self.current_q_values = None
    
    def create_egocentric_observation(self, goal_pos_red=None, goal_pos_blue=None, matrix_size=13):
        """
        Create an egocentric observation matrix where:
        - Agent is always at the bottom-middle cell, facing upward.
        - Goal positions (red, blue) are given in the agent's egocentric coordinates.
        
        Args:
            goal_pos_red: Tuple (x_right, z_forward) or None
            goal_pos_blue: Tuple (x_right, z_forward) or None
            matrix_size: Size of the square matrix (default 13x13)
        
        Returns:
            ego_matrix: numpy array of shape (matrix_size, matrix_size)
        """
        ego_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        
        # Agent position (bottom-center)
        agent_row = matrix_size - 1
        agent_col = matrix_size // 2
        
        def place_goal(pos, value):
            if pos is None:
                return
            gx, gz = pos  # (right, forward)
            # Convert to matrix coordinates
            ego_row = agent_row - gz
            ego_col = agent_col - gx
            
            # Check bounds and place marker
            if 0 <= ego_row < matrix_size and 0 <= ego_col < matrix_size:
                ego_matrix[int(ego_row), int(ego_col)] = value
        
        # Place red and blue goals
        place_goal(goal_pos_red, 1.0)
        place_goal(goal_pos_blue, 1.0)
        
        return ego_matrix