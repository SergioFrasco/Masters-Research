import numpy as np
from minigrid.core.world_object import Goal, Wall
from collections import deque

class ImprovedVisionOnlyAgent:
    """
    Improved vision-only agent with better value propagation and training
    """
    
    def __init__(self, env):
        self.env = env
        self.grid_size = env.size
        self.state_size = env.size * env.size
        
        # Action space
        self.action_size = env.action_space.n
        
        # Value map - this is what the autoencoder will predict
        self.predicted_value_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Ground truth value map based on TD learning
        self.true_value_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Experience buffer for better training
        self.experience_buffer = deque(maxlen=1000)
        
        # Visit counts for exploration bonus
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Learning parameters
        self.gamma = 0.95  # Higher discount for better long-term planning
        self.alpha = 0.1   # Learning rate for TD updates
        
        # Track training data quality
        self.training_examples = []
        
    def get_state_index(self, obs):
        """Get current state index based on agent position"""
        agent_pos = self.env.agent_pos
        return agent_pos[1] * self.grid_size + agent_pos[0]
    
    def get_position_from_state(self, state_idx):
        """Convert state index back to (x, y) position"""
        y = state_idx // self.grid_size
        x = state_idx % self.grid_size
        return x, y
    
    def sample_action_from_values(self, obs, epsilon=0.1):
        """
        Improved action selection using value map with exploration bonus
        """
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        
        # Get current position and direction
        current_pos = self.env.agent_pos
        x, y = current_pos
        current_dir = self.env.agent_dir
        
        action_values = np.zeros(self.action_size)
        
        # Evaluate each action
        for action in range(self.action_size):
            next_pos = self._simulate_action(x, y, current_dir, action)
            
            if next_pos is not None:
                next_x, next_y = next_pos
                
                # Double-check bounds before accessing value map
                if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                    # Base value from prediction
                    base_value = self.predicted_value_map[next_y, next_x]
                    
                    # Exploration bonus (encourage visiting less-visited states)
                    exploration_bonus = 0.1 / (1 + self.visit_counts[next_y, next_x])
                    
                    action_values[action] = base_value + exploration_bonus
                else:
                    # Out of bounds - strong penalty
                    action_values[action] = -1.0
            else:
                # Invalid action - strong penalty
                action_values[action] = -1.0
        
        return np.argmax(action_values)
    
    def _simulate_action(self, x, y, direction, action):
        """Simulate taking an action and return resulting position"""
        if action == 0:  # turn left
            new_dir = (direction - 1) % 4
            # After turning, the position stays the same
            return (x, y)
        elif action == 1:  # turn right
            new_dir = (direction + 1) % 4
            # After turning, the position stays the same
            return (x, y)
        elif action == 2:  # move forward
            next_pos = self._get_forward_pos(x, y, direction)
            next_x, next_y = next_pos
            # Check if the next position is valid
            if self._is_valid_pos(next_x, next_y):
                return next_pos
            else:
                # Can't move forward, stay in place
                return (x, y)
        else:
            # Other actions (pickup, drop, toggle, done) don't change position
            return (x, y)
    
    def _get_forward_pos(self, x, y, direction):
        """Get the position that would result from moving forward in given direction"""
        if direction == 0:  # facing right
            return x + 1, y
        elif direction == 1:  # facing down
            return x, y + 1
        elif direction == 2:  # facing left
            return x - 1, y
        elif direction == 3:  # facing up
            return x, y - 1
        return x, y
    
    def _is_valid_pos(self, x, y):
        """Check if position is valid (in bounds and not a wall)"""
        # First check bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        
        # Then check for walls
        try:
            cell = self.env.grid.get(x, y)
            return cell is None or not isinstance(cell, Wall)
        except:
            # If there's any error accessing the grid, assume invalid
            return False
    
    def update_value_map(self, prev_pos, action, reward, next_pos, done):
        """
        Improved TD learning update with better value propagation
        """
        px, py = prev_pos
        nx, ny = next_pos
        
        # Ensure positions are within bounds
        if not (0 <= px < self.grid_size and 0 <= py < self.grid_size):
            return
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return
        
        # Update visit counts
        self.visit_counts[py, px] += 1
        
        # Calculate TD target
        if done:
            td_target = reward
        else:
            # Use max value of valid neighbors for next state
            next_value = self._get_max_neighbor_value(nx, ny)
            td_target = reward + self.gamma * next_value
        
        # TD error
        current_value = self.true_value_map[py, px]
        td_error = td_target - current_value
        
        # Update value with TD learning
        self.true_value_map[py, px] = current_value + self.alpha * td_error
        
        # Store experience for training
        self.experience_buffer.append({
            'state': (px, py),
            'value': self.true_value_map[py, px],
            'reward': reward,
            'done': done
        })
        
        # Propagate values backwards using eligibility traces
        if abs(td_error) > 0.01:  # Only propagate significant updates
            self._propagate_value_update(px, py, td_error * 0.5)
    
    def _get_max_neighbor_value(self, x, y):
        """Get maximum value among valid neighboring positions"""
        max_value = 0
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        
        for nx, ny in neighbors:
            if self._is_valid_pos(nx, ny):
                # Use a combination of true value and prediction
                true_val = self.true_value_map[ny, nx]
                pred_val = self.predicted_value_map[ny, nx]
                
                # Weight based on visit count
                visit_weight = min(self.visit_counts[ny, nx] / 10.0, 1.0)
                combined_val = visit_weight * true_val + (1 - visit_weight) * pred_val
                
                max_value = max(max_value, combined_val)
        
        return max_value
    
    def _propagate_value_update(self, center_x, center_y, td_error):
        """Propagate TD error to nearby states with decay"""
        max_distance = 2
        
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = center_x + dx, center_y + dy
                
                # Check bounds
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                
                if not self._is_valid_pos(nx, ny):
                    continue
                
                # Distance-based decay
                distance = abs(dx) + abs(dy)
                decay_factor = (0.5 ** distance)
                
                # Propagate a fraction of the TD error
                self.true_value_map[ny, nx] += self.alpha * decay_factor * td_error
    
    def prepare_training_data(self):
        """Prepare training data for autoencoder with better target values"""
        if len(self.experience_buffer) < 10:
            return None, None
        
        # Create input grid
        grid = self.env.grid.encode()
        normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)
        
        # Normalize grid for autoencoder input
        object_layer = grid[..., 0]
        normalized_grid[object_layer == 2] = 0.0   # Wall
        normalized_grid[object_layer == 1] = 0.0   # Open space
        normalized_grid[object_layer == 8] = 1.0   # Goal
        
        # Create improved target map
        target_map = np.copy(self.true_value_map)
        
        # Apply smoothing to reduce noise
        target_map = self._smooth_value_map(target_map)
        
        # Ensure goal states have high values
        goal_positions = np.where(normalized_grid == 1.0)
        if len(goal_positions[0]) > 0:
            for gx, gy in zip(goal_positions[1], goal_positions[0]):
                target_map[gy, gx] = max(target_map[gy, gx], 1.0)
        
        input_grid = normalized_grid[np.newaxis, ..., np.newaxis]
        target_values = target_map[np.newaxis, ..., np.newaxis]
        
        return input_grid, target_values
    
    def _smooth_value_map(self, value_map):
        """Apply smoothing to reduce noise in value map"""
        smoothed = np.copy(value_map)
        
        # Simple 3x3 smoothing kernel
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if self._is_valid_pos(x, y):
                    # Average with neighbors
                    neighbors = [
                        value_map[y-1, x-1], value_map[y-1, x], value_map[y-1, x+1],
                        value_map[y, x-1],   value_map[y, x],   value_map[y, x+1],
                        value_map[y+1, x-1], value_map[y+1, x], value_map[y+1, x+1]
                    ]
                    smoothed[y, x] = np.mean(neighbors)
        
        return smoothed