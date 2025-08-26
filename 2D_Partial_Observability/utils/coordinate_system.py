import numpy as np

class CoordinateSystem:
    """
    Standardized coordinate system for the entire project.
    
    Conventions:
    - Positions are (x, y) tuples where x=column, y=row
    - Array indexing is [y, x] where first index=row, second=column
    - Agent direction: 0=right, 1=down, 2=left, 3=up
    - Egocentric view: agent at center-bottom, facing up in local coordinates
    """
    
    @staticmethod
    def position_to_state_index(pos, grid_size):
        """Convert (x, y) position to flat state index"""
        x, y = pos
        return y * grid_size + x
    
    @staticmethod
    def state_index_to_position(state_idx, grid_size):
        """Convert flat state index to (x, y) position"""
        y = state_idx // grid_size
        x = state_idx % grid_size
        return (x, y)
    
    @staticmethod
    def egocentric_to_global(ego_x, ego_y, agent_pos, agent_dir, view_size):
        agent_x, agent_y = agent_pos
        
        # In egocentric view, agent is at center-bottom facing up
        center_x = view_size // 2
        agent_ego_y = view_size - 1
        
        # Calculate relative position in egocentric frame (agent facing up)
        rel_x = ego_x - center_x
        rel_y = agent_ego_y - ego_y  # Positive y goes "forward" (up in ego frame)
        
        # Correct rotation matrices for each direction
        if agent_dir == 0:      # facing right
            global_offset_x, global_offset_y = rel_y, rel_x  # FIXED: was rel_y, -rel_x
        elif agent_dir == 1:    # facing down
            global_offset_x, global_offset_y = -rel_x, rel_y  # FIXED: was rel_x, rel_y
        elif agent_dir == 2:    # facing left
            global_offset_x, global_offset_y = -rel_y, -rel_x  # FIXED: was -rel_y, rel_x
        else:                   # facing up (agent_dir == 3)
            global_offset_x, global_offset_y = rel_x, -rel_y  # FIXED: was -rel_x, -rel_y
        
        global_x = agent_x + global_offset_x
        global_y = agent_y + global_offset_y
        
        return (global_x, global_y)
    
    @staticmethod
    def global_to_egocentric(global_x, global_y, agent_pos, agent_dir, view_size):
        """
        Convert global coordinates to egocentric coordinates.
        
        Returns:
            (ego_x, ego_y): Egocentric coordinates, or None if outside view
        """
        agent_x, agent_y = agent_pos
        
        # Calculate global offset
        offset_x = global_x - agent_x
        offset_y = global_y - agent_y
        
        if agent_dir == 0:      # facing right
            rel_x, rel_y = offset_y, offset_x  # FIXED: was -offset_y, offset_x
        elif agent_dir == 1:    # facing down
            rel_x, rel_y = -offset_x, offset_y  # FIXED: was offset_x, offset_y
        elif agent_dir == 2:    # facing left
            rel_x, rel_y = -offset_y, -offset_x  # FIXED: was offset_y, -offset_x
        else:                   # facing up (agent_dir == 3)
            rel_x, rel_y = offset_x, -offset_y  # FIXED: was -offset_x, -offset_y
        
        # Convert to egocentric coordinates
        center_x = view_size // 2
        agent_ego_y = view_size - 1
        
        ego_x = center_x + rel_x
        ego_y = agent_ego_y - rel_y
        
        # Check if within view bounds
        if 0 <= ego_x < view_size and 0 <= ego_y < view_size:
            return (int(ego_x), int(ego_y))
        else:
            return None
    
    @staticmethod
    def is_valid_position(pos, grid_size):
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < grid_size and 0 <= y < grid_size
    
    @staticmethod
    def array_get(array_2d, pos):
        """Get array value using (x, y) position"""
        x, y = pos
        return array_2d[y, x]
    
    @staticmethod
    def array_set(array_2d, pos, value):
        """Set array value using (x, y) position"""
        x, y = pos
        array_2d[y, x] = value
    
    @staticmethod
    def test_coordinate_system():
        """Test function to verify coordinate transformations"""
        print("Testing coordinate system...")
        
        # Test 1: State index conversion
        grid_size = 10
        pos = (3, 4)
        state_idx = CoordinateSystem.position_to_state_index(pos, grid_size)
        recovered_pos = CoordinateSystem.state_index_to_position(state_idx, grid_size)
        assert pos == recovered_pos, f"State index test failed: {pos} != {recovered_pos}"
        
        # Test 2: Egocentric transform roundtrip
        agent_pos = (5, 5)
        agent_dir = 0  # facing right
        view_size = 7
        global_pos = (6, 4)  # One right, one up from agent
        
        ego_coords = CoordinateSystem.global_to_egocentric(
            global_pos[0], global_pos[1], agent_pos, agent_dir, view_size
        )
        if ego_coords is not None:
            recovered_global = CoordinateSystem.egocentric_to_global(
                ego_coords[0], ego_coords[1], agent_pos, agent_dir, view_size
            )
            assert global_pos == recovered_global, f"Roundtrip test failed: {global_pos} != {recovered_global}"
        
        # Test 3: Array access
        test_array = np.zeros((10, 10))
        pos = (7, 3)
        CoordinateSystem.array_set(test_array, pos, 42)
        value = CoordinateSystem.array_get(test_array, pos)
        assert value == 42, f"Array access test failed: expected 42, got {value}"
        
        print("All coordinate system tests passed!")

# Usage example showing the correct patterns
def example_usage():
    """Example of how to use the standardized coordinate system"""
    
    # Initialize
    grid_size = 10
    agent_position = (5, 5)  # (x, y) position tuple
    agent_direction = 0      # facing right
    view_size = 7
    
    # Create arrays (always indexed as [y, x])
    reward_map = np.zeros((grid_size, grid_size))
    visited_positions = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Set values using position tuples
    CoordinateSystem.array_set(reward_map, agent_position, 1.0)
    CoordinateSystem.array_set(visited_positions, agent_position, True)
    
    # Convert between coordinate systems
    state_idx = CoordinateSystem.position_to_state_index(agent_position, grid_size)
    
    # Process egocentric view
    for ego_y in range(view_size):
        for ego_x in range(view_size):
            global_pos = CoordinateSystem.egocentric_to_global(
                ego_x, ego_y, agent_position, agent_direction, view_size
            )
            
            if CoordinateSystem.is_valid_position(global_pos, grid_size):
                # Access array using standardized methods
                reward_value = CoordinateSystem.array_get(reward_map, global_pos)
                print(f"Global {global_pos} -> Ego ({ego_x}, {ego_y}) -> Reward: {reward_value}")

if __name__ == "__main__":
    CoordinateSystem.test_coordinate_system()
    example_usage()