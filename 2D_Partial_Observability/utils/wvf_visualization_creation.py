"""
Value Function Generator with Dissipating Values
Creates value functions for each cell containing a reward,
with values that decrease as distance from the reward increases
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
from PIL import Image


class ValueFunctionGenerator:
    """
    Generates value functions with dissipating values from reward locations.
    """
    
    def __init__(self, outer_size=10, inner_size=10, gamma=0.9):
        self.outer_size = outer_size
        self.inner_size = inner_size
        self.gamma = gamma  # Discount factor for value dissipation
        self.total_size = outer_size * inner_size
    
    def calculate_value_function(self, goal_position, grid_size=10, max_value=1.0):
        """
        Calculate value function for a single goal.
        Values dissipate based on Manhattan distance.
        
        Args:
            goal_position: (row, col) position of the goal
            grid_size: size of the grid
            max_value: maximum value at the goal position
        
        Returns:
            numpy array of values
        """
        values = np.zeros((grid_size, grid_size))
        goal_row, goal_col = goal_position
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate distance (Manhattan distance)
                distance = abs(row - goal_row) + abs(col - goal_col)
                
                # Value dissipates exponentially with distance
                values[row, col] = max_value * (self.gamma ** distance)
        
        return values
    
    def calculate_value_function_euclidean(self, goal_position, grid_size=10, max_value=1.0):
        """
        Calculate value function using Euclidean distance.
        """
        values = np.zeros((grid_size, grid_size))
        goal_row, goal_col = goal_position
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate Euclidean distance
                distance = np.sqrt((row - goal_row)**2 + (col - goal_col)**2)
                
                # Value dissipates with distance
                values[row, col] = max_value * (self.gamma ** distance)
        
        return values
    
    def create_hierarchical_value_map(self, reward_cells, use_euclidean=False):
        """
        Create hierarchical value map.
        
        Args:
            reward_cells: Dict mapping (outer_row, outer_col) to (inner_row, inner_col)
            use_euclidean: If True, use Euclidean distance, else Manhattan
        
        Returns:
            numpy array of shape (total_size, total_size) with values
        """
        value_map = np.zeros((self.total_size, self.total_size))
        
        for (outer_row, outer_col), (inner_row, inner_col) in reward_cells.items():
            # Calculate value function for this cell's inner grid
            if use_euclidean:
                cell_values = self.calculate_value_function_euclidean(
                    (inner_row, inner_col), 
                    self.inner_size
                )
            else:
                cell_values = self.calculate_value_function(
                    (inner_row, inner_col), 
                    self.inner_size
                )
            
            # Place in the overall map
            row_start = outer_row * self.inner_size
            row_end = row_start + self.inner_size
            col_start = outer_col * self.inner_size
            col_end = col_start + self.inner_size
            
            value_map[row_start:row_end, col_start:col_end] = cell_values
        
        return value_map
    
    def visualize_value_map(self, value_map, save_path, title="Value Function Map", 
                           add_grid_lines=True, colormap='hot'):
        """
        Visualize the value map with proper coloring.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Create custom colormap (black to green through yellow)
        if colormap == 'custom_green':
            colors = ['black', 'darkgreen', 'green', 'lime', 'yellow']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        elif colormap == 'hot':
            cmap = 'hot'
        elif colormap == 'viridis':
            cmap = 'viridis'
        else:
            cmap = colormap
        
        # Plot the value map
        im = ax.imshow(value_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        
        # Add grid lines
        if add_grid_lines:
            # Outer grid lines only (thicker white lines)
            for i in range(0, self.total_size + 1, self.inner_size):
                ax.axhline(i - 0.5, color='white', linewidth=2.5, alpha=0.9)
                ax.axvline(i - 0.5, color='white', linewidth=2.5, alpha=0.9)
        
        # No title, no colorbar, no axis
        ax.axis('off')
        
        # Save with no padding
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0, facecolor='black')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def create_rgb_visualization(self, value_map, add_grid_lines=True):
        """
        Create RGB visualization directly as numpy array.
        """
        # Normalize values to 0-255
        normalized = (value_map * 255).astype(np.uint8)
        
        # Create RGB image (hot colormap: black -> red -> yellow -> white)
        rgb_image = np.zeros((self.total_size, self.total_size, 3), dtype=np.uint8)
        
        # Apply hot-like colormap
        for i in range(self.total_size):
            for j in range(self.total_size):
                val = normalized[i, j]
                if val == 0:
                    rgb_image[i, j] = [0, 0, 0]  # Black
                elif val < 85:
                    # Black to red
                    rgb_image[i, j] = [val * 3, 0, 0]
                elif val < 170:
                    # Red to yellow
                    rgb_image[i, j] = [255, (val - 85) * 3, 0]
                else:
                    # Yellow to white
                    rgb_image[i, j] = [255, 255, (val - 170) * 3]
        
        if add_grid_lines:
            # Add grid lines
            for i in range(0, self.total_size, self.inner_size):
                rgb_image[i, :] = [100, 100, 100]
                rgb_image[:, i] = [100, 100, 100]
        
        return rgb_image


def main():
    """
    Main function to generate value functions from specified reward positions.
    """
    
    print("\n" + "="*80)
    print("VALUE FUNCTION GENERATOR")
    print("="*80)
    
    # Define reward positions from your image
    # Format: (outer_row, outer_col): (inner_row, inner_col)
    reward_cells = {
        (2, 4): (5, 5),   # Top-middle reward (center of that cell)
        (2, 8): (5, 5),   # Top-right reward (center of that cell)
        (6, 8): (5, 5),   # Middle-right reward (center of that cell)
    }
    
    print(f"\n📍 Reward Cells:")
    for (o_row, o_col), (i_row, i_col) in reward_cells.items():
        print(f"  Outer ({o_row}, {o_col}) → Inner position ({i_row}, {i_col})")
    
    # Create save directory
    save_dir = Path("value_functions")
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate Manhattan distance with gamma=0.7
    gamma = 0.7
    print(f"\n🎨 Generating value function with gamma={gamma} (Manhattan distance)...")
    
    generator = ValueFunctionGenerator(
        outer_size=10, 
        inner_size=10, 
        gamma=gamma
    )
    
    # Manhattan distance version
    value_map = generator.create_hierarchical_value_map(
        reward_cells, 
        use_euclidean=False
    )
    
    # Save with matplotlib
    save_path = save_dir / f"value_function_clean_{timestamp}.png"
    generator.visualize_value_map(
        value_map,
        save_path,
        title="",  # No title
        add_grid_lines=True,
        colormap='hot'
    )
    
    print("\n" + "="*80)
    print(f"✨ Value function generated!")
    print(f"📁 Saved to: {save_path}")
    print("="*80 + "\n")
    
    # Print statistics
    print("\n📊 Value Function Statistics:")
    print(f"  Gamma: {gamma}")
    print(f"  Distance metric: Manhattan")
    print(f"  Max value: {value_map.max():.4f}")
    print(f"  Min value (non-zero): {value_map[value_map > 0].min():.6f}")
    print(f"  Mean value (non-zero): {value_map[value_map > 0].mean():.4f}")
    print(f"  Non-zero cells: {np.count_nonzero(value_map)} / {value_map.size}")
    print()
    
    return value_map


if __name__ == "__main__":
    value_map = main()