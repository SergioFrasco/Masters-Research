"""
Hierarchical Grid World Matrix Generator
Creates a 10x10 outer grid, where each cell contains a 10x10 inner grid
Rewards appear in the inner grids only where outer grid has rewards
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from datetime import datetime

# Configuration
OUTER_GRID_SIZE = 10  # 10x10 outer grid
INNER_GRID_SIZE = 10  # Each cell has a 10x10 inner grid
CELL_SIZE = 10        # Pixels per inner cell

# Define reward locations in the outer grid (from your image)
# These are (row, col) positions where green rewards appear
OUTER_REWARDS = [
    (2, 4),   # Top green reward
    (2, 8),   # Right green reward  
    (6, 8),   # Bottom green reward
]

# Define corresponding positions within each rewarded cell's inner grid
# These should match where the agent/reward appears within that cell
INNER_REWARD_POSITIONS = {
    (2, 4): (5, 5),  # Reward at center of this cell's inner grid
    (2, 8): (5, 5),  # Reward at center of this cell's inner grid
    (6, 8): (5, 5),  # Reward at center of this cell's inner grid
}


def create_hierarchical_world():
    """
    Create the full hierarchical world matrix.
    Returns a (100, 100, 3) RGB image.
    """
    # Total size in pixels
    total_size = OUTER_GRID_SIZE * INNER_GRID_SIZE
    world = np.zeros((total_size, total_size, 3), dtype=np.uint8)
    
    # Fill the world
    for outer_row in range(OUTER_GRID_SIZE):
        for outer_col in range(OUTER_GRID_SIZE):
            
            # Calculate pixel boundaries for this outer cell
            row_start = outer_row * INNER_GRID_SIZE
            row_end = row_start + INNER_GRID_SIZE
            col_start = outer_col * INNER_GRID_SIZE
            col_end = col_start + INNER_GRID_SIZE
            
            # Check if this outer cell has a reward
            if (outer_row, outer_col) in OUTER_REWARDS:
                # This cell should have a green reward in its inner grid
                inner_pos = INNER_REWARD_POSITIONS[(outer_row, outer_col)]
                inner_row, inner_col = inner_pos
                
                # Set the entire inner grid to dark (background)
                world[row_start:row_end, col_start:col_end] = [20, 20, 20]  # Dark gray
                
                # Place green reward at the specific position
                reward_row = row_start + inner_row
                reward_col = col_start + inner_col
                world[reward_row, reward_col] = [0, 255, 0]  # Green
            else:
                # This cell has no reward - make it all black
                world[row_start:row_end, col_start:col_end] = [0, 0, 0]  # Black
    
    return world


def create_hierarchical_world_with_grid_lines():
    """
    Create the hierarchical world with visible grid lines.
    """
    # Total size in pixels
    total_size = OUTER_GRID_SIZE * INNER_GRID_SIZE
    world = np.zeros((total_size, total_size, 3), dtype=np.uint8)
    
    # Fill the world
    for outer_row in range(OUTER_GRID_SIZE):
        for outer_col in range(OUTER_GRID_SIZE):
            
            # Calculate pixel boundaries for this outer cell
            row_start = outer_row * INNER_GRID_SIZE
            row_end = row_start + INNER_GRID_SIZE
            col_start = outer_col * INNER_GRID_SIZE
            col_end = col_start + INNER_GRID_SIZE
            
            # Check if this outer cell has a reward
            if (outer_row, outer_col) in OUTER_REWARDS:
                # This cell should have a green reward in its inner grid
                inner_pos = INNER_REWARD_POSITIONS[(outer_row, outer_col)]
                inner_row, inner_col = inner_pos
                
                # Set the entire inner grid to dark gray (background)
                world[row_start:row_end, col_start:col_end] = [30, 30, 30]
                
                # Add inner grid lines (lighter gray)
                for i in range(INNER_GRID_SIZE + 1):
                    # Horizontal lines
                    if row_start + i < total_size:
                        world[row_start + i, col_start:col_end] = [60, 60, 60]
                    # Vertical lines
                    if col_start + i < total_size:
                        world[row_start:row_end, col_start + i] = [60, 60, 60]
                
                # Place green reward at the specific position
                reward_row = row_start + inner_row
                reward_col = col_start + inner_col
                world[reward_row, reward_col] = [0, 255, 0]  # Green
            else:
                # This cell has no reward - make it all black
                world[row_start:row_end, col_start:col_end] = [0, 0, 0]
    
    # Add outer grid lines (white/bright)
    for i in range(0, total_size + 1, INNER_GRID_SIZE):
        if i < total_size:
            world[i, :] = [255, 255, 255]  # Horizontal outer grid lines
            world[:, i] = [255, 255, 255]  # Vertical outer grid lines
    
    return world


def visualize_and_save(world, filename="hierarchical_world.png"):
    """
    Visualize and save the hierarchical world.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(world, interpolation='nearest')
    # ax.set_title('', fontsize=14)
    ax.axis('off')
    
    # Save
    save_dir = Path("hierarchical_worlds")
    save_dir.mkdir(exist_ok=True)
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def create_individual_cell_visualizations():
    """
    Create individual visualizations for each outer cell showing its inner grid.
    """
    save_dir = Path("hierarchical_worlds") / "individual_cells"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for outer_row in range(OUTER_GRID_SIZE):
        for outer_col in range(OUTER_GRID_SIZE):
            
            # Create inner grid for this cell
            inner_grid = np.zeros((INNER_GRID_SIZE, INNER_GRID_SIZE, 3), dtype=np.uint8)
            
            if (outer_row, outer_col) in OUTER_REWARDS:
                # Has reward
                inner_grid[:, :] = [30, 30, 30]  # Dark gray background
                
                # Place reward
                inner_pos = INNER_REWARD_POSITIONS[(outer_row, outer_col)]
                inner_row, inner_col = inner_pos
                inner_grid[inner_row, inner_col] = [0, 255, 0]  # Green
                
                title = f"Outer Cell ({outer_row},{outer_col}) - WITH REWARD"
            else:
                # No reward - all black
                inner_grid[:, :] = [0, 0, 0]
                title = f"Outer Cell ({outer_row},{outer_col}) - Empty"
            
            # Save
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(inner_grid, interpolation='nearest')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
            filename = f"cell_{outer_row:02d}_{outer_col:02d}.png"
            plt.savefig(save_dir / filename, dpi=100, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Saved 100 individual cell visualizations to: {save_dir}")


def print_info():
    """Print information about the hierarchical structure."""
    print("\n" + "="*70)
    print("HIERARCHICAL GRID WORLD STRUCTURE")
    print("="*70)
    print(f"Outer Grid: {OUTER_GRID_SIZE}×{OUTER_GRID_SIZE} = {OUTER_GRID_SIZE**2} cells")
    print(f"Inner Grid (per cell): {INNER_GRID_SIZE}×{INNER_GRID_SIZE} = {INNER_GRID_SIZE**2} positions")
    print(f"Total Size: {OUTER_GRID_SIZE * INNER_GRID_SIZE}×{OUTER_GRID_SIZE * INNER_GRID_SIZE} pixels")
    print(f"\nReward Locations (Outer Grid):")
    for i, (row, col) in enumerate(OUTER_REWARDS, 1):
        inner_pos = INNER_REWARD_POSITIONS[(row, col)]
        print(f"  {i}. Outer cell ({row},{col}) → Inner position {inner_pos}")
    print("="*70 + "\n")


def main():
    print_info()
    
    print("🎨 Generating hierarchical world visualizations...\n")
    
    # Create basic version
    world_basic = create_hierarchical_world()
    visualize_and_save(world_basic, "hierarchical_world_basic.png")
    
    # Create version with grid lines
    world_grid = create_hierarchical_world_with_grid_lines()
    visualize_and_save(world_grid, "hierarchical_world_with_gridlines.png")
    
    # Create individual cell visualizations
    print("\n🎨 Generating individual cell visualizations...\n")
    create_individual_cell_visualizations()
    
    print("\n✨ All visualizations complete!")
    print("📁 Check the 'hierarchical_worlds' folder\n")
    
    return world_basic, world_grid


if __name__ == "__main__":
    world_basic, world_grid = main()