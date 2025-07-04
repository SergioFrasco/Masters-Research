import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


import os
from datetime import datetime

def get_run_directory():
    """
    Returns the current run directory, initializing it the first time it's called.
    Format: results/current/{day_month_year}/{run_number}
    """
    if not hasattr(get_run_directory, "_run_path"):
        today = datetime.today().strftime("%m_%d_%Y")
        base_dir = os.path.join("results", "current", today)
        os.makedirs(base_dir, exist_ok=True)

        # Get the next available run number
        existing_runs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
        ]
        next_run_number = str(max(map(int, existing_runs), default=0) + 1)

        # Set and create the run directory
        run_path = os.path.join(base_dir, next_run_number)
        os.makedirs(run_path, exist_ok=True)

        # Cache it
        get_run_directory._run_path = run_path

    return get_run_directory._run_path

def generate_save_path(name: str) -> str:
    """
    Returns a full path for a file to be saved under the current run directory.
    
    Parameters:
        name (str): Filename or relative path under the run folder
    
    Returns:
        str: Full path
    """
    run_dir = get_run_directory()
    full_path = os.path.join(run_dir, name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path




# Function to overlay values on the grid
def overlay_values_on_grid(grid, ax):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color='red', fontsize=8)


def visualize_sr():
    # Load the SR matrix
    sr_matrix = np.load("results/successor_representation.npy", allow_pickle=True).item()
    
    # Create a 2D grid to represent the environment
    # We'll use the first two dimensions of the image (height x width)
    sample_state = list(sr_matrix.keys())[0]
    grid_height = int(np.sqrt(len(sample_state[0]) // 3))  # Divide by 3 for RGB channels
    grid_width = grid_height
    
    # Initialize visualization grid
    visit_grid = np.zeros((grid_height, grid_width))
    
    # Aggregate visits across all directions and missions for each spatial position
    for state_key, visit_value in sr_matrix.items():
        image_array = np.array(state_key[0]).reshape(grid_height, grid_width, 3)
        # Sum the visitation values for each spatial position
        for i in range(grid_height):
            for j in range(grid_width):
                visit_grid[i, j] += visit_value
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_grid, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Successor Representation Heatmap')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Save the visualization
    plt.savefig('results/sr_visualization.png')
    plt.close()
    
    print("Visualization saved as sr_visualization.png!")

    # Optional: Create additional visualizations for different directions
    unique_directions = set(state_key[1] for state_key in sr_matrix.keys())
    
    for direction in unique_directions:
        visit_grid = np.zeros((grid_height, grid_width))
        
        # Filter states for this direction
        for state_key, visit_value in sr_matrix.items():
            if state_key[1] == direction:
                image_array = np.array(state_key[0]).reshape(grid_height, grid_width, 3)
                for i in range(grid_height):
                    for j in range(grid_width):
                        visit_grid[i, j] += visit_value
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(visit_grid, cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title(f'SR Heatmap - Direction {direction}')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.savefig(f'results/sr_visualization_direction_{direction}.png')
        plt.close()

def save_all_reward_maps(agent, maps_per_row=10, save_path="results/reward_maps.png"):
    num_maps = agent.state_size
    grid_size = agent.grid_size
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    for idx in range(num_maps):
        ax = axes[idx]
        ax.imshow(agent.reward_maps[idx], cmap='viridis', vmin=0, vmax=1)
        # ax.imshow(agent.reward_maps[idx], cmap='viridis')
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide any unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_all_wvf(agent, maps_per_row=10, save_path="results/wvf.png"):
    num_maps = agent.state_size
    grid_size = agent.grid_size
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    im = None  # Store the last imshow object for the colorbar anchor
    for idx in range(num_maps):
        ax = axes[idx]
        im = ax.imshow(agent.wvf[idx], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide any unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=axes[:num_maps], shrink=0.6, label="WVF Value")

    fig.savefig(save_path)
    plt.close()


def save_max_wvf_maps(max_wvfs, episode):
     # Compute a suitable grid size (square-ish)
    cols = int(np.ceil(np.sqrt(len(max_wvfs))))
            
    rows = int(np.ceil(len(max_wvfs) / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Flatten in case axs is a 2D array when rows, cols > 1
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

    for i in range(len(axs)):
        ax = axs[i]
        if i < len(max_wvfs):
            im = ax.imshow(max_wvfs[i], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Map {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/max_wvfs_{episode}")

def save_env_map_pred(agent, normalized_grid, predicted_reward_map_2d, episode, save_path):
    # Create visualization of current state
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    save_path = save_path
    # Original grid (environment)
    ax1.imshow(normalized_grid, cmap='gray')
    ax1.set_title("Environment Grid")
    
    # Agent's true reward map
    ax2.imshow(agent.true_reward_map, cmap='viridis')
    ax2.set_title("Agent's Understanding of Reward")
    
    # AE prediction
    ax3.imshow(predicted_reward_map_2d, cmap='viridis')
    ax3.set_title("AE Prediction")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# This broke when moved, had to comment out the line containing the word Goal
def visualize_agent_trajectory(env, wvf_grid, n_steps=100):
    """Visualize an agent following the World Value Function"""
    obs, _ = env.reset()
    trajectory = [env.agent_pos]
    
    for _ in range(n_steps):
        # Get current position
        x, y = env.agent_pos
        
        # Get neighboring positions
        neighbors = [
            (x-1, y), (x+1, y),  # Left, Right
            (x, y-1), (x, y+1)   # Up, Down
        ]
        
        # Filter valid positions and get their values
        valid_neighbors = []
        neighbor_values = []
        
        for nx, ny in neighbors:
            if 0 <= nx < env.size and 0 <= ny < env.size:
                valid_neighbors.append((nx, ny))
                neighbor_values.append(wvf_grid[ny, nx])
        
        # Choose direction with highest value
        if valid_neighbors:
            best_idx = np.argmax(neighbor_values)
            next_pos = valid_neighbors[best_idx]
            
            # Move agent (simplified)
            env.agent_pos = next_pos
            trajectory.append(next_pos)
        
        # Check if goal reached
        # if isinstance(env.grid.get(*env.agent_pos), Goal):
            break
    
    # Visualize trajectory
    plt.figure(figsize=(8, 8))
    plt.imshow(wvf_grid, cmap='hot')
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'w-', linewidth=2, label='Agent Path')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')
    
    plt.colorbar(label='Value')
    plt.legend()
    plt.title('Agent Trajectory on World Value Function')
    plt.savefig('results/agent_trajectory.png')
    plt.close()

