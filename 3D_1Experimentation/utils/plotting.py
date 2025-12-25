import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import termios
import tty


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


def plot_sr_matrix(agent, episode):
    """
    Plot the SR matrix for the forward action from each state
    Shows expected future occupancy heatmap
    """
    # Get SR for forward action only (index 2)
    sr_forward = agent.M[agent.MOVE_FORWARD, :, :]  # Shape: (state_size, state_size) = (100, 100)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Successor Representation - Episode {episode}', fontsize=16)
    
    # Plot 1: Average SR (expected occupancy averaged over all start states)
    avg_sr = sr_forward.mean(axis=0).reshape(agent.grid_size, agent.grid_size)
    im1 = axes[0, 0].imshow(avg_sr, cmap='hot', interpolation='nearest')
    axes[0, 0].set_title('Average Expected Occupancy (10x10)')
    axes[0, 0].set_xlabel('X position')
    axes[0, 0].set_ylabel('Z position')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: SR from agent's starting position (1, 1)
    start_state = 1 * agent.grid_size + 1  # State at (1,1)
    sr_from_start = sr_forward[start_state, :].reshape(agent.grid_size, agent.grid_size)
    im2 = axes[0, 1].imshow(sr_from_start, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title(f'SR from Start Position (1,1) - Spatial (10x10)')
    axes[0, 1].set_xlabel('X position')
    axes[0, 1].set_ylabel('Z position')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: SR from center of room
    center_x = agent.grid_size // 2
    center_z = agent.grid_size // 2
    center_state = center_z * agent.grid_size + center_x
    sr_from_center = sr_forward[center_state, :].reshape(agent.grid_size, agent.grid_size)
    im3 = axes[0, 2].imshow(sr_from_center, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title(f'SR from Center ({center_x},{center_z}) - Spatial (10x10)')
    axes[0, 2].set_xlabel('X position')
    axes[0, 2].set_ylabel('Z position')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Full raw SR matrix (100x100)
    im4 = axes[1, 0].imshow(sr_forward, cmap='hot', interpolation='nearest', aspect='auto')
    axes[1, 0].set_title(f'Full SR Matrix (100x100)')
    axes[1, 0].set_xlabel('To State Index')
    axes[1, 0].set_ylabel('From State Index')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Zoomed view of SR matrix (first 25x25 states)
    im5 = axes[1, 1].imshow(sr_forward[:25, :25], cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('SR Matrix Zoom (First 25 states)')
    axes[1, 1].set_xlabel('To State Index')
    axes[1, 1].set_ylabel('From State Index')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: SR matrix statistics
    axes[1, 2].hist(sr_forward.flatten(), bins=50, edgecolor='black')
    axes[1, 2].set_title('Distribution of SR Values')
    axes[1, 2].set_xlabel('SR Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(sr_forward.mean(), color='r', linestyle='--', 
                       label=f'Mean: {sr_forward.mean():.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save the plot
    from utils import generate_save_path
    save_path = generate_save_path(f'sr_plots/episode_{episode:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SR plot to: {save_path}")
    
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

def save_all_wvf(agent, maps_per_row=10, save_path="results/wvf.png"):
    num_maps = agent.state_size
    grid_size = agent.grid_size
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    im = None  # Store the last imshow object for the colorbar anchor
    for idx in range(num_maps):
        ax = axes[idx]
        im = ax.imshow(agent.wvf[idx], cmap='viridis')
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


