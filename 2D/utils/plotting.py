import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
    
    for idx in range(num_maps):
        ax = axes[idx]
        ax.imshow(agent.wvf[idx], cmap='viridis', vmin=0, vmax=1)
        # ax.imshow(agent.reward_maps[idx], cmap='viridis')
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide any unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

