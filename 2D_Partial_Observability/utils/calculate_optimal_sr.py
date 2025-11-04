import numpy as np
import matplotlib.pyplot as plt

def get_position_based_sr(grid_size=10, gamma=0.99):
    """
    Calculate SR based purely on position for a plain grid.
    Returns a 100x100 matrix for 10x10 grid.
    """
    n_states = grid_size * grid_size
    T = np.zeros((n_states, n_states)) # Transition matrix
    
    for state in range(n_states):
        row = state // grid_size
        col = state % grid_size
        
        # Find all accessible neighbors
        neighbors = []
        if row > 0: neighbors.append(state - grid_size)  # North
        if row < grid_size-1: neighbors.append(state + grid_size)  # South
        if col > 0: neighbors.append(state - 1)  # West
        if col < grid_size-1: neighbors.append(state + 1)  # East
        
        # Equal probability to each neighbor
        for neighbor in neighbors:
            T[state, neighbor] = 1.0 / len(neighbors)
    
    # Calculate SR
    I = np.eye(n_states)
    M = np.linalg.inv(I - gamma * T)

    # M = I + γT + γ²T² + γ³T³ + ...
    # Where:

    # I: You always occupy your starting state at time 0
    # γT: Expected occupancy at time 1 (discounted by γ)
    # γ²T²: Expected occupancy at time 2 (discounted by γ²)
    
    return M  # 100x100 matrix

if __name__ == "__main__":
    sr_matrix = get_position_based_sr()

    np.save('optimal_sr_10x10_gamma099.npy', sr_matrix)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(sr_matrix, cmap='hot')
    plt.title(f"OPTIMAL SR for Plain Grid")
    plt.colorbar(im, label="SR Value")
    plt.tight_layout()
    plt.savefig('optimal_M.png')
    plt.close()

    # Visualize SR from one starting position
    start_state = 44 
    sr_from_start = sr_matrix[start_state, :]

    # Reshape to grid
    sr_grid = sr_from_start.reshape(10, 10)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(sr_grid, cmap='hot', origin='upper')
    plt.title(f"SR from position (4,4) - γ=0.99")
    plt.colorbar(im, label="Expected Occupancy")
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.plot(4, 4, 'b*', markersize=20, label='Start')
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_sr_from_44.png")