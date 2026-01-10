
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

GRID_SIZE = 30          # 10x10 grid (100 states) like Stoewer paper
GAMMA = 0.9             # Discount factor
N_EIGENVECTORS = 30     # Show all 30 like Stoewer
SMOOTHING_SIGMA = 0.3   # Minimal smoothing

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def state_to_coords(state, grid_size):
    """Convert 1D state index to 2D (x, y) coordinates."""
    x = state % grid_size
    y = state // grid_size
    return x, y

def coords_to_state(x, y, grid_size):
    """Convert 2D (x, y) coordinates to 1D state index."""
    return y * grid_size + x

def get_neighbors(state, grid_size):
    """Get valid neighboring states (4-connected grid)."""
    x, y = state_to_coords(state, grid_size)
    neighbors = []
    
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            neighbors.append(coords_to_state(nx, ny, grid_size))
    
    return neighbors

# ============================================================================
# SUCCESSOR REPRESENTATION
# ============================================================================

def compute_transition_matrix(grid_size):
    """Compute transition matrix for uniform random walk."""
    n_states = grid_size * grid_size
    T = np.zeros((n_states, n_states))
    
    for state in range(n_states):
        neighbors = get_neighbors(state, grid_size)
        n_neighbors = len(neighbors)
        for neighbor in neighbors:
            T[state, neighbor] = 1.0 / n_neighbors
    
    return T

def compute_successor_representation(T, gamma):
    """Compute SR matrix: M = (I - gamma * T)^(-1)"""
    n_states = T.shape[0]
    I = np.eye(n_states)
    M = np.linalg.inv(I - gamma * T)
    return M

def extract_grid_cells(M, n_eigenvectors):
    """Extract eigenvectors sorted by eigenvalue magnitude."""
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvectors[:, :n_eigenvectors].real, eigenvalues[:n_eigenvectors].real

# ============================================================================
# VISUALIZATION - STOEWER STYLE
# ============================================================================

def plot_grid_cells_stoewer_style(eigenvectors, grid_size):
    """
    Plot 30 eigenvectors in a 5x6 grid layout exactly like Stoewer et al. Figure 4.
    """
    n_cells = eigenvectors.shape[1]
    
    # Create figure with 5 rows and 6 columns
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    fig.suptitle('Eigenvectors of SR Matrix (Stoewer et al. Style)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i in range(n_cells):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        
        # Get eigenvector and reshape to 2D
        eigvec = eigenvectors[:, i]
        rate_map = eigvec.reshape(grid_size, grid_size)
        
        # Threshold at zero (only positive values) and smooth
        rate_map = np.maximum(rate_map, 0)
        rate_map_smooth = gaussian_filter(rate_map, sigma=SMOOTHING_SIGMA)
        
        # Plot with jet colormap like Stoewer
        im = ax.imshow(rate_map_smooth, cmap='jet', interpolation='nearest')
        
        # Minimal styling - just show the pattern
        ax.set_xticks([0, grid_size-1])
        ax.set_yticks([0, grid_size-1])
        ax.set_xticklabels(['0', str(grid_size-1)], fontsize=8)
        ax.set_yticklabels(['0', str(grid_size-1)], fontsize=8)
        ax.tick_params(length=2)
    
    plt.tight_layout()
    return fig

def plot_autocorrelations_stoewer_style(eigenvectors, grid_size):
    """
    Plot autocorrelations of 30 eigenvectors in a 5x6 grid.
    """
    n_cells = eigenvectors.shape[1]
    
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    fig.suptitle('Spatial Autocorrelations', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i in range(n_cells):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        
        # Get eigenvector and reshape
        eigvec = eigenvectors[:, i]
        rate_map = eigvec.reshape(grid_size, grid_size)
        rate_map = np.maximum(rate_map, 0)
        rate_map_smooth = gaussian_filter(rate_map, sigma=SMOOTHING_SIGMA)
        
        # Compute autocorrelation
        rate_map_centered = rate_map_smooth - np.mean(rate_map_smooth)
        f = np.fft.fft2(rate_map_centered)
        autocorr = np.fft.ifft2(f * np.conj(f)).real
        autocorr = np.fft.fftshift(autocorr)
        
        # Normalize
        if autocorr.max() > 0:
            autocorr = autocorr / autocorr.max()
        
        # Plot
        im = ax.imshow(autocorr, cmap='jet', interpolation='nearest', 
                      vmin=-1, vmax=1)
        
        # Minimal styling
        ax.set_xticks([0, grid_size-1])
        ax.set_yticks([0, grid_size-1])
        ax.set_xticklabels(['0', str(grid_size-1)], fontsize=8)
        ax.set_yticklabels(['0', str(grid_size-1)], fontsize=8)
        ax.tick_params(length=2)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("GRID CELL EXTRACTION - STOEWER ET AL. STYLE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Grid size: {GRID_SIZE} x {GRID_SIZE} ({GRID_SIZE**2} states)")
    print(f"  Discount factor (γ): {GAMMA}")
    print(f"  Number of eigenvectors: {N_EIGENVECTORS}")
    
    # Step 1: Compute transition matrix
    print(f"\n{'Computing transition matrix...':<50}", end='')
    T = compute_transition_matrix(GRID_SIZE)
    print("✓")
    
    # Step 2: Compute successor representation
    print(f"{'Computing successor representation...':<50}", end='')
    M = compute_successor_representation(T, GAMMA)
    print("✓")
    
    # Step 3: Extract eigenvectors
    print(f"{'Extracting eigenvectors...':<50}", end='')
    eigenvectors, eigenvalues = extract_grid_cells(M, N_EIGENVECTORS)
    print("✓")
    
    # Step 4: Generate visualizations
    print(f"{'Generating visualizations...':<50}", end='')
    
    # Plot grid cells (rate maps)
    fig1 = plot_grid_cells_stoewer_style(eigenvectors, GRID_SIZE)
    
    # Plot autocorrelations
    fig2 = plot_autocorrelations_stoewer_style(eigenvectors, GRID_SIZE)
    
    print("✓")
    
    # Save figures
    import os
    output_dir = 'grid_cell_images'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'Saving figures...':<50}", end='')
    fig1.savefig(os.path.join(output_dir, 'stoewer_style_rate_maps.png'), 
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'stoewer_style_autocorrelations.png'), 
                dpi=150, bbox_inches='tight')
    print("✓")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nEigenvalue range:")
    print(f"  Largest:  {eigenvalues[0]:.4f}")
    print(f"  Smallest: {eigenvalues[-1]:.4f}")
    print(f"  Range:    {eigenvalues[0] - eigenvalues[-1]:.4f}")
    
    print("\nFiles saved to 'grid_cell_images/' folder:")
    print("  • stoewer_style_rate_maps.png - 30 eigenvector rate maps (5x6 grid)")
    print("  • stoewer_style_autocorrelations.png - Spatial autocorrelations")
    
    print("\nNOTE:")
    print("  Compare these patterns to Stoewer et al. (2022) Figure 4.")
    print("  Look for checkerboard and diagonal stripe patterns in cells 15-30.")
    print("="*70)
    
    plt.show()
    
    return M, eigenvectors, eigenvalues

if __name__ == "__main__":
    M, eigenvectors, eigenvalues = main()