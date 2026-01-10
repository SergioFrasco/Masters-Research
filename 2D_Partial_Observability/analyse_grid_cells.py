"""
Analyze Saved SR Matrices for Grid Cells
=========================================
Loads saved SR matrices and performs eigendecomposition to extract
potential grid cell patterns.

Usage:
    python analyze_sr_grid_cells.py --sr_path results/sr_matrices/sr_matrix_episode_5000.npy
    
Or analyze all SR matrices in a directory:
    python analyze_sr_grid_cells.py --sr_dir results/sr_matrices/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from scipy.stats import pearsonr
import os
import json
import argparse


def compute_squareness_score(rate_map, grid_size):
    """
    Compute squareness score (90° rotational symmetry).
    Higher scores indicate square-periodic patterns.
    """
    rate_map_smooth = gaussian_filter(rate_map, sigma=0.3)
    
    # Compute autocorrelation
    rate_map_centered = rate_map_smooth - np.mean(rate_map_smooth)
    f = np.fft.fft2(rate_map_centered)
    autocorr = np.fft.ifft2(f * np.conj(f)).real
    autocorr = np.fft.fftshift(autocorr)
    
    # Normalize
    if autocorr.max() > 0:
        autocorr = autocorr / autocorr.max()
    
    # Create ring mask
    h, w = autocorr.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    inner_radius = 2
    outer_radius = min(h, w) // 2 - 1
    ring_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    
    # Apply mask
    autocorr_masked = autocorr.copy()
    autocorr_masked[~ring_mask] = 0
    
    # Compute correlations at different angles
    angles = [30, 60, 90, 120, 150]
    correlations = {}
    
    for angle in angles:
        rotated = rotate(autocorr_masked, angle, reshape=False, order=1)
        mask_overlap = ring_mask & (np.abs(rotated) > 1e-10)
        
        if mask_overlap.sum() > 10:
            try:
                corr = pearsonr(autocorr_masked[mask_overlap].flatten(),
                              rotated[mask_overlap].flatten())[0]
                if np.isnan(corr):
                    corr = 0
            except:
                corr = 0
            correlations[angle] = corr
        else:
            correlations[angle] = 0
    
    # Gridness (hexagonal): min(60,120) - max(30,90,150)
    gridness = min(correlations[60], correlations[120]) - \
               max(correlations[30], correlations[90], correlations[150])
    
    # Squareness (square/rectangular): 90° high, 60°/120° low
    squareness = correlations[90] - max(correlations[60], correlations[120])
    
    return gridness, squareness, autocorr, correlations


def analyze_sr_matrix(sr_path, output_dir='grid_cell_analysis', n_eigenvectors=30):
    """
    Load an SR matrix and analyze it for grid cell patterns.
    
    Parameters:
    -----------
    sr_path : str
        Path to .npy file containing SR matrix
    output_dir : str  
        Directory to save analysis results
    n_eigenvectors : int
        Number of eigenvectors to extract and analyze
        
    Returns:
    --------
    results : dict
        Dictionary containing eigenvalues, eigenvectors, and scores
    """
    print(f"\n{'='*80}")
    print(f"GRID CELL ANALYSIS: {os.path.basename(sr_path)}")
    print(f"{'='*80}\n")
    
    # Load SR matrix
    M = np.load(sr_path)
    print(f"Loaded SR matrix: shape {M.shape}")
    
    # Load metadata if available
    metadata_path = sr_path.replace('sr_matrix_', 'sr_metadata_').replace('.npy', '.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Episode: {metadata['episode']}")
        print(f"  Grid size: {metadata['grid_size']}x{metadata['grid_size']}")
        print(f"  Gamma: {metadata['gamma']}")
        grid_size = metadata['grid_size']
    else:
        grid_size = int(np.sqrt(M.shape[0]))
        metadata = {'episode': '?', 'grid_size': grid_size}
        print(f"  Inferred grid size: {grid_size}x{grid_size}")
    
    print(f"\nSR Matrix Statistics:")
    print(f"  Mean: {np.mean(M):.4f}")
    print(f"  Std:  {np.std(M):.4f}")
    print(f"  Min:  {np.min(M):.4f}")
    print(f"  Max:  {np.max(M):.4f}")
    
    # Eigendecomposition
    print(f"\nPerforming eigendecomposition...")
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    
    n_eigenvectors = min(n_eigenvectors, eigenvectors.shape[1])
    
    print(f"  Extracted {n_eigenvectors} eigenvectors")
    print(f"  Top 10 eigenvalues: {eigenvalues[:10]}")
    
    # Compute scores for each eigenvector
    print(f"\nComputing gridness and squareness scores...")
    gridness_scores = []
    squareness_scores = []
    
    for i in range(n_eigenvectors):
        eigvec = eigenvectors[:, i]
        rate_map = eigvec.reshape(grid_size, grid_size)
        rate_map = np.maximum(rate_map, 0)  # Threshold at zero
        
        gridness, squareness, _, _ = compute_squareness_score(rate_map, grid_size)
        gridness_scores.append(gridness)
        squareness_scores.append(squareness)
    
    gridness_scores = np.array(gridness_scores)
    squareness_scores = np.array(squareness_scores)
    
    # Print results
    print(f"\nGRIDNESS (Hexagonal Symmetry) Statistics:")
    print(f"  Mean: {np.mean(gridness_scores):.4f}")
    print(f"  Max:  {np.max(gridness_scores):.4f}")
    print(f"  Cells > 0.3: {np.sum(gridness_scores > 0.3)}/{n_eigenvectors}")
    
    print(f"\nSQUARENESS (Square/Rectangular Symmetry) Statistics:")
    print(f"  Mean: {np.mean(squareness_scores):.4f}")
    print(f"  Max:  {np.max(squareness_scores):.4f}")
    print(f"  Cells > 0.3: {np.sum(squareness_scores > 0.3)}/{n_eigenvectors}")
    
    # Find top cells by squareness
    top_indices = np.argsort(squareness_scores)[::-1][:10]
    print(f"\nTop 10 Cells by SQUARENESS:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. Cell {idx+1:2d}: "
              f"squareness={squareness_scores[idx]:>7.4f}, "
              f"gridness={gridness_scores[idx]:>7.4f}, "
              f"λ={eigenvalues[idx]:>7.4f}")
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: All eigenvectors (5x6 grid)
    fig1, axes = plt.subplots(5, 6, figsize=(12, 10))
    fig1.suptitle(f'Eigenvectors from Learned SR (Episode {metadata["episode"]})', 
                  fontsize=14, fontweight='bold')
    
    for i in range(min(30, n_eigenvectors)):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        
        eigvec = eigenvectors[:, i]
        rate_map = eigvec.reshape(grid_size, grid_size)
        rate_map = np.maximum(rate_map, 0)
        rate_map_smooth = gaussian_filter(rate_map, sigma=0.3)
        
        im = ax.imshow(rate_map_smooth, cmap='jet', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'λ={eigenvalues[i]:.1f}', fontsize=8)
    
    plt.tight_layout()
    save_path1 = os.path.join(output_dir, f'all_eigenvectors_ep{metadata["episode"]}.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path1}")
    plt.close()
    
    # Plot 2: Top 9 cells by squareness with autocorrelations
    fig2, axes = plt.subplots(9, 3, figsize=(9, 18))
    fig2.suptitle(f'Top 9 Cells by Squareness (Episode {metadata["episode"]})', 
                  fontsize=14, fontweight='bold')
    
    for plot_idx in range(9):
        cell_idx = top_indices[plot_idx]
        
        eigvec = eigenvectors[:, cell_idx]
        rate_map = eigvec.reshape(grid_size, grid_size)
        rate_map = np.maximum(rate_map, 0)
        rate_map_smooth = gaussian_filter(rate_map, sigma=0.3)
        
        gridness, squareness, autocorr, correlations = compute_squareness_score(rate_map, grid_size)
        
        # Rate map
        ax1 = axes[plot_idx, 0]
        im1 = ax1.imshow(rate_map_smooth, cmap='jet', interpolation='nearest')
        ax1.set_title(f'Cell {cell_idx+1}\nλ={eigenvalues[cell_idx]:.2f}', fontsize=9)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Autocorrelation
        ax2 = axes[plot_idx, 1]
        im2 = ax2.imshow(autocorr, cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
        ax2.set_title(f'G={gridness:.3f}, S={squareness:.3f}', fontsize=9)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Rotational correlations
        ax3 = axes[plot_idx, 2]
        angles = list(correlations.keys())
        corrs = list(correlations.values())
        bars = ax3.bar(angles, corrs, width=20)
        
        # Color code: blue for 90° (square), green for 60/120° (hexagonal)
        for j, angle in enumerate(angles):
            if angle == 90:
                bars[j].set_facecolor('blue')
            elif angle in [60, 120]:
                bars[j].set_facecolor('green')
            else:
                bars[j].set_facecolor('red')
            bars[j].set_alpha(0.7)
        
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax3.set_ylim([-1, 1])
        ax3.set_xticks(angles)
        ax3.set_xticklabels(angles, fontsize=7)
        if plot_idx == 8:
            ax3.set_xlabel('Angle (°)', fontsize=8)
    
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f'top_cells_detailed_ep{metadata["episode"]}.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if np.max(squareness_scores) > 0.3:
        print("✓ Detected periodic structure with SQUARE symmetry")
        print(f"  Max squareness: {np.max(squareness_scores):.4f}")
        print(f"  This is expected for discrete square grids!")
    elif np.max(gridness_scores) > 0.3:
        print("✓ Detected periodic structure with HEXAGONAL symmetry")
        print(f"  Max gridness: {np.max(gridness_scores):.4f}")
    else:
        print("✗ No strong periodic structure detected")
        print("  Try analyzing SR matrices from later episodes (more training)")
    print(f"{'='*80}\n")
    
    # Return results
    results = {
        'eigenvalues': eigenvalues[:n_eigenvectors],
        'eigenvectors': eigenvectors[:, :n_eigenvectors],
        'gridness_scores': gridness_scores,
        'squareness_scores': squareness_scores,
        'grid_size': grid_size,
        'episode': metadata['episode']
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze SR matrices for grid cells')
    parser.add_argument('--sr_path', type=str, help='Path to specific SR matrix .npy file')
    parser.add_argument('--sr_dir', type=str, default='results/sr_matrices', 
                       help='Directory containing SR matrices')
    parser.add_argument('--output_dir', type=str, default='grid_cell_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--n_eigenvectors', type=int, default=30,
                       help='Number of eigenvectors to analyze')
    
    args = parser.parse_args()
    
    if args.sr_path:
        # Analyze specific file
        if os.path.exists(args.sr_path):
            analyze_sr_matrix(args.sr_path, args.output_dir, args.n_eigenvectors)
        else:
            print(f"Error: File not found: {args.sr_path}")
    else:
        # Analyze all files in directory
        if os.path.exists(args.sr_dir):
            sr_files = sorted([f for f in os.listdir(args.sr_dir) if f.endswith('.npy')])
            
            if not sr_files:
                print(f"No .npy files found in {args.sr_dir}")
                return
            
            print(f"Found {len(sr_files)} SR matrices to analyze\n")
            
            for sr_file in sr_files:
                sr_path = os.path.join(args.sr_dir, sr_file)
                analyze_sr_matrix(sr_path, args.output_dir, args.n_eigenvectors)
                print("\n" + "="*80 + "\n")
        else:
            print(f"Error: Directory not found: {args.sr_dir}")
            print("\nTo use this script:")
            print("1. Run your experiment to generate SR matrices")
            print("2. Ensure they're saved in results/sr_matrices/")
            print("3. Run: python analyze_sr_grid_cells.py")


if __name__ == "__main__":
    main()