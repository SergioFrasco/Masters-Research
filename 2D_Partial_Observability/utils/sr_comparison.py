import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import generate_save_path

class SRComparator:
    """Track and compare learned SR against optimal SR over time."""
    
    def __init__(self, optimal_sr):
        """
        Initialize SR comparator.
        
        Parameters:
        -----------
        optimal_sr : np.ndarray (100, 100)
            Pre-computed optimal SR matrix for the environment
        """
        self.optimal_sr = optimal_sr
        self.history = {
            'episodes': [],
            'mae': [],
            'rmse': [],
            'correlation': [],
            'max_error': [],
            'relative_error': []
        }
    
    def compare(self, learned_sr, episode):
        """
        Compare learned SR against optimal SR and record metrics.
        
        Parameters:
        -----------
        learned_sr : np.ndarray (100, 100)
            Current SR matrix from the agent
        episode : int
            Current episode number
        """
        # Ensure shapes match
        if learned_sr.shape != self.optimal_sr.shape:
            print(f"Warning: Shape mismatch. Learned: {learned_sr.shape}, Optimal: {self.optimal_sr.shape}")
            return None
        
        # Calculate metrics
        mae = np.mean(np.abs(learned_sr - self.optimal_sr))
        rmse = np.sqrt(np.mean((learned_sr - self.optimal_sr) ** 2))
        correlation = np.corrcoef(learned_sr.flatten(), self.optimal_sr.flatten())[0, 1]
        max_error = np.max(np.abs(learned_sr - self.optimal_sr))
        relative_error = np.mean(np.abs(learned_sr - self.optimal_sr) / (np.abs(self.optimal_sr) + 1e-8))
        
        # Store in history
        self.history['episodes'].append(episode)
        self.history['mae'].append(mae)
        self.history['rmse'].append(rmse)
        self.history['correlation'].append(correlation)
        self.history['max_error'].append(max_error)
        self.history['relative_error'].append(relative_error)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Max_Error': max_error,
            'Relative_Error': relative_error
        }
        
        return metrics
    
    def visualize_comparison(self, learned_sr, episode, save_path=None):
        """
        Create comprehensive visualization comparing learned vs optimal SR.
        
        Parameters:
        -----------
        learned_sr : np.ndarray (100, 100)
            Current SR matrix from the agent
        episode : int
            Current episode number
        save_path : str, optional
            Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Optimal SR
        im1 = axes[0, 0].imshow(self.optimal_sr, cmap='hot', aspect='auto')
        axes[0, 0].set_title('Optimal SR (Ground Truth)')
        axes[0, 0].set_xlabel('State')
        axes[0, 0].set_ylabel('State')
        plt.colorbar(im1, ax=axes[0, 0], label='SR Value')
        
        # 2. Learned SR
        im2 = axes[0, 1].imshow(learned_sr, cmap='hot', aspect='auto')
        axes[0, 1].set_title(f'Learned SR (Episode {episode})')
        axes[0, 1].set_xlabel('State')
        axes[0, 1].set_ylabel('State')
        plt.colorbar(im2, ax=axes[0, 1], label='SR Value')
        
        # 3. Absolute Difference
        diff = np.abs(learned_sr - self.optimal_sr)
        im3 = axes[0, 2].imshow(diff, cmap='Reds', aspect='auto')
        axes[0, 2].set_title(f'Absolute Difference\n(MAE: {np.mean(diff):.4f})')
        axes[0, 2].set_xlabel('State')
        axes[0, 2].set_ylabel('State')
        plt.colorbar(im3, ax=axes[0, 2], label='Absolute Error')
        
        # 4. SR from center position (optimal)
        center_state = 44  # Middle of 10x10 grid
        sr_opt_center = self.optimal_sr[center_state, :].reshape(10, 10)
        im4 = axes[1, 0].imshow(sr_opt_center, cmap='hot', origin='upper')
        axes[1, 0].set_title('Optimal SR from Center (4,4)')
        axes[1, 0].plot(4, 4, 'b*', markersize=15, label='Start')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].legend()
        plt.colorbar(im4, ax=axes[1, 0], label='Expected Occupancy')
        
        # 5. SR from center position (learned)
        sr_learned_center = learned_sr[center_state, :].reshape(10, 10)
        im5 = axes[1, 1].imshow(sr_learned_center, cmap='hot', origin='upper')
        axes[1, 1].set_title(f'Learned SR from Center (Ep {episode})')
        axes[1, 1].plot(4, 4, 'b*', markersize=15, label='Start')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].legend()
        plt.colorbar(im5, ax=axes[1, 1], label='Expected Occupancy')
        
        # 6. Correlation scatter plot
        axes[1, 2].scatter(self.optimal_sr.flatten(), learned_sr.flatten(), 
                          alpha=0.3, s=1, c='blue')
        axes[1, 2].plot([self.optimal_sr.min(), self.optimal_sr.max()], 
                       [self.optimal_sr.min(), self.optimal_sr.max()], 
                       'r--', linewidth=2, label='Perfect match')
        
        # Calculate and display correlation
        corr = np.corrcoef(self.optimal_sr.flatten(), learned_sr.flatten())[0, 1]
        axes[1, 2].set_xlabel('Optimal SR Value')
        axes[1, 2].set_ylabel('Learned SR Value')
        axes[1, 2].set_title(f'Correlation: {corr:.4f}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = generate_save_path(f'sr_comparison/comparison_ep{episode}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"SR comparison visualization saved to {save_path}")
    
    def plot_learning_curves(self, save_path=None):
        """Plot how SR metrics evolve over training."""
        if not self.history['episodes']:
            print("No comparison history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        episodes = self.history['episodes']
        
        # MAE
        axes[0, 0].plot(episodes, self.history['mae'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('MAE vs Optimal SR')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE
        axes[0, 1].plot(episodes, self.history['rmse'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE vs Optimal SR')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation
        axes[0, 2].plot(episodes, self.history['correlation'], 'r-', linewidth=2)
        axes[0, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Correlation')
        axes[0, 2].set_title('Correlation with Optimal SR')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0, 1.05])
        
        # Max Error
        axes[1, 0].plot(episodes, self.history['max_error'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Max Absolute Error')
        axes[1, 0].set_title('Maximum Error vs Optimal SR')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Relative Error
        axes[1, 1].plot(episodes, self.history['relative_error'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_title('Relative Error vs Optimal SR')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics text
        axes[1, 2].axis('off')
        if len(episodes) > 0:
            summary_text = f"""
            SR Learning Progress Summary

            Episodes tracked: {len(episodes)}
            From episode {episodes[0]} to {episodes[-1]}

            Final Metrics (Episode {episodes[-1]}):
            MAE: {self.history['mae'][-1]:.6f}
            RMSE: {self.history['rmse'][-1]:.6f}
            Correlation: {self.history['correlation'][-1]:.6f}
            Max Error: {self.history['max_error'][-1]:.6f}
            Relative Error: {self.history['relative_error'][-1]:.6f}

            Best Correlation: {max(self.history['correlation']):.6f}
            (at episode {episodes[np.argmax(self.history['correlation'])]})

            Lowest MAE: {min(self.history['mae']):.6f}
            (at episode {episodes[np.argmin(self.history['mae'])]})
            """
            axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, 
                          verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = generate_save_path('sr_comparison/learning_curves.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"SR learning curves saved to {save_path}")
    
    def save_history(self, filename=None):
        """Save comparison history to file."""
        if filename is None:
            filename = generate_save_path('sr_comparison/comparison_history.npz')
        
        np.savez(filename, **self.history)
        print(f"SR comparison history saved to {filename}")