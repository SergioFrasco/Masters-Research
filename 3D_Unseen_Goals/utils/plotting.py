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


"""
Standalone Timeline Plotting Utility

Import this and call plot_experiment_timeline() with your results.

Usage:
    from timeline_plotter import plot_experiment_timeline
    
    plot_experiment_timeline(
        training_history=history,
        seen_eval_results=seen_results,
        unseen_eval_results=unseen_results,
        save_path="my_timeline.png"
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_experiment_timeline(training_history, seen_eval_results, unseen_eval_results, 
                             save_path, eval_episodes_per_task=100):
    """
    Create a comprehensive timeline plot showing:
    1. Training rewards over episodes
    2. Vertical line marking end of training
    3. Evaluation on seen tasks (with rewards if available)
    4. Vertical line marking start of unseen task evaluation
    5. Evaluation on unseen tasks (with rewards if available)
    
    Args:
        training_history: Dict with at least 'episode_rewards' key
        seen_eval_results: Dict of {task_name: {'success_rate': float, 
                                                 'episode_rewards': list (optional)}}
        unseen_eval_results: Dict of {task_name: {'success_rate': float,
                                                   'episode_rewards': list (optional)}}
        save_path: Path to save the figure
        eval_episodes_per_task: Number of episodes per evaluation task (default 100)
    """
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25, height_ratios=[2, 1, 1, 1])
    
    # ========================================================================
    # MAIN TIMELINE PLOT (Full Width, Top)
    # ========================================================================
    ax_main = fig.add_subplot(gs[0, :])
    
    train_rewards = training_history['episode_rewards']
    n_train = len(train_rewards)
    
    # Plot training rewards (raw + smoothed)
    ax_main.plot(range(n_train), train_rewards, alpha=0.2, color='gray', 
                linewidth=0.5, label='Training (raw)')
    
    if len(train_rewards) >= 100:
        smoothed = pd.Series(train_rewards).rolling(100, min_periods=1).mean()
        ax_main.plot(range(n_train), smoothed, color='black', linewidth=2.5, 
                    label='Training (100-ep MA)', zorder=10)
    
    # Track current episode number
    current_ep = n_train
    
    # === VERTICAL LINE 1: End of Training ===
    ax_main.axvline(x=n_train, color='red', linestyle='--', linewidth=3, 
                   label='End of Training', zorder=20, alpha=0.8)
    ax_main.text(n_train - 50, 0.98, '◄ Training Phase', ha='right', va='top',
                transform=ax_main.get_xaxis_transform(), fontsize=12, fontweight='bold',
                color='red', bbox=dict(boxstyle='round', facecolor='white', 
                                      edgecolor='red', linewidth=2))
    
    # === SEEN EVALUATION REGION ===
    seen_start = current_ep
    task_colors = {
        'red': '#E74C3C', 'blue': '#3498DB', 'box': '#E67E22', 
        'sphere': '#2ECC71', 'green': '#27AE60'
    }
    
    for task_name, results in seen_eval_results.items():
        # Use episode rewards if available, otherwise create horizontal line
        if 'episode_rewards' in results and results['episode_rewards']:
            ep_rewards = results['episode_rewards']
            n_ep = len(ep_rewards)
        else:
            # Create synthetic episode data from success rate
            n_ep = eval_episodes_per_task
            success_rate = results['success_rate']
            ep_rewards = [success_rate] * n_ep  # Horizontal line at success rate
        
        x_coords = range(current_ep, current_ep + n_ep)
        
        # Determine color
        color = task_colors.get(task_name.split('_')[0], 'purple')
        
        # Plot episodes
        ax_main.plot(x_coords, ep_rewards, alpha=0.3, color=color, linewidth=1)
        
        # Add mean line
        mean_val = np.mean(ep_rewards)
        ax_main.hlines(mean_val, current_ep, current_ep + n_ep,
                      colors=color, linewidth=2.5, label=f'{task_name}', zorder=5)
        
        # Add text label at end
        ax_main.text(current_ep + n_ep/2, mean_val + 0.05, f'{mean_val:.1%}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=color, bbox=dict(boxstyle='round', facecolor='white', 
                                          alpha=0.7, pad=0.3))
        
        current_ep += n_ep
    
    seen_end = current_ep
    
    # Shade seen evaluation region
    ax_main.axvspan(seen_start, seen_end, alpha=0.08, color='blue', zorder=0)
    ax_main.text((seen_start + seen_end)/2, 0.02, 'SEEN TASKS\nEVALUATION',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                transform=ax_main.get_xaxis_transform(), color='blue', alpha=0.6)
    
    # === VERTICAL LINE 2: Start of Unseen Tasks ===
    ax_main.axvline(x=seen_end, color='green', linestyle='--', linewidth=3,
                   label='Unseen Tasks Begin', zorder=20, alpha=0.8)
    ax_main.text(seen_end + 50, 0.98, 'Unseen (GREEN) ►', ha='left', va='top',
                transform=ax_main.get_xaxis_transform(), fontsize=12, fontweight='bold',
                color='green', bbox=dict(boxstyle='round', facecolor='white',
                                        edgecolor='green', linewidth=2))
    
    # === UNSEEN EVALUATION REGION ===
    unseen_start = current_ep
    
    for task_name, results in unseen_eval_results.items():
        if 'episode_rewards' in results and results['episode_rewards']:
            ep_rewards = results['episode_rewards']
            n_ep = len(ep_rewards)
        else:
            n_ep = eval_episodes_per_task
            success_rate = results['success_rate']
            ep_rewards = [success_rate] * n_ep
        
        x_coords = range(current_ep, current_ep + n_ep)
        
        color = '#27AE60' if 'green' in task_name else '#8E44AD'
        
        ax_main.plot(x_coords, ep_rewards, alpha=0.4, color=color, linewidth=1)
        
        mean_val = np.mean(ep_rewards)
        ax_main.hlines(mean_val, current_ep, current_ep + n_ep,
                      colors=color, linewidth=2.5, linestyle=':', 
                      label=f'{task_name} (unseen)', zorder=5)
        
        ax_main.text(current_ep + n_ep/2, mean_val + 0.05, f'{mean_val:.1%}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=color, bbox=dict(boxstyle='round', facecolor='white',
                                          alpha=0.7, pad=0.3))
        
        current_ep += n_ep
    
    unseen_end = current_ep
    
    # Shade unseen evaluation region
    ax_main.axvspan(unseen_start, unseen_end, alpha=0.08, color='green', zorder=0)
    ax_main.text((unseen_start + unseen_end)/2, 0.02, 'UNSEEN TASKS\nEVALUATION',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                transform=ax_main.get_xaxis_transform(), color='green', alpha=0.6)
    
    # Formatting
    ax_main.set_xlabel('Episode Number', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Reward (1 = Success, 0 = Failure)', fontsize=13, fontweight='bold')
    ax_main.set_title('Complete Experiment Timeline: Training → Seen Evaluation → Unseen Evaluation',
                     fontsize=15, fontweight='bold', pad=20)
    ax_main.set_ylim([-0.05, 1.15])
    ax_main.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle=':')
    
    # Add chance line
    ax_main.axhline(y=0.167, color='red', linestyle=':', linewidth=1.5, 
                   alpha=0.4, label='Chance (16.7%)')
    ax_main.text(0.99, 0.167, '16.7% (random)', ha='right', va='bottom',
                transform=ax_main.get_yaxis_transform(), fontsize=8, 
                style='italic', color='red', alpha=0.6)
    
    # ========================================================================
    # TRAINING CURVE DETAIL
    # ========================================================================
    ax_train = fig.add_subplot(gs[1, 0])
    
    ax_train.plot(train_rewards, alpha=0.3, color='gray', linewidth=0.5)
    if len(train_rewards) >= 100:
        smoothed = pd.Series(train_rewards).rolling(100, min_periods=1).mean()
        ax_train.plot(smoothed, color='black', linewidth=2)
    
    ax_train.set_xlabel('Training Episode')
    ax_train.set_ylabel('Reward')
    ax_train.set_title('Training Phase Detail', fontweight='bold')
    ax_train.grid(True, alpha=0.3)
    ax_train.set_ylim([-0.05, 1.05])
    
    # Add final training performance
    final_100 = train_rewards[-100:] if len(train_rewards) >= 100 else train_rewards
    final_success = np.mean([r > 0 for r in final_100])
    ax_train.text(0.95, 0.95, f'Final 100 eps:\n{final_success:.1%}',
                 transform=ax_train.transAxes, ha='right', va='top',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========================================================================
    # EPSILON DECAY (if available)
    # ========================================================================
    ax_eps = fig.add_subplot(gs[1, 1])
    
    if 'episode_epsilons' in training_history:
        epsilons = training_history['episode_epsilons']
        ax_eps.plot(epsilons, color='purple', linewidth=2)
        ax_eps.set_xlabel('Training Episode')
        ax_eps.set_ylabel('Epsilon (Exploration Rate)')
        ax_eps.set_title('Exploration Decay', fontweight='bold')
        ax_eps.grid(True, alpha=0.3)
        ax_eps.set_ylim([0, 1.05])
        
        # Annotate start and end
        ax_eps.text(0, epsilons[0], f'{epsilons[0]:.2f}', ha='right', va='bottom',
                   fontsize=9, fontweight='bold', color='purple')
        ax_eps.text(len(epsilons)-1, epsilons[-1], f'{epsilons[-1]:.3f}', 
                   ha='left', va='top', fontsize=9, fontweight='bold', color='purple')
    else:
        ax_eps.text(0.5, 0.5, 'Epsilon history\nnot available', 
                   ha='center', va='center', transform=ax_eps.transAxes,
                   fontsize=12, style='italic', color='gray')
        ax_eps.set_xticks([])
        ax_eps.set_yticks([])
    
    # ========================================================================
    # SUCCESS RATES BAR CHART
    # ========================================================================
    ax_bars = fig.add_subplot(gs[2, :])
    
    all_tasks = []
    all_rates = []
    all_colors = []
    all_types = []
    
    # Get training final performance
    if 'per_task_final_success' in training_history:
        for task in ['red', 'blue', 'box', 'sphere']:
            if task in training_history['per_task_final_success']:
                all_tasks.append(task)
                all_rates.append(training_history['per_task_final_success'][task])
                all_colors.append(task_colors.get(task, 'gray'))
                all_types.append('Train')
    
    # Seen eval
    for task, results in seen_eval_results.items():
        all_tasks.append(task)
        all_rates.append(results['success_rate'])
        all_colors.append(task_colors.get(task.split('_')[0], 'steelblue'))
        all_types.append('Seen')
    
    # Unseen eval
    for task, results in unseen_eval_results.items():
        all_tasks.append(task)
        all_rates.append(results['success_rate'])
        all_colors.append('#27AE60')
        all_types.append('UNSEEN')
    
    x_pos = np.arange(len(all_tasks))
    bars = ax_bars.bar(x_pos, all_rates, color=all_colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, all_rates):
        height = bar.get_height()
        ax_bars.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add type labels
    for i, task_type in enumerate(all_types):
        color = 'gray' if task_type == 'Train' else ('steelblue' if task_type == 'Seen' else '#27AE60')
        ax_bars.text(i, -0.12, task_type, ha='center', va='top',
                    transform=ax_bars.get_xaxis_transform(), fontsize=8,
                    fontweight='bold', color=color)
    
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax_bars.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax_bars.set_title('All Tasks: Success Rates', fontsize=13, fontweight='bold')
    ax_bars.set_ylim([0, 1.15])
    ax_bars.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax_bars.axhline(y=0.167, color='red', linestyle=':', alpha=0.4)
    ax_bars.grid(True, alpha=0.3, axis='y')
    
    # Add separators
    n_train = len([t for t in all_types if t == 'Train'])
    n_seen = len([t for t in all_types if t == 'Seen'])
    if n_train > 0:
        ax_bars.axvline(x=n_train - 0.5, color='red', linestyle='-', linewidth=2, alpha=0.3)
    if n_seen > 0:
        ax_bars.axvline(x=n_train + n_seen - 0.5, color='green', linestyle='-', 
                       linewidth=2, alpha=0.3)
    
    # ========================================================================
    # GENERALIZATION GAP ANALYSIS
    # ========================================================================
    ax_gap = fig.add_subplot(gs[3, :])
    
    # Calculate averages
    seen_simple_tasks = [t for t in seen_eval_results if len(seen_eval_results[t].get('features', [])) <= 1]
    seen_comp_tasks = [t for t in seen_eval_results if len(seen_eval_results[t].get('features', [])) == 2]
    unseen_simple_tasks = [t for t in unseen_eval_results if len(unseen_eval_results[t].get('features', [])) <= 1]
    unseen_comp_tasks = [t for t in unseen_eval_results if len(unseen_eval_results[t].get('features', [])) == 2]
    
    seen_simple_avg = np.mean([seen_eval_results[t]['success_rate'] for t in seen_simple_tasks]) if seen_simple_tasks else 0
    seen_comp_avg = np.mean([seen_eval_results[t]['success_rate'] for t in seen_comp_tasks]) if seen_comp_tasks else 0
    unseen_simple_avg = np.mean([unseen_eval_results[t]['success_rate'] for t in unseen_simple_tasks]) if unseen_simple_tasks else 0
    unseen_comp_avg = np.mean([unseen_eval_results[t]['success_rate'] for t in unseen_comp_tasks]) if unseen_comp_tasks else 0
    
    categories = ['Simple Tasks', 'Compositional Tasks']
    seen_vals = [seen_simple_avg, seen_comp_avg]
    unseen_vals = [unseen_simple_avg, unseen_comp_avg]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax_gap.bar(x - width/2, seen_vals, width, label='Seen (Red/Blue)',
                      color='steelblue', edgecolor='black', linewidth=2)
    bars2 = ax_gap.bar(x + width/2, unseen_vals, width, label='UNSEEN (Green)',
                      color='#27AE60', edgecolor='black', linewidth=2)
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_gap.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.1%}', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    # Add gap annotations
    for i in range(len(categories)):
        gap = seen_vals[i] - unseen_vals[i]
        if gap > 0:
            # Arrow showing gap
            y1, y2 = seen_vals[i], unseen_vals[i]
            ax_gap.annotate('', xy=(i + width/2, y2), xytext=(i - width/2, y1),
                          arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
            
            # Gap label
            mid_y = (y1 + y2) / 2
            ax_gap.text(i - 0.6, mid_y, f'Gap:\n{gap:.1%}',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', 
                                edgecolor='red', linewidth=2, alpha=0.8))
    
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax_gap.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax_gap.set_title('Generalization Gap: Seen vs Unseen', fontsize=13, fontweight='bold')
    ax_gap.set_ylim([0, 1.15])
    ax_gap.legend(fontsize=11, loc='upper right')
    ax_gap.grid(True, alpha=0.3, axis='y')
    ax_gap.axhline(y=0.167, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    
    # Overall title
    fig.suptitle('Complete Experiment Timeline: Training → Seen Eval → Unseen (Green) Eval',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"Timeline plot saved to: {save_path}")
    print(f"{'='*70}")


# Example usage
if __name__ == "__main__":
    # Create dummy data for demonstration
    np.random.seed(42)
    
    # Training history
    train_rewards = []
    for i in range(8000):
        # Simulate learning curve
        base_rate = min(0.9, 0.1 + 0.8 * (i / 8000))
        reward = 1.0 if np.random.random() < base_rate else 0.0
        train_rewards.append(reward)
    
    training_history = {
        'episode_rewards': train_rewards,
        'episode_epsilons': [1.0 * (0.9995 ** i) for i in range(8000)],
        'per_task_final_success': {
            'red': 0.92,
            'blue': 0.89,
            'box': 0.91,
            'sphere': 0.88
        }
    }
    
    # Seen evaluation results
    seen_eval_results = {
        'red': {'success_rate': 0.91, 'features': ['red']},
        'blue': {'success_rate': 0.88, 'features': ['blue']},
        'box': {'success_rate': 0.90, 'features': ['box']},
        'sphere': {'success_rate': 0.87, 'features': ['sphere']},
        'red_box': {'success_rate': 0.78, 'features': ['red', 'box']},
        'blue_sphere': {'success_rate': 0.75, 'features': ['blue', 'sphere']},
    }
    
    # Unseen evaluation results
    unseen_eval_results = {
        'green': {'success_rate': 0.32, 'features': ['green']},
        'green_box': {'success_rate': 0.28, 'features': ['green', 'box']},
        'green_sphere': {'success_rate': 0.25, 'features': ['green', 'sphere']},
    }
    
    plot_experiment_timeline(
        training_history,
        seen_eval_results,
        unseen_eval_results,
        'demo_timeline.png'
    )