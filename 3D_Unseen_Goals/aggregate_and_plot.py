"""
Aggregate Results and Create Comparison Plot

Collects results from all (algorithm, seed) runs and creates a unified comparison plot
showing rewards over episodes with smoothing.

Usage:
    python aggregate_and_plot.py <experiment_dir>
    
Example:
    python aggregate_and_plot.py experiment_results/comparison_20241218_143022
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_all_results(experiment_dir):
    """
    Load results from all algorithm/seed combinations.
    
    Returns:
        dict: {algorithm: {seed: results_dict}}
    """
    
    experiment_dir = Path(experiment_dir)
    
    print(f"\n{'='*70}")
    print(f"LOADING RESULTS FROM: {experiment_dir}")
    print(f"{'='*70}\n")
    
    # Load metadata
    with open(experiment_dir / "experiment_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    algorithms = metadata["algorithms"]
    num_seeds = metadata["num_seeds"]
    
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {list(range(num_seeds))}\n")
    
    # Load results for each (algorithm, seed) pair
    all_results = defaultdict(dict)
    
    for algorithm in algorithms:
        for seed in range(num_seeds):
            run_dir = experiment_dir / f"{algorithm}_seed{seed}"
            results_file = run_dir / "results.json"
            
            if not results_file.exists():
                print(f"⚠️  Missing results: {algorithm} seed={seed}")
                continue
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Convert lists back to numpy arrays
            results['all_rewards'] = np.array(results['all_rewards'])
            
            all_results[algorithm][seed] = results
            print(f"✓ Loaded: {algorithm} seed={seed} ({len(results['all_rewards'])} episodes)")
    
    print(f"\n{'='*70}\n")
    
    return all_results, metadata


def aggregate_rewards(all_results):
    """
    Aggregate rewards across seeds for each algorithm.
    
    Returns:
        dict: {algorithm: {'mean': array, 'std': array, 'training_episodes': int}}
    """
    
    aggregated = {}
    
    for algorithm, seed_results in all_results.items():
        # Collect all reward arrays
        reward_arrays = [results['all_rewards'] for results in seed_results.values()]
        
        # Stack and compute statistics
        rewards_stacked = np.stack(reward_arrays, axis=0)  # (num_seeds, num_episodes)
        
        mean_rewards = rewards_stacked.mean(axis=0)
        std_rewards = rewards_stacked.std(axis=0)
        
        # Get training episodes count (same across all seeds)
        training_episodes = list(seed_results.values())[0]['training_episodes']
        
        aggregated[algorithm] = {
            'mean': mean_rewards,
            'std': std_rewards,
            'training_episodes': training_episodes,
            'num_seeds': len(seed_results)
        }
        
        print(f"{algorithm}:")
        print(f"  Seeds: {len(seed_results)}")
        print(f"  Episodes: {len(mean_rewards)} (training={training_episodes}, eval={len(mean_rewards) - training_episodes})")
        print(f"  Final training reward: {mean_rewards[training_episodes-1]:.3f} ± {std_rewards[training_episodes-1]:.3f}")
        print(f"  Final eval reward: {mean_rewards[-1]:.3f} ± {std_rewards[-1]:.3f}\n")
    
    return aggregated


def create_comparison_plot(aggregated, metadata, output_path, window=50):
    """
    Create the main comparison plot.
    
    Shows all algorithms with:
    - Smoothed rewards (window=50)
    - Shaded std error regions
    - Vertical line at training/eval boundary
    - Task labels above graph
    """
    
    print(f"\n{'='*70}")
    print(f"CREATING COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Algorithm colors and styles
    colors = {
        'SR': '#E74C3C',       # Red
        'DQN': '#3498DB',      # Blue
        'LSTM': '#2ECC71',     # Green
        'WVF': '#9B59B6'       # Purple
    }
    
    linestyles = {
        'SR': '-',
        'DQN': '-',
        'LSTM': '-',
        'WVF': '-'
    }
    
    # Get training episodes (should be same for all)
    training_episodes = list(aggregated.values())[0]['training_episodes']
    total_episodes = len(list(aggregated.values())[0]['mean'])
    eval_episodes = total_episodes - training_episodes
    
    print(f"Training episodes: {training_episodes}")
    print(f"Eval episodes: {eval_episodes}")
    print(f"Total episodes: {total_episodes}")
    print(f"Smoothing window: {window}\n")
    
    # Plot each algorithm
    for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
        if algorithm not in aggregated:
            print(f"⚠️  Skipping {algorithm} (no data)")
            continue
        
        data = aggregated[algorithm]
        mean_rewards = data['mean']
        std_rewards = data['std']
        num_seeds = data['num_seeds']
        
        # Smooth rewards
        mean_smoothed = pd.Series(mean_rewards).rolling(window, min_periods=1).mean().values
        std_smoothed = pd.Series(std_rewards).rolling(window, min_periods=1).mean().values
        
        # Standard error
        se_smoothed = std_smoothed / np.sqrt(num_seeds)
        
        episodes = np.arange(len(mean_smoothed))
        
        # Plot mean line
        ax.plot(episodes, mean_smoothed, 
                color=colors[algorithm], 
                linestyle=linestyles[algorithm],
                linewidth=2.5, 
                label=algorithm,
                alpha=0.9)
        
        # Plot shaded error region
        ax.fill_between(episodes,
                        mean_smoothed - se_smoothed,
                        mean_smoothed + se_smoothed,
                        color=colors[algorithm],
                        alpha=0.2)
        
        print(f"✓ Plotted {algorithm}: {len(mean_smoothed)} episodes")
    
    # Add vertical line at training/eval boundary
    ax.axvline(x=training_episodes, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, label='Start Evaluation')
    
    # Add task labels above the graph
    compositional_tasks = ['blue_sphere', 'red_sphere', 'blue_box', 'red_box']
    eval_episodes_per_task = eval_episodes // len(compositional_tasks)
    
    # Create a second x-axis on top for task labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Position ticks at the middle of each task's evaluation period
    task_positions = []
    for i, task in enumerate(compositional_tasks):
        start_ep = training_episodes + i * eval_episodes_per_task
        end_ep = training_episodes + (i + 1) * eval_episodes_per_task
        mid_ep = (start_ep + end_ep) / 2
        task_positions.append(mid_ep)
    
    ax2.set_xticks(task_positions)
    ax2.set_xticklabels(compositional_tasks, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Compositional Task (Evaluation)', fontsize=12, fontweight='bold')
    
    # Add vertical lines between tasks (lighter)
    for i in range(1, len(compositional_tasks)):
        boundary = training_episodes + i * eval_episodes_per_task
        ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    
    # Main axis labels
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward (Smoothed)', fontsize=13, fontweight='bold')
    ax.set_title(f'Algorithm Comparison: Compositional RL\n' +
                 f'Training on Random Primitive Tasks → Zero-Shot Compositional Evaluation\n' +
                 f'(Smoothing window={window}, {aggregated[list(aggregated.keys())[0]]["num_seeds"]} seeds)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set y-limits
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close()


def create_summary_statistics(aggregated, output_path):
    """Create a text summary of key statistics."""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
            if algorithm not in aggregated:
                continue
            
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            std_rewards = data['std']
            num_seeds = data['num_seeds']
            training_episodes = data['training_episodes']
            
            # Training statistics
            train_mean = mean_rewards[:training_episodes].mean()
            train_std = std_rewards[:training_episodes].mean()
            train_final = mean_rewards[training_episodes - 1]
            train_final_std = std_rewards[training_episodes - 1]
            
            # Eval statistics
            eval_mean = mean_rewards[training_episodes:].mean()
            eval_std = std_rewards[training_episodes:].mean()
            eval_final = mean_rewards[-1]
            eval_final_std = std_rewards[-1]
            
            f.write(f"{algorithm}:\n")
            f.write(f"  Number of seeds: {num_seeds}\n")
            f.write(f"  Training episodes: {training_episodes}\n")
            f.write(f"  Eval episodes: {len(mean_rewards) - training_episodes}\n")
            f.write(f"\n")
            f.write(f"  Training phase:\n")
            f.write(f"    Average reward: {train_mean:.4f} ± {train_std:.4f}\n")
            f.write(f"    Final episode: {train_final:.4f} ± {train_final_std:.4f}\n")
            f.write(f"\n")
            f.write(f"  Evaluation phase (compositional):\n")
            f.write(f"    Average reward: {eval_mean:.4f} ± {eval_std:.4f}\n")
            f.write(f"    Final episode: {eval_final:.4f} ± {eval_final_std:.4f}\n")
            f.write(f"\n")
            f.write(f"  Generalization gap: {train_final - eval_mean:.4f}\n")
            f.write(f"\n" + "-"*80 + "\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Summary statistics saved to: {output_path}")


def aggregate_and_plot(experiment_dir):
    """Main aggregation and plotting function."""
    
    experiment_dir = Path(experiment_dir)
    
    # Load all results
    all_results, metadata = load_all_results(experiment_dir)
    
    # Aggregate across seeds
    print(f"\n{'='*70}")
    print(f"AGGREGATING ACROSS SEEDS")
    print(f"{'='*70}\n")
    aggregated = aggregate_rewards(all_results)
    
    # Create comparison plot
    plot_path = experiment_dir / "comparison_plot.png"
    create_comparison_plot(aggregated, metadata, plot_path, window=50)
    
    # Create summary statistics
    summary_path = experiment_dir / "summary_statistics.txt"
    create_summary_statistics(aggregated, summary_path)
    
    print(f"\n{'='*70}")
    print(f"AGGREGATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Main plot: {plot_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*70}\n")


def main():
    """Command line interface."""
    
    if len(sys.argv) < 2:
        print("Usage: python aggregate_and_plot.py <experiment_dir>")
        print("\nExample:")
        print("  python aggregate_and_plot.py experiment_results/comparison_20241218_143022")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not Path(experiment_dir).exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    aggregate_and_plot(experiment_dir)


if __name__ == "__main__":
    main()