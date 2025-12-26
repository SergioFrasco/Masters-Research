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
        dict: {algorithm: {'mean': array, 'std': array, 'training_episodes': int, ...}}
    """
    
    aggregated = {}
    
    for algorithm, seed_results in all_results.items():
        # Collect all reward arrays
        reward_arrays = [results['all_rewards'] for results in seed_results.values()]
        
        # Stack and compute statistics
        rewards_stacked = np.stack(reward_arrays, axis=0)  # (num_seeds, num_episodes)
        
        mean_rewards = rewards_stacked.mean(axis=0)
        std_rewards = rewards_stacked.std(axis=0)
        
        # Get episode counts (same across all seeds)
        first_result = list(seed_results.values())[0]
        training_episodes = first_result['training_episodes']
        
        # Handle both old format (eval_episodes) and new format (primitive_eval_episodes, comp_eval_episodes)
        if 'primitive_eval_episodes' in first_result:
            primitive_eval_episodes = first_result['primitive_eval_episodes']
            comp_eval_episodes = first_result['comp_eval_episodes']
        else:
            # Backward compatibility: old format only had compositional eval
            primitive_eval_episodes = 0
            comp_eval_episodes = first_result.get('eval_episodes', len(mean_rewards) - training_episodes)
        
        aggregated[algorithm] = {
            'mean': mean_rewards,
            'std': std_rewards,
            'training_episodes': training_episodes,
            'primitive_eval_episodes': primitive_eval_episodes,
            'comp_eval_episodes': comp_eval_episodes,
            'num_seeds': len(seed_results)
        }
        
        total_eval = primitive_eval_episodes + comp_eval_episodes
        
        print(f"{algorithm}:")
        print(f"  Seeds: {len(seed_results)}")
        print(f"  Episodes: {len(mean_rewards)} (training={training_episodes}, prim_eval={primitive_eval_episodes}, comp_eval={comp_eval_episodes})")
        print(f"  Final training reward: {mean_rewards[training_episodes-1]:.3f} ± {std_rewards[training_episodes-1]:.3f}")
        
        if primitive_eval_episodes > 0:
            prim_eval_end = training_episodes + primitive_eval_episodes
            prim_eval_mean = mean_rewards[training_episodes:prim_eval_end].mean()
            prim_eval_std = std_rewards[training_episodes:prim_eval_end].mean()
            print(f"  Primitive eval reward: {prim_eval_mean:.3f} ± {prim_eval_std:.3f}")
        
        if comp_eval_episodes > 0:
            comp_eval_start = training_episodes + primitive_eval_episodes
            comp_eval_mean = mean_rewards[comp_eval_start:].mean()
            comp_eval_std = std_rewards[comp_eval_start:].mean()
            print(f"  Compositional eval reward: {comp_eval_mean:.3f} ± {comp_eval_std:.3f}")
        
        print()
    
    return aggregated


def create_comparison_plot(aggregated, metadata, output_path, window=50):
    """
    Create the main comparison plot.
    
    Shows all algorithms with:
    - Smoothed rewards (window=50)
    - Shaded std error regions
    - Vertical lines at training/primitive_eval/comp_eval boundaries
    - Task labels above graph
    """
    
    print(f"\n{'='*70}")
    print(f"CREATING COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
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
    
    # Get episode counts (should be same for all)
    first_algo = list(aggregated.values())[0]
    training_episodes = first_algo['training_episodes']
    primitive_eval_episodes = first_algo['primitive_eval_episodes']
    comp_eval_episodes = first_algo['comp_eval_episodes']
    total_episodes = len(first_algo['mean'])
    
    print(f"Training episodes: {training_episodes}")
    print(f"Primitive eval episodes: {primitive_eval_episodes}")
    print(f"Compositional eval episodes: {comp_eval_episodes}")
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
               linewidth=2, alpha=0.7, label='Start Primitive Eval')
    
    # Add vertical line at primitive/compositional eval boundary (if primitive eval exists)
    if primitive_eval_episodes > 0:
        prim_eval_end = training_episodes + primitive_eval_episodes
        ax.axvline(x=prim_eval_end, color='darkgray', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Start Compositional Eval')
    
    # Task definitions
    primitive_tasks = ['red', 'blue', 'box', 'sphere']
    compositional_tasks = ['blue_sphere', 'red_sphere', 'blue_box', 'red_box']
    
    # Create a second x-axis on top for task labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    task_positions = []
    task_labels = []
    
    # Add primitive eval task labels (if exists)
    if primitive_eval_episodes > 0:
        prim_eval_per_task = primitive_eval_episodes // len(primitive_tasks)
        for i, task in enumerate(primitive_tasks):
            start_ep = training_episodes + i * prim_eval_per_task
            end_ep = training_episodes + (i + 1) * prim_eval_per_task
            mid_ep = (start_ep + end_ep) / 2
            task_positions.append(mid_ep)
            task_labels.append(f"[P] {task}")
        
        # Add vertical lines between primitive tasks
        for i in range(1, len(primitive_tasks)):
            boundary = training_episodes + i * prim_eval_per_task
            ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    # Add compositional eval task labels
    if comp_eval_episodes > 0:
        comp_eval_start = training_episodes + primitive_eval_episodes
        comp_eval_per_task = comp_eval_episodes // len(compositional_tasks)
        for i, task in enumerate(compositional_tasks):
            start_ep = comp_eval_start + i * comp_eval_per_task
            end_ep = comp_eval_start + (i + 1) * comp_eval_per_task
            mid_ep = (start_ep + end_ep) / 2
            task_positions.append(mid_ep)
            task_labels.append(f"[C] {task}")
        
        # Add vertical lines between compositional tasks
        for i in range(1, len(compositional_tasks)):
            boundary = comp_eval_start + i * comp_eval_per_task
            ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax2.set_xticks(task_positions)
    ax2.set_xticklabels(task_labels, fontsize=9, fontweight='bold', rotation=45, ha='left')
    ax2.set_xlabel('Evaluation Tasks ([P]=Primitive, [C]=Compositional)', fontsize=11, fontweight='bold')
    
    # Main axis labels
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward (Smoothed)', fontsize=13, fontweight='bold')
    ax.set_title(f'Algorithm Comparison: Compositional RL\n' +
                 f'Training → Primitive Eval → Compositional Eval (Zero-Shot)\n' +
                 f'(Smoothing window={window}, {first_algo["num_seeds"]} seeds)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set y-limits
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
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
            primitive_eval_episodes = data['primitive_eval_episodes']
            comp_eval_episodes = data['comp_eval_episodes']
            
            # Training statistics
            train_mean = mean_rewards[:training_episodes].mean()
            train_std = std_rewards[:training_episodes].mean()
            train_final = mean_rewards[training_episodes - 1]
            train_final_std = std_rewards[training_episodes - 1]
            
            f.write(f"{algorithm}:\n")
            f.write(f"  Number of seeds: {num_seeds}\n")
            f.write(f"  Training episodes: {training_episodes}\n")
            f.write(f"  Primitive eval episodes: {primitive_eval_episodes}\n")
            f.write(f"  Compositional eval episodes: {comp_eval_episodes}\n")
            f.write(f"\n")
            f.write(f"  Training phase:\n")
            f.write(f"    Average reward: {train_mean:.4f} ± {train_std:.4f}\n")
            f.write(f"    Final episode: {train_final:.4f} ± {train_final_std:.4f}\n")
            f.write(f"\n")
            
            # Primitive eval statistics
            if primitive_eval_episodes > 0:
                prim_start = training_episodes
                prim_end = training_episodes + primitive_eval_episodes
                prim_mean = mean_rewards[prim_start:prim_end].mean()
                prim_std = std_rewards[prim_start:prim_end].mean()
                
                f.write(f"  Primitive evaluation phase:\n")
                f.write(f"    Average reward: {prim_mean:.4f} ± {prim_std:.4f}\n")
                f.write(f"    Train→Prim gap: {train_final - prim_mean:.4f}\n")
                f.write(f"\n")
                
                # Per-task primitive breakdown
                primitive_tasks = ['red', 'blue', 'box', 'sphere']
                prim_per_task = primitive_eval_episodes // len(primitive_tasks)
                f.write(f"    Per-task breakdown:\n")
                for i, task in enumerate(primitive_tasks):
                    task_start = prim_start + i * prim_per_task
                    task_end = prim_start + (i + 1) * prim_per_task
                    task_mean = mean_rewards[task_start:task_end].mean()
                    task_std = std_rewards[task_start:task_end].mean()
                    f.write(f"      {task}: {task_mean:.4f} ± {task_std:.4f}\n")
                f.write(f"\n")
            
            # Compositional eval statistics
            if comp_eval_episodes > 0:
                comp_start = training_episodes + primitive_eval_episodes
                comp_mean = mean_rewards[comp_start:].mean()
                comp_std = std_rewards[comp_start:].mean()
                comp_final = mean_rewards[-1]
                comp_final_std = std_rewards[-1]
                
                f.write(f"  Compositional evaluation phase (zero-shot):\n")
                f.write(f"    Average reward: {comp_mean:.4f} ± {comp_std:.4f}\n")
                f.write(f"    Final episode: {comp_final:.4f} ± {comp_final_std:.4f}\n")
                
                if primitive_eval_episodes > 0:
                    f.write(f"    Prim→Comp gap: {prim_mean - comp_mean:.4f}\n")
                f.write(f"    Train→Comp gap: {train_final - comp_mean:.4f}\n")
                f.write(f"\n")
                
                # Per-task compositional breakdown
                compositional_tasks = ['blue_sphere', 'red_sphere', 'blue_box', 'red_box']
                comp_per_task = comp_eval_episodes // len(compositional_tasks)
                f.write(f"    Per-task breakdown:\n")
                for i, task in enumerate(compositional_tasks):
                    task_start = comp_start + i * comp_per_task
                    task_end = comp_start + (i + 1) * comp_per_task
                    task_mean = mean_rewards[task_start:task_end].mean()
                    task_std = std_rewards[task_start:task_end].mean()
                    f.write(f"      {task}: {task_mean:.4f} ± {task_std:.4f}\n")
                f.write(f"\n")
            
            f.write("-"*80 + "\n\n")
        
        # Summary comparison table
        f.write("="*80 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Algorithm':<10} {'Train':>12} {'Prim Eval':>12} {'Comp Eval':>12} {'Prim Gap':>12} {'Comp Gap':>12}\n")
        f.write("-"*70 + "\n")
        
        for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
            if algorithm not in aggregated:
                continue
            
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            training_episodes = data['training_episodes']
            primitive_eval_episodes = data['primitive_eval_episodes']
            comp_eval_episodes = data['comp_eval_episodes']
            
            train_final = mean_rewards[training_episodes - 1]
            
            if primitive_eval_episodes > 0:
                prim_start = training_episodes
                prim_end = training_episodes + primitive_eval_episodes
                prim_mean = mean_rewards[prim_start:prim_end].mean()
                prim_gap = train_final - prim_mean
            else:
                prim_mean = float('nan')
                prim_gap = float('nan')
            
            if comp_eval_episodes > 0:
                comp_start = training_episodes + primitive_eval_episodes
                comp_mean = mean_rewards[comp_start:].mean()
                comp_gap = train_final - comp_mean
            else:
                comp_mean = float('nan')
                comp_gap = float('nan')
            
            f.write(f"{algorithm:<10} {train_final:>12.4f} {prim_mean:>12.4f} {comp_mean:>12.4f} {prim_gap:>12.4f} {comp_gap:>12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
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