"""
Aggregate Results and Create Comparison Plot for Green Object Experiments

Collects results from all (algorithm, seed) runs and creates a unified comparison plot
showing rewards over episodes with smoothing for green object generalization experiments.

Evaluation structure:
- Training: Random primitive tasks (red, blue, box, sphere)
- Eval Phase 1: Primitive tasks (seen)
- Eval Phase 2: Seen compositional (red/blue + shape)
- Eval Phase 3: GREEN simple (unseen)
- Eval Phase 4: GREEN compositional (unseen)

Usage:
    python aggregate_and_plot_green.py <experiment_dir>
    
Example:
    python aggregate_and_plot_green.py experiment_results_green/green_comparison_20241218_143022
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
    print(f"LOADING GREEN EXPERIMENT RESULTS FROM: {experiment_dir}")
    print(f"{'='*70}\n")
    
    # Load metadata
    with open(experiment_dir / "experiment_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    algorithms = metadata["algorithms"]
    num_seeds = metadata["num_seeds"]
    
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {list(range(num_seeds))}")
    print(f"Experiment type: {metadata.get('experiment_type', 'unknown')}\n")
    
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
        sample_results = list(seed_results.values())[0]
        training_episodes = sample_results['training_episodes']
        primitive_eval_episodes = sample_results['primitive_eval_episodes']
        seen_comp_eval_episodes = sample_results['seen_comp_eval_episodes']
        green_simple_eval_episodes = sample_results['green_simple_eval_episodes']
        green_comp_eval_episodes = sample_results['green_comp_eval_episodes']
        
        aggregated[algorithm] = {
            'mean': mean_rewards,
            'std': std_rewards,
            'training_episodes': training_episodes,
            'primitive_eval_episodes': primitive_eval_episodes,
            'seen_comp_eval_episodes': seen_comp_eval_episodes,
            'green_simple_eval_episodes': green_simple_eval_episodes,
            'green_comp_eval_episodes': green_comp_eval_episodes,
            'num_seeds': len(seed_results)
        }
        
        # Calculate phase boundaries
        prim_start = training_episodes
        prim_end = prim_start + primitive_eval_episodes
        seen_comp_start = prim_end
        seen_comp_end = seen_comp_start + seen_comp_eval_episodes
        green_simple_start = seen_comp_end
        green_simple_end = green_simple_start + green_simple_eval_episodes
        green_comp_start = green_simple_end
        green_comp_end = green_comp_start + green_comp_eval_episodes
        
        print(f"{algorithm}:")
        print(f"  Seeds: {len(seed_results)}")
        print(f"  Training episodes: {training_episodes}")
        print(f"  Primitive eval: {primitive_eval_episodes} (episodes {prim_start}-{prim_end})")
        print(f"  Seen comp eval: {seen_comp_eval_episodes} (episodes {seen_comp_start}-{seen_comp_end})")
        print(f"  Green simple eval: {green_simple_eval_episodes} (episodes {green_simple_start}-{green_simple_end})")
        print(f"  Green comp eval: {green_comp_eval_episodes} (episodes {green_comp_start}-{green_comp_end})")
        print(f"  Total episodes: {len(mean_rewards)}")
        
        # Phase statistics
        train_final = mean_rewards[training_episodes-1]
        train_final_std = std_rewards[training_episodes-1]
        prim_mean = mean_rewards[prim_start:prim_end].mean()
        seen_comp_mean = mean_rewards[seen_comp_start:seen_comp_end].mean()
        green_simple_mean = mean_rewards[green_simple_start:green_simple_end].mean()
        green_comp_mean = mean_rewards[green_comp_start:green_comp_end].mean()
        
        print(f"  Training final: {train_final:.3f} ± {train_final_std:.3f}")
        print(f"  Primitive eval mean: {prim_mean:.3f}")
        print(f"  Seen comp eval mean: {seen_comp_mean:.3f}")
        print(f"  Green simple eval mean: {green_simple_mean:.3f} (UNSEEN)")
        print(f"  Green comp eval mean: {green_comp_mean:.3f} (UNSEEN)")
        print()
    
    return aggregated


def create_comparison_plot(aggregated, metadata, output_path, window=50):
    """
    Create the main comparison plot for green object experiments.
    
    Shows all algorithms with:
    - Smoothed rewards (window=50)
    - Shaded std error regions
    - Vertical lines separating phases
    - Task labels for evaluation phases
    - Special highlighting for GREEN (unseen) tasks
    """
    
    print(f"\n{'='*70}")
    print(f"CREATING GREEN EXPERIMENT COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
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
    
    # Get episode counts (should be same for all algorithms)
    sample_data = list(aggregated.values())[0]
    training_episodes = sample_data['training_episodes']
    primitive_eval_episodes = sample_data['primitive_eval_episodes']
    seen_comp_eval_episodes = sample_data['seen_comp_eval_episodes']
    green_simple_eval_episodes = sample_data['green_simple_eval_episodes']
    green_comp_eval_episodes = sample_data['green_comp_eval_episodes']
    
    # Calculate phase boundaries
    prim_eval_start = training_episodes
    prim_eval_end = prim_eval_start + primitive_eval_episodes
    seen_comp_start = prim_eval_end
    seen_comp_end = seen_comp_start + seen_comp_eval_episodes
    green_simple_start = seen_comp_end
    green_simple_end = green_simple_start + green_simple_eval_episodes
    green_comp_start = green_simple_end
    green_comp_end = green_comp_start + green_comp_eval_episodes
    
    total_episodes = green_comp_end
    
    print(f"Phase boundaries:")
    print(f"  Training: 0 - {training_episodes}")
    print(f"  Primitive eval: {prim_eval_start} - {prim_eval_end}")
    print(f"  Seen comp eval: {seen_comp_start} - {seen_comp_end}")
    print(f"  Green simple eval: {green_simple_start} - {green_simple_end} (UNSEEN)")
    print(f"  Green comp eval: {green_comp_start} - {green_comp_end} (UNSEEN)")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Smoothing window: {window}\n")
    
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
                alpha=0.9,
                zorder=5)
        
        # Plot shaded error region
        ax.fill_between(episodes,
                        mean_smoothed - se_smoothed,
                        mean_smoothed + se_smoothed,
                        color=colors[algorithm],
                        alpha=0.2,
                        zorder=4)
        
        print(f"✓ Plotted {algorithm}: {len(mean_smoothed)} episodes")
    
    # Add vertical lines at phase boundaries
    ax.axvline(x=training_episodes, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, label='Training → Eval', zorder=3)
    ax.axvline(x=seen_comp_end, color='red', linestyle='--',
               linewidth=2.5, alpha=0.8, label='Seen → Unseen (GREEN)', zorder=3)
    
    # Lighter boundaries between eval phases
    ax.axvline(x=prim_eval_end, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)
    ax.axvline(x=green_simple_end, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)
    
    # Highlight GREEN (unseen) region
    ax.axvspan(green_simple_start, green_comp_end, alpha=0.1, color='green', 
               label='GREEN (Unseen)', zorder=1)
    
    # Create task labels on top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Define task groups and their positions
    task_groups = [
        ('Primitives\n(Seen)', prim_eval_start, prim_eval_end),
        ('Seen\nCompositional', seen_comp_start, seen_comp_end),
        ('GREEN\nSimple\n(UNSEEN)', green_simple_start, green_simple_end),
        ('GREEN\nCompositional\n(UNSEEN)', green_comp_start, green_comp_end),
    ]
    
    tick_positions = []
    tick_labels = []
    
    for label, start, end in task_groups:
        mid_point = (start + end) / 2
        tick_positions.append(mid_point)
        tick_labels.append(label)
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Evaluation Phase', fontsize=13, fontweight='bold')
    
    # Main axis labels
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward (Smoothed)', fontsize=13, fontweight='bold')
    
    title = (f'Green Object Generalization: Zero-Shot Compositional RL\n'
             f'Training: Random Primitives (red, blue, box, sphere) → '
             f'Eval: Seen Tasks + Unseen GREEN Tasks\n'
             f'(Smoothing window={window}, {sample_data["num_seeds"]} seeds)')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set y-limits
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close()


def create_summary_statistics(aggregated, output_path):
    """Create a text summary of key statistics for green experiments."""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GREEN OBJECT GENERALIZATION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT DESIGN:\n")
        f.write("  Training: Random primitive tasks (red, blue, box, sphere)\n")
        f.write("  Evaluation Phases:\n")
        f.write("    1. Primitive tasks (seen)\n")
        f.write("    2. Seen compositional (red/blue + shape)\n")
        f.write("    3. GREEN simple (UNSEEN - zero-shot)\n")
        f.write("    4. GREEN compositional (UNSEEN - zero-shot)\n")
        f.write("\n" + "="*80 + "\n\n")
        
        for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
            if algorithm not in aggregated:
                continue
            
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            std_rewards = data['std']
            num_seeds = data['num_seeds']
            
            # Episode boundaries
            training_episodes = data['training_episodes']
            prim_start = training_episodes
            prim_end = prim_start + data['primitive_eval_episodes']
            seen_comp_start = prim_end
            seen_comp_end = seen_comp_start + data['seen_comp_eval_episodes']
            green_simple_start = seen_comp_end
            green_simple_end = green_simple_start + data['green_simple_eval_episodes']
            green_comp_start = green_simple_end
            green_comp_end = green_comp_start + data['green_comp_eval_episodes']
            
            # Training statistics
            train_mean = mean_rewards[:training_episodes].mean()
            train_std = std_rewards[:training_episodes].mean()
            train_final = mean_rewards[training_episodes - 1]
            train_final_std = std_rewards[training_episodes - 1]
            
            # Primitive eval statistics
            prim_mean = mean_rewards[prim_start:prim_end].mean()
            prim_std = std_rewards[prim_start:prim_end].mean()
            
            # Seen compositional statistics
            seen_comp_mean = mean_rewards[seen_comp_start:seen_comp_end].mean()
            seen_comp_std = std_rewards[seen_comp_start:seen_comp_end].mean()
            
            # Green simple statistics (UNSEEN)
            green_simple_mean = mean_rewards[green_simple_start:green_simple_end].mean()
            green_simple_std = std_rewards[green_simple_start:green_simple_end].mean()
            
            # Green compositional statistics (UNSEEN)
            green_comp_mean = mean_rewards[green_comp_start:green_comp_end].mean()
            green_comp_std = std_rewards[green_comp_start:green_comp_end].mean()
            
            f.write(f"{algorithm}:\n")
            f.write(f"  Number of seeds: {num_seeds}\n")
            f.write(f"  Training episodes: {training_episodes}\n")
            f.write(f"\n")
            f.write(f"  TRAINING PHASE (red/blue only):\n")
            f.write(f"    Average reward: {train_mean:.4f} ± {train_std:.4f}\n")
            f.write(f"    Final episode: {train_final:.4f} ± {train_final_std:.4f}\n")
            f.write(f"\n")
            f.write(f"  EVALUATION - PRIMITIVE TASKS (seen):\n")
            f.write(f"    Episodes: {data['primitive_eval_episodes']}\n")
            f.write(f"    Average reward: {prim_mean:.4f} ± {prim_std:.4f}\n")
            f.write(f"\n")
            f.write(f"  EVALUATION - SEEN COMPOSITIONAL (red/blue + shape):\n")
            f.write(f"    Episodes: {data['seen_comp_eval_episodes']}\n")
            f.write(f"    Average reward: {seen_comp_mean:.4f} ± {seen_comp_std:.4f}\n")
            f.write(f"\n")
            f.write(f"  EVALUATION - GREEN SIMPLE (UNSEEN - zero-shot):\n")
            f.write(f"    Episodes: {data['green_simple_eval_episodes']}\n")
            f.write(f"    Average reward: {green_simple_mean:.4f} ± {green_simple_std:.4f}\n")
            f.write(f"    Generalization gap (seen prim → green): {prim_mean - green_simple_mean:+.4f}\n")
            f.write(f"\n")
            f.write(f"  EVALUATION - GREEN COMPOSITIONAL (UNSEEN - zero-shot):\n")
            f.write(f"    Episodes: {data['green_comp_eval_episodes']}\n")
            f.write(f"    Average reward: {green_comp_mean:.4f} ± {green_comp_std:.4f}\n")
            f.write(f"    Generalization gap (seen comp → green comp): {seen_comp_mean - green_comp_mean:+.4f}\n")
            f.write(f"\n")
            f.write(f"  ZERO-SHOT GENERALIZATION ANALYSIS:\n")
            f.write(f"    Seen primitives → Unseen green simple: {prim_mean - green_simple_mean:+.4f}\n")
            f.write(f"    Seen compositional → Unseen green comp: {seen_comp_mean - green_comp_mean:+.4f}\n")
            f.write(f"    Overall seen → Overall unseen: {((prim_mean + seen_comp_mean)/2) - ((green_simple_mean + green_comp_mean)/2):+.4f}\n")
            f.write(f"\n" + "-"*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("CROSS-ALGORITHM COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Compare algorithms on green tasks
        f.write("GREEN SIMPLE (UNSEEN) PERFORMANCE:\n")
        for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
            if algorithm not in aggregated:
                continue
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            green_simple_start = (data['training_episodes'] + 
                                 data['primitive_eval_episodes'] + 
                                 data['seen_comp_eval_episodes'])
            green_simple_end = green_simple_start + data['green_simple_eval_episodes']
            green_simple_mean = mean_rewards[green_simple_start:green_simple_end].mean()
            f.write(f"  {algorithm:6s}: {green_simple_mean:.4f}\n")
        
        f.write("\nGREEN COMPOSITIONAL (UNSEEN) PERFORMANCE:\n")
        for algorithm in ['SR', 'DQN', 'LSTM', 'WVF']:
            if algorithm not in aggregated:
                continue
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            green_comp_start = (data['training_episodes'] + 
                               data['primitive_eval_episodes'] + 
                               data['seen_comp_eval_episodes'] +
                               data['green_simple_eval_episodes'])
            green_comp_end = green_comp_start + data['green_comp_eval_episodes']
            green_comp_mean = mean_rewards[green_comp_start:green_comp_end].mean()
            f.write(f"  {algorithm:6s}: {green_comp_mean:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Summary statistics saved to: {output_path}")


def create_bar_comparison_plot(aggregated, output_path):
    """Create a bar plot comparing performance across different evaluation phases."""
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    algorithms = ['SR', 'DQN', 'LSTM', 'WVF']
    colors = {
        'SR': '#E74C3C',
        'DQN': '#3498DB',
        'LSTM': '#2ECC71',
        'WVF': '#9B59B6'
    }
    
    phases = ['Primitive\n(Seen)', 'Seen\nCompositional', 'GREEN\nSimple\n(UNSEEN)', 'GREEN\nCompositional\n(UNSEEN)']
    
    for phase_idx, (ax, phase_name) in enumerate(zip(axes, phases)):
        phase_means = []
        phase_stds = []
        
        for algorithm in algorithms:
            if algorithm not in aggregated:
                phase_means.append(0)
                phase_stds.append(0)
                continue
            
            data = aggregated[algorithm]
            mean_rewards = data['mean']
            std_rewards = data['std']
            num_seeds = data['num_seeds']
            
            # Calculate phase boundaries
            training_episodes = data['training_episodes']
            prim_start = training_episodes
            prim_end = prim_start + data['primitive_eval_episodes']
            seen_comp_start = prim_end
            seen_comp_end = seen_comp_start + data['seen_comp_eval_episodes']
            green_simple_start = seen_comp_end
            green_simple_end = green_simple_start + data['green_simple_eval_episodes']
            green_comp_start = green_simple_end
            green_comp_end = green_comp_start + data['green_comp_eval_episodes']
            
            # Get phase data
            if phase_idx == 0:  # Primitive
                phase_data = mean_rewards[prim_start:prim_end]
                phase_std_data = std_rewards[prim_start:prim_end]
            elif phase_idx == 1:  # Seen comp
                phase_data = mean_rewards[seen_comp_start:seen_comp_end]
                phase_std_data = std_rewards[seen_comp_start:seen_comp_end]
            elif phase_idx == 2:  # Green simple
                phase_data = mean_rewards[green_simple_start:green_simple_end]
                phase_std_data = std_rewards[green_simple_start:green_simple_end]
            else:  # Green comp
                phase_data = mean_rewards[green_comp_start:green_comp_end]
                phase_std_data = std_rewards[green_comp_start:green_comp_end]
            
            phase_means.append(phase_data.mean())
            phase_stds.append(phase_std_data.mean() / np.sqrt(num_seeds))
        
        # Plot bars
        x = np.arange(len(algorithms))
        bars = ax.bar(x, phase_means, yerr=phase_stds, 
                     color=[colors[alg] for alg in algorithms],
                     edgecolor='black', linewidth=1.5, alpha=0.8, capsize=5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, phase_means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax.set_title(phase_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight unseen phases
        if phase_idx >= 2:
            ax.set_facecolor('#f0fff0')  # Light green background
    
    plt.suptitle('Algorithm Performance Across Evaluation Phases\n(Green = UNSEEN zero-shot tasks)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bar comparison plot saved to: {output_path}")


def aggregate_and_plot_green(experiment_dir):
    """Main aggregation and plotting function for green experiments."""
    
    experiment_dir = Path(experiment_dir)
    
    # Load all results
    all_results, metadata = load_all_results(experiment_dir)
    
    # Aggregate across seeds
    print(f"\n{'='*70}")
    print(f"AGGREGATING ACROSS SEEDS")
    print(f"{'='*70}\n")
    aggregated = aggregate_rewards(all_results)
    
    # Create main comparison plot
    plot_path = experiment_dir / "green_comparison_plot.png"
    create_comparison_plot(aggregated, metadata, plot_path, window=50)
    
    # Create bar comparison plot
    bar_plot_path = experiment_dir / "green_bar_comparison.png"
    create_bar_comparison_plot(aggregated, bar_plot_path)
    
    # Create summary statistics
    summary_path = experiment_dir / "green_summary_statistics.txt"
    create_summary_statistics(aggregated, summary_path)
    
    print(f"\n{'='*70}")
    print(f"GREEN EXPERIMENT AGGREGATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Main plot: {plot_path}")
    print(f"Bar comparison: {bar_plot_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*70}\n")


def main():
    """Command line interface."""
    
    if len(sys.argv) < 2:
        print("Usage: python aggregate_and_plot_green.py <experiment_dir>")
        print("\nExample:")
        print("  python aggregate_and_plot_green.py experiment_results_green/green_comparison_20241218_143022")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not Path(experiment_dir).exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    aggregate_and_plot_green(experiment_dir)


if __name__ == "__main__":
    main()