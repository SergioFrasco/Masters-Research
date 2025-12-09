"""
DQN Training and Evaluation Script for MiniWorld

This script trains DQN on simple tasks and evaluates on compositional tasks.
Focus: Clean visualization of DQN training progress and evaluation results.
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from tqdm import tqdm
import json
import time
import gc
import torch

# Import your environment and agent
from env import DiscreteMiniWorldWrapper
from agents import DQNAgent3D


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

SIMPLE_TASKS = [
    {"name": "red", "features": ["red"], "type": "simple"},
    {"name": "blue", "features": ["blue"], "type": "simple"},
    {"name": "box", "features": ["box"], "type": "simple"},
    {"name": "sphere", "features": ["sphere"], "type": "simple"},
]

COMPOSITIONAL_TASKS = [
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies current task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    
    # Single feature tasks (simple)
    if len(features) == 1:
        feature = features[0]
        
        if feature == "blue":
            return contacted_object in ["blue_box", "blue_sphere"]
        elif feature == "red":
            return contacted_object in ["red_box", "red_sphere"]
        elif feature == "box":
            return contacted_object in ["blue_box", "red_box"]
        elif feature == "sphere":
            return contacted_object in ["blue_sphere", "red_sphere"]
    
    # Compositional tasks (2 features - AND logic)
    elif len(features) == 2:
        if set(features) == {"blue", "sphere"}:
            return contacted_object == "blue_sphere"
        elif set(features) == {"red", "sphere"}:
            return contacted_object == "red_sphere"
        elif set(features) == {"blue", "box"}:
            return contacted_object == "blue_box"
        elif set(features) == {"red", "box"}:
            return contacted_object == "red_box"
    
    return False


# ============================================================================
# TRAINING SCHEDULE
# ============================================================================

def create_training_schedule(total_episodes, shuffle=True):
    """Create training schedule with ONLY simple tasks."""
    episodes_per_task = total_episodes // len(SIMPLE_TASKS)
    
    schedule = []
    for task in SIMPLE_TASKS:
        for _ in range(episodes_per_task):
            schedule.append(task.copy())
    
    # Handle remainder
    remaining = total_episodes - len(schedule)
    for i in range(remaining):
        schedule.append(SIMPLE_TASKS[i % len(SIMPLE_TASKS)].copy())
    
    if shuffle:
        np.random.shuffle(schedule)
    
    return schedule


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_save_path(filename):
    """Generate save path"""
    os.makedirs("results_dqn", exist_ok=True)
    return os.path.join("results_dqn", filename)


def plot_training_and_evaluation(training_history, eval_results, save_path, window=100):
    """
    Single comprehensive plot showing:
    - Training rewards over time
    - Evaluation performance (simple vs compositional)
    - Training loss
    - Epsilon decay
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    episode_rewards = training_history["episode_rewards"]
    episode_lengths = training_history["episode_lengths"]
    episode_losses = training_history["episode_losses"]
    
    # ===== Plot 1: Training Rewards (top left) =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(episode_rewards) >= window:
        smoothed = pd.Series(episode_rewards).rolling(window).mean()
        ax1.plot(smoothed, color='darkblue', linewidth=2, label=f'Smoothed (w={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards (Simple Tasks Only)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Evaluation Bar Chart (top right) =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Simple tasks
    simple_names = [t['name'] for t in SIMPLE_TASKS]
    simple_success = [eval_results['simple'][name]['success_rate'] for name in simple_names]
    
    # Compositional tasks
    comp_names = [t['name'] for t in COMPOSITIONAL_TASKS]
    comp_success = [eval_results['compositional'][name]['success_rate'] for name in comp_names]
    
    x = np.arange(len(simple_names) + len(comp_names))
    colors = ['steelblue'] * len(simple_names) + ['coral'] * len(comp_names)
    all_names = simple_names + comp_names
    all_success = simple_success + comp_success
    
    bars = ax2.bar(x, all_success, color=colors)
    
    # Add value labels on bars
    for bar, val in zip(bars, all_success):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Evaluation: Simple (blue) vs Compositional (orange)')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.axvline(x=len(simple_names) - 0.5, color='black', linestyle='-', alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 3: Training Loss (middle left) =====
    ax3 = fig.add_subplot(gs[1, 0])
    if episode_losses and any(l > 0 for l in episode_losses):
        ax3.plot(episode_losses, alpha=0.3, color='red', label='Raw Loss')
        if len(episode_losses) >= window:
            smoothed_loss = pd.Series(episode_losses).rolling(window).mean()
            ax3.plot(smoothed_loss, color='darkred', linewidth=2, label=f'Smoothed (w={window})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('DQN Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('DQN Training Loss')
    
    # ===== Plot 4: Episode Length (middle right) =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episode_lengths, alpha=0.3, color='green', label='Raw Length')
    if len(episode_lengths) >= window:
        smoothed_len = pd.Series(episode_lengths).rolling(window).mean()
        ax4.plot(smoothed_len, color='darkgreen', linewidth=2, label=f'Smoothed (w={window})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.set_title('Episode Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Success Rate Over Training (bottom left) =====
    ax5 = fig.add_subplot(gs[2, 0])
    success_rates = []
    for i in range(window, len(episode_rewards)):
        recent = episode_rewards[i-window:i]
        success_rate = np.mean([r > 0 for r in recent])
        success_rates.append(success_rate)
    if success_rates:
        ax5.plot(range(window, window + len(success_rates)), success_rates, 
                color='purple', linewidth=2)
        ax5.axhline(y=np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates),
                   color='purple', linestyle='--', alpha=0.5, label=f'Final: {success_rates[-1]:.1%}')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Success Rate')
    ax5.set_title(f'Training Success Rate (rolling window={window})')
    ax5.set_ylim([0, 1])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ===== Plot 6: Summary Statistics (bottom right) =====
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate summary stats
    avg_simple = np.mean(simple_success)
    avg_comp = np.mean(comp_success)
    final_train_success = success_rates[-1] if success_rates else 0
    final_epsilon = training_history.get('final_epsilon', 0)
    
    summary_text = f"""
    ══════════════════════════════════════
                SUMMARY STATISTICS
    ══════════════════════════════════════
    
    TRAINING (Simple Tasks):
      • Final Success Rate: {final_train_success:.1%}
      • Final Epsilon: {final_epsilon:.4f}
      • Total Episodes: {len(episode_rewards)}
    
    EVALUATION:
      • Simple Tasks Avg: {avg_simple:.1%}
      • Compositional Tasks Avg: {avg_comp:.1%}
      • Performance Drop: {avg_simple - avg_comp:.1%}
    
    ──────────────────────────────────────
    Simple Task Breakdown:
      {', '.join([f'{n}: {s:.0%}' for n, s in zip(simple_names, simple_success)])}
    
    Compositional Task Breakdown:
      {', '.join([f'{n}: {s:.0%}' for n, s in zip(comp_names, comp_success)])}
    ══════════════════════════════════════
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DQN Training & Evaluation Results', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to: {save_path}")


def plot_loss_detailed(episode_losses, save_path, window=100):
    """Detailed loss plot with multiple views"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter out zeros for better visualization
    nonzero_losses = [l for l in episode_losses if l > 0]
    
    # Plot 1: Full loss curve
    ax1 = axes[0, 0]
    ax1.plot(episode_losses, alpha=0.3, color='red', label='Raw')
    if len(episode_losses) >= window:
        smoothed = pd.Series(episode_losses).rolling(window).mean()
        ax1.plot(smoothed, color='darkred', linewidth=2, label=f'Smoothed (w={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale loss
    ax2 = axes[0, 1]
    if nonzero_losses:
        ax2.semilogy(range(len(nonzero_losses)), nonzero_losses, alpha=0.3, color='red')
        if len(nonzero_losses) >= window:
            smoothed = pd.Series(nonzero_losses).rolling(window).mean()
            ax2.semilogy(smoothed, color='darkred', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss distribution
    ax3 = axes[1, 0]
    if nonzero_losses:
        ax3.hist(nonzero_losses, bins=50, color='red', alpha=0.7, edgecolor='darkred')
        ax3.axvline(np.mean(nonzero_losses), color='black', linestyle='--', 
                   label=f'Mean: {np.mean(nonzero_losses):.4f}')
        ax3.axvline(np.median(nonzero_losses), color='blue', linestyle='--',
                   label=f'Median: {np.median(nonzero_losses):.4f}')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Loss Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss over last 1000 episodes (zoomed)
    ax4 = axes[1, 1]
    last_n = min(1000, len(episode_losses))
    recent_losses = episode_losses[-last_n:]
    ax4.plot(range(len(episode_losses) - last_n, len(episode_losses)), 
             recent_losses, alpha=0.5, color='red')
    if len(recent_losses) >= 50:
        smoothed = pd.Series(recent_losses).rolling(50).mean()
        ax4.plot(range(len(episode_losses) - last_n, len(episode_losses)),
                smoothed, color='darkred', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title(f'Loss (Last {last_n} Episodes)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Detailed loss plot saved to: {save_path}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn_simple_tasks(env, agent, total_episodes=5000, max_steps=200, 
                           save_interval=1000, verbose=True,
                           step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train DQN agent on SIMPLE tasks ONLY.
    
    Uses reward shaping to help learning:
    - step_penalty: Small negative reward each step (encourages efficiency)
    - wrong_object_penalty: Penalty for contacting wrong object (encourages selectivity)
    
    Tracks two types of rewards:
    - shaped_reward: What DQN learns from (includes shaping)
    - true_reward: Task success only (for fair plotting/comparison)
    """
    
    print("\n" + "="*60)
    print("TRAINING DQN ON SIMPLE TASKS ONLY")
    print("="*60)
    print(f"Total episodes: {total_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Simple tasks: {[t['name'] for t in SIMPLE_TASKS]}")
    print(f"Reward shaping: step_penalty={step_penalty}, wrong_object_penalty={wrong_object_penalty}")
    print("="*60 + "\n")
    
    # Create training schedule
    training_schedule = create_training_schedule(total_episodes, shuffle=True)
    
    # Tracking
    episode_rewards = []       # TRUE rewards (task success only) - for plotting
    episode_shaped_rewards = [] # Shaped rewards (what DQN sees) - for debugging
    episode_lengths = []
    episode_losses = []
    task_performance = {task['name']: [] for task in SIMPLE_TASKS}
    
    for episode in tqdm(range(total_episodes), desc="Training DQN"):
        current_task = training_schedule[episode]
        env.set_task(current_task)
        
        obs, info = env.reset()
        true_reward_total = 0      # For plotting (task success only)
        shaped_reward_total = 0    # For debugging (what DQN learns from)
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get contacted object (if any)
            contacted_object = info.get('contacted_object', None)
            
            # Check task satisfaction
            task_satisfied = check_task_satisfaction(info, current_task)
            
            # === TRUE REWARD (for plotting) ===
            true_reward = 1.0 if task_satisfied else 0.0
            
            # === SHAPED REWARD (for DQN learning) ===
            if task_satisfied:
                # Success! Big positive reward
                shaped_reward = 1.0
            elif contacted_object is not None:
                # Contacted wrong object - penalty to learn selectivity
                shaped_reward = wrong_object_penalty
            else:
                # Normal step - small penalty to encourage efficiency
                shaped_reward = step_penalty
            
            # Store shaped reward for DQN training
            agent.remember(obs, action, shaped_reward, next_obs, done)
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            # Track both reward types
            true_reward_total += true_reward
            shaped_reward_total += shaped_reward
            obs = next_obs
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Store metrics
        episode_rewards.append(true_reward_total)  # Plot this (fair comparison)
        episode_shaped_rewards.append(shaped_reward_total)  # Debug info
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        task_performance[current_task['name']].append(true_reward_total)
        
        # Logging
        if verbose and (episode + 1) % 500 == 0:
            recent_true_reward = np.mean(episode_rewards[-500:])
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            print(f"Episode {episode+1}: Success={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
        
        # Checkpoints
        if (episode + 1) % save_interval == 0:
            agent.save_model(generate_save_path(f"dqn_checkpoint_ep{episode+1}.pt"))
    
    # Final save
    agent.save_model(generate_save_path("dqn_final.pt"))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for task_name, rewards in task_performance.items():
        success_rate = np.mean([r > 0 for r in rewards])
        print(f"  {task_name}: {success_rate:.1%} success")
    
    return {
        "episode_rewards": episode_rewards,           # TRUE rewards (for plotting)
        "episode_shaped_rewards": episode_shaped_rewards,  # Shaped rewards (for debug)
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "task_performance": task_performance,
        "final_epsilon": agent.epsilon
    }


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_dqn(env, agent, tasks, episodes_per_task=100, max_steps=200, verbose=True):
    """Evaluate DQN agent on given tasks (no training, no exploration)."""
    
    results = {}
    
    for task in tasks:
        if verbose:
            print(f"Evaluating: {task['name']}")
        
        env.set_task(task)
        
        task_rewards = []
        task_successes = []
        task_lengths = []
        
        for _ in range(episodes_per_task):
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(obs, epsilon=0.0)
                obs, env_reward, terminated, truncated, info = env.step(action)
                
                task_satisfied = check_task_satisfaction(info, task)
                if task_satisfied:
                    total_reward += 1.0
                
                if terminated or truncated:
                    break
            
            task_rewards.append(total_reward)
            task_successes.append(total_reward > 0)
            task_lengths.append(step + 1)
        
        results[task['name']] = {
            "mean_reward": np.mean(task_rewards),
            "std_reward": np.std(task_rewards),
            "success_rate": np.mean(task_successes),
            "mean_length": np.mean(task_lengths),
        }
        
        if verbose:
            print(f"  → Success: {results[task['name']]['success_rate']:.1%}")
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_dqn_experiment(
    env_size=10,
    training_episodes=5000,
    eval_episodes_per_task=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run DQN training and evaluation experiment."""
    
    print("\n" + "="*70)
    print("DQN EXPERIMENT: SIMPLE TASK TRAINING → COMPOSITIONAL EVALUATION")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # Create agent
    print("Creating DQN agent...")
    agent = DQNAgent3D(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=50000,
        batch_size=32,
        target_update_freq=1000,
        hidden_size=256,
        use_dueling=True
    )
    
    # ===== TRAINING =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING ON SIMPLE TASKS")
    print("="*60)
    training_history = train_dqn_simple_tasks(
        env, agent,
        total_episodes=training_episodes,
        max_steps=max_steps,
        save_interval=1000,
        verbose=True
    )
    
    # ===== EVALUATION ON SIMPLE TASKS =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION ON SIMPLE TASKS")
    print("="*60)
    simple_results = evaluate_dqn(
        env, agent, SIMPLE_TASKS,
        episodes_per_task=eval_episodes_per_task,
        max_steps=max_steps,
        verbose=True
    )
    
    # ===== EVALUATION ON COMPOSITIONAL TASKS =====
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION ON COMPOSITIONAL TASKS")
    print("="*60)
    comp_results = evaluate_dqn(
        env, agent, COMPOSITIONAL_TASKS,
        episodes_per_task=eval_episodes_per_task,
        max_steps=max_steps,
        verbose=True
    )
    
    # ===== GENERATE PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    eval_results = {
        'simple': simple_results,
        'compositional': comp_results
    }
    
    # Main combined plot
    plot_training_and_evaluation(
        training_history, eval_results,
        generate_save_path("dqn_results_combined.png")
    )
    
    # Detailed loss plot
    plot_loss_detailed(
        training_history["episode_losses"],
        generate_save_path("dqn_loss_detailed.png")
    )
    
    # ===== SAVE RESULTS =====
    all_results = {
        "training": {
            "total_episodes": training_episodes,
            "final_epsilon": training_history["final_epsilon"],
            "final_success_rate": np.mean([r > 0 for r in training_history["episode_rewards"][-100:]]),
        },
        "evaluation_simple": simple_results,
        "evaluation_compositional": comp_results,
        "summary": {
            "avg_simple_success": np.mean([m['success_rate'] for m in simple_results.values()]),
            "avg_comp_success": np.mean([m['success_rate'] for m in comp_results.values()]),
        }
    }
    
    results_path = generate_save_path("experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to: {results_path}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Simple Tasks Average Success:        {all_results['summary']['avg_simple_success']:.1%}")
    print(f"Compositional Tasks Average Success: {all_results['summary']['avg_comp_success']:.1%}")
    print("="*70)
    print("\nOutput files in results_dqn/:")
    print("  • dqn_results_combined.png - Main results plot")
    print("  • dqn_loss_detailed.png - Detailed loss analysis")
    print("  • experiment_results.json - All numerical results")
    print("  • dqn_final.pt - Trained model")
    print("="*70)
    
    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_dqn_experiment(
        env_size=10,
        training_episodes=8000,
        eval_episodes_per_task=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.9995,
        seed=42
    )