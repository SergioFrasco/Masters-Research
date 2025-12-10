"""
DQN Training with SEPARATE MODELS per Task

Each simple task (red, blue, box, sphere) gets its own independent DQN.
This eliminates catastrophic forgetting since models don't interfere with each other.

This serves as a clean baseline to show:
1. DQN CAN learn each simple task individually
2. Performance on compositional tasks (using... which model? This is the problem!)

Key insight: With separate models, we can't easily handle compositional tasks
because "red_box" would need to combine knowledge from the "red" model and "box" model.
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
from utils import generate_save_path


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
# TRAINING FUNCTION FOR SINGLE TASK
# ============================================================================

def train_single_task_dqn(env, task, episodes=2000, max_steps=200, 
                          learning_rate=0.0001, gamma=0.99,
                          epsilon_start=1.0, epsilon_end=0.05,
                          epsilon_decay=0.9995, verbose=True,
                          step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train a fresh DQN agent on a SINGLE task.
    
    IMPORTANT: Each task gets a COMPLETELY FRESH agent with:
    - Fresh neural network weights
    - Fresh replay buffer (empty)
    - Fresh epsilon starting at epsilon_start
    
    Returns:
        agent: Trained DQN agent
        history: Training history dict
    """
    
    task_name = task['name']
    print(f"\n{'='*60}")
    print(f"TRAINING DQN FOR TASK: {task_name.upper()}")
    print(f"{'='*60}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"  Episodes: {episodes}, Max steps: {max_steps}")
    print(f"  Fresh agent with empty replay buffer")
    print(f"{'='*60}")
    
    # Create COMPLETELY FRESH agent for this task
    agent = DQNAgent3D(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=50000,
        batch_size=32,
        target_update_freq=1000,
        hidden_size=256,
        use_dueling=True
    )
    
    # Verify agent is fresh
    assert agent.epsilon == epsilon_start, f"Epsilon should be {epsilon_start}, got {agent.epsilon}"
    assert len(agent.memory) == 0, f"Replay buffer should be empty, got {len(agent.memory)}"
    
    # Set the task (stays constant for all episodes)
    env.set_task(task)
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []  # Track epsilon over time
    
    for episode in tqdm(range(episodes), desc=f"Training '{task_name}'"):
        obs, info = env.reset()
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            contacted_object = info.get('contacted_object', None)
            task_satisfied = check_task_satisfaction(info, task)
            
            # TRUE REWARD
            true_reward = 1.0 if task_satisfied else 0.0
            
            # SHAPED REWARD
            if task_satisfied:
                shaped_reward = 1.0
            elif contacted_object is not None:
                shaped_reward = wrong_object_penalty
            else:
                shaped_reward = step_penalty
            
            agent.remember(obs, action, shaped_reward, next_obs, done)
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            true_reward_total += true_reward
            obs = next_obs
            
            if done:
                break
        
        # Track epsilon BEFORE decay
        episode_epsilons.append(agent.epsilon)
        agent.decay_epsilon()
        
        episode_rewards.append(true_reward_total)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        
        # Periodic logging
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode+1}: Success={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
    
    # Save model
    model_path = generate_save_path(f"dqn_model_{task_name}.pt")
    agent.save_model(model_path)
    
    # Final stats
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    print(f"\nTask '{task_name}' training complete!")
    print(f"  Final success rate (last 100 eps): {final_success:.1%}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Replay buffer size: {len(agent.memory)}")
    print(f"  Model saved to: {model_path}")
    
    return agent, {
        "task_name": task_name,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
        "final_epsilon": agent.epsilon,
        "final_success_rate": final_success
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_agent_on_task(env, agent, task, episodes=100, max_steps=200):
    """Evaluate a single agent on a single task."""
    
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        
        for step in range(max_steps):
            action = agent.select_action(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths)
    }


def evaluate_compositional_tasks(env, trained_agents, episodes=100, max_steps=200):
    """
    Evaluate on compositional tasks using the COLOR MODEL approach.
    
    For "red_box" -> use the "red" model
    For "blue_sphere" -> use the "blue" model
    
    Expected result: ~50% success rate, because the color model will find
    any object of that color, and there's a 50% chance it's the right shape.
    """
    
    results = {}
    
    for comp_task in COMPOSITIONAL_TASKS:
        task_name = comp_task['name']
        features = comp_task['features']
        
        # Use the color model (red or blue)
        color_feature = [f for f in features if f in ['red', 'blue']][0]
        color_agent = trained_agents[color_feature]
        
        print(f"Evaluating '{task_name}' using '{color_feature}' model")
        
        eval_results = evaluate_agent_on_task(env, color_agent, comp_task, episodes, max_steps)
        results[task_name] = {
            'success_rate': eval_results['success_rate'],
            'mean_length': eval_results['mean_length'],
            'std_length': eval_results['std_length'],
            'model_used': color_feature
        }
        
        print(f"  → Success: {eval_results['success_rate']:.1%}")
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_all_training_curves(all_histories, save_path, window=100):
    """Plot training curves for all tasks side by side."""
    
    n_tasks = len(all_histories)
    fig, axes = plt.subplots(2, n_tasks, figsize=(5*n_tasks, 10))
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    for i, (task_name, history) in enumerate(all_histories.items()):
        color = task_colors.get(task_name, 'gray')
        rewards = history['episode_rewards']
        losses = history['episode_losses']
        
        # Top row: Rewards
        ax1 = axes[0, i]
        ax1.plot(rewards, alpha=0.3, color=color)
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(smoothed, color=color, linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f"'{task_name}' - Rewards")
        ax1.grid(True, alpha=0.3)
        
        # Add final success rate
        final_success = history['final_success_rate']
        ax1.text(0.95, 0.05, f'Final: {final_success:.1%}', 
                transform=ax1.transAxes, ha='right', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom row: Loss
        ax2 = axes[1, i]
        if losses and any(l > 0 for l in losses):
            ax2.plot(losses, alpha=0.3, color=color)
            if len(losses) >= window:
                smoothed_loss = pd.Series(losses).rolling(window).mean()
                ax2.plot(smoothed_loss, color=color, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title(f"'{task_name}' - Loss")
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Separate DQN Models - Training Curves per Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_combined_training_curve(all_histories, save_path, window=100):
    """
    Plot all training curves on a single plot as if it were one continuous training run.
    Vertical lines mark where each task's training begins.
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    task_order = ['red', 'blue', 'box', 'sphere']
    
    # Concatenate all rewards and track boundaries
    all_rewards = []
    all_losses = []
    task_boundaries = [0]
    
    for task_name in task_order:
        history = all_histories[task_name]
        all_rewards.extend(history['episode_rewards'])
        all_losses.extend(history['episode_losses'])
        task_boundaries.append(len(all_rewards))
    
    total_episodes = len(all_rewards)
    
    # ===== Plot 1: Rewards over time =====
    ax1 = axes[0]
    
    # Plot each task's rewards in its color
    for i, task_name in enumerate(task_order):
        start = task_boundaries[i]
        end = task_boundaries[i + 1]
        task_rewards = all_rewards[start:end]
        color = task_colors[task_name]
        
        # Raw rewards (faded)
        ax1.plot(range(start, end), task_rewards, alpha=0.15, color=color)
        
        # Smoothed rewards
        if len(task_rewards) >= window:
            smoothed = pd.Series(task_rewards).rolling(window).mean()
            ax1.plot(range(start, end), smoothed, color=color, linewidth=2.5, label=f'{task_name}')
    
    # Add vertical lines at task boundaries
    for i, boundary in enumerate(task_boundaries[1:-1], 1):  # Skip first (0) and last
        ax1.axvline(x=boundary, color='black', linestyle='--', linewidth=2, alpha=0.7)
        # Add task label at the top
        ax1.text(boundary, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1.0, 
                f'→ {task_order[i]}', rotation=90, va='top', ha='right', 
                fontsize=9, fontweight='bold')
    
    # Add task labels in the middle of each block
    for i, task_name in enumerate(task_order):
        start = task_boundaries[i]
        end = task_boundaries[i + 1]
        mid = (start + end) // 2
        ax1.text(mid, -0.15, task_name, ha='center', va='top', fontsize=11, 
                fontweight='bold', color=task_colors[task_name],
                transform=ax1.get_xaxis_transform())
    
    ax1.set_xlabel('Episode (Continuous)')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards Over Time - Separate Models (Sequential Training)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, total_episodes])
    
    # ===== Plot 2: Loss over time =====
    ax2 = axes[1]
    
    # Plot each task's loss in its color
    for i, task_name in enumerate(task_order):
        start = task_boundaries[i]
        end = task_boundaries[i + 1]
        task_losses = all_losses[start:end]
        color = task_colors[task_name]
        
        # Filter out zeros for better visualization
        if any(l > 0 for l in task_losses):
            # Raw losses (faded)
            ax2.plot(range(start, end), task_losses, alpha=0.15, color=color)
            
            # Smoothed losses
            if len(task_losses) >= window:
                smoothed = pd.Series(task_losses).rolling(window).mean()
                ax2.plot(range(start, end), smoothed, color=color, linewidth=2.5, label=f'{task_name}')
    
    # Add vertical lines at task boundaries
    for i, boundary in enumerate(task_boundaries[1:-1], 1):
        ax2.axvline(x=boundary, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Episode (Continuous)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over Time', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, total_episodes])
    
    # Add note about separate models
    fig.text(0.5, 0.02, 
             "Note: Each colored section is a SEPARATE model being trained. "
             "Vertical lines show task switches. No catastrophic forgetting since models are independent.",
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined training curve saved to: {save_path}")


def plot_summary(all_histories, simple_results, comp_results, save_path):
    """Create a comprehensive summary plot."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Row 1: Training curves for each task (4 plots)
    for i, (task_name, history) in enumerate(all_histories.items()):
        ax = fig.add_subplot(gs[0, i])
        color = task_colors.get(task_name, 'gray')
        rewards = history['episode_rewards']
        
        ax.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= 50:
            smoothed = pd.Series(rewards).rolling(50).mean()
            ax.plot(smoothed, color=color, linewidth=2)
        
        final_success = history['final_success_rate']
        ax.set_title(f"'{task_name}'\nFinal: {final_success:.0%}", fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Simple task evaluation (left) and compositional breakdown (right)
    ax_simple = fig.add_subplot(gs[1, :2])
    task_names = list(simple_results.keys())
    success_rates = [simple_results[t]['success_rate'] for t in task_names]
    colors = [task_colors[t] for t in task_names]
    
    bars = ax_simple.bar(task_names, success_rates, color=colors, edgecolor='black')
    for bar, val in zip(bars, success_rates):
        ax_simple.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax_simple.set_ylabel('Success Rate')
    ax_simple.set_title('Simple Tasks Evaluation\n(Each model tested on its trained task)')
    ax_simple.set_ylim([0, 1.15])
    ax_simple.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_simple.grid(True, alpha=0.3, axis='y')
    
    # Compositional results (using color model only)
    ax_comp = fig.add_subplot(gs[1, 2:])
    comp_task_names = list(comp_results.keys())
    comp_success = [comp_results[t]['success_rate'] for t in comp_task_names]
    models_used = [comp_results[t]['model_used'] for t in comp_task_names]
    
    bars = ax_comp.bar(comp_task_names, comp_success, color='coral', edgecolor='black')
    for bar, val, model in zip(bars, comp_success, models_used):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}\n({model})', ha='center', va='bottom', fontsize=8)
    
    ax_comp.set_ylabel('Success Rate')
    ax_comp.set_title('Compositional Tasks\n(Using color model only)')
    ax_comp.set_ylim([0, 1.15])
    ax_comp.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Expected (~50%)')
    ax_comp.tick_params(axis='x', rotation=45)
    ax_comp.legend(loc='upper right')
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Summary statistics and insights
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    avg_simple = np.mean([simple_results[t]['success_rate'] for t in simple_results])
    avg_comp = np.mean([comp_results[t]['success_rate'] for t in comp_results])
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                           SEPARATE MODELS EXPERIMENT SUMMARY                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║  APPROACH: Train 4 independent DQN models, one per simple task                       ║
    ║            For compositional tasks, use the COLOR model (red/blue)                   ║
    ║                                                                                       ║
    ║  SIMPLE TASKS (each model on its own task):                                          ║
    ║    • red:    {simple_results['red']['success_rate']:.1%}    • blue:   {simple_results['blue']['success_rate']:.1%}    • box:    {simple_results['box']['success_rate']:.1%}    • sphere: {simple_results['sphere']['success_rate']:.1%}     ║
    ║    • Average: {avg_simple:.1%}                                                                    ║
    ║                                                                                       ║
    ║  COMPOSITIONAL TASKS (using color model):                                            ║
    ║    • red_box:    {comp_results['red_box']['success_rate']:.1%} (red model)                                                  ║
    ║    • red_sphere: {comp_results['red_sphere']['success_rate']:.1%} (red model)                                                  ║
    ║    • blue_box:   {comp_results['blue_box']['success_rate']:.1%} (blue model)                                                  ║
    ║    • blue_sphere:{comp_results['blue_sphere']['success_rate']:.1%} (blue model)                                                  ║
    ║    • Average: {avg_comp:.1%}  (expected ~50% by chance)                                          ║
    ║                                                                                       ║
    ║  GENERALIZATION GAP: {avg_simple - avg_comp:.1%} drop from simple to compositional               ║
    ║                                                                                       ║
    ║  KEY INSIGHT: The color model finds ANY object of that color. Since there's a        ║
    ║               red_box and red_sphere, it has ~50% chance of finding the right one.   ║
    ║               This shows DQN cannot compose features - it only learned "go to red".  ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('DQN with Separate Models per Task', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_separate_models_experiment(
    env_size=10,
    episodes_per_task=2000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run the separate models experiment."""
    
    print("\n" + "="*70)
    print("DQN SEPARATE MODELS EXPERIMENT")
    print("="*70)
    print("Training one independent DQN per simple task")
    print("Then evaluating on both simple and compositional tasks")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # ===== TRAINING: One model per task =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING SEPARATE MODELS")
    print("="*60)
    
    trained_agents = {}
    all_histories = {}
    
    for task in SIMPLE_TASKS:
        agent, history = train_single_task_dqn(
            env, task,
            episodes=episodes_per_task,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=epsilon_decay,
            verbose=True
        )
        
        trained_agents[task['name']] = agent
        all_histories[task['name']] = history
        
        # Clear memory between tasks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== EVALUATION ON SIMPLE TASKS =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATING ON SIMPLE TASKS")
    print("="*60)
    
    simple_results = {}
    for task in SIMPLE_TASKS:
        task_name = task['name']
        agent = trained_agents[task_name]
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        simple_results[task_name] = results
        print(f"  {task_name}: {results['success_rate']:.1%}")
    
    # ===== EVALUATION ON COMPOSITIONAL TASKS =====
    print("\n" + "="*60)
    print("PHASE 3: EVALUATING ON COMPOSITIONAL TASKS")
    print("="*60)
    
    comp_results = evaluate_compositional_tasks(env, trained_agents, eval_episodes, max_steps)
    
    # ===== GENERATE PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_all_training_curves(all_histories, generate_save_path("training_curves.png"))
    plot_combined_training_curve(all_histories, generate_save_path("training_rewards_over_time.png"))
    plot_summary(all_histories, simple_results, comp_results, generate_save_path("summary.png"))
    
    # ===== SAVE RESULTS =====
    all_results = {
        "training": {task: {
            "episodes": episodes_per_task,
            "final_success_rate": h["final_success_rate"],
            "final_epsilon": h["final_epsilon"]
        } for task, h in all_histories.items()},
        "evaluation_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in simple_results.items()},
        "evaluation_compositional": {t: {
            "success_rate": comp_results[t]["success_rate"],
            "model_used": comp_results[t]["model_used"],
            "mean_length": comp_results[t]["mean_length"]
        } for t in comp_results},
        "summary": {
            "avg_simple_success": np.mean([r["success_rate"] for r in simple_results.values()]),
            "avg_comp_success": np.mean([comp_results[t]["success_rate"] for t in comp_results]),
        }
    }
    
    results_path = generate_save_path("experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Simple Tasks Average Success:        {all_results['summary']['avg_simple_success']:.1%}")
    print(f"Compositional Tasks Average Success: {all_results['summary']['avg_comp_success']:.1%}")
    print(f"Generalization Gap:                  {all_results['summary']['avg_simple_success'] - all_results['summary']['avg_comp_success']:.1%}")
    print("="*70)
    print("\nOutput files in results_dqn_separate/:")
    print("  • training_curves.png - Individual learning curves for each task")
    print("  • training_rewards_over_time.png - Combined view with task boundaries")
    print("  • summary.png - Comprehensive summary")
    print("  • experiment_results.json - All numerical results")
    print("  • dqn_model_<task>.pt - Trained models for each task")
    print("="*70)
    
    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_separate_models_experiment(
        env_size=10,
        episodes_per_task=3000,  # More episodes for better learning
        eval_episodes=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.999,  # Slower decay: 0.999^3000 ≈ 0.05
        seed=42
    )