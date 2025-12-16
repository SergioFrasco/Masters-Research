"""
Unified World Value Functions (WVF) Training Script

KEY CHANGE: Single unified model instead of 4 separate models
- Random task sampling each episode to prevent catastrophic forgetting
- Task conditioning: primitive task encoded as one-hot and tiled
- Same composition logic at evaluation time

Based on Nangue Tasse et al.'s Boolean Task Algebra (NeurIPS 2020)
"""

import os
import sys

# Force headless mode
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm
import json
import gc
import torch

from env import DiscreteMiniWorldWrapper
from agents import UnifiedWorldValueFunctionAgent
from utils import generate_save_path


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

PRIMITIVE_TASKS = {
    "red": {"name": "red", "features": ["red"], "type": "primitive"},
    "blue": {"name": "blue", "features": ["blue"], "type": "primitive"},
    "box": {"name": "box", "features": ["box"], "type": "primitive"},
    "sphere": {"name": "sphere", "features": ["sphere"], "type": "primitive"},
}

# These are NEVER seen during training - only used for zero-shot evaluation
COMPOSITIONAL_TASKS = {
    "red_box": {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    "red_sphere": {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    "blue_box": {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    "blue_sphere": {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
}


def check_primitive_satisfaction(contacted_object, primitive):
    """Check if contacted object satisfies primitive task."""
    if contacted_object is None:
        return False
    
    valid_goals = UnifiedWorldValueFunctionAgent.VALID_GOALS[primitive]
    return contacted_object in valid_goals


def check_compositional_satisfaction(contacted_object, task_name):
    """Check if contacted object satisfies compositional task (exact match)."""
    return contacted_object == task_name


# ============================================================================
# TRAINING
# ============================================================================

def train_unified_wvf(env, agent, total_episodes=8000, max_steps=200,
                      train_every=4, verbose=True):
    """
    Train UNIFIED WVF with random task sampling each episode.
    
    KEY CHANGE:
    - Sample a random primitive task at the start of each episode
    - This prevents catastrophic forgetting
    - All tasks train the same unified network
    """
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFIED WVF")
    print(f"{'='*60}")
    print(f"Total episodes: {total_episodes}")
    print(f"Random task sampling each episode")
    print(f"{'='*60}")
    
    # Tracking per task
    task_episode_counts = {p: 0 for p in agent.PRIMITIVES}
    task_success_rates = {p: [] for p in agent.PRIMITIVES}
    task_episode_rewards = {p: [] for p in agent.PRIMITIVES}
    
    # Global tracking
    global_losses = []
    global_epsilons = []
    episode_lengths = []
    
    # Track which goals are reached per task
    goal_reached_counts = {p: {g: 0 for g in agent.GOALS} for p in agent.PRIMITIVES}
    goal_conditioned_counts = {p: {g: 0 for g in agent.GOALS} for p in agent.PRIMITIVES}
    
    for episode in tqdm(range(total_episodes), desc="Training Unified WVF"):
        # CRITICAL: Sample random task for this episode
        current_task = agent.sample_task()
        task_idx = agent.TASK_TO_IDX[current_task]
        task_episode_counts[current_task] += 1
        
        # Set environment task
        task_config = PRIMITIVE_TASKS[current_task]
        env.set_task(task_config)
        
        # Reset environment
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        # Sample target goal from entire goal space
        target_goal_idx = agent.sample_target_goal(current_task)
        target_goal_name = agent.IDX_TO_GOAL[target_goal_idx]
        goal_conditioned_counts[current_task][target_goal_name] += 1
        
        true_reward_episode = 0.0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action conditioned on BOTH goal AND task
            action = agent.select_action(stacked_obs, target_goal_idx, task_idx)
            
            # Step environment
            next_obs, _, terminated, truncated, info = env.step(action)
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Compute rewards
            true_reward, extended_reward, goal_reached = agent.compute_rewards(
                info, target_goal_idx, current_task
            )
            
            # Track TRUE reward
            true_reward_episode = max(true_reward_episode, true_reward)
            
            # Track which goal was reached
            contacted = info.get('contacted_object', None)
            if contacted in goal_reached_counts[current_task]:
                goal_reached_counts[current_task][contacted] += 1
            
            done = goal_reached or terminated or truncated
            
            # Store with EXTENDED reward and task index
            agent.remember(stacked_obs, target_goal_idx, task_idx, action,
                          extended_reward, next_stacked_obs, done)
            
            # Train periodically
            if step % train_every == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
            
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Track per-task metrics
        task_episode_rewards[current_task].append(true_reward_episode)
        
        # Global tracking
        episode_lengths.append(step + 1)
        if episode_loss:
            global_losses.append(np.mean(episode_loss))
        else:
            global_losses.append(0.0)
        global_epsilons.append(agent.epsilon)
        
        # Compute rolling success rates per task
        if len(task_episode_rewards[current_task]) >= 100:
            recent_success = np.mean(task_episode_rewards[current_task][-100:])
            task_success_rates[current_task].append(recent_success)
        
        # Periodic logging
        if verbose and (episode + 1) % 500 == 0:
            print(f"\n  Episode {episode+1}:")
            print(f"    Task counts: {task_episode_counts}")
            for task in agent.PRIMITIVES:
                if len(task_episode_rewards[task]) > 0:
                    recent = task_episode_rewards[task][-min(100, len(task_episode_rewards[task])):]
                    success_rate = np.mean(recent)
                    print(f"      {task}: {success_rate:.1%} ({len(recent)} recent episodes)")
            print(f"    Epsilon: {agent.epsilon:.3f}")
    
    # Final statistics
    print(f"\nUnified WVF training complete!")
    print(f"\nTask distribution:")
    for task, count in task_episode_counts.items():
        print(f"  {task}: {count} episodes ({count/total_episodes:.1%})")
    
    print(f"\nFinal success rates (last 100 episodes per task):")
    final_success_rates = {}
    for task in agent.PRIMITIVES:
        if len(task_episode_rewards[task]) >= 100:
            final_success = np.mean(task_episode_rewards[task][-100:])
            final_success_rates[task] = final_success
            print(f"  {task}: {final_success:.1%}")
            print(f"    Goals reached: {goal_reached_counts[task]}")
    
    return {
        "task_episode_counts": task_episode_counts,
        "task_episode_rewards": task_episode_rewards,
        "task_success_rates": task_success_rates,
        "global_losses": global_losses,
        "global_epsilons": global_epsilons,
        "episode_lengths": episode_lengths,
        "goal_reached_counts": goal_reached_counts,
        "goal_conditioned_counts": goal_conditioned_counts,
        "final_success_rates": final_success_rates,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_primitive_unified(env, agent, primitive, episodes=100, max_steps=200):
    """
    Evaluate unified model on primitive task.
    
    We condition on the specific task and maximize over valid goals.
    """
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    task_idx = agent.TASK_TO_IDX[primitive]
    valid_goal_indices = agent.get_valid_goal_indices(primitive)
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        # For evaluation, we can either:
        # 1. Pick a specific valid goal and stick with it
        # 2. Maximize over valid goals at each step
        # Let's use option 2 for better exploration
        
        ep_reward = 0.0
        
        for step in range(max_steps):
            # Evaluate Q(s, g, task, a) for all valid goals, pick max action
            state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(agent.device)
            task_one_hot = agent.get_task_one_hot(task_idx)
            
            best_action = None
            best_q = float('-inf')
            
            with torch.no_grad():
                for goal_idx in valid_goal_indices:
                    goal_one_hot = agent.get_goal_one_hot(goal_idx)
                    hidden = agent.q_network.init_hidden(1, agent.device)
                    q_vals, _ = agent.q_network(state, goal_one_hot, task_one_hot, hidden)
                    max_q = q_vals.max().item()
                    
                    if max_q > best_q:
                        best_q = max_q
                        best_action = q_vals.argmax().item()
            
            action = best_action
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            contacted = info.get('contacted_object', None)
            
            if check_primitive_satisfaction(contacted, primitive):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1.0
                break
            
            if contacted is not None or terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "episode_rewards": episode_rewards
    }


def evaluate_compositional_unified(env, agent, task_name, episodes=100, max_steps=200):
    """
    Zero-shot evaluation on compositional task using Boolean composition.
    
    For conjunction (AND):
        Q_composed(s, g, a) = min(Q_feature1(s, g, a), Q_feature2(s, g, a))
        action = argmax_a Q_composed(s, target_goal, a)
    
    The agent has NEVER trained on this task - this is zero-shot generalization.
    """
    task = COMPOSITIONAL_TASKS[task_name]
    features = task["features"]
    
    env.set_task(task)
    
    # The target goal IS the task name (e.g., "red_box")
    target_goal_idx = agent.GOAL_TO_IDX[task_name]
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        ep_reward = 0.0
        
        for step in range(max_steps):
            # Zero-shot composed action selection
            action = agent.select_action_composed(stacked_obs, features, target_goal_idx)
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            contacted = info.get('contacted_object', None)
            
            if check_compositional_satisfaction(contacted, task_name):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1.0
                break
            
            if contacted is not None or terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "features_used": features,
        "episode_rewards": episode_rewards
    }


def evaluate_all_unified(env, agent, episodes=100, max_steps=200):
    """Evaluate on all tasks."""
    
    print("\n" + "="*60)
    print("EVALUATING PRIMITIVES")
    print("="*60)
    
    primitive_results = {}
    for primitive in agent.PRIMITIVES:
        results = evaluate_primitive_unified(env, agent, primitive, episodes, max_steps)
        primitive_results[primitive] = results
        print(f"  {primitive}: {results['success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("ZERO-SHOT COMPOSITIONAL EVALUATION")
    print("(These tasks were NEVER seen during training)")
    print("="*60)
    
    compositional_results = {}
    for task_name in COMPOSITIONAL_TASKS:
        results = evaluate_compositional_unified(env, agent, task_name, episodes, max_steps)
        compositional_results[task_name] = results
        print(f"  {task_name}: {results['success_rate']:.1%} "
              f"(min of {results['features_used']})")
    
    return primitive_results, compositional_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_rewards_over_time(history, save_path, window=100):
    """Plot simple rewards over time (all episodes)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Flatten all rewards into chronological order
    # We need to reconstruct the chronological order from the history
    total_episodes = sum(history['task_episode_counts'].values())
    
    # Create arrays to store chronological data
    all_rewards = []
    all_tasks = []
    
    # We'll reconstruct episode order by tracking how many episodes of each task we've seen
    task_counters = {p: 0 for p in UnifiedWorldValueFunctionAgent.PRIMITIVES}
    
    # Read from task_episode_rewards in the order they would have been sampled
    # Note: Since we randomly sampled, we lost the exact order, but we can approximate
    # by interleaving based on the counts
    
    # Actually, let's just concatenate all rewards and color by task
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Top plot: All rewards colored by task
    episode_num = 0
    for primitive in UnifiedWorldValueFunctionAgent.PRIMITIVES:
        rewards = history['task_episode_rewards'][primitive]
        episodes = list(range(episode_num, episode_num + len(rewards)))
        episode_num += len(rewards)
        
        ax1.scatter(episodes, rewards, alpha=0.3, s=10, color=colors[primitive], label=primitive)
        
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(episodes, smoothed, color=colors[primitive], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Episode Number', fontsize=11)
    ax1.set_ylabel('Reward (0 or 1)', fontsize=11)
    ax1.set_title('Rewards Over Time (All Tasks)', fontsize=12, fontweight='bold')
    ax1.set_ylim([-0.1, 1.1])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Bottom plot: Moving average across ALL tasks combined
    # Flatten rewards in approximate chronological order
    all_task_rewards = []
    for primitive in UnifiedWorldValueFunctionAgent.PRIMITIVES:
        all_task_rewards.extend(history['task_episode_rewards'][primitive])
    
    ax2.plot(all_task_rewards, alpha=0.3, color='purple', linewidth=0.5)
    if len(all_task_rewards) >= window:
        smoothed_all = pd.Series(all_task_rewards).rolling(window).mean()
        ax2.plot(smoothed_all, color='purple', linewidth=3, label=f'Smoothed (window={window})')
    
    ax2.set_xlabel('Episode Number (Approximate)', fontsize=11)
    ax2.set_ylabel('Reward (0 or 1)', fontsize=11)
    ax2.set_title('Overall Training Progress (All Tasks Combined)', fontsize=12, fontweight='bold')
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('Unified WVF - Rewards Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rewards over time plot saved to: {save_path}")


def plot_unified_training_curves(history, save_path, window=100):
    """Plot training curves for unified model."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Top row: Per-task success rates
    for i, primitive in enumerate(UnifiedWorldValueFunctionAgent.PRIMITIVES):
        ax = axes[0, i]
        rewards = history['task_episode_rewards'][primitive]
        
        if len(rewards) > 0:
            # Plot raw rewards
            ax.plot(rewards, alpha=0.3, color=colors[primitive], linewidth=0.5)
            
            # Plot smoothed
            if len(rewards) >= window:
                smoothed = pd.Series(rewards).rolling(window).mean()
                ax.plot(smoothed, color=colors[primitive], linewidth=2, label=f'Smoothed ({window})')
            
            # Final success rate
            if len(rewards) >= 100:
                final_success = np.mean(rewards[-100:])
                ax.axhline(y=final_success, color='black', linestyle='--', linewidth=1,
                          label=f'Final: {final_success:.0%}')
        
        ax.set_xlabel('Episode (for this task)', fontsize=9)
        ax.set_ylabel('Success Rate', fontsize=9)
        ax.set_title(f"'{primitive}' Training\n({history['task_episode_counts'][primitive]} total episodes)",
                    fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    
    # Bottom row: Global metrics
    # Loss
    ax = axes[1, 0]
    losses = history['global_losses']
    if losses and any(l > 0 for l in losses):
        ax.plot(losses, alpha=0.3, color='purple')
        if len(losses) >= window:
            smoothed = pd.Series(losses).rolling(window).mean()
            ax.plot(smoothed, color='purple', linewidth=2)
    ax.set_xlabel('Global Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (All Tasks)')
    ax.grid(True, alpha=0.3)
    
    # Epsilon
    ax = axes[1, 1]
    ax.plot(history['global_epsilons'], color='navy', linewidth=1)
    ax.set_xlabel('Global Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.grid(True, alpha=0.3)
    
    # Episode length
    ax = axes[1, 2]
    lengths = history['episode_lengths']
    if len(lengths) >= window:
        smoothed = pd.Series(lengths).rolling(window).mean()
        ax.plot(smoothed, color='teal', linewidth=2)
    ax.set_xlabel('Global Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Task distribution
    ax = axes[1, 3]
    task_counts = history['task_episode_counts']
    ax.bar(task_counts.keys(), task_counts.values(),
           color=[colors[t] for t in task_counts.keys()], edgecolor='black')
    ax.set_ylabel('Number of Episodes')
    ax.set_title('Task Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Unified WVF - Training with Random Task Sampling\n' +
                 '(Single model, task-conditioned, prevents catastrophic forgetting)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_evaluation_comparison(primitive_results, compositional_results, save_path):
    """Plot evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Primitives
    ax1 = axes[0]
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    colors = ['red', 'blue', 'orange', 'green']
    
    bars = ax1.bar(primitives, prim_success, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, prim_success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Primitive Tasks\n(Trained - Unified Model)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.15])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper right')
    
    # Compositional (zero-shot)
    ax2 = axes[1]
    comp_tasks = list(compositional_results.keys())
    comp_success = [compositional_results[t]['success_rate'] for t in comp_tasks]
    
    bars = ax2.bar(comp_tasks, comp_success, color='coral', edgecolor='black', linewidth=1.5)
    for bar, val, task in zip(bars, comp_success, comp_tasks):
        features = compositional_results[task]['features_used']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}\nmin({features[0]},{features[1]})',
                ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title('Compositional Tasks\n(Zero-Shot - Never Trained)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random baseline (1/4)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    plt.suptitle(f'Unified WVF Evaluation\nPrimitives: {avg_prim:.0%} | '
                 f'Zero-Shot Compositional: {avg_comp:.0%} | Gap: {avg_prim - avg_comp:.0%}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation comparison saved to: {save_path}")


def plot_full_summary(history, primitive_results, compositional_results, save_path):
    """Comprehensive summary plot."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Row 1: Training curves per task
    for i, primitive in enumerate(UnifiedWorldValueFunctionAgent.PRIMITIVES):
        ax = fig.add_subplot(gs[0, i])
        rewards = history['task_episode_rewards'][primitive]
        
        if len(rewards) > 0:
            ax.plot(rewards, alpha=0.2, color=colors[primitive])
            if len(rewards) >= 100:
                smoothed = pd.Series(rewards).rolling(100).mean()
                ax.plot(smoothed, color=colors[primitive], linewidth=2)
            
            if primitive in history['final_success_rates']:
                final = history['final_success_rates'][primitive]
                ax.set_title(f"'{primitive}'\nFinal: {final:.0%}", fontsize=10)
        
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Success', fontsize=8)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
    
    # Row 2: Evaluation comparisons
    ax_prim = fig.add_subplot(gs[1, :2])
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    prim_colors = [colors[p] for p in primitives]
    
    bars = ax_prim.bar(primitives, prim_success, color=prim_colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax_prim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax_prim.set_ylabel('Success Rate')
    ax_prim.set_title('Primitive Tasks (Trained - Unified Model)', fontweight='bold')
    ax_prim.set_ylim([0, 1.15])
    ax_prim.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_prim.grid(True, alpha=0.3, axis='y')
    
    ax_comp = fig.add_subplot(gs[1, 2:])
    comp_tasks = list(compositional_results.keys())
    comp_success = [compositional_results[t]['success_rate'] for t in comp_tasks]
    
    bars = ax_comp.bar(comp_tasks, comp_success, color='coral', edgecolor='black')
    for bar, val in zip(bars, comp_success):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax_comp.set_ylabel('Success Rate')
    ax_comp.set_title('Compositional Tasks (Zero-Shot)', fontweight='bold')
    ax_comp.set_ylim([0, 1.15])
    ax_comp.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    ax_comp.tick_params(axis='x', rotation=45)
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Task distribution and goals reached
    ax_dist = fig.add_subplot(gs[2, :2])
    task_counts = history['task_episode_counts']
    ax_dist.bar(task_counts.keys(), task_counts.values(),
               color=[colors[t] for t in task_counts.keys()], edgecolor='black')
    ax_dist.set_ylabel('Episodes')
    ax_dist.set_title('Training Task Distribution', fontweight='bold')
    ax_dist.grid(True, alpha=0.3, axis='y')
    
    # Goals reached per task
    ax_goals = fig.add_subplot(gs[2, 2:])
    goal_names = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    goal_colors_map = {'red_box': 'darkred', 'blue_box': 'darkblue',
                       'red_sphere': 'lightcoral', 'blue_sphere': 'lightblue'}
    
    x = np.arange(len(UnifiedWorldValueFunctionAgent.PRIMITIVES))
    width = 0.2
    
    for j, goal in enumerate(goal_names):
        counts = [history['goal_reached_counts'][p].get(goal, 0)
                  for p in UnifiedWorldValueFunctionAgent.PRIMITIVES]
        ax_goals.bar(x + j*width, counts, width, label=goal, color=goal_colors_map[goal])
    
    ax_goals.set_xlabel('Primitive Task')
    ax_goals.set_ylabel('Goals Reached (count)')
    ax_goals.set_title('Goal Distribution During Training', fontweight='bold')
    ax_goals.set_xticks(x + width*1.5)
    ax_goals.set_xticklabels(UnifiedWorldValueFunctionAgent.PRIMITIVES)
    ax_goals.legend(loc='upper right', fontsize=8)
    ax_goals.grid(True, alpha=0.3, axis='y')
    
    # Row 4: Summary text
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    summary_text = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                         UNIFIED WORLD VALUE FUNCTIONS (WVF) EXPERIMENT                                         ║
    ║                    Single Model with Task Conditioning - Zero-Shot Composition                                 ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                ║
    ║  KEY INNOVATION:                                                                                               ║
    ║    • UNIFIED model instead of 4 separate models                                                               ║
    ║    • Task conditioning: primitive task encoded as one-hot and tiled across image                              ║
    ║    • Random task sampling each episode prevents catastrophic forgetting                                       ║
    ║    • Extended reward: +1 for ANY valid goal that satisfies the task                                           ║
    ║                                                                                                                ║
    ║  PRIMITIVE RESULTS (trained on unified model):                                                                 ║
    ║    • red:    {primitive_results['red']['success_rate']:.1%}    • blue:   {primitive_results['blue']['success_rate']:.1%}    • box:    {primitive_results['box']['success_rate']:.1%}    • sphere: {primitive_results['sphere']['success_rate']:.1%}              ║
    ║    • Average: {avg_prim:.1%}                                                                                          ║
    ║                                                                                                                ║
    ║  ZERO-SHOT COMPOSITIONAL RESULTS (never trained, composed via min):                                           ║
    ║    • red_box:     {compositional_results['red_box']['success_rate']:.1%} = min(Q_red, Q_box)                                                            ║
    ║    • red_sphere:  {compositional_results['red_sphere']['success_rate']:.1%} = min(Q_red, Q_sphere)                                                         ║
    ║    • blue_box:    {compositional_results['blue_box']['success_rate']:.1%} = min(Q_blue, Q_box)                                                           ║
    ║    • blue_sphere: {compositional_results['blue_sphere']['success_rate']:.1%} = min(Q_blue, Q_sphere)                                                        ║
    ║    • Average: {avg_comp:.1%}                                                                                          ║
    ║                                                                                                                ║
    ║  GENERALIZATION GAP: {avg_prim - avg_comp:.1%}                                                                                ║
    ║                                                                                                                ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Unified World Value Functions - Full Summary',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Full summary saved to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_unified_wvf_experiment(
    env_size=10,
    total_episodes=8000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run the UNIFIED World Value Functions experiment."""
    
    print("\n" + "="*70)
    print("UNIFIED WORLD VALUE FUNCTIONS (WVF) EXPERIMENT")
    print("="*70)
    print("Single Model with Task Conditioning")
    print("Zero-Shot Compositional Generalization via Boolean Task Algebra")
    print("Based on Nangue Tasse et al. (NeurIPS 2020)")
    print("="*70)
    print("\nKEY INNOVATION:")
    print("  • Unified model (not 4 separate models)")
    print("  • Task conditioning: tiled as extra channels")
    print("  • Random task sampling prevents catastrophic forgetting")
    print("  • Extended reward: +1 for ANY valid goal")
    print("="*70 + "\n")
    
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # Unified WVF Agent
    print("Creating UnifiedWorldValueFunctionAgent...")
    agent = UnifiedWorldValueFunctionAgent(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=2000,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        tau=0.005,
        grad_clip=10.0,
        r_correct=1.0,
        r_wrong=-1.0,
        step_penalty=-0.01
    )
    
    # Training
    print("\n" + "="*60)
    print("PHASE 1: TRAINING UNIFIED MODEL")
    print("="*60)
    
    history = train_unified_wvf(
        env, agent,
        total_episodes=total_episodes,
        max_steps=max_steps
    )
    
    # Save model
    model_path = generate_save_path("unified_wvf_model.pt")
    agent.save_model(model_path)
    
    # Evaluation
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION (including zero-shot compositional)")
    print("="*60)
    
    primitive_results, compositional_results = evaluate_all_unified(
        env, agent, episodes=eval_episodes, max_steps=max_steps
    )
    
    # Plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_rewards_over_time(history, generate_save_path("unified_wvf_rewards_over_time.png"))
    plot_unified_training_curves(history, generate_save_path("unified_wvf_training.png"))
    plot_evaluation_comparison(primitive_results, compositional_results,
                               generate_save_path("unified_wvf_evaluation.png"))
    plot_full_summary(history, primitive_results, compositional_results,
                     generate_save_path("unified_wvf_summary.png"))
    
    # Save results
    results = {
        "method": "Unified World Value Functions",
        "innovation": "Single model with task conditioning, random task sampling per episode",
        "training": {
            "total_episodes": total_episodes,
            "task_distribution": history["task_episode_counts"],
            "final_success_rates": history["final_success_rates"],
        },
        "evaluation_primitives": {
            p: {"success_rate": r["success_rate"], "mean_length": r["mean_length"]}
            for p, r in primitive_results.items()
        },
        "evaluation_compositional_zero_shot": {
            t: {
                "success_rate": r["success_rate"],
                "mean_length": r["mean_length"],
                "composition": f"min({r['features_used'][0]}, {r['features_used'][1]})"
            }
            for t, r in compositional_results.items()
        },
        "summary": {
            "avg_primitive_success": np.mean([r["success_rate"] for r in primitive_results.values()]),
            "avg_compositional_success": np.mean([r["success_rate"] for r in compositional_results.values()]),
        }
    }
    
    results["summary"]["generalization_gap"] = (
        results["summary"]["avg_primitive_success"] -
        results["summary"]["avg_compositional_success"]
    )
    
    results_path = generate_save_path("unified_wvf_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("UNIFIED WVF EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Primitive Tasks Average:              {results['summary']['avg_primitive_success']:.1%}")
    print(f"Zero-Shot Compositional Average:      {results['summary']['avg_compositional_success']:.1%}")
    print(f"Generalization Gap:                   {results['summary']['generalization_gap']:.1%}")
    print("="*70)
    
    return results, agent


if __name__ == "__main__":
    results, agent = run_unified_wvf_experiment(
        env_size=10,
        total_episodes=8000,  # 2000 per task on average
        eval_episodes=200,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.999,
        seed=42
    )