"""
Corrected World Value Functions (WVF) Training Script

KEY FIXES:
1. Sample goals from ENTIRE goal space during training
2. Track TRUE environment rewards (0/1) separately from extended rewards
3. Proper plotting showing real success rates
4. Correct composition at evaluation time

CLUSTER-FRIENDLY:
- Episode-based replay buffer (2000 episodes)
- Small LSTM (64 units)
- Small batch size (16)
- Networks cleared between primitive training

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
from collections import deque
from tqdm import tqdm
import json
import gc
import torch

from env import DiscreteMiniWorldWrapper
from agents import WorldValueFunctionAgent
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
    
    valid_goals = WorldValueFunctionAgent.VALID_GOALS[primitive]
    return contacted_object in valid_goals


def check_compositional_satisfaction(contacted_object, task_name):
    """Check if contacted object satisfies compositional task (exact match)."""
    return contacted_object == task_name


# ============================================================================
# TRAINING
# ============================================================================

def train_primitive_wvf(env, agent, primitive, episodes=2000, max_steps=200,
                        train_every=4, verbose=True):
    """
    Train WVF on a single primitive task.
    
    KEY DIFFERENCES from original:
    1. Sample goals from ENTIRE goal space (all 4 objects)
    2. Extended reward depends on: did we reach the goal we conditioned on?
    3. Track TRUE task completion (0/1) separately for plotting
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WVF FOR PRIMITIVE: {primitive.upper()}")
    print(f"{'='*60}")
    
    agent.set_training_primitive(primitive)
    
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    # Tracking - TRUE env rewards only (did we complete the primitive task?)
    episode_true_rewards = []  # 0 or 1 per episode
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []
    
    # Track which goals the agent reaches (for debugging)
    goal_reached_counts = {g: 0 for g in agent.GOALS}
    goal_conditioned_counts = {g: 0 for g in agent.GOALS}
    
    for episode in tqdm(range(episodes), desc=f"Training '{primitive}'"):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        # Sample target goal from ENTIRE goal space (not just valid goals!)
        target_goal_idx = agent.sample_target_goal()
        target_goal_name = agent.IDX_TO_GOAL[target_goal_idx]
        goal_conditioned_counts[target_goal_name] += 1
        
        true_reward_episode = 0.0  # Track TRUE task success
        episode_loss = []
        
        for step in range(max_steps):
            # Select action conditioned on target goal
            action = agent.select_action(stacked_obs, target_goal_idx)
            
            # Step environment
            next_obs, _, terminated, truncated, info = env.step(action)
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Compute rewards
            true_reward, extended_reward, goal_reached = agent.compute_rewards(
                info, target_goal_idx
            )
            
            # Track TRUE reward for plotting
            true_reward_episode = max(true_reward_episode, true_reward)
            
            # Track which goal was reached
            contacted = info.get('contacted_object', None)
            if contacted in goal_reached_counts:
                goal_reached_counts[contacted] += 1
            
            done = goal_reached or terminated or truncated
            
            # Store with EXTENDED reward (for training)
            agent.remember(stacked_obs, target_goal_idx, action, extended_reward,
                          next_stacked_obs, done)
            
            # Train periodically
            if step % train_every == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
            
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Track TRUE rewards (for plotting)
        episode_true_rewards.append(true_reward_episode)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        episode_epsilons.append(agent.epsilon)
        
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean(episode_true_rewards[-500:])
            recent_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode+1}: Success Rate={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
            print(f"    Goals reached: {goal_reached_counts}")
            print(f"    Goals conditioned: {goal_conditioned_counts}")
    
    # Save the trained network
    agent.save_trained_network(primitive)
    
    final_success = np.mean(episode_true_rewards[-100:])
    print(f"\nPrimitive '{primitive}' training complete!")
    print(f"  Final success rate (last 100): {final_success:.1%}")
    print(f"  Total goals reached: {goal_reached_counts}")
    
    return {
        "primitive": primitive,
        "episode_rewards": episode_true_rewards,  # TRUE rewards only!
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
        "goal_reached_counts": goal_reached_counts,
        "goal_conditioned_counts": goal_conditioned_counts,
        "final_success_rate": final_success,
    }


def train_all_primitives_wvf(env, agent, episodes_per_primitive=2000, max_steps=200):
    """Train all primitives sequentially."""
    all_histories = {}
    
    for primitive in agent.PRIMITIVES:
        history = train_primitive_wvf(
            env, agent, primitive,
            episodes=episodes_per_primitive,
            max_steps=max_steps
        )
        all_histories[primitive] = history
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_histories


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_primitive_wvf(env, agent, primitive, episodes=100, max_steps=200):
    """
    Evaluate WVF on primitive task.
    
    Policy: argmax_a max_{g in valid_goals} Q(s, g, a)
    
    We only consider valid goals for this primitive, which makes sense
    because the agent was trained to value those goals highly.
    """
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    # Load the trained network for this primitive
    agent.load_trained_network(primitive)
    
    # Get valid goal indices for this primitive
    valid_goal_indices = agent.get_valid_goal_indices(primitive)
    
    successes = []
    lengths = []
    episode_rewards = []  # For plotting evaluation performance
    
    for ep in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        ep_reward = 0.0
        
        for step in range(max_steps):
            # Use greedy policy over VALID goals only
            action = agent.select_action_greedy_over_goals(stacked_obs, valid_goal_indices)
            
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


def evaluate_compositional_wvf(env, agent, task_name, episodes=100, max_steps=200):
    """
    Zero-shot evaluation on compositional task using Boolean composition.
    
    For conjunction (AND):
        Q_composed(s, g, a) = min(Q_feature1(s, g, a), Q_feature2(s, g, a))
        action = argmax_a Q_composed(s, target_goal, a)
    
    Since compositional task has exactly ONE valid goal (e.g., red_box),
    we directly use that goal instead of max over goals.
    
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
            # Zero-shot composed action selection for specific target goal
            action = agent.select_action_composed_for_goal(stacked_obs, features, target_goal_idx)
            
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


def evaluate_all_wvf(env, agent, episodes=100, max_steps=200):
    """Evaluate on all tasks."""
    
    print("\n" + "="*60)
    print("EVALUATING PRIMITIVES")
    print("="*60)
    
    primitive_results = {}
    for primitive in agent.PRIMITIVES:
        results = evaluate_primitive_wvf(env, agent, primitive, episodes, max_steps)
        primitive_results[primitive] = results
        print(f"  {primitive}: {results['success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("ZERO-SHOT COMPOSITIONAL EVALUATION")
    print("(These tasks were NEVER seen during training)")
    print("="*60)
    
    compositional_results = {}
    for task_name in COMPOSITIONAL_TASKS:
        results = evaluate_compositional_wvf(env, agent, task_name, episodes, max_steps)
        compositional_results[task_name] = results
        print(f"  {task_name}: {results['success_rate']:.1%} "
              f"(min of {results['features_used']})")
    
    return primitive_results, compositional_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(all_histories, save_path, window=100):
    """Plot training curves showing TRUE environment rewards."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    for i, (primitive, history) in enumerate(all_histories.items()):
        color = colors[primitive]
        rewards = history['episode_rewards']  # TRUE rewards (0 or 1)
        losses = history['episode_losses']
        
        # Top row: Success rate (smoothed rewards)
        ax1 = axes[0, i]
        ax1.plot(rewards, alpha=0.2, color=color, label='Raw')
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(smoothed, color=color, linewidth=2, label=f'Smoothed ({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate')
        ax1.set_title(f"'{primitive}' - Training Success")
        ax1.set_ylim([-0.05, 1.05])
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=8)
        
        # Bottom row: Loss
        ax2 = axes[1, i]
        if losses and any(l > 0 for l in losses):
            ax2.plot(losses, alpha=0.3, color=color)
            if len(losses) >= window:
                smoothed = pd.Series(losses).rolling(window).mean()
                ax2.plot(smoothed, color=color, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title(f"'{primitive}' - Training Loss")
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Corrected WVF - Training on Primitive Tasks\n(Showing TRUE environment rewards)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_evaluation_comparison(primitive_results, compositional_results, save_path):
    """Plot evaluation results comparing primitives and zero-shot compositional."""
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
    ax1.set_title('Primitive Tasks\n(Trained)', fontsize=12, fontweight='bold')
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
    
    plt.suptitle(f'Corrected WVF Evaluation\nPrimitives: {avg_prim:.0%} | '
                 f'Zero-Shot Compositional: {avg_comp:.0%} | Gap: {avg_prim - avg_comp:.0%}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation comparison saved to: {save_path}")


def plot_evaluation_episode_rewards(primitive_results, compositional_results, save_path):
    """Plot evaluation episode rewards over time."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Top row: Primitives
    for i, (prim, results) in enumerate(primitive_results.items()):
        ax = axes[0, i]
        rewards = results['episode_rewards']
        ax.plot(rewards, 'o', alpha=0.5, color=colors[prim], markersize=3)
        ax.axhline(y=results['success_rate'], color=colors[prim], linewidth=2, 
                  label=f"Mean: {results['success_rate']:.0%}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (0 or 1)')
        ax.set_title(f"'{prim}' Evaluation", fontsize=10)
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Compositional
    comp_list = list(compositional_results.items())
    for i, (task, results) in enumerate(comp_list):
        ax = axes[1, i]
        rewards = results['episode_rewards']
        ax.plot(rewards, 'o', alpha=0.5, color='coral', markersize=3)
        ax.axhline(y=results['success_rate'], color='coral', linewidth=2,
                  label=f"Mean: {results['success_rate']:.0%}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (0 or 1)')
        ax.set_title(f"'{task}' (Zero-Shot)", fontsize=10)
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Evaluation Episode Rewards\n(TRUE environment rewards: 0 = fail, 1 = success)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation episode rewards saved to: {save_path}")


def plot_full_summary(all_histories, primitive_results, compositional_results, save_path):
    """Comprehensive summary plot."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Row 1: Training curves
    for i, (primitive, history) in enumerate(all_histories.items()):
        ax = fig.add_subplot(gs[0, i])
        color = colors[primitive]
        rewards = history['episode_rewards']
        
        ax.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= 100:
            smoothed = pd.Series(rewards).rolling(100).mean()
            ax.plot(smoothed, color=color, linewidth=2)
        
        final = history['final_success_rate']
        ax.set_title(f"'{primitive}' Training\nFinal: {final:.0%}", fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Success', fontsize=8)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
    
    # Row 2: Evaluation bar charts
    ax_prim = fig.add_subplot(gs[1, :2])
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    prim_colors = [colors[p] for p in primitives]
    
    bars = ax_prim.bar(primitives, prim_success, color=prim_colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax_prim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax_prim.set_ylabel('Success Rate')
    ax_prim.set_title('Primitive Tasks (Trained)', fontweight='bold')
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
    
    # Row 3: Goal distribution during training
    ax_goals = fig.add_subplot(gs[2, :])
    x = np.arange(len(all_histories))
    width = 0.2
    goal_names = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    goal_colors_map = {'red_box': 'darkred', 'blue_box': 'darkblue', 
                       'red_sphere': 'lightcoral', 'blue_sphere': 'lightblue'}
    
    for j, goal in enumerate(goal_names):
        counts = [all_histories[p]['goal_reached_counts'].get(goal, 0) 
                  for p in all_histories.keys()]
        ax_goals.bar(x + j*width, counts, width, label=goal, color=goal_colors_map[goal])
    
    ax_goals.set_xlabel('Primitive Task')
    ax_goals.set_ylabel('Goals Reached (count)')
    ax_goals.set_title('Goal Distribution During Training', fontweight='bold')
    ax_goals.set_xticks(x + width*1.5)
    ax_goals.set_xticklabels(list(all_histories.keys()))
    ax_goals.legend(loc='upper right')
    ax_goals.grid(True, alpha=0.3, axis='y')
    
    # Row 4: Summary text
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    summary_text = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                         CORRECTED WORLD VALUE FUNCTIONS (WVF) EXPERIMENT                                       ║
    ║                              Zero-Shot Compositional Generalization                                            ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                ║
    ║  KEY CORRECTIONS FROM ORIGINAL:                                                                                ║
    ║    1. Sample goals from ENTIRE goal space (not just valid goals for current task)                             ║
    ║    2. Extended reward r_min applied when reaching DIFFERENT goal than conditioned on                          ║
    ║    3. Track TRUE environment rewards (0/1) separately for plotting                                            ║
    ║    4. Network learns Q(s,g,a) = "value of reaching goal g when solving task T"                                ║
    ║                                                                                                                ║
    ║  PRIMITIVE RESULTS (each trained separately):                                                                  ║
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
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Corrected World Value Functions - Full Summary',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Full summary saved to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_corrected_wvf_experiment(
    env_size=10,
    episodes_per_primitive=2000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    r_min=-10.0,
    seed=42
):
    """Run the corrected World Value Functions experiment."""
    
    print("\n" + "="*70)
    print("WORLD VALUE FUNCTIONS (WVF) EXPERIMENT")
    print("="*70)
    print("Zero-Shot Compositional Generalization via Boolean Task Algebra")
    print("Based on Nangue Tasse et al. (NeurIPS 2020)")
    print("="*70)
    print("\nKEY CORRECTIONS:")
    print("  1. Sample goals from ENTIRE goal space during training")
    print("  2. Extended reward r_min when reaching different goal than conditioned")
    print("  3. TRUE env rewards tracked separately for plotting")
    print("\nCLUSTER-FRIENDLY SETTINGS:")
    print("  - Episode-based replay (2000 episodes)")
    print("  - Small LSTM (64 units), Small batch (16)")
    print("  - Networks cleared between primitives")
    print("="*70 + "\n")
    
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # Corrected WVF Agent (cluster-friendly settings)
    print("Creating WorldValueFunctionAgent...")
    agent = WorldValueFunctionAgent(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=2000,       # Episode-based, cluster-friendly
        batch_size=16,          # Small batch
        seq_len=4,              # Short sequences
        hidden_size=128,
        lstm_size=64,           # Small LSTM
        tau=0.005,
        grad_clip=10.0,
        r_min=r_min,
        r_correct=1.0,
        r_wrong=-1.0,
        step_penalty=-0.01
    )
    
    # Training
    print("\n" + "="*60)
    print("PHASE 1: TRAINING ON PRIMITIVE TASKS")
    print("="*60)
    
    all_histories = train_all_primitives_wvf(
        env, agent,
        episodes_per_primitive=episodes_per_primitive,
        max_steps=max_steps
    )
    
    # Save model
    model_path = generate_save_path("corrected_wvf_model.pt")
    agent.save_model(model_path)
    
    # Evaluation
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION (including zero-shot compositional)")
    print("="*60)
    
    primitive_results, compositional_results = evaluate_all_wvf(
        env, agent, episodes=eval_episodes, max_steps=max_steps
    )
    
    # Plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_training_curves(all_histories, generate_save_path("corrected_wvf_training.png"))
    plot_evaluation_comparison(primitive_results, compositional_results,
                               generate_save_path("corrected_wvf_evaluation.png"))
    plot_evaluation_episode_rewards(primitive_results, compositional_results,
                                    generate_save_path("corrected_wvf_eval_episodes.png"))
    plot_full_summary(all_histories, primitive_results, compositional_results,
                     generate_save_path("corrected_wvf_summary.png"))
    
    # Save results
    results = {
        "method": "Corrected World Value Functions",
        "corrections": [
            "Sample goals from entire goal space",
            "Extended reward when reaching different goal than conditioned",
            "TRUE env rewards tracked separately"
        ],
        "r_min": r_min,
        "training": {
            primitive: {
                "episodes": episodes_per_primitive,
                "final_success_rate": history["final_success_rate"],
                "goal_reached_counts": history["goal_reached_counts"],
                "goal_conditioned_counts": history["goal_conditioned_counts"]
            }
            for primitive, history in all_histories.items()
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
    
    results_path = generate_save_path("corrected_wvf_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("CORRECTED WVF EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Primitive Tasks Average:              {results['summary']['avg_primitive_success']:.1%}")
    print(f"Zero-Shot Compositional Average:      {results['summary']['avg_compositional_success']:.1%}")
    print(f"Generalization Gap:                   {results['summary']['generalization_gap']:.1%}")
    print("="*70)
    
    return results, agent


if __name__ == "__main__":
    results, agent = run_corrected_wvf_experiment(
        env_size=10,
        episodes_per_primitive=3000,
        eval_episodes=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.999,
        r_min=-10.0,
        seed=42
    )