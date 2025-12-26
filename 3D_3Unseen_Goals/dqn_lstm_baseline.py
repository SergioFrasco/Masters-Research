"""
Unified LSTM-DQN Training with Unseen Goal Evaluation

Key modifications from original:
1. Training ONLY on red/blue objects (4 tasks)
2. Evaluation on UNSEEN green objects using compositional encoding
3. Tests true zero-shot compositional generalization
4. TIMELINE PLOTTING with clear phase demarcation
5. Comprehensive summary plot matching DQN baseline

Architecture: LSTM-DQN with frame stacking and 5-dim task conditioning
Training: Uniform task sampling to prevent catastrophic forgetting
Evaluation: Seen tasks (red/blue) + Unseen tasks (green - zero-shot)
"""

import os
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
import random

# Import environment and unified LSTM agent
from env import DiscreteMiniWorldWrapper
from agents import UnifiedLSTMDQNAgent3D
from utils import generate_save_path


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# TRAINING TASKS - Only red and blue objects (NO GREEN!)
TRAINING_TASKS = [
    {"name": "red", "features": ["red"], "type": "simple"},
    {"name": "blue", "features": ["blue"], "type": "simple"},
    {"name": "box", "features": ["box"], "type": "simple"},
    {"name": "sphere", "features": ["sphere"], "type": "simple"},
]

# SEEN COMPOSITIONAL (for sanity check - combinations of trained primitives)
SEEN_COMPOSITIONAL = [
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
]

# UNSEEN TASKS - Green objects (never seen during training!)
UNSEEN_SIMPLE_TASKS = [
    {"name": "green", "features": ["green"], "type": "simple_unseen"},
]

UNSEEN_COMPOSITIONAL_TASKS = [
    {"name": "green_box", "features": ["green", "box"], "type": "compositional_unseen"},
    {"name": "green_sphere", "features": ["green", "sphere"], "type": "compositional_unseen"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies current task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    
    # Single feature tasks
    if len(features) == 1:
        feature = features[0]
        
        if feature == "blue":
            return contacted_object in ["blue_box", "blue_sphere"]
        elif feature == "red":
            return contacted_object in ["red_box", "red_sphere"]
        elif feature == "green":
            return contacted_object in ["green_box", "green_sphere"]
        elif feature == "box":
            return contacted_object in ["blue_box", "red_box", "green_box"]
        elif feature == "sphere":
            return contacted_object in ["blue_sphere", "red_sphere", "green_sphere"]
    
    # Compositional tasks (2 features)
    elif len(features) == 2:
        feature_set = set(features)
        
        mappings = {
            frozenset({"blue", "sphere"}): "blue_sphere",
            frozenset({"red", "sphere"}): "red_sphere",
            frozenset({"green", "sphere"}): "green_sphere",
            frozenset({"blue", "box"}): "blue_box",
            frozenset({"red", "box"}): "red_box",
            frozenset({"green", "box"}): "green_box",
        }
        
        expected_object = mappings.get(frozenset(feature_set))
        return contacted_object == expected_object
    
    return False


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# TRAINING FUNCTION (NO GREEN OBJECTS)
# ============================================================================

def train_unified_lstm_dqn(env, episodes=8000, max_steps=200,
                           learning_rate=0.0001, gamma=0.99,
                           epsilon_start=1.0, epsilon_end=0.05,
                           epsilon_decay=0.9995, verbose=True,
                           step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train unified LSTM-DQN ONLY on red/blue objects.
    Green objects are excluded from training entirely.
    
    Key features:
    - Uniform task sampling per episode (prevents catastrophic forgetting)
    - Frame stacking with LSTM for temporal memory
    - 5-dim task encoding (green reserved for zero-shot)
    - Balanced replay buffer sampling
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED LSTM-DQN (RED/BLUE ONLY)")
    print(f"{'='*70}")
    print(f"  Architecture: LSTM-DQN with frame stacking")
    print(f"  Total episodes: {episodes}")
    print(f"  Training tasks: {[t['name'] for t in TRAINING_TASKS]}")
    print(f"  EXCLUDED: All green objects (reserved for zero-shot evaluation)")
    print(f"  Task encoding: 5-dim [red, blue, green, box, sphere]")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"{'='*70}")
    
    # Ensure environment is in training mode (no green objects)
    env.set_training_mode(True)
    
    # Create unified LSTM agent
    agent = UnifiedLSTMDQNAgent3D(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=2000,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []
    episode_tasks = []
    
    task_rewards = defaultdict(list)
    task_lengths = defaultdict(list)
    task_counts = defaultdict(int)
    
    for episode in tqdm(range(episodes), desc="Training LSTM-DQN (Red/Blue Only)"):
        # UNIFORM TASK SAMPLING - prevents catastrophic forgetting
        task = random.choice(TRAINING_TASKS)
        task_name = task['name']
        task_counts[task_name] += 1
        
        # Set task in environment
        env.set_task(task)
        
        # Reset environment
        obs, info = env.reset()
        
        # Reset agent with task-conditioned observation
        stacked_obs = agent.reset_episode(obs, task_name)
        
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(stacked_obs)
            
            # Step environment
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get next stacked observation
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Compute rewards
            contacted_object = info.get('contacted_object', None)
            task_satisfied = check_task_satisfaction(info, task)
            
            # TRUE REWARD (for tracking)
            true_reward = 1.0 if task_satisfied else 0.0
            
            # SHAPED REWARD (for training)
            if task_satisfied:
                shaped_reward = 1.0
            elif contacted_object is not None:
                shaped_reward = wrong_object_penalty
            else:
                shaped_reward = step_penalty
            
            # Remember transition
            agent.remember(stacked_obs, action, shaped_reward, next_stacked_obs, done)
            
            # Train every 4 steps
            if step % 4 == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
            
            true_reward_total += true_reward
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        # Update tracking
        success = true_reward_total > 0
        agent.update_task_success(task_name, success)
        agent.decay_epsilon()
        
        episode_tasks.append(task_name)
        episode_rewards.append(true_reward_total)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        episode_epsilons.append(agent.epsilon)
        
        # Per-task tracking
        task_rewards[task_name].append(true_reward_total)
        task_lengths[task_name].append(step + 1)
        
        # Periodic logging
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            task_success_rates = agent.get_task_success_rates()
            buffer_dist = agent.memory.get_task_distribution()
            
            print(f"\n  Episode {episode+1}:")
            print(f"    Overall Success: {recent_success:.1%}, Avg Length: {recent_length:.1f}")
            print(f"    Epsilon: {agent.epsilon:.3f}")
            print(f"    Per-task success (last 100 each):")
            for t_name in ['red', 'blue', 'box', 'sphere']:
                rate = task_success_rates.get(t_name, 0.0)
                count = task_counts[t_name]
                print(f"      {t_name:7s}: {rate:.1%} ({count} episodes)")
            print(f"    Buffer distribution: {buffer_dist}")
            
            clear_gpu_memory()
    
    # Save model
    model_path = generate_save_path("unified_lstm_dqn_no_green_model.pt")
    agent.save_model(model_path)
    
    # Calculate per-task final success
    per_task_final_success = {}
    for t in TRAINING_TASKS:
        tname = t['name']
        if tname in task_rewards and len(task_rewards[tname]) > 0:
            recent = task_rewards[tname][-min(100, len(task_rewards[tname])):]
            per_task_final_success[tname] = np.mean([r > 0 for r in recent])
    
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE (RED/BLUE ONLY)")
    print(f"{'='*70}")
    print(f"  Final success rate: {final_success:.1%}")
    print(f"  Per-task final success:")
    for tname, rate in per_task_final_success.items():
        print(f"    {tname:7s}: {rate:.1%}")
    print(f"  Model saved: {model_path}")
    print(f"{'='*70}")
    
    return agent, {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
        "episode_tasks": episode_tasks,
        "task_rewards": dict(task_rewards),
        "task_lengths": dict(task_lengths),
        "task_counts": dict(task_counts),
        "final_epsilon": agent.epsilon,
        "final_success_rate": final_success,
        "per_task_final_success": per_task_final_success,
        "model_path": model_path
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_agent_on_task(env, agent, task, episodes=100, max_steps=200):
    """
    Evaluate LSTM agent on a single task.
    Returns both summary stats AND per-episode rewards for timeline plotting.
    
    Handles both simple tasks (string encoding) and compositional tasks (list encoding).
    """
    
    task_name = task['name']
    features = task.get('features', [task_name])
    task_type = task.get('type', 'simple')
    
    env.set_task(task)
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for ep in range(episodes):
        # Reset environment every episode
        obs, info = env.reset()
        
        # Use correct encoding based on task type
        if len(features) > 1:
            # Compositional task - use list of features
            stacked_obs = agent.reset_episode(obs, features)
        else:
            # Simple task - use task name string
            stacked_obs = agent.reset_episode(obs, task_name)
        
        for step in range(max_steps):
            # Select action (greedy, no exploration)
            action = agent.select_action(stacked_obs, epsilon=0.0)
            
            # Step environment
            obs, _, terminated, truncated, info = env.step(action)
            
            # Update stacked observation
            stacked_obs = agent.step_episode(obs)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                episode_rewards.append(1.0)
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                episode_rewards.append(0.0)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
            episode_rewards.append(0.0)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "features": features,
        "episode_rewards": episode_rewards
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_experiment_timeline(training_history, seen_eval_results, unseen_eval_results,
                            save_path, eval_episodes_per_task=100):
    """
    Create a comprehensive timeline plot showing:
    1. Training phase with per-task performance
    2. Evaluation on seen tasks (red/blue)
    3. Evaluation on unseen tasks (green - zero-shot)
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    task_colors = {
        'red': 'red', 'blue': 'blue', 'green': 'green',
        'box': 'orange', 'sphere': 'purple',
        'red_box': 'darkred', 'red_sphere': 'lightcoral',
        'blue_box': 'darkblue', 'blue_sphere': 'lightblue',
        'green_box': 'darkgreen', 'green_sphere': 'lightgreen'
    }
    
    # === Panel 1: Training Phase ===
    ax1 = axes[0]
    
    # Plot overall training rewards
    rewards = training_history['episode_rewards']
    ax1.plot(rewards, alpha=0.2, color='gray', label='Episode reward')
    
    # Smoothed overall
    if len(rewards) >= 100:
        smoothed = pd.Series(rewards).rolling(100).mean()
        ax1.plot(smoothed, color='black', linewidth=2, label='Smoothed (100)')
    
    # Per-task smoothed curves
    task_rewards = training_history.get('task_rewards', {})
    for task_name in ['red', 'blue', 'box', 'sphere']:
        if task_name in task_rewards and len(task_rewards[task_name]) > 0:
            task_rews = task_rewards[task_name]
            if len(task_rews) >= 50:
                # Create x-axis for task-specific episodes
                task_x = np.linspace(0, len(rewards), len(task_rews))
                smoothed_task = pd.Series(task_rews).rolling(50).mean()
                ax1.plot(task_x, smoothed_task, color=task_colors[task_name], 
                        linewidth=1.5, alpha=0.7, label=f'{task_name}')
    
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('PHASE 1: Training (Red/Blue Only) - LSTM-DQN', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    # Add training stats annotation
    final_success = training_history.get('final_success_rate', 0)
    ax1.annotate(f'Final Success: {final_success:.1%}', 
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # === Panel 2: Seen Tasks Evaluation ===
    ax2 = axes[1]
    
    seen_tasks = list(seen_eval_results.keys())
    seen_success = [seen_eval_results[t]['success_rate'] for t in seen_tasks]
    seen_colors = [task_colors.get(t, 'gray') for t in seen_tasks]
    
    bars2 = ax2.bar(seen_tasks, seen_success, color=seen_colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars2, seen_success):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Success Rate')
    ax2.set_title('PHASE 2: Evaluation on SEEN Tasks (Red/Blue) - LSTM-DQN', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # Average annotation
    avg_seen = np.mean(seen_success)
    ax2.annotate(f'Average: {avg_seen:.1%}', 
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # === Panel 3: Unseen Tasks Evaluation (Zero-Shot) ===
    ax3 = axes[2]
    
    unseen_tasks = list(unseen_eval_results.keys())
    unseen_success = [unseen_eval_results[t]['success_rate'] for t in unseen_tasks]
    unseen_colors = [task_colors.get(t, 'green') for t in unseen_tasks]
    
    bars3 = ax3.bar(unseen_tasks, unseen_success, color=unseen_colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars3, unseen_success):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_ylabel('Success Rate')
    ax3.set_title('PHASE 3: ZERO-SHOT Evaluation on UNSEEN Tasks (Green) - LSTM-DQN', 
                 fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.15])
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # Average annotation
    avg_unseen = np.mean(unseen_success)
    ax3.annotate(f'Average: {avg_unseen:.1%}\n(ZERO-SHOT)', 
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('LSTM-DQN Experiment Timeline: Training → Seen Eval → Unseen Eval', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timeline plot saved to: {save_path}")


def plot_comprehensive_summary(training_history, seen_simple_results, seen_comp_results,
                               unseen_simple_results, unseen_comp_results, save_path):
    """
    Create the massive comprehensive summary plot matching the DQN version.
    """
    
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    task_colors = {
        'red': 'red', 'blue': 'blue', 'green': 'green',
        'box': 'orange', 'sphere': 'purple',
        'red_box': 'darkred', 'red_sphere': 'lightcoral',
        'blue_box': 'darkblue', 'blue_sphere': 'lightblue',
        'green_box': 'darkgreen', 'green_sphere': 'lightgreen'
    }
    
    # === Row 1: Training convergence per task ===
    for i, task_name in enumerate(['red', 'blue', 'box', 'sphere']):
        ax = fig.add_subplot(gs[0, i])
        task_rews = training_history.get('task_rewards', {}).get(task_name, [])
        
        if len(task_rews) > 0:
            ax.plot(task_rews, alpha=0.2, color=task_colors[task_name])
            if len(task_rews) >= 50:
                smoothed = pd.Series(task_rews).rolling(50).mean()
                ax.plot(smoothed, color=task_colors[task_name], linewidth=2)
        
        final_rate = training_history.get('per_task_final_success', {}).get(task_name, 0.0)
        ax.set_title(f"'{task_name}'\nFinal: {final_rate:.0%}", fontsize=10)
        ax.set_xlabel('Episode (per task)', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])
    
    # === Row 2: Seen simple tasks evaluation ===
    ax_seen_simple = fig.add_subplot(gs[1, :2])
    task_names = list(seen_simple_results.keys())
    success_rates = [seen_simple_results[t]['success_rate'] for t in task_names]
    colors = [task_colors.get(t, 'gray') for t in task_names]
    
    bars = ax_seen_simple.bar(task_names, success_rates, color=colors, edgecolor='black')
    for bar, val in zip(bars, success_rates):
        ax_seen_simple.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_seen_simple.set_ylabel('Success Rate')
    ax_seen_simple.set_title('SEEN Simple Tasks (Training Primitives)', fontsize=11, fontweight='bold')
    ax_seen_simple.set_ylim([0, 1.15])
    ax_seen_simple.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_seen_simple.grid(True, alpha=0.3, axis='y')
    
    # === Row 2: Seen compositional tasks ===
    ax_seen_comp = fig.add_subplot(gs[1, 2:])
    comp_names = list(seen_comp_results.keys())
    comp_success = [seen_comp_results[t]['success_rate'] for t in comp_names]
    comp_colors = [task_colors.get(t, 'gray') for t in comp_names]
    
    bars = ax_seen_comp.bar(comp_names, comp_success, color=comp_colors, edgecolor='black')
    for bar, val in zip(bars, comp_success):
        ax_seen_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_seen_comp.set_ylabel('Success Rate')
    ax_seen_comp.set_title('SEEN Compositional Tasks (Red/Blue + Shape)', fontsize=11, fontweight='bold')
    ax_seen_comp.set_ylim([0, 1.15])
    ax_seen_comp.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_seen_comp.grid(True, alpha=0.3, axis='y')
    ax_seen_comp.tick_params(axis='x', rotation=45)
    
    # === Row 3: Unseen simple tasks (green) ===
    ax_unseen_simple = fig.add_subplot(gs[2, :2])
    unseen_simple_names = list(unseen_simple_results.keys())
    unseen_simple_success = [unseen_simple_results[t]['success_rate'] for t in unseen_simple_names]
    
    bars = ax_unseen_simple.bar(unseen_simple_names, unseen_simple_success, 
                                color='green', edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, unseen_simple_success):
        ax_unseen_simple.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_unseen_simple.set_ylabel('Success Rate')
    ax_unseen_simple.set_title('UNSEEN Simple Tasks (Green - ZERO-SHOT)', fontsize=11, fontweight='bold')
    ax_unseen_simple.set_ylim([0, 1.15])
    ax_unseen_simple.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_unseen_simple.grid(True, alpha=0.3, axis='y')
    
    # === Row 3: Unseen compositional tasks (green + shape) ===
    ax_unseen_comp = fig.add_subplot(gs[2, 2:])
    unseen_comp_names = list(unseen_comp_results.keys())
    unseen_comp_success = [unseen_comp_results[t]['success_rate'] for t in unseen_comp_names]
    unseen_comp_colors = [task_colors.get(t, 'green') for t in unseen_comp_names]
    
    bars = ax_unseen_comp.bar(unseen_comp_names, unseen_comp_success, 
                              color=unseen_comp_colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, unseen_comp_success):
        ax_unseen_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_unseen_comp.set_ylabel('Success Rate')
    ax_unseen_comp.set_title('UNSEEN Compositional Tasks (Green + Shape - ZERO-SHOT)', 
                            fontsize=11, fontweight='bold')
    ax_unseen_comp.set_ylim([0, 1.15])
    ax_unseen_comp.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_unseen_comp.grid(True, alpha=0.3, axis='y')
    ax_unseen_comp.tick_params(axis='x', rotation=45)
    
    # === Row 4: Summary text ===
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    avg_seen_simple = np.mean([seen_simple_results[t]['success_rate'] for t in seen_simple_results])
    avg_seen_comp = np.mean([seen_comp_results[t]['success_rate'] for t in seen_comp_results])
    avg_unseen_simple = np.mean([unseen_simple_results[t]['success_rate'] for t in unseen_simple_results])
    avg_unseen_comp = np.mean([unseen_comp_results[t]['success_rate'] for t in unseen_comp_results])
    
    # Get individual results for display
    seen_simple_str = "  ".join([f"{t}={seen_simple_results[t]['success_rate']:.0%}" for t in seen_simple_results])
    seen_comp_str = "  ".join([f"{t}={seen_comp_results[t]['success_rate']:.0%}" for t in seen_comp_results])
    unseen_simple_str = "  ".join([f"{t}={unseen_simple_results[t]['success_rate']:.0%}" for t in unseen_simple_results])
    unseen_comp_str = "  ".join([f"{t}={unseen_comp_results[t]['success_rate']:.0%}" for t in unseen_comp_results])
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                      UNIFIED LSTM-DQN WITH UNSEEN GOAL EVALUATION                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                    ║
    ║  ARCHITECTURE: LSTM-DQN with Frame Stacking + Goal Tiling                                         ║
    ║    • Input: Stacked frames (12 ch) + Task encoding (5 ch) = 17 channels                          ║
    ║    • Task space: [red, blue, GREEN, box, sphere] - GREEN reserved for zero-shot                  ║
    ║    • LSTM: 64 hidden units for temporal memory                                                    ║
    ║    • Training: Uniform task sampling (prevents catastrophic forgetting)                           ║
    ║                                                                                                    ║
    ║  TRAINING (Red/Blue Only):                                                                        ║
    ║    • Final success rate: {training_history.get('final_success_rate', 0):.1%}                                                                  ║
    ║                                                                                                    ║
    ║  SEEN TASKS (Trained primitives):                                                                 ║
    ║    • Simple:        {seen_simple_str}                                     ║
    ║    • Average:       {avg_seen_simple:.1%}                                                                             ║
    ║    • Compositional: {seen_comp_str}              ║
    ║    • Average:       {avg_seen_comp:.1%}                                                                             ║
    ║                                                                                                    ║
    ║  UNSEEN TASKS (Zero-Shot - Never saw GREEN during training!):                                     ║
    ║    • Simple:        {unseen_simple_str}                                                           ║
    ║    • Average:       {avg_unseen_simple:.1%}                                                                             ║
    ║    • Compositional: {unseen_comp_str}                               ║
    ║    • Average:       {avg_unseen_comp:.1%}                                                                             ║
    ║                                                                                                    ║
    ║  GENERALIZATION GAPS:                                                                             ║
    ║    • Seen Simple → Seen Comp:       {avg_seen_simple - avg_seen_comp:+.1%}  (compositional generalization)                    ║
    ║    • Seen Simple → Unseen Simple:   {avg_seen_simple - avg_unseen_simple:+.1%}  (color generalization)                        ║
    ║    • Seen Comp → Unseen Comp:       {avg_seen_comp - avg_unseen_comp:+.1%}  (full zero-shot)                             ║
    ║                                                                                                    ║
    ║  KEY FINDINGS:                                                                                    ║
    ║    • LSTM-DQN with goal tiling learns primitive tasks through uniform sampling                    ║
    ║    • Model encodes green in task space but NEVER sees it during training                          ║
    ║    • Zero-shot performance reveals true generalization capability                                  ║
    ║    • Comparison with vanilla DQN shows impact of temporal memory on generalization                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.suptitle('LSTM-DQN Unseen Goal Generalization Experiment - Complete Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive summary saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_lstm_experiment_with_unseen_goals(
    env_size=10,
    total_episodes=8000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run LSTM-DQN experiment: train on red/blue, evaluate on green (unseen)."""
    
    print("\n" + "="*70)
    print("LSTM-DQN UNSEEN GOAL GENERALIZATION EXPERIMENT")
    print("="*70)
    print("Architecture: LSTM-DQN with frame stacking + goal tiling")
    print("Training: red/blue objects only (4 primitive tasks)")
    print("Evaluation: includes GREEN objects (never seen during training)")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment in TRAINING mode (red/blue only)
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array", training_mode=True)
    
    # ===== PHASE 1: TRAINING (NO GREEN) =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING ON RED/BLUE OBJECTS")
    print("="*60)
    
    agent, history = train_unified_lstm_dqn(
        env,
        episodes=total_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        verbose=True
    )
    
    clear_gpu_memory()
    
    # ===== PHASE 2: EVALUATE ON SEEN TASKS (RED/BLUE ONLY) =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATING ON SEEN TASKS (RED/BLUE)")
    print("="*60)
    print("Environment: Training mode (red/blue only) - same as training")
    print("="*60)
    
    # Keep training mode for seen tasks (no green objects)
    env.set_training_mode(True)
    
    seen_simple_results = {}
    for task in TRAINING_TASKS:
        print(f"  Evaluating {task['name']}...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        seen_simple_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%}")
    
    seen_comp_results = {}
    for task in SEEN_COMPOSITIONAL:
        print(f"  Evaluating {task['name']}...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        seen_comp_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%}")
    
    # Combine seen results for timeline plotting
    seen_results = {**seen_simple_results, **seen_comp_results}
    
    # ===== PHASE 3: EVALUATE ON UNSEEN TASKS (GREEN) =====
    print("\n" + "="*60)
    print("PHASE 3: ZERO-SHOT EVALUATION ON UNSEEN GREEN OBJECTS")
    print("="*60)
    print("NOTE: Model has NEVER seen green objects during training!")
    print("Testing if learned representations generalize to novel color")
    print("="*60)
    
    # Switch to evaluation mode (spawn green objects)
    print("Switching environment to EVALUATION mode (spawning green objects)...")
    env.set_training_mode(False)
    obs, info = env.reset()
    print(f"Objects now in environment: {[k for k in info.keys() if 'distance_to' in k]}")
    print("="*60)
    
    unseen_simple_results = {}
    for task in UNSEEN_SIMPLE_TASKS:
        print(f"  Evaluating {task['name']} (UNSEEN)...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        unseen_simple_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%} (UNSEEN COLOR)")
    
    unseen_comp_results = {}
    for task in UNSEEN_COMPOSITIONAL_TASKS:
        print(f"  Evaluating {task['name']} (UNSEEN)...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        unseen_comp_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%} (UNSEEN COMPOSITION)")
    
    # Combine unseen results for timeline plotting
    unseen_results = {**unseen_simple_results, **unseen_comp_results}
    
    # ===== GENERATE PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Timeline plot
    timeline_path = generate_save_path("lstm_dqn_experiment_timeline.png")
    plot_experiment_timeline(
        training_history=history,
        seen_eval_results=seen_results,
        unseen_eval_results=unseen_results,
        save_path=timeline_path,
        eval_episodes_per_task=eval_episodes
    )
    
    # Comprehensive summary plot
    summary_path = generate_save_path("lstm_dqn_comprehensive_summary.png")
    plot_comprehensive_summary(
        training_history=history,
        seen_simple_results=seen_simple_results,
        seen_comp_results=seen_comp_results,
        unseen_simple_results=unseen_simple_results,
        unseen_comp_results=unseen_comp_results,
        save_path=summary_path
    )
    
    # ===== SAVE RESULTS =====
    all_results = {
        "architecture": "Unified LSTM-DQN with Frame Stacking + Goal Tiling",
        "training": {
            "total_episodes": total_episodes,
            "final_success_rate": history["final_success_rate"],
            "training_tasks": [t['name'] for t in TRAINING_TASKS],
            "per_task_final_success": history["per_task_final_success"],
            "model_path": history["model_path"]
        },
        "evaluation_seen_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in seen_simple_results.items()},
        "evaluation_seen_compositional": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in seen_comp_results.items()},
        "evaluation_unseen_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"],
            "note": "ZERO-SHOT: Never saw green during training"
        } for t, r in unseen_simple_results.items()},
        "evaluation_unseen_compositional": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"],
            "note": "ZERO-SHOT: Novel green+shape composition"
        } for t, r in unseen_comp_results.items()},
        "summary": {
            "avg_seen_simple": np.mean([r["success_rate"] for r in seen_simple_results.values()]),
            "avg_seen_comp": np.mean([r["success_rate"] for r in seen_comp_results.values()]),
            "avg_unseen_simple": np.mean([r["success_rate"] for r in unseen_simple_results.values()]),
            "avg_unseen_comp": np.mean([r["success_rate"] for r in unseen_comp_results.values()]),
        },
        "plots": {
            "timeline": timeline_path,
            "summary": summary_path
        }
    }
    
    results_path = generate_save_path("lstm_dqn_unseen_goal_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("LSTM-DQN EXPERIMENT COMPLETE - GENERALIZATION ANALYSIS")
    print("="*70)
    print(f"SEEN (Red/Blue):")
    print(f"  Simple tasks:        {all_results['summary']['avg_seen_simple']:.1%}")
    print(f"  Compositional tasks: {all_results['summary']['avg_seen_comp']:.1%}")
    print(f"\nUNSEEN (Green - Zero-Shot):")
    print(f"  Simple tasks:        {all_results['summary']['avg_unseen_simple']:.1%}")
    print(f"  Compositional tasks: {all_results['summary']['avg_unseen_comp']:.1%}")
    print(f"\nGeneralization Gaps:")
    print(f"  Seen→Unseen Simple:  {all_results['summary']['avg_seen_simple'] - all_results['summary']['avg_unseen_simple']:.1%} drop")
    print(f"  Seen→Unseen Comp:    {all_results['summary']['avg_seen_comp'] - all_results['summary']['avg_unseen_comp']:.1%} drop")
    print(f"\nPlots:")
    print(f"  Timeline: {timeline_path}")
    print(f"  Summary:  {summary_path}")
    print("="*70)
    
    return all_results, agent


if __name__ == "__main__":
    results, agent = run_lstm_experiment_with_unseen_goals(
        env_size=10,
        total_episodes=8000,  # Matching DQN experiment
        eval_episodes=400,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.9995,
        seed=42
    )