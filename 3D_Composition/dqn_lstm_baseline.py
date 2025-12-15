"""
DQN Training with LSTM + Frame Stacking - SEPARATE MODELS per Task

This version uses the hybrid LSTM-DQN agent that combines:
1. Frame stacking (k=4) for short-term spatial memory
2. Small LSTM (128 units) for medium-term temporal reasoning

This should help the agent remember objects it saw before turning around.
"""

import os
import sys

# Force headless mode BEFORE any other imports
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Disable display entirely
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
import time
import gc
import torch

# Import environment and LSTM agent
from env import DiscreteMiniWorldWrapper
from agents import LSTMDQNAgent3D
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

def train_single_task_lstm_dqn(env, task, episodes=2000, max_steps=200, 
                               learning_rate=0.0001, gamma=0.99,
                               epsilon_start=1.0, epsilon_end=0.05,
                               epsilon_decay=0.9995, verbose=True,
                               step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train a fresh LSTM-DQN agent on a SINGLE task.
    """
    
    task_name = task['name']
    print(f"\n{'='*60}")
    print(f"TRAINING LSTM-DQN FOR TASK: {task_name.upper()}")
    print(f"{'='*60}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"  Episodes: {episodes}, Max steps: {max_steps}")
    print(f"  Fresh agent with frame stacking + LSTM")
    print(f"{'='*60}")
    
    # Create LSTM-DQN agent with OPTIMIZED hyperparameters for speed
    agent = LSTMDQNAgent3D(
        env,
        k_frames=4,              # Stack 4 frames
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=2000,        # REDUCED: Smaller buffer = less memory
        batch_size=16,           # REDUCED: Smaller batch = faster training
        seq_len=4,               # REDUCED: Shorter sequences = faster
        hidden_size=128,         # REDUCED: Smaller network = faster
        lstm_size=64,            # REDUCED: Smaller LSTM = much faster
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    # Set the task
    env.set_task(task)
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []
    
    for episode in tqdm(range(episodes), desc=f"Training '{task_name}'"):
        obs, info = env.reset()
        
        # IMPORTANT: Reset frame stack and LSTM hidden state
        stacked_obs = agent.reset_episode(obs)
        
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action (maintains hidden state internally)
            action = agent.select_action(stacked_obs)
            
            # Step environment
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get next stacked observation
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Compute rewards
            contacted_object = info.get('contacted_object', None)
            task_satisfied = check_task_satisfaction(info, task)
            
            true_reward = 1.0 if task_satisfied else 0.0
            
            if task_satisfied:
                shaped_reward = 1.0
            elif contacted_object is not None:
                shaped_reward = wrong_object_penalty
            else:
                shaped_reward = step_penalty
            
            # Remember transition (stores in episode buffer)
            agent.remember(stacked_obs, action, shaped_reward, next_stacked_obs, done)
            
            # Train less frequently (every 4 steps instead of every step)
            if step % 4 == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
            
            true_reward_total += true_reward
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        # Track metrics
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
    model_path = generate_save_path(f"lstm_dqn_model_{task_name}.pt")
    agent.save_model(model_path)
    
    # Final stats
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    print(f"\nTask '{task_name}' training complete!")
    print(f"  Final success rate (last 100 eps): {final_success:.1%}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Episode buffer size: {len(agent.memory)}")
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
        
        # Reset frame stack and LSTM state
        stacked_obs = agent.reset_episode(obs)
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            # Update frame stack
            stacked_obs = agent.step_episode(obs)
            
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
    """Evaluate on compositional tasks using the COLOR MODEL approach."""
    
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
# PLOTTING (reuse from original)
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
    
    plt.suptitle('LSTM-DQN with Frame Stacking - Training Curves per Task', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_summary(all_histories, simple_results, comp_results, save_path):
    """Create a comprehensive summary plot."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Row 1: Training curves for each task
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
    
    # Row 2: Evaluation results
    ax_simple = fig.add_subplot(gs[1, :2])
    task_names = list(simple_results.keys())
    success_rates = [simple_results[t]['success_rate'] for t in task_names]
    colors = [task_colors[t] for t in task_names]
    
    bars = ax_simple.bar(task_names, success_rates, color=colors, edgecolor='black')
    for bar, val in zip(bars, success_rates):
        ax_simple.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax_simple.set_ylabel('Success Rate')
    ax_simple.set_title('Simple Tasks Evaluation\n(LSTM-DQN with Frame Stacking)')
    ax_simple.set_ylim([0, 1.15])
    ax_simple.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_simple.grid(True, alpha=0.3, axis='y')
    
    # Compositional results
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
    
    # Row 3: Summary text
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    avg_simple = np.mean([simple_results[t]['success_rate'] for t in simple_results])
    avg_comp = np.mean([comp_results[t]['success_rate'] for t in comp_results])
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                      LSTM-DQN WITH FRAME STACKING EXPERIMENT                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║  ARCHITECTURE: Frame Stacking (k=4) + CNN + Small LSTM (128) + FC                   ║
    ║                                                                                       ║
    ║  MEMORY FEATURES:                                                                     ║
    ║    • Frame stacking: Agent sees last 4 frames → perceives motion & recent history   ║
    ║    • LSTM: Maintains hidden state across timesteps → temporal reasoning              ║
    ║    • Episode buffer: Trains on sequences to learn temporal dependencies              ║
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
    ║    • Average: {avg_comp:.1%}                                                                         ║
    ║                                                                                       ║
    ║  GENERALIZATION GAP: {avg_simple - avg_comp:.1%}                                                            ║
    ║                                                                                       ║
    ║  KEY INSIGHT: Memory helps the agent remember objects after turning, but doesn't     ║
    ║               solve compositional generalization. The agent still only learned       ║
    ║               "go to red" rather than composing "red" + "box" features.              ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('LSTM-DQN with Frame Stacking - Separate Models', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_lstm_experiment(
    env_size=10,
    episodes_per_task=2000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run the LSTM-DQN experiment."""
    
    print("\n" + "="*70)
    print("LSTM-DQN WITH FRAME STACKING EXPERIMENT")
    print("="*70)
    print("Architecture: Frame Stacking (k=4) + CNN + LSTM (128) + FC")
    print("Training one independent model per simple task")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # ===== TRAINING =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING SEPARATE LSTM-DQN MODELS")
    print("="*60)
    
    trained_agents = {}
    all_histories = {}
    
    for task in SIMPLE_TASKS:
        agent, history = train_single_task_lstm_dqn(
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
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== EVALUATION =====
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
    
    print("\n" + "="*60)
    print("PHASE 3: EVALUATING ON COMPOSITIONAL TASKS")
    print("="*60)
    
    comp_results = evaluate_compositional_tasks(env, trained_agents, eval_episodes, max_steps)
    
    # ===== PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_all_training_curves(all_histories, generate_save_path("lstm_training_curves.png"))
    plot_summary(all_histories, simple_results, comp_results, generate_save_path("lstm_summary.png"))
    
    # ===== SAVE RESULTS =====
    all_results = {
        "architecture": "Frame Stacking (k=4) + CNN + LSTM (128) + FC",
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
    
    results_path = generate_save_path("lstm_experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("LSTM-DQN EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Simple Tasks Average Success:        {all_results['summary']['avg_simple_success']:.1%}")
    print(f"Compositional Tasks Average Success: {all_results['summary']['avg_comp_success']:.1%}")
    print(f"Generalization Gap:                  {all_results['summary']['avg_simple_success'] - all_results['summary']['avg_comp_success']:.1%}")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = run_lstm_experiment(
        env_size=10,
        episodes_per_task=2000,
        eval_episodes=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.999,
        seed=42
    )