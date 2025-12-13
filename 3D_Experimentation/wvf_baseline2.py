"""
World Value Functions (WVF) Training Script

Implements Nangue Tasse et al.'s Boolean Task Algebra for zero-shot compositional generalization:

1. Train goal-conditioned Q-networks with extended rewards on primitive tasks
2. Each primitive network learns values for ALL goals (not just valid ones)
3. Zero-shot compose via min operation at evaluation for conjunction tasks

Key result: Agent can solve compositional tasks (red_box, blue_sphere, etc.)
without ever having trained on them directly.
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


def check_primitive_satisfaction(info, primitive):
    """Check if contacted object satisfies primitive task."""
    contacted = info.get('contacted_object', None)
    if contacted is None:
        return False
    
    valid_goals = WorldValueFunctionAgent.PRIMITIVE_GOALS[primitive]
    return contacted in valid_goals


def check_compositional_satisfaction(info, task_name):
    """Check if contacted object satisfies compositional task."""
    contacted = info.get('contacted_object', None)
    return contacted == task_name


# ============================================================================
# TRAINING
# ============================================================================

def train_primitive_wvf(env, agent, primitive, episodes=2000, max_steps=200,
                        step_penalty=-0.005, verbose=True):
    """
    Train WVF on a single primitive task.
    
    Key difference from regular DQN training:
    - Each episode samples a target goal from VALID goals for this primitive
    - Extended rewards penalize reaching wrong goals (for learning)
    - We track TRUE env rewards separately (for plotting)
    - Network learns Q(s, g, a) for all goals g
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WVF FOR PRIMITIVE: {primitive.upper()}")
    print(f"{'='*60}")
    
    agent.set_training_primitive(primitive)
    
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    # Tracking - TRUE env rewards (did we complete the primitive task?)
    episode_rewards = []  # True task success: 1 if primitive satisfied, 0 otherwise
    episode_lengths = []
    episode_losses = []
    goal_reached_counts = {g: 0 for g in agent.GOALS}
    
    for episode in tqdm(range(episodes), desc=f"Training '{primitive}'"):
        obs, info = env.reset()
        
        # Reset episode and sample a target goal (only from valid goals)
        stacked_obs, target_goal_idx = agent.reset_episode(obs)
        target_goal_name = agent.GOALS[target_goal_idx]
        
        true_reward = 0  # Track actual task completion
        episode_loss = []
        
        for step in range(max_steps):
            # Select action conditioned on target goal
            action = agent.select_action(stacked_obs, target_goal_idx)
            
            # Step environment
            next_obs, _, terminated, truncated, info = env.step(action)
            
            # Compute extended reward for LEARNING (can use r_min, shaping, etc.)
            shaped_reward, goal_reached = agent.compute_extended_reward(
                info, target_goal_idx, step_penalty
            )
            
            # Check TRUE task completion (for plotting)
            contacted = info.get('contacted_object', None)
            if contacted is not None:
                if check_primitive_satisfaction(info, primitive):
                    true_reward = 1.0
            
            done = goal_reached or terminated or truncated
            
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Store transition with SHAPED reward (for learning)
            agent.remember(stacked_obs, target_goal_idx, action, shaped_reward,
                          next_stacked_obs, done)
            
            # Train every 4 steps
            if step % 4 == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
            
            stacked_obs = next_stacked_obs
            
            if done:
                if contacted in goal_reached_counts:
                    goal_reached_counts[contacted] += 1
                break
        
        agent.decay_epsilon()
        
        # Track TRUE rewards (task completion), not shaped rewards
        episode_rewards.append(true_reward)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean(episode_rewards[-500:])
            recent_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode+1}: Success Rate={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
            print(f"    Goals reached: {goal_reached_counts}")
    
    final_success = np.mean(episode_rewards[-100:])
    print(f"\nPrimitive '{primitive}' training complete!")
    print(f"  Final success rate (last 100): {final_success:.1%}")
    print(f"  Goal distribution: {goal_reached_counts}")
    
    return {
        "primitive": primitive,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "goal_reached_counts": goal_reached_counts,
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

        agent.memories[primitive].clear()
        
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
    
    For primitive evaluation: argmax_a max_g Q(s, g, a)
    """
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs, _ = agent.reset_episode(obs)
        
        for step in range(max_steps):
            # Use composed action selection with single feature
            action = agent.select_action_composed(stacked_obs, [primitive])
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_primitive_satisfaction(info, primitive):
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


def evaluate_compositional_wvf(env, agent, task_name, episodes=100, max_steps=200):
    """
    Zero-shot evaluation on compositional task using Boolean composition.
    
    For conjunction (AND):
        Q_composed(s, g, a) = min(Q_red(s, g, a), Q_box(s, g, a))
        action = argmax_a max_g Q_composed(s, g, a)
    
    The agent has NEVER trained on this task - this is zero-shot generalization.
    """
    task = COMPOSITIONAL_TASKS[task_name]
    features = task["features"]
    
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs, _ = agent.reset_episode(obs)
        
        for step in range(max_steps):
            # Zero-shot composed action selection
            action = agent.select_action_composed(stacked_obs, features)
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_compositional_satisfaction(info, task_name):
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
        "std_length": np.std(lengths),
        "features_used": features
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

def plot_training_curves_wvf(all_histories, save_path, window=100):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    for i, (primitive, history) in enumerate(all_histories.items()):
        color = colors[primitive]
        rewards = history['episode_rewards']
        losses = history['episode_losses']
        
        ax1 = axes[0, i]
        ax1.plot(rewards, alpha=0.3, color=color)
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(smoothed, color=color, linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f"'{primitive}' - Rewards")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1, i]
        if losses and any(l > 0 for l in losses):
            ax2.plot(losses, alpha=0.3, color=color)
            if len(losses) >= window:
                smoothed = pd.Series(losses).rolling(window).mean()
                ax2.plot(smoothed, color=color, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title(f"'{primitive}' - Loss")
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('World Value Functions - Training on Primitive Tasks', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_evaluation_summary_wvf(primitive_results, compositional_results, save_path):
    """Plot evaluation comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Primitives
    ax1 = axes[0]
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    colors = ['red', 'blue', 'orange', 'green']
    
    bars = ax1.bar(primitives, prim_success, color=colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Primitive Tasks\n(Trained)')
    ax1.set_ylim([0, 1.15])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Compositional (zero-shot)
    ax2 = axes[1]
    comp_tasks = list(compositional_results.keys())
    comp_success = [compositional_results[t]['success_rate'] for t in comp_tasks]
    
    bars = ax2.bar(comp_tasks, comp_success, color='coral', edgecolor='black')
    for bar, val, task in zip(bars, comp_success, comp_tasks):
        features = compositional_results[task]['features_used']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}\nmin({features[0]},{features[1]})',
                ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Compositional Tasks\n(Zero-Shot - Never Trained)')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    plt.suptitle(f'WVF Evaluation | Primitives: {avg_prim:.0%} | '
                 f'Zero-Shot Compositional: {avg_comp:.0%} | Gap: {avg_prim - avg_comp:.0%}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation saved to: {save_path}")


def plot_full_summary_wvf(all_histories, primitive_results, compositional_results, save_path):
    """Comprehensive summary plot."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Row 1: Training curves
    for i, (primitive, history) in enumerate(all_histories.items()):
        ax = fig.add_subplot(gs[0, i])
        color = colors[primitive]
        rewards = history['episode_rewards']
        
        ax.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= 50:
            smoothed = pd.Series(rewards).rolling(50).mean()
            ax.plot(smoothed, color=color, linewidth=2)
        
        ax.set_title(f"'{primitive}'", fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Evaluation
    ax_prim = fig.add_subplot(gs[1, :2])
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    prim_colors = [colors[p] for p in primitives]
    
    bars = ax_prim.bar(primitives, prim_success, color=prim_colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax_prim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax_prim.set_ylabel('Success Rate')
    ax_prim.set_title('Primitive Tasks (Trained)')
    ax_prim.set_ylim([0, 1.15])
    ax_prim.grid(True, alpha=0.3, axis='y')
    
    ax_comp = fig.add_subplot(gs[1, 2:])
    comp_tasks = list(compositional_results.keys())
    comp_success = [compositional_results[t]['success_rate'] for t in comp_tasks]
    
    bars = ax_comp.bar(comp_tasks, comp_success, color='coral', edgecolor='black')
    for bar, val in zip(bars, comp_success):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax_comp.set_ylabel('Success Rate')
    ax_comp.set_title('Compositional Tasks (Zero-Shot)')
    ax_comp.set_ylim([0, 1.15])
    ax_comp.tick_params(axis='x', rotation=45)
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    summary_text = f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                    WORLD VALUE FUNCTIONS (WVF) EXPERIMENT                                           ║
    ║                    Zero-Shot Compositional Generalization                                           ║
    ╠════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                     ║
    ║  APPROACH: Boolean Task Algebra (Nangue Tasse et al.)                                               ║
    ║                                                                                                     ║
    ║  ARCHITECTURE:                                                                                      ║
    ║    Goal-Conditioned Q-Network: Q(s, g, a)                                                           ║
    ║    • State s: visual observation (frame-stacked)                                                    ║
    ║    • Goal g: one-hot encoding of target object                                                      ║
    ║    • Action a: turn_left, turn_right, move_forward                                                  ║
    ║                                                                                                     ║
    ║  TRAINING (on primitives only):                                                                     ║
    ║    • Each episode samples random target goal from all 4 objects                                     ║
    ║    • Extended reward: +1 if reach correct goal, r_min penalty otherwise                             ║
    ║    • Network learns value of reaching ANY goal under each primitive task                            ║
    ║                                                                                                     ║
    ║  ZERO-SHOT COMPOSITION (never trained on these):                                                    ║
    ║    Q_red_box(s, g, a) = min(Q_red(s, g, a), Q_box(s, g, a))                                         ║
    ║    action = argmax_a max_g Q_red_box(s, g, a)                                                       ║
    ║                                                                                                     ║
    ║  PRIMITIVE RESULTS:                                                                                 ║
    ║    • red:    {primitive_results['red']['success_rate']:.1%}    • blue:   {primitive_results['blue']['success_rate']:.1%}                                                           ║
    ║    • box:    {primitive_results['box']['success_rate']:.1%}    • sphere: {primitive_results['sphere']['success_rate']:.1%}                                                           ║
    ║    • Average: {avg_prim:.1%}                                                                               ║
    ║                                                                                                     ║
    ║  ZERO-SHOT COMPOSITIONAL RESULTS:                                                                   ║
    ║    • red_box:     {compositional_results['red_box']['success_rate']:.1%}    • red_sphere:  {compositional_results['red_sphere']['success_rate']:.1%}                                                     ║
    ║    • blue_box:    {compositional_results['blue_box']['success_rate']:.1%}    • blue_sphere: {compositional_results['blue_sphere']['success_rate']:.1%}                                                     ║
    ║    • Average: {avg_comp:.1%}                                                                               ║
    ║                                                                                                     ║
    ║  GENERALIZATION GAP: {avg_prim - avg_comp:.1%}                                                                         ║
    ║                                                                                                     ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('World Value Functions - Zero-Shot Compositional Generalization',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Full summary saved to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_wvf_experiment(
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
    """Run the World Value Functions experiment."""
    
    print("\n" + "="*70)
    print("WORLD VALUE FUNCTIONS (WVF) EXPERIMENT")
    print("="*70)
    print("Zero-Shot Compositional Generalization via Boolean Task Algebra")
    print("Based on Nangue Tasse et al.")
    print("="*70 + "\n")
    
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # WVF Agent
    print("Creating WVF agent...")
    agent = WorldValueFunctionAgent(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=1500,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        tau=0.005,
        grad_clip=10.0,
        r_min=r_min
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
    model_path = generate_save_path("wvf_model.pt")
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
    
    plot_training_curves_wvf(all_histories, generate_save_path("wvf_training_curves.png"))
    plot_evaluation_summary_wvf(primitive_results, compositional_results,
                                generate_save_path("wvf_evaluation.png"))
    plot_full_summary_wvf(all_histories, primitive_results, compositional_results,
                         generate_save_path("wvf_summary.png"))
    
    # Save results
    results = {
        "method": "World Value Functions (Boolean Task Algebra)",
        "zero_shot": True,
        "composition": "min(Q_f1(s,g,a), Q_f2(s,g,a)), then max_g",
        "r_min": r_min,
        "training": {
            primitive: {
                "episodes": episodes_per_primitive,
                "goal_distribution": history["goal_reached_counts"]
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
    
    results_path = generate_save_path("wvf_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("WVF EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Primitive Tasks Average:              {results['summary']['avg_primitive_success']:.1%}")
    print(f"Zero-Shot Compositional Average:      {results['summary']['avg_compositional_success']:.1%}")
    print(f"Generalization Gap:                   {results['summary']['generalization_gap']:.1%}")
    print("="*70)
    
    return results, agent


if __name__ == "__main__":
    results, agent = run_wvf_experiment(
        env_size=10,
        episodes_per_primitive=2000,
        eval_episodes=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.999,
        r_min=-10.0,
        seed=42
    )