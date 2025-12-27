"""
Standalone Unified World Value Functions (WVF) Training Script

This matches the implementation in experiment_utils.py:
- Option A: Pure Task Conditioning (no goal conditioning during training)
- Learn Q(s, a, task) for each primitive task
- Simple reward: +1 for correct object, -0.1 for wrong, small step penalty
- Composition via min(Q_task1, Q_task2) for AND operations at evaluation
- Uses target network for stable evaluation

Based on Boolean Task Algebra approach
"""

import os

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
import torch

from env import DiscreteMiniWorldWrapper
from agents import UnifiedWorldValueFunctionAgent
from utils import generate_save_path


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

PRIMITIVE_TASKS = [
    {"name": "red", "features": ["red"], "type": "primitive"},
    {"name": "blue", "features": ["blue"], "type": "primitive"},
    {"name": "box", "features": ["box"], "type": "primitive"},
    {"name": "sphere", "features": ["sphere"], "type": "primitive"},
]

COMPOSITIONAL_TASKS = [
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies task requirements."""
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
# TRAINING
# ============================================================================

def train_unified_wvf(env, agent, training_episodes=20000, eval_episodes_per_task=1000,
                      max_steps=200):
    """
    Train Unified WVF agent (Option A - Pure Task Conditioning).
    
    Key approach:
    1. Learn Q(s, a, task) for each primitive task
    2. Simple reward: +1 for valid object, -0.1 for wrong, small step penalty
    3. Random task sampling each episode
    4. Uses target network for stable evaluation during training
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED WVF - OPTION A")
    print(f"Pure Task Conditioning (No Goal Conditioning)")
    print(f"{'='*70}")
    print(f"Training episodes: {training_episodes}")
    print(f"Training on primitive tasks: red, blue, box, sphere")
    print(f"{'='*70}\n")
    
    # Tracking
    all_rewards = []
    episode_labels = []
    task_episode_counts = {task['name']: 0 for task in PRIMITIVE_TASKS}
    
    print("Starting training phase...")
    
    for episode in tqdm(range(training_episodes), desc="Training WVF"):
        # Sample random primitive task
        current_task = agent.sample_task()
        task_idx = agent.TASK_TO_IDX[current_task]
        task_episode_counts[current_task] += 1
        
        task_config = {"name": current_task, "features": [current_task], "type": "primitive"}
        env.set_task(task_config)
        
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs, current_task)
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, task_idx)
            
            next_obs, _, terminated, truncated, info = env.step(action)
            next_stacked_obs = agent.step_episode(next_obs)
            
            reward, goal_reached = agent.compute_reward(info, current_task)
            
            if reward > 0:
                episode_reward = 1.0
            
            done = goal_reached or terminated or truncated
            agent.remember(stacked_obs, task_idx, action, reward, next_stacked_obs, done)
            
            if step % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.train_step()
            
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        episode_labels.append(current_task)
        
        if (episode + 1) % 500 == 0:
            recent_rewards = all_rewards[-500:]
            print(f"  Episode {episode+1}: Recent success rate = {np.mean(recent_rewards):.2%}")
    
    print(f"\n✓ WVF training complete!")
    print(f"\nTask distribution during training:")
    for task_name, count in task_episode_counts.items():
        print(f"  {task_name}: {count} episodes ({count/training_episodes:.1%})")
    
    # Calculate training success rates per task
    training_results = {}
    for task in PRIMITIVE_TASKS:
        task_name = task['name']
        task_rewards = [all_rewards[i] for i, label in enumerate(episode_labels) if label == task_name]
        if len(task_rewards) >= 100:
            final_success = np.mean(task_rewards[-100:])
            training_results[task_name] = final_success
            print(f"  {task_name} final training success: {final_success:.1%}")
    
    return {
        "all_rewards": all_rewards,
        "episode_labels": episode_labels,
        "task_episode_counts": task_episode_counts,
        "training_results": training_results,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_primitive(env, agent, prim_task, episodes=1000, max_steps=200):
    """Evaluate on primitive task using target network."""
    env.set_task(prim_task)
    task_name = prim_task['name']
    
    print(f"Evaluating primitive task: {task_name}...")
    
    successes = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        for step in range(max_steps):
            action = agent.select_action_primitive(stacked_obs, task_name, use_target=True)
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_task_satisfaction(info, prim_task):
                successes += 1
                break
            
            if terminated or truncated:
                break
    
    success_rate = successes / episodes
    print(f"  {task_name}: {success_rate:.1%} success rate ({successes}/{episodes})")
    
    return success_rate


def evaluate_compositional(env, agent, comp_task, episodes=1000, max_steps=200):
    """Zero-shot evaluation on compositional task using min composition."""
    env.set_task(comp_task)
    task_name = comp_task['name']
    features = comp_task['features']
    
    print(f"Evaluating {task_name} = min(Q_{features[0]}, Q_{features[1]})")
    
    successes = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        for step in range(max_steps):
            action = agent.select_action_composed(stacked_obs, features, use_target=True)
            
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_task_satisfaction(info, comp_task):
                successes += 1
                break
            
            if terminated or truncated:
                break
    
    success_rate = successes / episodes
    print(f"  {task_name}: {success_rate:.1%} success rate ({successes}/{episodes})")
    
    return {
        "success_rate": success_rate,
        "features": features,
    }


def evaluate_all(env, agent, eval_episodes_per_task=1000, max_steps=200):
    """Evaluate on all tasks."""
    
    print(f"\n{'='*70}")
    print("PRIMITIVE EVALUATION")
    print("Using target network for stable Q-estimates")
    print(f"{'='*70}\n")
    
    primitive_results = {}
    for prim_task in PRIMITIVE_TASKS:
        success_rate = evaluate_primitive(env, agent, prim_task, eval_episodes_per_task, max_steps)
        primitive_results[prim_task['name']] = success_rate
    
    print(f"\n{'='*70}")
    print("COMPOSITIONAL EVALUATION (Zero-Shot)")
    print("Using min(Q_task1, Q_task2) composition with target network")
    print(f"{'='*70}\n")
    
    compositional_results = {}
    for comp_task in COMPOSITIONAL_TASKS:
        result = evaluate_compositional(env, agent, comp_task, eval_episodes_per_task, max_steps)
        compositional_results[comp_task['name']] = result
    
    return primitive_results, compositional_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(history, save_path):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Rewards over time per task
    ax = axes[0, 0]
    for task_name in ['red', 'blue', 'box', 'sphere']:
        task_rewards = [history['all_rewards'][i] for i, label in enumerate(history['episode_labels']) 
                       if label == task_name]
        if task_rewards:
            episodes = list(range(len(task_rewards)))
            ax.plot(episodes, task_rewards, alpha=0.3, color=colors[task_name], linewidth=0.5)
            if len(task_rewards) >= 100:
                smoothed = pd.Series(task_rewards).rolling(100).mean()
                ax.plot(episodes, smoothed, color=colors[task_name], linewidth=2, label=task_name)
    
    ax.set_xlabel('Episode (per task)')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards per Task')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overall training progress
    ax = axes[0, 1]
    all_rewards = history['all_rewards']
    ax.plot(all_rewards, alpha=0.3, color='purple', linewidth=0.5)
    if len(all_rewards) >= 100:
        smoothed = pd.Series(all_rewards).rolling(100).mean()
        ax.plot(smoothed, color='purple', linewidth=2, label='Smoothed (100 ep)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Overall Training Progress')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Task distribution
    ax = axes[1, 0]
    task_counts = history['task_episode_counts']
    ax.bar(task_counts.keys(), task_counts.values(),
           color=[colors[t] for t in task_counts.keys()], edgecolor='black')
    ax.set_ylabel('Number of Episodes')
    ax.set_title('Task Distribution During Training')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Success rates per task
    ax = axes[1, 1]
    if 'training_results' in history:
        training_results = history['training_results']
        tasks = list(training_results.keys())
        success_rates = list(training_results.values())
        ax.bar(tasks, success_rates, color=[colors[t] for t in tasks], edgecolor='black')
        for i, (task, rate) in enumerate(zip(tasks, success_rates)):
            ax.text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
    
    ax.set_ylabel('Success Rate')
    ax.set_title('Final Training Success (last 100 episodes)')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('WVF Training Progress (Option A - Pure Task Conditioning)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_evaluation_results(primitive_results, compositional_results, save_path):
    """Plot evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_prim = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Primitives
    ax1 = axes[0]
    primitives = list(primitive_results.keys())
    prim_success = list(primitive_results.values())
    prim_colors = [colors_prim[p] for p in primitives]
    
    bars = ax1.bar(primitives, prim_success, color=prim_colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, prim_success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
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
        features = compositional_results[task]['features']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}\nmin({features[0]},{features[1]})',
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
    
    plt.suptitle(f'WVF Evaluation (Option A)\nPrimitives: {avg_prim:.1%} | '
                 f'Zero-Shot Compositional: {avg_comp:.1%} | Gap: {avg_prim - avg_comp:.1%}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation results saved to: {save_path}")


def plot_summary(history, primitive_results, compositional_results, save_path):
    """Create comprehensive summary plot."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Training progress
    ax = fig.add_subplot(gs[0, :])
    all_rewards = history['all_rewards']
    ax.plot(all_rewards, alpha=0.2, color='purple', linewidth=0.5)
    if len(all_rewards) >= 100:
        smoothed = pd.Series(all_rewards).rolling(100).mean()
        ax.plot(smoothed, color='purple', linewidth=2, label='Smoothed (100 ep)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress (All Tasks)', fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Primitive evaluation
    ax_prim = fig.add_subplot(gs[1, 0])
    primitives = list(primitive_results.keys())
    prim_success = list(primitive_results.values())
    prim_colors = [colors[p] for p in primitives]
    
    bars = ax_prim.bar(primitives, prim_success, color=prim_colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax_prim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
    ax_prim.set_ylabel('Success Rate')
    ax_prim.set_title('Primitive Tasks (Trained)', fontweight='bold')
    ax_prim.set_ylim([0, 1.15])
    ax_prim.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_prim.grid(True, alpha=0.3, axis='y')
    
    # Compositional evaluation
    ax_comp = fig.add_subplot(gs[1, 1])
    comp_tasks = list(compositional_results.keys())
    comp_success = [compositional_results[t]['success_rate'] for t in comp_tasks]
    
    bars = ax_comp.bar(comp_tasks, comp_success, color='coral', edgecolor='black')
    for bar, val in zip(bars, comp_success):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
    ax_comp.set_ylabel('Success Rate')
    ax_comp.set_title('Compositional Tasks (Zero-Shot)', fontweight='bold')
    ax_comp.set_ylim([0, 1.15])
    ax_comp.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    ax_comp.tick_params(axis='x', rotation=45)
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Summary text
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    summary_text = f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════╗
    ║                    UNIFIED WORLD VALUE FUNCTIONS - OPTION A                            ║
    ║                         Pure Task Conditioning Approach                                 ║
    ╠════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                         ║
    ║  APPROACH:                                                                              ║
    ║    • Learn Q(s, a, task) for each primitive task                                       ║
    ║    • Simple reward: +1 for correct object, -0.1 for wrong, small step penalty         ║
    ║    • Random task sampling each episode                                                 ║
    ║    • Composition via min(Q_task1, Q_task2) for AND operations                          ║
    ║    • Uses target network for stable evaluation                                         ║
    ║                                                                                         ║
    ║  PRIMITIVE RESULTS (trained):                                                          ║
    ║    • red:    {primitive_results['red']:.1%}    • blue:   {primitive_results['blue']:.1%}    • box:    {primitive_results['box']:.1%}    • sphere: {primitive_results['sphere']:.1%}    ║
    ║    • Average: {avg_prim:.1%}                                                                   ║
    ║                                                                                         ║
    ║  ZERO-SHOT COMPOSITIONAL RESULTS (never trained):                                      ║
    ║    • blue_sphere:  {compositional_results['blue_sphere']['success_rate']:.1%} = min(Q_blue, Q_sphere)                                      ║
    ║    • red_sphere:   {compositional_results['red_sphere']['success_rate']:.1%} = min(Q_red, Q_sphere)                                       ║
    ║    • blue_box:     {compositional_results['blue_box']['success_rate']:.1%} = min(Q_blue, Q_box)                                         ║
    ║    • red_box:      {compositional_results['red_box']['success_rate']:.1%} = min(Q_red, Q_box)                                          ║
    ║    • Average: {avg_comp:.1%}                                                                   ║
    ║                                                                                         ║
    ║  GENERALIZATION GAP: {avg_prim - avg_comp:.1%}                                                         ║
    ║                                                                                         ║
    ╚════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('WVF (Option A) - Complete Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_wvf_experiment(
    env_size=10,
    training_episodes=20000,
    eval_episodes_per_task=1000,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run the WVF experiment matching experiment_utils.py implementation."""
    
    print("\n" + "="*70)
    print("UNIFIED WORLD VALUE FUNCTIONS (WVF) - STANDALONE")
    print("="*70)
    print("Option A: Pure Task Conditioning")
    print("Zero-Shot Compositional Generalization via min() Composition")
    print("="*70)
    print("\nAPPROACH:")
    print("  • Learn Q(s, a, task) for each primitive task")
    print("  • Simple reward: +1 correct, -0.1 wrong, -0.005 step penalty")
    print("  • Random task sampling prevents catastrophic forgetting")
    print("  • Composition: min(Q_task1, Q_task2) for AND")
    print("  • Target network for stable evaluation")
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
        r_wrong=-0.1,
        step_penalty=-0.005
    )
    
    # Training
    print("\n" + "="*70)
    print("PHASE 1: TRAINING")
    print("="*70)
    
    history = train_unified_wvf(
        env, agent,
        training_episodes=training_episodes,
        eval_episodes_per_task=eval_episodes_per_task,
        max_steps=max_steps
    )
    
    # Save model
    model_path = generate_save_path("wvf_model.pt")
    agent.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluation
    print("\n" + "="*70)
    print("PHASE 2: EVALUATION")
    print("="*70)
    
    primitive_results, compositional_results = evaluate_all(
        env, agent,
        eval_episodes_per_task=eval_episodes_per_task,
        max_steps=max_steps
    )
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_training_curves(history, generate_save_path("wvf_training.png"))
    plot_evaluation_results(primitive_results, compositional_results,
                           generate_save_path("wvf_evaluation.png"))
    plot_summary(history, primitive_results, compositional_results,
                generate_save_path("wvf_summary.png"))
    
    # Save results
    results = {
        "method": "Unified World Value Functions (Option A)",
        "approach": "Pure task conditioning, min() composition",
        "training": {
            "total_episodes": training_episodes,
            "task_distribution": history["task_episode_counts"],
            "final_training_success": history.get("training_results", {}),
        },
        "evaluation_primitives": {
            p: prim_success
            for p, prim_success in primitive_results.items()
        },
        "evaluation_compositional_zero_shot": {
            t: {
                "success_rate": r["success_rate"],
                "composition": f"min({r['features'][0]}, {r['features'][1]})"
            }
            for t, r in compositional_results.items()
        },
        "summary": {
            "avg_primitive_success": np.mean(list(primitive_results.values())),
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
    print(f"\nResults saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("WVF EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Primitive Tasks Average:              {results['summary']['avg_primitive_success']:.1%}")
    print(f"Zero-Shot Compositional Average:      {results['summary']['avg_compositional_success']:.1%}")
    print(f"Generalization Gap:                   {results['summary']['generalization_gap']:.1%}")
    print("="*70 + "\n")
    
    return results, agent


if __name__ == "__main__":
    results, agent = run_wvf_experiment(
        env_size=10,
        training_episodes=20000,
        eval_episodes_per_task=1000,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.9995,
        seed=42
    )