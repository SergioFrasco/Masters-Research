"""
Unified DQN Training with Task Conditioning

Key changes from separate models approach:
1. Single model trained on all 4 primitive tasks
2. Tasks sampled uniformly at random each episode
3. Task information tiled and concatenated to observation
4. Prevents catastrophic forgetting via continuous multi-task training
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
from collections import deque, defaultdict
from tqdm import tqdm
import json
import time
import gc
import torch

# Import environment and agent
from env import DiscreteMiniWorldWrapper
from agents import UnifiedDQNAgent
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


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# UNIFIED TRAINING FUNCTION
# ============================================================================

def train_unified_dqn(env, episodes=8000, max_steps=200,
                      learning_rate=0.0001, gamma=0.99,
                      epsilon_start=1.0, epsilon_end=0.05,
                      epsilon_decay=0.9995, verbose=True,
                      step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train a single unified DQN on all 4 primitive tasks.
    Tasks are sampled uniformly at random each episode.
    
    Returns:
        agent: Trained unified DQN agent
        history: Training history dict with per-task breakdowns
    """
    
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFIED DQN ON ALL TASKS")
    print(f"{'='*60}")
    print(f"  Total episodes: {episodes}")
    print(f"  Tasks per episode: Sampled uniformly from {[t['name'] for t in SIMPLE_TASKS]}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"  Max steps per episode: {max_steps}")
    print(f"{'='*60}")
    
    # Print GPU memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print(f"{'='*60}")
    
    # Create unified agent
    agent = UnifiedDQNAgent(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=100000,
        batch_size=64,
        target_update_freq=1,
        hidden_size=256,
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    # Tracking - overall and per-task
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []
    episode_tasks = []  # Track which task was used each episode
    
    # Per-task tracking
    task_rewards = defaultdict(list)
    task_lengths = defaultdict(list)
    task_counts = defaultdict(int)
    
    for episode in tqdm(range(episodes), desc="Training Unified Model"):
        # UNIFORM RANDOM TASK SAMPLING
        task = np.random.choice(SIMPLE_TASKS)
        task_name = task['name']
        task_counts[task_name] += 1
        
        # Set task in environment
        env.set_task(task)
        
        obs, info = env.reset()
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action with task conditioning
            action = agent.select_action(obs, task_name)
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
            
            # Store with task conditioning
            agent.remember(obs, task_name, action, shaped_reward, next_obs, done)
            
            # Train step
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            true_reward_total += true_reward
            obs = next_obs
            
            if done:
                break
        
        # Decay epsilon globally (not per-task)
        episode_epsilons.append(agent.epsilon)
        agent.decay_epsilon()
        
        # Track overall
        episode_rewards.append(true_reward_total)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        episode_tasks.append(task_name)
        
        # Track per-task
        task_rewards[task_name].append(true_reward_total)
        task_lengths[task_name].append(step + 1)
        
        # Periodic logging
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            
            # Per-task stats
            task_stats = []
            for t in SIMPLE_TASKS:
                tname = t['name']
                if tname in task_rewards and len(task_rewards[tname]) > 0:
                    recent_task_rewards = task_rewards[tname][-100:]  # Last ~100 for this task
                    task_success = np.mean([r > 0 for r in recent_task_rewards]) if recent_task_rewards else 0
                    task_stats.append(f"{tname}={task_success:.1%}")
            
            print(f"  Episode {episode+1}: Overall Success={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
            print(f"    Per-task: {', '.join(task_stats)}")
    
    # Save model
    model_path = generate_save_path("unified_dqn_model.pt")
    agent.save_model(model_path)
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"UNIFIED TRAINING COMPLETE")
    print(f"{'='*60}")
    
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    print(f"  Final overall success rate (last 100 eps): {final_success:.1%}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Replay buffer size: {len(agent.memory)}")
    
    print(f"\n  Task distribution:")
    for task_name, count in task_counts.items():
        pct = count / episodes * 100
        print(f"    {task_name}: {count} episodes ({pct:.1f}%)")
    
    print(f"\n  Per-task final success rates (last ~100 episodes per task):")
    per_task_final = {}
    for t in SIMPLE_TASKS:
        tname = t['name']
        if tname in task_rewards and len(task_rewards[tname]) > 0:
            recent = task_rewards[tname][-min(100, len(task_rewards[tname])):]
            success_rate = np.mean([r > 0 for r in recent])
            per_task_final[tname] = success_rate
            print(f"    {tname}: {success_rate:.1%}")
    
    print(f"\n  Model saved to: {model_path}")
    print(f"{'='*60}")
    
    return agent, {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
        "episode_tasks": episode_tasks,
        "task_rewards": dict(task_rewards),
        "task_lengths": dict(task_lengths),
        "task_counts": dict(task_counts),
        "per_task_final_success": per_task_final,
        "final_epsilon": agent.epsilon,
        "final_success_rate": final_success,
        "model_path": model_path
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_agent_on_task(env, agent, task, episodes=100, max_steps=200):
    """Evaluate unified agent on a single task."""
    
    task_name = task['name']
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        
        for step in range(max_steps):
            # Use task conditioning during evaluation
            action = agent.select_action(obs, task_name, epsilon=0.0)
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


def evaluate_compositional_tasks(env, agent, episodes=100, max_steps=200):
    """
    Evaluate on compositional tasks.
    For unified model, we use the color feature as task conditioning.
    """
    
    results = {}
    
    for comp_task in COMPOSITIONAL_TASKS:
        task_name = comp_task['name']
        features = comp_task['features']
        
        # Use the color feature (red or blue) for conditioning
        color_feature = [f for f in features if f in ['red', 'blue']][0]
        
        print(f"Evaluating '{task_name}' using '{color_feature}' conditioning")
        
        env.set_task(comp_task)
        
        successes = []
        lengths = []
        
        for _ in range(episodes):
            obs, info = env.reset()
            
            for step in range(max_steps):
                # Condition on color feature
                action = agent.select_action(obs, color_feature, epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, comp_task):
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
        
        results[task_name] = {
            'success_rate': np.mean(successes),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'conditioning': color_feature
        }
        
        print(f"  → Success: {np.mean(successes):.1%}")
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_curves(history, save_path, window=100):
    """Plot training curves with per-task breakdown."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    episode_rewards = history['episode_rewards']
    episode_losses = history['episode_losses']
    episode_epsilons = history['episode_epsilons']
    task_rewards = history['task_rewards']
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Plot 1: Overall rewards
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.2, color='gray')
    if len(episode_rewards) >= window:
        smoothed = pd.Series(episode_rewards).rolling(window).mean()
        ax.plot(smoothed, color='black', linewidth=2, label=f'{window}-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Overall Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Overall loss
    ax = axes[0, 1]
    if episode_losses and any(l > 0 for l in episode_losses):
        ax.plot(episode_losses, alpha=0.2, color='gray')
        if len(episode_losses) >= window:
            smoothed_loss = pd.Series(episode_losses).rolling(window).mean()
            ax.plot(smoothed_loss, color='black', linewidth=2, label=f'{window}-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    ax = axes[1, 0]
    ax.plot(episode_epsilons, color='purple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon Decay (Global)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Per-task success rates (rolling window)
    ax = axes[1, 1]
    for task_name, rewards in task_rewards.items():
        color = task_colors.get(task_name, 'gray')
        # Calculate success rate with rolling window
        success_indicators = [1 if r > 0 else 0 for r in rewards]
        if len(success_indicators) >= 50:
            smoothed = pd.Series(success_indicators).rolling(50).mean()
            # Create x-axis that corresponds to episode numbers
            episodes_for_task = [i for i, t in enumerate(history['episode_tasks']) if t == task_name]
            ax.plot(episodes_for_task, smoothed, color=color, linewidth=2, label=task_name, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (50-ep rolling)')
    ax.set_title('Per-Task Success Rates Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 5: Task distribution
    ax = axes[2, 0]
    task_counts = history['task_counts']
    tasks = list(task_counts.keys())
    counts = [task_counts[t] for t in tasks]
    colors_list = [task_colors[t] for t in tasks]
    bars = ax.bar(tasks, counts, color=colors_list, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Episode Count')
    ax.set_title('Task Distribution Across Training')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Final per-task performance
    ax = axes[2, 1]
    per_task_final = history['per_task_final_success']
    tasks = list(per_task_final.keys())
    success_rates = [per_task_final[t] for t in tasks]
    colors_list = [task_colors[t] for t in tasks]
    bars = ax.bar(tasks, success_rates, color=colors_list, edgecolor='black')
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Success Rate')
    ax.set_title('Final Per-Task Success (last ~100 eps per task)')
    ax.set_ylim([0, 1.15])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Unified DQN Training - Multi-Task Learning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_summary(history, simple_results, comp_results, save_path):
    """Create comprehensive summary plot."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    task_colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    # Plot 1: Overall training curve
    ax1 = fig.add_subplot(gs[0, :])
    episode_rewards = history['episode_rewards']
    ax1.plot(episode_rewards, alpha=0.15, color='gray', label='Raw')
    if len(episode_rewards) >= 100:
        smoothed = pd.Series(episode_rewards).rolling(100).mean()
        ax1.plot(smoothed, color='black', linewidth=2.5, label='100-ep MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Unified Model Training Progress', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Simple task evaluation
    ax2 = fig.add_subplot(gs[1, 0])
    task_names = list(simple_results.keys())
    success_rates = [simple_results[t]['success_rate'] for t in task_names]
    colors = [task_colors[t] for t in task_names]
    
    bars = ax2.bar(task_names, success_rates, color=colors, edgecolor='black')
    for bar, val in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Simple Tasks Evaluation\n(Single unified model)')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Compositional task evaluation
    ax3 = fig.add_subplot(gs[1, 1])
    comp_task_names = list(comp_results.keys())
    comp_success = [comp_results[t]['success_rate'] for t in comp_task_names]
    conditioning = [comp_results[t]['conditioning'] for t in comp_task_names]
    
    bars = ax3.bar(comp_task_names, comp_success, color='coral', edgecolor='black')
    for bar, val, cond in zip(bars, comp_success, conditioning):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}\n({cond})', ha='center', va='bottom', fontsize=8)
    
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Compositional Tasks\n(Using color conditioning)')
    ax3.set_ylim([0, 1.15])
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (~50%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics as text
    avg_simple = np.mean([simple_results[t]['success_rate'] for t in simple_results])
    avg_comp = np.mean([comp_results[t]['success_rate'] for t in comp_results])
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    UNIFIED MODEL EXPERIMENT SUMMARY                       ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  APPROACH: Single DQN with task conditioning via goal tiling             ║
    ║            Tasks sampled uniformly each episode (prevents forgetting)    ║
    ║                                                                           ║
    ║  SIMPLE TASKS (unified model):                                           ║
    ║    • red:    {simple_results['red']['success_rate']:.1%}    • blue:   {simple_results['blue']['success_rate']:.1%}                                    ║
    ║    • box:    {simple_results['box']['success_rate']:.1%}    • sphere: {simple_results['sphere']['success_rate']:.1%}                                    ║
    ║    • Average: {avg_simple:.1%}                                                          ║
    ║                                                                           ║
    ║  COMPOSITIONAL TASKS (color conditioning):                               ║
    ║    • red_box:     {comp_results['red_box']['success_rate']:.1%} (red)                                           ║
    ║    • red_sphere:  {comp_results['red_sphere']['success_rate']:.1%} (red)                                           ║
    ║    • blue_box:    {comp_results['blue_box']['success_rate']:.1%} (blue)                                          ║
    ║    • blue_sphere: {comp_results['blue_sphere']['success_rate']:.1%} (blue)                                          ║
    ║    • Average: {avg_comp:.1%}  (expected ~50% by chance)                             ║
    ║                                                                           ║
    ║  GENERALIZATION GAP: {avg_simple - avg_comp:.1%} drop                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    
    fig.text(0.5, 0.01, summary_text, transform=fig.transFigure,
            fontsize=9, fontfamily='monospace', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Unified DQN with Task Conditioning', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_unified_experiment(
    env_size=10,
    total_episodes=8000,  # 2000 per task equivalent
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run the unified model experiment."""
    
    print("\n" + "="*70)
    print("UNIFIED DQN EXPERIMENT")
    print("="*70)
    print("Training single unified model on all 4 primitive tasks")
    print("Tasks sampled uniformly each episode")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # ===== TRAINING: Single unified model =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING UNIFIED MODEL")
    print("="*60)
    
    agent, history = train_unified_dqn(
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
    
    # ===== EVALUATION ON SIMPLE TASKS =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATING ON SIMPLE TASKS")
    print("="*60)
    
    simple_results = {}
    for task in SIMPLE_TASKS:
        task_name = task['name']
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        simple_results[task_name] = results
        print(f"  {task_name}: {results['success_rate']:.1%}")
    
    # ===== EVALUATION ON COMPOSITIONAL TASKS =====
    print("\n" + "="*60)
    print("PHASE 3: EVALUATING ON COMPOSITIONAL TASKS")
    print("="*60)
    
    comp_results = evaluate_compositional_tasks(env, agent, eval_episodes, max_steps)
    
    # ===== GENERATE PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_training_curves(history, generate_save_path("unified_training_curves.png"))
    plot_summary(history, simple_results, comp_results, generate_save_path("unified_summary.png"))
    
    # ===== SAVE RESULTS =====
    all_results = {
        "training": {
            "total_episodes": total_episodes,
            "final_success_rate": history["final_success_rate"],
            "final_epsilon": history["final_epsilon"],
            "task_distribution": history["task_counts"],
            "per_task_final_success": history["per_task_final_success"],
            "model_path": history["model_path"]
        },
        "evaluation_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in simple_results.items()},
        "evaluation_compositional": {t: {
            "success_rate": comp_results[t]["success_rate"],
            "conditioning": comp_results[t]["conditioning"],
            "mean_length": comp_results[t]["mean_length"]
        } for t in comp_results},
        "summary": {
            "avg_simple_success": np.mean([r["success_rate"] for r in simple_results.values()]),
            "avg_comp_success": np.mean([comp_results[t]["success_rate"] for t in comp_results]),
        }
    }
    
    results_path = generate_save_path("unified_experiment_results.json")
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
    
    return all_results, agent


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results, agent = run_unified_experiment(
        env_size=10,
        total_episodes=8000,  # 2000 per task on average
        eval_episodes=100,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.9995,
        seed=42
    )