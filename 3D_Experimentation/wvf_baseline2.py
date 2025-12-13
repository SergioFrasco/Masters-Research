"""
World Value Functions Training Script

Trains separate Q-networks for each primitive feature (red, blue, box, sphere)
then evaluates compositional generalization using min composition.
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
import time
import gc
import torch

# Import environment and WVF agent
from env import DiscreteMiniWorldWrapper
from agents import WorldValueFunctionAgent


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

PRIMITIVE_TASKS = {
    "red": {"name": "red", "features": ["red"], "type": "primitive"},
    "blue": {"name": "blue", "features": ["blue"], "type": "primitive"},
    "box": {"name": "box", "features": ["box"], "type": "primitive"},
    "sphere": {"name": "sphere", "features": ["sphere"], "type": "primitive"},
}

COMPOSITIONAL_TASKS = {
    "red_box": {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    "red_sphere": {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    "blue_box": {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    "blue_sphere": {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
}


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    
    # Single feature (primitive)
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
    
    # Compositional (2 features)
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


def generate_save_path(filename):
    """Generate save path in outputs directory."""
    output_dir = "/mnt/user-data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


# ============================================================================
# TRAINING
# ============================================================================

def train_primitive(env, agent, primitive, episodes=2000, max_steps=200,
                    step_penalty=-0.005, wrong_object_penalty=-0.1, verbose=True):
    """
    Train the agent on a single primitive task.
    """
    task = PRIMITIVE_TASKS[primitive]
    
    print(f"\n{'='*60}")
    print(f"TRAINING PRIMITIVE: {primitive.upper()}")
    print(f"{'='*60}")
    
    # Set primitive and task
    agent.set_training_primitive(primitive)
    env.set_task(task)
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    for episode in tqdm(range(episodes), desc=f"Training '{primitive}'"):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_stacked_obs = agent.step_episode(next_obs)
            
            # Compute rewards
            contacted = info.get('contacted_object', None)
            task_satisfied = check_task_satisfaction(info, task)
            
            true_reward = 1.0 if task_satisfied else 0.0
            
            if task_satisfied:
                shaped_reward = 1.0
            elif contacted is not None:
                shaped_reward = wrong_object_penalty
            else:
                shaped_reward = step_penalty
            
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
        
        agent.decay_epsilon()
        
        episode_rewards.append(true_reward_total)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode+1}: Success={recent_success:.1%}, "
                  f"Avg Length={recent_length:.1f}, Epsilon={agent.epsilon:.3f}")
    
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    print(f"\nPrimitive '{primitive}' training complete!")
    print(f"  Final success rate: {final_success:.1%}")
    print(f"  Q-value range: [{agent.q_stats[primitive]['min']:.2f}, "
          f"{agent.q_stats[primitive]['max']:.2f}]")
    
    return {
        "primitive": primitive,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "final_success_rate": final_success,
        "q_stats": agent.q_stats[primitive].copy()
    }


def train_all_primitives(env, agent, episodes_per_primitive=2000, max_steps=200):
    """Train all 4 primitives sequentially."""
    
    all_histories = {}
    
    for primitive in agent.PRIMITIVES:
        history = train_primitive(
            env, agent, primitive,
            episodes=episodes_per_primitive,
            max_steps=max_steps
        )
        all_histories[primitive] = history
        
        # Clear memory between primitives
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_histories


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_primitive(env, agent, primitive, episodes=100, max_steps=200):
    """Evaluate agent on a primitive task using that primitive's network."""
    
    task = PRIMITIVE_TASKS[primitive]
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        # Reset hidden for this primitive
        agent.current_hidden = agent.q_networks[primitive].init_hidden(
            batch_size=1, device=agent.device
        )
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, epsilon=0.0, primitive=primitive)
            obs, _, terminated, truncated, info = env.step(action)
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


def evaluate_compositional(env, agent, task_name, episodes=100, max_steps=200, 
                           normalize=True):
    """
    Evaluate agent on a compositional task using Q-value composition.
    
    Uses min(Q_feature1, Q_feature2) to select actions.
    """
    
    task = COMPOSITIONAL_TASKS[task_name]
    features = task["features"]
    
    env.set_task(task)
    
    successes = []
    lengths = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        
        for step in range(max_steps):
            # Use composed Q-values for action selection
            action = agent.select_action_composed(
                stacked_obs, features, normalize=normalize
            )
            
            obs, _, terminated, truncated, info = env.step(action)
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
        "std_length": np.std(lengths),
        "features_used": features
    }


def evaluate_all(env, agent, episodes=100, max_steps=200):
    """Evaluate on all primitives and compositional tasks."""
    
    print("\n" + "="*60)
    print("EVALUATING PRIMITIVES")
    print("="*60)
    
    primitive_results = {}
    for primitive in agent.PRIMITIVES:
        results = evaluate_primitive(env, agent, primitive, episodes, max_steps)
        primitive_results[primitive] = results
        print(f"  {primitive}: {results['success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("EVALUATING COMPOSITIONAL (with min composition)")
    print("="*60)
    
    compositional_results = {}
    for task_name in COMPOSITIONAL_TASKS:
        results = evaluate_compositional(env, agent, task_name, episodes, max_steps)
        compositional_results[task_name] = results
        print(f"  {task_name}: {results['success_rate']:.1%} "
              f"(composed: {results['features_used']})")
    
    return primitive_results, compositional_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(all_histories, save_path, window=100):
    """Plot training curves for all primitives."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = {'red': 'red', 'blue': 'blue', 'box': 'orange', 'sphere': 'green'}
    
    for i, (primitive, history) in enumerate(all_histories.items()):
        color = colors[primitive]
        rewards = history['episode_rewards']
        losses = history['episode_losses']
        
        # Rewards
        ax1 = axes[0, i]
        ax1.plot(rewards, alpha=0.3, color=color)
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(smoothed, color=color, linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f"'{primitive}' - Success: {history['final_success_rate']:.0%}")
        ax1.grid(True, alpha=0.3)
        
        # Loss
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
    
    plt.suptitle('WVF Agent - Primitive Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_evaluation_summary(primitive_results, compositional_results, save_path):
    """Plot evaluation results comparing primitives vs compositional."""
    
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
    ax1.set_title('Primitive Tasks\n(Each network on its own task)')
    ax1.set_ylim([0, 1.15])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Compositional
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
    ax2.set_title('Compositional Tasks\n(Using min(Q₁, Q₂) composition)')
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Stats
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    plt.suptitle(f'WVF Evaluation | Primitives: {avg_prim:.0%} | '
                 f'Compositional: {avg_comp:.0%} | Gap: {avg_prim - avg_comp:.0%}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation summary saved to: {save_path}")


def plot_full_summary(all_histories, primitive_results, compositional_results, save_path):
    """Create comprehensive summary plot."""
    
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
        
        ax.set_title(f"'{primitive}'\nFinal: {history['final_success_rate']:.0%}", fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Evaluation bars
    ax_prim = fig.add_subplot(gs[1, :2])
    primitives = list(primitive_results.keys())
    prim_success = [primitive_results[p]['success_rate'] for p in primitives]
    prim_colors = [colors[p] for p in primitives]
    
    bars = ax_prim.bar(primitives, prim_success, color=prim_colors, edgecolor='black')
    for bar, val in zip(bars, prim_success):
        ax_prim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax_prim.set_ylabel('Success Rate')
    ax_prim.set_title('Primitive Tasks Evaluation')
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
    ax_comp.set_title('Compositional Tasks (min composition)')
    ax_comp.set_ylim([0, 1.15])
    ax_comp.tick_params(axis='x', rotation=45)
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Summary text
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    avg_prim = np.mean(prim_success)
    avg_comp = np.mean(comp_success)
    
    summary_text = f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                          WORLD VALUE FUNCTIONS EXPERIMENT                                   ║
    ╠════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                             ║
    ║  APPROACH: Train separate Q-networks for each primitive, compose at evaluation              ║
    ║                                                                                             ║
    ║  ARCHITECTURE (per primitive):                                                              ║
    ║    Frame Stacking (k=4) → CNN → LSTM (64) → Dueling FC → Q-values                          ║
    ║                                                                                             ║
    ║  COMPOSITION METHOD:                                                                        ║
    ║    Q_composed(s,a) = min(Q_feature1(s,a), Q_feature2(s,a))                                  ║
    ║    With normalization to handle different Q-value scales                                    ║
    ║                                                                                             ║
    ║  PRIMITIVE RESULTS:                                                                         ║
    ║    • red:    {primitive_results['red']['success_rate']:.1%}    • blue:   {primitive_results['blue']['success_rate']:.1%}                                                     ║
    ║    • box:    {primitive_results['box']['success_rate']:.1%}    • sphere: {primitive_results['sphere']['success_rate']:.1%}                                                     ║
    ║    • Average: {avg_prim:.1%}                                                                         ║
    ║                                                                                             ║
    ║  COMPOSITIONAL RESULTS (using min composition):                                             ║
    ║    • red_box:     {compositional_results['red_box']['success_rate']:.1%}  (min of red, box)                                            ║
    ║    • red_sphere:  {compositional_results['red_sphere']['success_rate']:.1%}  (min of red, sphere)                                         ║
    ║    • blue_box:    {compositional_results['blue_box']['success_rate']:.1%}  (min of blue, box)                                           ║
    ║    • blue_sphere: {compositional_results['blue_sphere']['success_rate']:.1%}  (min of blue, sphere)                                        ║
    ║    • Average: {avg_comp:.1%}                                                                         ║
    ║                                                                                             ║
    ║  GENERALIZATION GAP: {avg_prim - avg_comp:.1%}                                                                   ║
    ║                                                                                             ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('World Value Functions - Compositional RL Experiment',
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
    seed=42
):
    """Run the complete WVF experiment."""
    
    print("\n" + "="*70)
    print("WORLD VALUE FUNCTIONS EXPERIMENT")
    print("="*70)
    print("Training separate Q-networks for each primitive feature")
    print("Composing Q-values using min operation at evaluation")
    print("="*70 + "\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment
    print("Creating environment...")
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # Create WVF agent
    print("Creating WVF agent...")
    agent = WorldValueFunctionAgent(
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
        grad_clip=10.0
    )
    
    # ===== TRAINING =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING PRIMITIVES SEQUENTIALLY")
    print("="*60)
    
    all_histories = train_all_primitives(
        env, agent,
        episodes_per_primitive=episodes_per_primitive,
        max_steps=max_steps
    )
    
    # Save model
    model_path = generate_save_path("wvf_model.pt")
    agent.save_model(model_path)
    
    # ===== EVALUATION =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION")
    print("="*60)
    
    primitive_results, compositional_results = evaluate_all(
        env, agent, episodes=eval_episodes, max_steps=max_steps
    )
    
    # ===== PLOTS =====
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_training_curves(all_histories, generate_save_path("wvf_training_curves.png"))
    plot_evaluation_summary(primitive_results, compositional_results, 
                           generate_save_path("wvf_evaluation.png"))
    plot_full_summary(all_histories, primitive_results, compositional_results,
                     generate_save_path("wvf_summary.png"))
    
    # ===== SAVE RESULTS =====
    results = {
        "method": "World Value Functions",
        "composition": "min(Q_f1, Q_f2) with normalization",
        "training": {
            primitive: {
                "episodes": episodes_per_primitive,
                "final_success_rate": h["final_success_rate"],
                "q_range": [h["q_stats"]["min"], h["q_stats"]["max"]]
            }
            for primitive, h in all_histories.items()
        },
        "evaluation_primitives": {
            p: {"success_rate": r["success_rate"], "mean_length": r["mean_length"]}
            for p, r in primitive_results.items()
        },
        "evaluation_compositional": {
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
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("WVF EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Primitive Tasks Average:       {results['summary']['avg_primitive_success']:.1%}")
    print(f"Compositional Tasks Average:   {results['summary']['avg_compositional_success']:.1%}")
    print(f"Generalization Gap:            {results['summary']['generalization_gap']:.1%}")
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
        seed=42
    )