"""
Training script for Compositional WVF Agent - FIXED VERSION with Separate Evaluation.

Key Changes from Original:
1. Training phase: Only simple tasks (red, blue, box, sphere) - networks ARE trained
2. Evaluation phase: Only compositional tasks - networks NOT trained, just composed

This ensures:
- All networks learn identical navigation dynamics (G*) during training
- Composition via min() is evaluated WITHOUT further training
- Clean separation of learning vs composition testing

Usage:
    python train_compositional_wvf_fixed.py
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gc
import pandas as pd

# Import fixed components
from agents import CompositionalWVFAgent
from models import WVF_MLP
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from train_vision import CubeDetector
from utils import generate_save_path

from diagnose_wvf import diagnose_composition, check_q_value_scales, diagnose_value_maps


# ==================== Task System ====================

def create_task_schedule(total_episodes):
    """
    Create task schedule for TRAINING phase only (simple tasks).
    Compositional tasks are handled separately in evaluation.
    """
    simple_tasks = [
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"}
    ]
    
    episodes_per_simple = total_episodes // len(simple_tasks)
    
    tasks = []
    for task in simple_tasks:
        task_copy = task.copy()
        task_copy["duration"] = episodes_per_simple
        tasks.append(task_copy)

    return tasks, episodes_per_simple


def get_compositional_tasks():
    """Get list of compositional tasks for evaluation."""
    return [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]


def get_current_task(episode, tasks):
    """Get the current task for a given episode."""
    cumulative = 0
    for idx, task in enumerate(tasks):
        cumulative += task["duration"]
        if episode < cumulative:
            return task, idx
    return tasks[-1], len(tasks) - 1


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies current task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False

    features = task["features"]

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


# ==================== Vision Model ====================

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained 4-object cube detector model."""
    from train_vision import CubeDetector
    
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CubeDetector().to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
        model.load_state_dict(checkpoint)
        pos_mean = 0.0
        pos_std = 1.0

    model.eval()
    print(f"✓ Cube detector loaded on {device}")
    return model, device, pos_mean, pos_std


def detect_cube(model, obs, device, transform, pos_mean=0.0, pos_std=1.0):
    """Run cube detection with classification + regression output."""
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs

    if isinstance(img, np.ndarray):
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, pos_preds = model(img_tensor)
        
        probs = torch.sigmoid(cls_logits)
        predictions = (probs > 0.5).cpu().numpy()[0]
        regression_values = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        label_names = ["red_box", "blue_box", "red_sphere", "blue_sphere"]
        detected_objects = [label_names[i] for i in range(4) if predictions[i]]

    positions = {
        'red_box': (regression_values[0], regression_values[1]) if predictions[0] else None,
        'blue_box': (regression_values[2], regression_values[3]) if predictions[1] else None,
        'red_sphere': (regression_values[4], regression_values[5]) if predictions[2] else None,
        'blue_sphere': (regression_values[6], regression_values[7]) if predictions[3] else None,
    }

    probabilities = {label_names[i]: float(probs[0, i]) for i in range(4)}

    return {
        "detected_objects": detected_objects,
        "predictions": predictions,
        "probabilities": probabilities,
        "positions": positions,
        "regression_raw": regression_values
    }


# ==================== Visualization ====================

def plot_training_results(results, save_path='training_results.png', window=100):
    """Plot training phase results (simple tasks only)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    tasks = results["tasks"]
    
    # Plot 1: Episode rewards with task boundaries
    ax1 = axes[0, 0]
    mean_smooth = pd.Series(results["rewards"]).rolling(window).mean()
    ax1.plot(results["rewards"], alpha=0.3, label='Raw')
    ax1.plot(mean_smooth, linewidth=2, label='Smoothed')
    
    # Add task boundaries
    cumulative = 0
    colors = ['red', 'blue', 'brown', 'purple']
    for i, task in enumerate(tasks[:-1]):
        cumulative += task["duration"]
        ax1.axvline(x=cumulative, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training: Episode Rewards (Simple Tasks)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Episode lengths
    ax2 = axes[0, 1]
    mean_smooth = pd.Series(results["lengths"]).rolling(window).mean()
    ax2.plot(results["lengths"], alpha=0.3, label='Raw')
    ax2.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Training: Episode Lengths")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Task-specific rewards
    ax3 = axes[1, 0]
    mean_smooth = pd.Series(results["task_rewards"]).rolling(window).mean()
    ax3.plot(results["task_rewards"], alpha=0.3, label='Raw')
    ax3.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Task Reward")
    ax3.set_title("Training: Task-Specific Rewards")
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Feature losses
    ax4 = axes[1, 1]
    colors = ['red', 'blue', 'brown', 'purple']
    for idx, feature in enumerate(['red', 'blue', 'box', 'sphere']):
        losses = results["feature_losses"][feature]
        if len(losses) >= window:
            smoothed = pd.Series(losses).rolling(window).mean()
            ax4.plot(smoothed, label=feature, color=colors[idx], linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Loss")
    ax4.set_title("Training: Feature WVF Losses")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Training results plot saved: {save_path}")


def plot_evaluation_results(eval_results, save_path='evaluation_results.png'):
    """Plot evaluation phase results (compositional tasks only)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    task_names = list(eval_results["per_task_metrics"].keys())
    
    # Plot 1: Success rate per task
    ax1 = axes[0, 0]
    success_rates = [eval_results["per_task_metrics"][t]["success_rate"] * 100 
                    for t in task_names]
    colors = ['cyan', 'magenta', 'lightblue', 'lightcoral']
    bars = ax1.bar(task_names, success_rates, color=colors, edgecolor='black')
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Evaluation: Compositional Task Success Rates")
    ax1.set_ylim(0, 100)
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average reward per task
    ax2 = axes[0, 1]
    avg_rewards = [eval_results["per_task_metrics"][t]["avg_reward"] 
                  for t in task_names]
    bars = ax2.bar(task_names, avg_rewards, color=colors, edgecolor='black')
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Evaluation: Average Reward per Compositional Task")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average episode length per task
    ax3 = axes[1, 0]
    avg_lengths = [eval_results["per_task_metrics"][t]["avg_length"] 
                  for t in task_names]
    bars = ax3.bar(task_names, avg_lengths, color=colors, edgecolor='black')
    ax3.set_ylabel("Average Episode Length")
    ax3.set_title("Evaluation: Average Episode Length per Compositional Task")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode rewards over evaluation
    ax4 = axes[1, 1]
    for task_name in task_names:
        rewards = eval_results["per_task_rewards"][task_name]
        episodes = range(len(rewards))
        ax4.plot(episodes, rewards, label=task_name, alpha=0.7, linewidth=2)
    ax4.set_xlabel("Episode (within task)")
    ax4.set_ylabel("Reward")
    ax4.set_title("Evaluation: Rewards Over Episodes")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Evaluation results plot saved: {save_path}")


def plot_combined_results(train_results, eval_results, save_path='combined_results.png', window=50):
    """Plot combined training and evaluation results."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    train_episodes = len(train_results["rewards"])
    
    # Combine rewards for continuous plot
    all_rewards = train_results["rewards"].copy()
    all_task_rewards = train_results["task_rewards"].copy()
    
    # Add evaluation rewards
    for task_name in eval_results["per_task_rewards"]:
        all_rewards.extend(eval_results["per_task_rewards"][task_name])
        all_task_rewards.extend(eval_results["per_task_rewards"][task_name])
    
    # Plot 1: All rewards
    ax1 = axes[0]
    episodes = range(len(all_rewards))
    ax1.plot(episodes, all_rewards, alpha=0.3, color='blue', label='Raw Reward')
    
    if len(all_rewards) >= window:
        smoothed = pd.Series(all_rewards).rolling(window).mean()
        ax1.plot(episodes, smoothed, color='darkblue', linewidth=2, 
                 label=f'Smoothed (window={window})')
    
    # Add task boundaries for training
    tasks = train_results["tasks"]
    cumulative = 0
    for i, task in enumerate(tasks[:-1]):
        cumulative += task["duration"]
        ax1.axvline(x=cumulative, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add EVALUATION boundary
    ax1.axvline(x=train_episodes, color='green', linestyle='-', alpha=0.9, linewidth=3)
    ax1.text(train_episodes + 5, ax1.get_ylim()[1] * 0.9, 'EVALUATION →', 
            fontsize=12, color='green', fontweight='bold')
    
    # Add compositional task boundaries
    eval_tasks = get_compositional_tasks()
    eps_per_eval = eval_results.get("episodes_per_task", 100)
    for i in range(len(eval_tasks) - 1):
        boundary = train_episodes + (i + 1) * eps_per_eval
        ax1.axvline(x=boundary, color='purple', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.axvspan(0, train_episodes, alpha=0.1, color='blue', label='Training Phase')
    ax1.axvspan(train_episodes, len(all_rewards), alpha=0.1, color='green', label='Evaluation Phase')
    
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training (Simple Tasks) → Evaluation (Compositional Tasks)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Task success summary
    ax2 = axes[1]
    
    # Training success (approximate from rewards)
    train_task_names = [t["name"] for t in tasks]
    train_success = []
    cumulative = 0
    for task in tasks:
        task_rewards = train_results["task_rewards"][cumulative:cumulative + task["duration"]]
        success_rate = sum(1 for r in task_rewards if r > 0) / len(task_rewards) * 100 if task_rewards else 0
        train_success.append(success_rate)
        cumulative += task["duration"]
    
    # Evaluation success
    eval_task_names = list(eval_results["per_task_metrics"].keys())
    eval_success = [eval_results["per_task_metrics"][t]["success_rate"] * 100 
                   for t in eval_task_names]
    
    # Combined bar plot
    all_names = train_task_names + eval_task_names
    all_success = train_success + eval_success
    colors = ['lightblue'] * len(train_task_names) + ['lightgreen'] * len(eval_task_names)
    
    x = range(len(all_names))
    bars = ax2.bar(x, all_success, color=colors, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_names, rotation=45, ha='right')
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Success Rates: Training (Blue) vs Evaluation (Green)")
    ax2.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, rate in zip(bars, all_success):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.axvline(x=len(train_task_names) - 0.5, color='black', linestyle='-', linewidth=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Combined results plot saved: {save_path}")


# ==================== Training Phase ====================

def run_training_phase(env, agent, cube_model, cube_device, transform, pos_mean, pos_std,
                       max_episodes=1200, max_steps_per_episode=200):
    """
    Run TRAINING phase with simple tasks only.
    Networks ARE updated during this phase.
    """
    print("\n" + "="*60)
    print("TRAINING PHASE (Simple Tasks Only)")
    print("="*60)
    print("Training 4 feature-specific value functions")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60 + "\n")

    tasks, eps_per_simple = create_task_schedule(max_episodes)
    
    print(f"Training schedule:")
    print(f"  Episodes per simple task: {eps_per_simple}")
    cumulative = 0
    for i, task in enumerate(tasks):
        cumulative += task["duration"]
        print(f"  Task {i}: {task['name']:12} episodes {cumulative - task['duration']:4}-{cumulative:4}")
    print()

    obs, info = env.reset()
    agent.reset()

    # Tracking
    episode_rewards = []
    episode_task_rewards = []
    episode_lengths = []
    feature_losses = {f: [] for f in agent.feature_names}
    feature_update_counts = {f: 0 for f in agent.feature_names}

    # Exploration
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995

    total_steps = 0

    for episode in tqdm(range(max_episodes), desc="Training (Simple Tasks)"):
        step = 0
        episode_reward = 0
        episode_task_reward = 0
        episode_feature_losses = {f: [] for f in agent.feature_names}
        
        current_task, task_idx = get_current_task(episode, tasks)
        features_to_train = current_task["features"]

        env.set_task(current_task)
        agent.reset()

        detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
        agent.update_from_detection(detection_result)

        current_state_features = agent.get_all_state_features()

        while step < max_steps_per_episode:
            action = agent.sample_action_with_wvf(obs, current_task, epsilon=epsilon)

            obs, env_reward, terminated, truncated, info = env.step(action)
            step += 1
            total_steps += 1
            done = terminated or truncated

            feature_rewards = agent.compute_feature_rewards(info)

            task_satisfied = check_task_satisfaction(info, current_task)
            if task_satisfied:
                episode_task_reward += env_reward
            
            episode_reward += env_reward
            
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            agent.update_from_detection(detection_result)

            next_state_features = agent.get_all_state_features()

            experience = [
                current_state_features,
                action,
                next_state_features,
                feature_rewards,
                done
            ]

            # TRAIN the networks
            losses = agent.update_selected_features(experience, features_to_train)
            
            for f, loss in losses.items():
                if loss > 0:
                    episode_feature_losses[f].append(loss)
                    feature_update_counts[f] += 1

            current_state_features = next_state_features

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_task_rewards.append(episode_task_reward)
        episode_lengths.append(step)

        for f in agent.feature_names:
            if len(episode_feature_losses[f]) > 0:
                feature_losses[f].append(np.mean(episode_feature_losses[f]))
            else:
                feature_losses[f].append(0.0)

        obs, info = env.reset()
        agent.reset()

    print(f"\n✓ Training complete!")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Final epsilon: {epsilon:.4f}")
    
    print("\nTraining updates per feature:")
    for f in agent.feature_names:
        print(f"  {f}: {feature_update_counts[f]} updates")

    return {
        "rewards": episode_rewards,
        "task_rewards": episode_task_rewards,
        "lengths": episode_lengths,
        "feature_losses": feature_losses,
        "tasks": tasks,
        "final_epsilon": epsilon,
        "feature_update_counts": feature_update_counts,
    }


# ==================== Evaluation Phase ====================

def run_evaluation_phase(env, agent, cube_model, cube_device, transform, pos_mean, pos_std,
                         episodes_per_task=100, max_steps_per_episode=200, 
                         run_diagnostics=True):
    """
    Run EVALUATION phase with compositional tasks.
    Networks are NOT updated - only composed via min().
    """
    print("\n" + "="*60)
    print("EVALUATION PHASE (Compositional Tasks - No Training)")
    print("="*60)
    print("Evaluating composition of learned value functions")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60 + "\n")

    compositional_tasks = get_compositional_tasks()
    
    print("Evaluation tasks:")
    for task in compositional_tasks:
        print(f"  {task['name']}: {task['features']}")
    print()

    # Fixed low epsilon for evaluation (minimal exploration)
    epsilon = 0.05
    
    # Results tracking
    per_task_metrics = {}
    per_task_rewards = {}
    all_rewards = []
    all_lengths = []

    # =========================================================
    # RUN DIAGNOSTICS ONCE BEFORE EVALUATION (if enabled)
    # =========================================================
    if run_diagnostics:
        print("\n" + "="*60)
        print("RUNNING DIAGNOSTICS")
        print("="*60)
        
        # Check Q-value scales across all features
        check_q_value_scales(agent)
        
        # Run detailed diagnostic for each compositional task
        for task in compositional_tasks:
            # Reset for fresh episode
            obs, info = env.reset()
            agent.reset()
            
            # Update agent's observation
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            agent.update_from_detection(detection_result)
            
            # Run diagnostic for this task
            diagnose_composition(agent, task, obs, info)
            
            # Visualize value maps (saves to file)
            diagnose_value_maps(agent, task)
        
        print("\n" + "="*60)
        print("DIAGNOSTICS COMPLETE - Starting Evaluation")
        print("="*60 + "\n")

    # =========================================================
    # MAIN EVALUATION LOOP
    # =========================================================
    obs, info = env.reset()
    agent.reset()

    for task in compositional_tasks:
        task_name = task["name"]
        print(f"\nEvaluating: {task_name}")
        
        task_rewards = []
        task_lengths = []
        task_successes = 0
        
        # Track diagnostic info for first few episodes
        no_goals_count = 0
        
        for episode in tqdm(range(episodes_per_task), desc=f"  {task_name}"):
            step = 0
            episode_reward = 0
            episode_success = False
            
            agent.reset()
            
            # Initial detection
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            agent.update_from_detection(detection_result)
            
            # Check if we have valid goals (diagnostic)
            if episode < 5:  # Only check first 5 episodes
                goals = agent._get_goals_for_task(task)
                if len(goals) == 0:
                    no_goals_count += 1

            while step < max_steps_per_episode:
                # Use composed Q-values (NO TRAINING)
                action = agent.sample_action_with_wvf(obs, task, epsilon=epsilon)

                obs, env_reward, terminated, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated

                # Check task satisfaction
                if check_task_satisfaction(info, task):
                    episode_success = True
                    episode_reward += env_reward
                
                # Update detection (for next action selection)
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                agent.update_from_detection(detection_result)

                if done:
                    break

            task_rewards.append(episode_reward)
            task_lengths.append(step)
            if episode_success:
                task_successes += 1
            
            all_rewards.append(episode_reward)
            all_lengths.append(step)

            obs, info = env.reset()
            # Note: agent.reset() happens at start of next episode

        # Warn if goals were missing
        if no_goals_count > 0:
            print(f"  ⚠ WARNING: {no_goals_count}/5 initial episodes had NO valid goals!")
            print(f"    This suggests vision/detection issues for {task_name}")

        # Compute metrics for this task
        per_task_metrics[task_name] = {
            "success_rate": task_successes / episodes_per_task,
            "avg_reward": np.mean(task_rewards),
            "avg_length": np.mean(task_lengths),
            "total_successes": task_successes,
        }
        per_task_rewards[task_name] = task_rewards
        
        print(f"  Success rate: {task_successes}/{episodes_per_task} = {task_successes/episodes_per_task*100:.1f}%")
        print(f"  Avg reward: {np.mean(task_rewards):.3f}")
        print(f"  Avg length: {np.mean(task_lengths):.1f}")

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task_name, metrics in per_task_metrics.items():
        print(f"  {task_name:12}: {metrics['success_rate']*100:5.1f}% success, "
              f"avg reward {metrics['avg_reward']:.3f}")
    
    overall_success = sum(m["total_successes"] for m in per_task_metrics.values())
    total_episodes = episodes_per_task * len(compositional_tasks)
    print(f"\n  OVERALL: {overall_success}/{total_episodes} = {overall_success/total_episodes*100:.1f}%")

    return {
        "per_task_metrics": per_task_metrics,
        "per_task_rewards": per_task_rewards,
        "all_rewards": all_rewards,
        "all_lengths": all_lengths,
        "episodes_per_task": episodes_per_task,
    }

# ==================== Main ====================

if __name__ == "__main__":
    from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
    from utils import generate_save_path
    
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model kwargs with shared state dimension
    model_kwargs = {
        'state_dim': 682,    # 4*13*13 + 2 + 4 = 682
        'num_actions': 3,
        'hidden_dim': 128
    }
    
    # Create agent
    agent = CompositionalWVFAgent(
        env=env,
        wvf_model_class=WVF_MLP,
        model_kwargs=model_kwargs,
        lr=0.0005,
        gamma=0.99,
        device=device,
        grid_size=env.size,
        target_update_freq=100,
        confidence_threshold=0.5
    )
    
    print(f"\nAgent created with {len(agent.feature_names)} feature networks:")
    print(f"  State dimension: {model_kwargs['state_dim']} (shared across all features)")
    for f in agent.feature_names:
        print(f"  - {f}: {sum(p.numel() for p in agent.wvf_models[f].parameters())} parameters")
    print()
    
    # Load vision model
    print("Loading cube detector model...")
    cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
        'models/advanced_cube_detector.pth', force_cpu=False
    )

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==================== TRAINING PHASE ====================
    train_results = run_training_phase(
        env, agent, cube_model, cube_device, transform, pos_mean, pos_std,
        max_episodes=1000,  # Simple tasks only
        max_steps_per_episode=200,
    )
    
    # Plot training results
    plot_training_results(
        train_results,
        save_path=generate_save_path('compositional_wvf_fixed/training_results.png')
    )
    
    # Save model checkpoints AFTER training
    print("\nSaving model checkpoints...")
    for feature in agent.feature_names:
        checkpoint_path = generate_save_path(f"compositional_wvf_fixed/wvf_{feature}.pth")
        torch.save({
            'model_state_dict': agent.wvf_models[feature].state_dict(),
            'optimizer_state_dict': agent.optimizers[feature].state_dict(),
            'update_count': agent.update_counts[feature],
        }, checkpoint_path)
        print(f"  ✓ Saved {feature} model: {checkpoint_path}")

    # ==================== EVALUATION PHASE ====================
    eval_results = run_evaluation_phase(
        env, agent, cube_model, cube_device, transform, pos_mean, pos_std,
        episodes_per_task=100,  # 100 episodes per compositional task
        max_steps_per_episode=200,
    )
    
    # Plot evaluation results
    plot_evaluation_results(
        eval_results,
        save_path=generate_save_path('compositional_wvf_fixed/evaluation_results.png')
    )
    
    # Plot combined results
    plot_combined_results(
        train_results, eval_results,
        save_path=generate_save_path('compositional_wvf_fixed/combined_results.png')
    )
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("✓ TRAINING AND EVALUATION COMPLETE!")
    print("="*60)