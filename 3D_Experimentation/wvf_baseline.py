"""
Training script for Compositional WVF Agent - FIXED VERSION.

Key Fix: Uses shared state representation across all feature networks.

This ensures:
- All networks learn identical navigation dynamics (G*)
- Composition via min() correctly combines value functions
- Transfer to compositional tasks works as intended

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


# ==================== Task System ====================

def create_task_schedule(total_episodes, simple_ratio=0.6):
    """
    Create SEQUENTIAL task schedule: all simple tasks first, then compositional.
    """
    simple_tasks = [
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"},
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
    ]
    
    compositional_tasks = [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]

    simple_episodes = int(total_episodes * simple_ratio)
    comp_episodes = total_episodes - simple_episodes
    
    episodes_per_simple = simple_episodes // len(simple_tasks)
    episodes_per_comp = comp_episodes // len(compositional_tasks)
    
    tasks = []
    
    for task in simple_tasks:
        task_copy = task.copy()
        task_copy["duration"] = episodes_per_simple
        tasks.append(task_copy)
    
    for task in compositional_tasks:
        task_copy = task.copy()
        task_copy["duration"] = episodes_per_comp
        tasks.append(task_copy)

    return tasks, episodes_per_simple, episodes_per_comp


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
    # Import here to avoid circular imports
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

def plot_task_rewards(task_rewards, tasks, max_episodes, save_path='task_rewards_fixed.png'):
    """Plot rewards with task boundaries and labels."""
    episodes = [r[0] for r in task_rewards]
    task_rewards_values = [r[1] for r in task_rewards]
    env_rewards_values = [r[2] for r in task_rewards]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Task-specific rewards
    ax1.plot(episodes, task_rewards_values, alpha=0.3, color='blue', label='Raw Task Reward')
    
    window = 50
    if len(task_rewards_values) >= window:
        smoothed = pd.Series(task_rewards_values).rolling(window).mean()
        ax1.plot(episodes, smoothed, color='darkblue', linewidth=2, 
                 label=f'Smoothed (window={window})')
    
    # Add vertical lines for task boundaries
    cumulative = 0
    for i, task in enumerate(tasks[:-1]):
        cumulative += task["duration"]
        ax1.axvline(x=cumulative, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        if i == 3:  # After last simple task
            ax1.axvline(x=cumulative, color='green', linestyle='-', alpha=0.9, linewidth=2.5)
            ax1.text(cumulative, ax1.get_ylim()[1] * 0.9, ' COMPOSITIONAL→', 
                    fontsize=10, color='green', fontweight='bold')
    
    ax1.set_ylabel('Task-Specific Reward', fontsize=12)
    ax1.set_title('FIXED: Reward Over Episodes (Simple → Compositional)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Environment rewards
    ax2.plot(episodes, env_rewards_values, alpha=0.3, color='green', label='Raw Env Reward')
    
    if len(env_rewards_values) >= window:
        smoothed_env = pd.Series(env_rewards_values).rolling(window).mean()
        ax2.plot(episodes, smoothed_env, color='darkgreen', linewidth=2, 
                 label=f'Smoothed (window={window})')
    
    cumulative = 0
    for i, task in enumerate(tasks[:-1]):
        cumulative += task["duration"]
        ax2.axvline(x=cumulative, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Environment Reward', fontsize=12)
    ax2.set_title('Environment Reward (All Objects)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Task reward plot saved: {save_path}")


def plot_final_results(results, save_path='final_results_fixed.png', window=100):
    """Plot final training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Episode rewards
    ax1 = axes[0, 0]
    mean_smooth = pd.Series(results["rewards"]).rolling(window).mean()
    ax1.plot(results["rewards"], alpha=0.3, label='Raw')
    ax1.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards (FIXED)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Episode lengths
    ax2 = axes[0, 1]
    mean_smooth = pd.Series(results["lengths"]).rolling(window).mean()
    ax2.plot(results["lengths"], alpha=0.3, label='Raw')
    ax2.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Lengths")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Task-specific rewards
    ax3 = axes[1, 0]
    mean_smooth = pd.Series(results["task_rewards"]).rolling(window).mean()
    ax3.plot(results["task_rewards"], alpha=0.3, label='Raw')
    ax3.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Task Reward")
    ax3.set_title("Task-Specific Rewards")
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
    ax4.set_title("Feature WVF Losses")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Final results plot saved: {save_path}")


# ==================== Main Training Loop ====================

def run_compositional_wvf_agent_fixed(env, agent, max_episodes=2000, max_steps_per_episode=200,
                                       simple_ratio=0.6, selective_training=True):
    """
    Run FIXED Compositional WVF agent training.
    
    Key difference: Agent uses shared state representation, enabling proper composition.
    """
    print("\n" + "="*60)
    print("COMPOSITIONAL WVF AGENT (FIXED - SHARED STATE)")
    print("="*60)
    print("4 feature-specific value functions with SHARED world view")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Simple task ratio: {simple_ratio*100:.0f}%")
    print(f"Selective training: {selective_training}")
    print("="*60 + "\n")

    tasks, eps_per_simple, eps_per_comp = create_task_schedule(max_episodes, simple_ratio)
    
    print(f"Task schedule (SEQUENTIAL):")
    print(f"  Simple tasks: {eps_per_simple} episodes each")
    print(f"  Compositional tasks: {eps_per_comp} episodes each")
    print()
    cumulative = 0
    for i, task in enumerate(tasks):
        cumulative += task["duration"]
        print(f"  Task {i}: {task['name']:12} ({task['type']:13}) episodes {cumulative - task['duration']:4}-{cumulative:4}")
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

    obs, info = env.reset()
    agent.reset()

    # Tracking
    episode_rewards = []
    episode_task_rewards = []
    episode_lengths = []
    feature_losses = {f: [] for f in agent.feature_names}
    task_rewards_log = []
    feature_update_counts = {f: 0 for f in agent.feature_names}

    # Exploration
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995

    total_steps = 0
    total_detections = 0

    for episode in tqdm(range(max_episodes), desc="Training Fixed WVF Agent"):
        step = 0
        episode_reward = 0
        episode_task_reward = 0
        episode_feature_losses = {f: [] for f in agent.feature_names}
        
        current_task, task_idx = get_current_task(episode, tasks)
        
        if selective_training:
            features_to_train = current_task["features"]
        else:
            features_to_train = agent.feature_names
        
        if hasattr(env, 'set_task'):
            env.set_task(current_task)

        agent.reset()

        detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
        agent.update_from_detection(detection_result)

        if detection_result['detected_objects']:
            total_detections += len(detection_result['detected_objects'])

        # Get SHARED state for all features
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
            
            if detection_result['detected_objects']:
                total_detections += len(detection_result['detected_objects'])

            # Get SHARED next state
            next_state_features = agent.get_all_state_features()

            experience = [
                current_state_features,
                action,
                next_state_features,
                feature_rewards,
                done
            ]

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
        task_rewards_log.append((episode, episode_task_reward, episode_reward, current_task['name']))

        for f in agent.feature_names:
            if len(episode_feature_losses[f]) > 0:
                feature_losses[f].append(np.mean(episode_feature_losses[f]))
            else:
                feature_losses[f].append(0.0)

        obs, info = env.reset()
        agent.reset()

    print(f"\n✓ Training complete!")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Final epsilon: {epsilon:.4f}")
    
    print("\nTraining updates per feature:")
    for f in agent.feature_names:
        print(f"  {f}: {feature_update_counts[f]} updates")

    return {
        "rewards": episode_rewards,
        "task_rewards": episode_task_rewards,
        "lengths": episode_lengths,
        "feature_losses": feature_losses,
        "task_rewards_log": task_rewards_log,
        "tasks": tasks,
        "final_epsilon": epsilon,
        "feature_update_counts": feature_update_counts,
    }


# ==================== Main ====================

if __name__ == "__main__":
    # Import environment (adjust path as needed)
    from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
    from utils import generate_save_path
    
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # UPDATED model kwargs with new state dimension
    model_kwargs = {
        'state_dim': 682,    # 4*13*13 + 2 + 4 = 682 (was 175)
        'num_actions': 3,
        'hidden_dim': 128
    }
    
    # Create FIXED agent
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
    
    print(f"\nFIXED Agent created with {len(agent.feature_names)} feature networks:")
    print(f"  State dimension: {model_kwargs['state_dim']} (shared across all features)")
    for f in agent.feature_names:
        print(f"  - {f}: {sum(p.numel() for p in agent.wvf_models[f].parameters())} parameters")
    print()
    
    # Run training
    results = run_compositional_wvf_agent_fixed(
        env,
        agent,
        max_episodes=2000,
        max_steps_per_episode=200,
        simple_ratio=0.6,
        selective_training=True
    )
    
    # Plot results
    plot_task_rewards(
        results["task_rewards_log"], 
        results["tasks"], 
        4000,
        save_path=generate_save_path('compositional_wvf_fixed/task_rewards.png')
    )

    plot_final_results(
        results,
        save_path=generate_save_path('compositional_wvf_fixed/final_results.png')
    )
    
    # Save checkpoints
    print("\nSaving model checkpoints...")
    for feature in agent.feature_names:
        checkpoint_path = generate_save_path(f"compositional_wvf_fixed/wvf_{feature}.pth")
        torch.save({
            'model_state_dict': agent.wvf_models[feature].state_dict(),
            'optimizer_state_dict': agent.optimizers[feature].state_dict(),
            'update_count': agent.update_counts[feature],
        }, checkpoint_path)
        print(f"  ✓ Saved {feature} model: {checkpoint_path}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✓ Training complete!")