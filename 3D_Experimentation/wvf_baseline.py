"""
Training script for Compositional WVF Agent.

This agent uses 4 feature-specific value functions (red, blue, box, sphere)
that can be composed at runtime to solve compositional tasks.

Tasks:
    - Simple: ['blue'], ['red'], ['box'], ['sphere']
    - Compositional: ['blue', 'sphere'], ['red', 'box'], etc.

The agent learns each feature's value function independently, then composes
them via min() to solve multi-feature tasks.
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import miniworld
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import CompositionalWVFAgent
from models import WVF_MLP
from tqdm import tqdm
from utils import generate_save_path
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

# Import the 4-object cube detector from train_vision
from train_vision import CubeDetector


# ==================== Task System ====================

def create_task_schedule(total_episodes):
    """
    Create interleaved simple and compositional tasks.
    
    Simple tasks have 1 feature requirement.
    Compositional tasks have 2 feature requirements (AND logic).
    
    Args:
        total_episodes: Total number of training episodes
        
    Returns:
        tasks: List of task dicts
        episodes_per_task: Number of episodes per task
    """
    # Define task pools
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
    
    # Interleave: simple, compositional, simple, compositional, ...
    interleaved = []
    for i in range(max(len(simple_tasks), len(compositional_tasks))):
        if i < len(simple_tasks):
            interleaved.append(simple_tasks[i])
        if i < len(compositional_tasks):
            interleaved.append(compositional_tasks[i])
    
    # Calculate episodes per task
    num_tasks = len(interleaved)
    episodes_per_task = total_episodes // num_tasks
    
    # Assign durations
    for task in interleaved:
        task["duration"] = episodes_per_task
    
    return interleaved, episodes_per_task


def check_task_satisfaction(info, task):
    """
    Check if contacted object satisfies current task requirements.
    
    Args:
        info: Environment info dict with 'contacted_object' key
        task: Task dict with 'features' list
        
    Returns:
        bool: True if task is satisfied
    """
    contacted_object = info.get('contacted_object', None)
    
    # No contact = no satisfaction
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


# ==================== Vision Model ====================

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """
    Load the trained 4-object cube detector model.
    
    Args:
        model_path: Path to saved model
        force_cpu: If True, force CPU mode
        
    Returns:
        model: Loaded CubeDetector
        device: torch device
        pos_mean: Position normalization mean
        pos_std: Position normalization std
    """
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    
    # Load checkpoint
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
    """
    Run cube detection with classification + regression output.
    
    Detects 4 object types: red_box, blue_box, red_sphere, blue_sphere
    
    Args:
        model: CubeDetector model
        obs: Observation (image or dict with 'image' key)
        device: torch device
        transform: Image transform
        pos_mean: Position denormalization mean
        pos_std: Position denormalization std
        
    Returns:
        dict with:
            - detected_objects: list of detected object names
            - predictions: boolean array of predictions
            - probabilities: dict of confidence scores per object
            - positions: dict of (dx, dz) positions per object
            - regression_raw: raw regression values
    """
    # Extract image
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
    # Convert to PIL Image
    if isinstance(img, np.ndarray):
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img)
    
    # Apply transform and move to device
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        cls_logits, pos_preds = model(img_tensor)
        
        # Multi-label classification with sigmoid
        probs = torch.sigmoid(cls_logits)
        predictions = (probs > 0.5).cpu().numpy()[0]
        
        # Denormalize regression output (8 values)
        regression_values = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        # Label names match the model's output order
        label_names = ["red_box", "blue_box", "red_sphere", "blue_sphere"]
        detected_objects = [label_names[i] for i in range(4) if predictions[i]]
    
    # Extract positions for detected objects
    # Positions are: [red_box_dx, red_box_dz, blue_box_dx, blue_box_dz, 
    #                 red_sphere_dx, red_sphere_dz, blue_sphere_dx, blue_sphere_dz]
    positions = {
        'red_box': (regression_values[0], regression_values[1]) if predictions[0] else None,
        'blue_box': (regression_values[2], regression_values[3]) if predictions[1] else None,
        'red_sphere': (regression_values[4], regression_values[5]) if predictions[2] else None,
        'blue_sphere': (regression_values[6], regression_values[7]) if predictions[3] else None,
    }
    
    # Probabilities dict
    probabilities = {label_names[i]: float(probs[0, i]) for i in range(4)}
    
    return {
        "detected_objects": detected_objects,
        "predictions": predictions,
        "probabilities": probabilities,
        "positions": positions,
        "regression_raw": regression_values
    }


# ==================== Visualization ====================

def plot_task_rewards(task_rewards, tasks, episodes_per_task, max_episodes):
    """
    Plot rewards with task boundaries and labels.
    
    Args:
        task_rewards: List of (episode, task_reward, env_reward, task_name) tuples
        tasks: List of task dicts
        episodes_per_task: Episodes per task
        max_episodes: Total episodes
    """
    # Extract data
    episodes = [r[0] for r in task_rewards]
    task_rewards_values = [r[1] for r in task_rewards]
    env_rewards_values = [r[2] for r in task_rewards]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Task-specific rewards
    ax1.plot(episodes, task_rewards_values, alpha=0.3, color='blue', label='Raw Task Reward')
    
    # Smooth
    window = 50
    if len(task_rewards_values) >= window:
        smoothed = pd.Series(task_rewards_values).rolling(window).mean()
        ax1.plot(episodes, smoothed, color='darkblue', linewidth=2, 
                 label=f'Smoothed (window={window})')
    
    # Add vertical lines for task boundaries
    for i in range(1, len(tasks)):
        boundary = i * episodes_per_task
        ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_ylabel('Task-Specific Reward', fontsize=12)
    ax1.set_title('Reward Over Episodes with Task Composition', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Environment rewards (for comparison)
    ax2.plot(episodes, env_rewards_values, alpha=0.3, color='green', label='Raw Env Reward')
    
    if len(env_rewards_values) >= window:
        smoothed_env = pd.Series(env_rewards_values).rolling(window).mean()
        ax2.plot(episodes, smoothed_env, color='darkgreen', linewidth=2, 
                 label=f'Smoothed (window={window})')
    
    # Add vertical lines for task boundaries
    for i in range(1, len(tasks)):
        boundary = i * episodes_per_task
        ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Environment Reward', fontsize=12)
    ax2.set_title('Environment Reward (All Objects)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add task labels on x-axis
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    
    # Create tick positions at middle of each task period
    tick_positions = [(i * episodes_per_task + (i + 1) * episodes_per_task) / 2 
                     for i in range(len(tasks))]
    tick_labels = [task['name'] for task in tasks]
    
    ax2_twin.set_xticks(tick_positions)
    ax2_twin.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax2_twin.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = generate_save_path('compositional_wvf/task_rewards.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Task reward plot saved: {save_path}")


def visualize_compositional_progress(agent, env, episode, current_task, feature_losses):
    """
    Save visualization plots for compositional WVF agent.
    
    Args:
        agent: CompositionalWVFAgent
        env: Environment
        episode: Current episode number
        current_task: Current task dict
        feature_losses: Dict of loss lists per feature
    """
    # Plot 1: All 4 feature reward maps + composed map
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    try:
        agent_x, agent_z = agent._get_agent_pos_from_env()
    except:
        agent_x, agent_z = 0, 0
    
    for idx, feature in enumerate(agent.feature_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        im = ax.imshow(agent.feature_reward_maps[feature], cmap='viridis', origin='lower')
        ax.plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
        ax.set_title(f'Feature: {feature}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Composed map for current task
    composed_map = agent.get_composed_reward_map(current_task)
    ax = axes[1, 2]
    im = ax.imshow(composed_map, cmap='viridis', origin='lower')
    ax.plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
    ax.set_title(f'Composed: {current_task["name"]}')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplot
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, f'Episode {episode}\nTask: {current_task["name"]}',
                    ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'compositional_wvf/feature_maps_ep{episode}.png'), dpi=150)
    plt.close()
    
    # Plot 2: Loss curves for all features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, feature in enumerate(agent.feature_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        losses = feature_losses[feature]
        if len(losses) > 0:
            ax.plot(losses, alpha=0.3, label='Raw')
            window = min(50, len(losses))
            if len(losses) >= window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), smoothed, 'r-', 
                        linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'WVF Loss: {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'compositional_wvf/losses_ep{episode}.png'), dpi=150)
    plt.close()
    
    # Plot 3: Egocentric observations for each feature
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for idx, feature in enumerate(agent.feature_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        im = ax.imshow(agent.feature_ego_obs[feature], cmap='viridis', origin='upper')
        ax.plot(6, 12, 'ro', markersize=8, label='Agent')  # Agent at (6, 12)
        ax.set_title(f'Ego Obs: {feature}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'compositional_wvf/ego_obs_ep{episode}.png'), dpi=150)
    plt.close()


def visualize_q_values(agent, episode, task):
    """
    Visualize Q-values for each feature network.
    
    Args:
        agent: CompositionalWVFAgent
        episode: Current episode
        task: Current task dict
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    action_names = ['Turn Left', 'Turn Right', 'Move Forward']
    
    for idx, feature in enumerate(agent.feature_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        q_values = agent.get_all_q_values_for_feature(feature)
        # Show max Q across actions
        max_q = np.max(q_values, axis=2)
        im = ax.imshow(max_q, cmap='viridis', origin='lower')
        ax.set_title(f'Max Q-values: {feature}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'compositional_wvf/q_values_ep{episode}.png'), dpi=150)
    plt.close()


# ==================== Main Training Loop ====================

def run_compositional_wvf_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """
    Run Compositional WVF agent training with task scheduling.
    
    Args:
        env: Environment
        agent: CompositionalWVFAgent
        max_episodes: Total training episodes
        max_steps_per_episode: Max steps before truncation
        
    Returns:
        dict with training results
    """
    print("\n" + "="*60)
    print("COMPOSITIONAL WVF AGENT")
    print("="*60)
    print("4 feature-specific value functions: red, blue, box, sphere")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60 + "\n")
    
    # Create task schedule
    tasks, episodes_per_task = create_task_schedule(max_episodes)
    print(f"Task schedule: {len(tasks)} tasks, {episodes_per_task} episodes each")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task['name']} ({task['type']}) - features: {task['features']}")
    print()
    
    # Load cube detector (4-object version)
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
    task_rewards_log = []  # For plotting
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    total_detections = 0
    
    for episode in tqdm(range(max_episodes), desc="Training Compositional WVF Agent"):
        step = 0
        episode_reward = 0
        episode_task_reward = 0
        episode_feature_losses = {f: [] for f in agent.feature_names}
        
        # Determine current task
        task_idx = episode // episodes_per_task
        current_task = tasks[min(task_idx, len(tasks) - 1)]
        
        # Set task in environment (for termination logic)
        if hasattr(env, 'set_task'):
            env.set_task(current_task)
        
        # Reset agent maps
        agent.reset()
        
        # Get initial detection and update agent
        detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
        agent.update_from_detection(detection_result)
        
        if detection_result['detected_objects']:
            total_detections += len(detection_result['detected_objects'])
        
        # Store initial state features for all features
        current_state_features = agent.get_all_state_features()
        
        while step < max_steps_per_episode:
            # Select action using composed WVF for current task
            action = agent.sample_action_with_wvf(obs, current_task, epsilon=epsilon)
            
            # Step environment
            obs, env_reward, terminated, truncated, info = env.step(action)
            step += 1
            total_steps += 1
            done = terminated or truncated
            
            # Compute feature-specific rewards
            feature_rewards = agent.compute_feature_rewards(info)
            
            # Check if task was satisfied
            task_satisfied = check_task_satisfaction(info, current_task)
            if task_satisfied:
                episode_task_reward += env_reward
            
            episode_reward += env_reward
            
            # Update detection and agent state
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            agent.update_from_detection(detection_result)
            
            if detection_result['detected_objects']:
                total_detections += len(detection_result['detected_objects'])
            
            # Get next state features
            next_state_features = agent.get_all_state_features()
            
            # Create experience tuple (with feature-specific components)
            experience = [
                current_state_features,   # Dict: feature -> state_features
                action,                    # int
                next_state_features,       # Dict: feature -> state_features
                feature_rewards,           # Dict: feature -> reward
                done                       # bool
            ]
            
            # Update ALL feature networks
            losses = agent.update_all_features(experience)
            
            for f, loss in losses.items():
                if loss > 0:
                    episode_feature_losses[f].append(loss)
            
            # Move to next step
            current_state_features = next_state_features
            
            if done:
                break
        
        # Episode complete
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_task_rewards.append(episode_task_reward)
        episode_lengths.append(step)
        task_rewards_log.append((episode, episode_task_reward, episode_reward, current_task['name']))
        
        # Track losses
        for f in agent.feature_names:
            if len(episode_feature_losses[f]) > 0:
                feature_losses[f].append(np.mean(episode_feature_losses[f]))
            else:
                feature_losses[f].append(0.0)
        
        # Visualization every N episodes
        if episode % 250 == 0 or episode == max_episodes - 1:
            visualize_compositional_progress(agent, env, episode, current_task, feature_losses)
            
            # Also visualize Q-values occasionally
            if episode % 500 == 0 or episode == max_episodes - 1:
                visualize_q_values(agent, episode, current_task)
        
        # Reset for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Final epsilon: {epsilon:.4f}")
    
    # Final losses summary
    print("\nFinal losses per feature (last 100 episodes):")
    for f in agent.feature_names:
        if len(feature_losses[f]) >= 100:
            mean_loss = np.mean(feature_losses[f][-100:])
            print(f"  {f}: {mean_loss:.4f}")
    
    return {
        "rewards": episode_rewards,
        "task_rewards": episode_task_rewards,
        "lengths": episode_lengths,
        "feature_losses": feature_losses,
        "task_rewards_log": task_rewards_log,
        "tasks": tasks,
        "episodes_per_task": episodes_per_task,
        "final_epsilon": epsilon,
    }


def plot_final_results(results, window=100):
    """
    Plot final training results.
    
    Args:
        results: Dict from run_compositional_wvf_agent
        window: Smoothing window size
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Episode rewards
    ax1 = axes[0, 0]
    mean_smooth = pd.Series(results["rewards"]).rolling(window).mean()
    ax1.plot(results["rewards"], alpha=0.3, label='Raw')
    ax1.plot(mean_smooth, linewidth=2, label='Smoothed')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards")
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
    
    # Plot 4: Feature losses (all on same plot)
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
    save_path = generate_save_path("compositional_wvf/final_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Final results plot saved: {save_path}")

if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model kwargs (same architecture as before)
    model_kwargs = {
        'state_dim': 175,    # 13*13 + 2 + 4
        'num_actions': 3,
        'hidden_dim': 128
    }
    
    # Create compositional WVF agent
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
    for f in agent.feature_names:
        print(f"  - {f}: {sum(p.numel() for p in agent.wvf_models[f].parameters())} parameters")
    print()
    
    # Run training
    results = run_compositional_wvf_agent(
        env,
        agent,
        max_episodes=5000,
        max_steps_per_episode=200
    )
    
    # Plot task rewards with boundaries
    plot_task_rewards(
        results["task_rewards_log"], 
        results["tasks"], 
        results["episodes_per_task"], 
        5000
    )
    
    # Plot final results
    plot_final_results(results)
    
    # Save model checkpoints
    print("\nSaving model checkpoints...")
    for feature in agent.feature_names:
        checkpoint_path = generate_save_path(f"compositional_wvf/wvf_{feature}.pth")
        torch.save({
            'model_state_dict': agent.wvf_models[feature].state_dict(),
            'optimizer_state_dict': agent.optimizers[feature].state_dict(),
            'update_count': agent.update_counts[feature],
        }, checkpoint_path)
        print(f"  ✓ Saved {feature} model: {checkpoint_path}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✓ Training complete!")