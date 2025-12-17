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

import matplotlib
matplotlib.use('Agg')  

import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import SuccessorAgent
from tqdm import tqdm
import math
from utils import plot_sr_matrix, generate_save_path, save_all_wvf
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from models import Autoencoder
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque
import gc
import pandas as pd
from train_vision import CubeDetector


def plot_training_and_eval_rewards(training_rewards, eval_rewards, eval_task_names):
    """Plot training rewards (0-7999) and evaluation rewards (8000-8799) with task labels"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Training rewards (episodes 0-7999)
    training_episodes = list(range(len(training_rewards)))
    ax1.plot(training_episodes, training_rewards, alpha=0.3, color='blue', label='Raw Training Reward')
    
    # Smooth training
    window = 50
    if len(training_rewards) >= window:
        smoothed = pd.Series(training_rewards).rolling(window).mean()
        ax1.plot(training_episodes, smoothed, color='darkblue', linewidth=2, label=f'Smoothed (window={window})')
    
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_xlabel('Training Episodes', fontsize=12)
    ax1.set_title('Training Phase: Random Primitive Tasks', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation rewards (episodes 8000-8799)
    eval_episodes = list(range(8000, 8000 + len(eval_rewards)))
    ax2.plot(eval_episodes, eval_rewards, alpha=0.3, color='green', label='Raw Eval Reward')
    
    # Smooth eval
    if len(eval_rewards) >= window:
        smoothed_eval = pd.Series(eval_rewards).rolling(window).mean()
        ax2.plot(eval_episodes, smoothed_eval, color='darkgreen', linewidth=2, label=f'Smoothed (window={window})')
    
    # Add vertical lines between eval tasks (every 200 episodes)
    for i in range(1, 4):
        boundary = 8000 + i * 200
        ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Evaluation Phase: Compositional Tasks', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add task labels on top x-axis
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    
    # Create tick positions at middle of each eval task period
    tick_positions = [8000 + i * 200 + 100 for i in range(4)]
    
    ax2_twin.set_xticks(tick_positions)
    ax2_twin.set_xticklabels(eval_task_names, rotation=0, ha='center', fontsize=10)
    ax2_twin.set_xlabel('Compositional Task', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = generate_save_path('training_and_eval_rewards.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training/Eval reward plot saved: {save_path}")


def plot_sr_matrix_forward(M_forward, episode, grid_size):
    """Plot the SR matrix for MOVE_FORWARD action only"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # M_forward is shape (state_size, state_size) = (100, 100)
    im = ax.imshow(M_forward, cmap='viridis', aspect='auto')
    ax.set_title(f'SR Matrix M[MOVE_FORWARD] - Episode {episode}', fontsize=14, fontweight='bold')
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, label='Successor Value')
    
    plt.tight_layout()
    save_path = generate_save_path(f'sr_matrix/M_forward_ep{episode}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SR matrix plot saved: {save_path}")


def create_task_pools():
    """Create primitive and compositional task pools"""
    
    # Primitive tasks for training
    primitive_tasks = [
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"},
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
    ]
    
    # Compositional tasks for evaluation (blocked, 200 episodes each)
    compositional_tasks = [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]
    
    return primitive_tasks, compositional_tasks


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies current task requirements"""
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

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model"""
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode to avoid CUDA compatibility issues")
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
    """Run cube detection with classification + regression output"""
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
        
        # Multi-label classification
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
    
    return {
        "detected_objects": detected_objects,
        "predictions": predictions,
        "probabilities": {label_names[i]: float(probs[0, i]) for i in range(4)},
        "positions": positions,
        "regression_raw": regression_values
    }

    
def run_successor_agent(env, agent, training_episodes=8000, eval_episodes_per_task=200, max_steps_per_episode=200):
    print("\n=== SUCCESSOR REPRESENTATION AGENT WITH RANDOM PRIMITIVES + COMPOSITIONAL EVAL ===")
    print(f"Training episodes: {training_episodes} (random primitive tasks)")
    print(f"Eval episodes: {eval_episodes_per_task * 4} (blocked compositional tasks)")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    # Create task pools
    primitive_tasks, compositional_tasks = create_task_pools()
    print(f"Primitive tasks for training: {[t['name'] for t in primitive_tasks]}")
    print(f"Compositional tasks for eval: {[t['name'] for t in compositional_tasks]}")
    print()
    
    print("Loading cube detector model...")
    cube_model, device, pos_mean, pos_std = load_cube_detector('models/advanced_cube_detector.pth', force_cpu=False)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tracking
    training_rewards = []
    eval_rewards = []
    episode_lengths = []
    
    obs, info = env.reset()
    agent.reset()

    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    total_cubes_detected = 0
    
    # Total episodes = training + eval
    total_episodes = training_episodes + (eval_episodes_per_task * len(compositional_tasks))
    
    # SR matrix saved flag
    sr_saved = False
    
    for episode in tqdm(range(total_episodes), desc="Training + Eval"):
        step = 0
        episode_reward = 0
        episode_cubes = 0
        
        # Determine current task and phase
        if episode < training_episodes:
            # TRAINING PHASE: Random primitive task
            current_task = np.random.choice(primitive_tasks)
            is_training = True
            
            # Epsilon decays during training
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
        else:
            # EVALUATION PHASE: Blocked compositional tasks
            if not sr_saved:
                # Save SR matrix after training
                sr_save_path = generate_save_path('sr_matrix_after_training.npy')
                np.save(sr_save_path, agent.M)
                print(f"\n✓ SR matrix saved after training: {sr_save_path}")
                sr_saved = True
            
            is_training = False
            epsilon = epsilon_end  # Pure exploitation during eval
            
            # Determine which compositional task (200 episodes each, blocked)
            eval_episode_idx = episode - training_episodes
            task_idx = eval_episode_idx // eval_episodes_per_task
            current_task = compositional_tasks[task_idx]

        # Pass task to environment for termination checks
        env.set_task(current_task)
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        
        while step < max_steps_per_episode:
            agent_pos = agent._get_agent_pos_from_env()

            # ========== DETECTION BEFORE STEP ==========
            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            detected_objects = detection_result['detected_objects']
            positions = detection_result['positions']

            # Update feature map with detections
            agent.update_feature_map(detected_objects, positions)
            
            # Compose reward map based on current task
            agent.compose_reward_map(current_task)
            
            # Compute WVF for action selection
            agent.compute_wvf()

            # Count detections
            if detected_objects:
                episode_cubes += len(detected_objects)
                total_cubes_detected += len(detected_objects)
    
            # ========== STEP ENVIRONMENT ==========
            obs, env_reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1

            # Check task satisfaction using info from environment
            task_satisfied = check_task_satisfaction(info, current_task)
            
            # Only count reward if task satisfied
            if task_satisfied:
                episode_reward += env_reward
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (SARSA)
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            done = terminated or truncated
            
            # Update SR matrix ONLY during training
            if is_training:
                td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)
            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            if terminated or truncated:
                break
        
        # Episode ended - track rewards
        episode_lengths.append(step)
        
        if is_training:
            training_rewards.append(episode_reward)
        else:
            eval_rewards.append(episode_reward)
        
        # Generate visualizations occasionally
        if episode % 250 == 0 or episode == total_episodes - 1:
            # Plot SR matrix (MOVE_FORWARD only)
            M_forward = agent.M[agent.MOVE_FORWARD, :, :]
            plot_sr_matrix_forward(M_forward, episode, agent.grid_size)
            
        if episode % 100 == 0 or episode == total_episodes - 1 or episode == 0:
            # Save feature maps
            agent_x, agent_z = agent_pos
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot each feature map
            for idx, (feature_name, feature_map) in enumerate(agent.feature_map.items()):
                row = idx // 3
                col = idx % 3
                im = axes[row, col].imshow(feature_map, cmap='viridis', origin='lower')
                axes[row, col].set_title(f'Feature Map: {feature_name}')
                axes[row, col].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            
            # Plot composed reward map
            im = axes[1, 1].imshow(agent.composed_reward_map, cmap='viridis', origin='lower')
            axes[1, 1].set_title(f'Composed Map: {current_task["name"]}')
            axes[1, 1].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            # Plot WVF
            im = axes[1, 2].imshow(agent.wvf, cmap='hot', origin='lower')
            axes[1, 2].set_title(f'WVF (Task: {current_task["name"]})')
            axes[1, 2].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f"feature_maps/ep{episode}.png"), dpi=150)
            plt.close()
        
        # Reset environment for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training + Evaluation complete!")
    print(f"✓ Total episodes: {total_episodes}")
    print(f"✓ Training episodes: {training_episodes}")
    print(f"✓ Eval episodes: {len(eval_rewards)}")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total cubes detected: {total_cubes_detected}")
    
    # Create training/eval reward plot
    eval_task_names = [t['name'] for t in compositional_tasks]
    plot_training_and_eval_rewards(training_rewards, eval_rewards, eval_task_names)
    
    # Compute eval statistics per task
    print("\n=== EVALUATION RESULTS ===")
    for i, task in enumerate(compositional_tasks):
        start_idx = i * eval_episodes_per_task
        end_idx = start_idx + eval_episodes_per_task
        task_rewards = eval_rewards[start_idx:end_idx]
        mean_reward = np.mean(task_rewards)
        std_reward = np.std(task_rewards)
        success_rate = np.sum(np.array(task_rewards) > 0) / len(task_rewards) * 100
        print(f"{task['name']:15s} | Mean: {mean_reward:.3f} ± {std_reward:.3f} | Success: {success_rate:.1f}%")

    return {
        "training_rewards": training_rewards,
        "eval_rewards": eval_rewards,
        "lengths": episode_lengths,
        "final_epsilon": epsilon_end,
        "algorithm": "Random Primitive Training + Compositional Eval",
        "primitive_tasks": primitive_tasks,
        "compositional_tasks": compositional_tasks
    }


def _find_convergence_episode(all_rewards, window):
    """Find approximate convergence episode"""
    mean_rewards = np.mean(all_rewards, axis=0)
    smoothed = pd.Series(mean_rewards).rolling(window).mean()

    # Simple heuristic: convergence when slope becomes small
    if len(smoothed) < window * 2:
        return len(smoothed)

    slopes = np.diff(smoothed[window:])
    convergence_threshold = 0.001

    for i, slope in enumerate(slopes):
        if abs(slope) < convergence_threshold:
            return i + window

    return len(smoothed)
    
if __name__ == "__main__":
    # create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")
    
    # create agent
    agent = SuccessorAgent(env)
    
    all_results = {}

    # Run training with random primitives + compositional eval
    results = run_successor_agent(
        env, 
        agent, 
        training_episodes=8000,
        eval_episodes_per_task=200,
        max_steps_per_episode=200 
    )

    # Store results
    all_results['Successor Agent'] = [results]

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✓ All experiments complete!")