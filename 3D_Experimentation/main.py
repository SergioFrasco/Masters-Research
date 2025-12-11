import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True" 

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


def plot_task_rewards(task_rewards, tasks, episodes_per_task, max_episodes, phase="training"):
    """Plot rewards with task boundaries and labels"""
    
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
        ax1.plot(episodes, smoothed, color='darkblue', linewidth=2, label=f'Smoothed (window={window})')
    
    # Add vertical lines for task boundaries
    for i in range(1, len(tasks)):
        boundary = i * episodes_per_task
        ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_ylabel('Task-Specific Reward', fontsize=12)
    ax1.set_title(f'{phase.capitalize()} Reward Over Episodes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Environment rewards (for comparison)
    ax2.plot(episodes, env_rewards_values, alpha=0.3, color='green', label='Raw Env Reward')
    
    if len(env_rewards_values) >= window:
        smoothed_env = pd.Series(env_rewards_values).rolling(window).mean()
        ax2.plot(episodes, smoothed_env, color='darkgreen', linewidth=2, label=f'Smoothed (window={window})')
    
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
    save_path = generate_save_path(f'{phase}_task_rewards.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {phase.capitalize()} task reward plot saved: {save_path}")


def create_training_schedule(episodes_per_task=2000):
    """Create sequential simple tasks for training (no interleaving)"""
    
    simple_tasks = [
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"},
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
    ]
    
    # Assign durations
    for task in simple_tasks:
        task["duration"] = episodes_per_task
    
    total_episodes = len(simple_tasks) * episodes_per_task
    
    return simple_tasks, episodes_per_task, total_episodes


def create_evaluation_schedule(episodes_per_task=500):
    """Create compositional tasks for evaluation (zero-shot generalization test)"""
    
    compositional_tasks = [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]
    
    # Assign durations
    for task in compositional_tasks:
        task["duration"] = episodes_per_task
    
    total_episodes = len(compositional_tasks) * episodes_per_task
    
    return compositional_tasks, episodes_per_task, total_episodes


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


def plot_wvf(wvf, episode, grid_size, maps_per_row=10):
    """Plot all world value functions in a grid"""
    num_maps = wvf.shape[0]
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    im = None
    for idx in range(num_maps):
        ax = axes[idx]
        im = ax.imshow(wvf[idx], cmap='viridis')
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=axes[:num_maps], shrink=0.6, label="WVF Value")

    save_path = generate_save_path(f'wvf_episode_{episode}.png')
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"✓ WVF plot saved: {save_path}")


def _create_target_view_with_reward(past_agent_pos, past_agent_dir, reward_pos, reward_map):
    """Create 13x13 target view from past agent position showing reward location"""
    target_13x13 = np.zeros((13, 13), dtype=np.float32)
    
    ego_center_x, ego_center_z = 6, 12
    past_x, past_z = past_agent_pos
    reward_x, reward_z = reward_pos
    
    for view_z in range(13):
        for view_x in range(13):
            dx_ego = view_x - ego_center_x
            dz_ego = view_z - ego_center_z
            
            if past_agent_dir == 3:  # North
                dx_world, dz_world = dx_ego, dz_ego
            elif past_agent_dir == 0:  # East
                dx_world, dz_world = -dz_ego, dx_ego
            elif past_agent_dir == 1:  # South
                dx_world, dz_world = -dx_ego, -dz_ego
            elif past_agent_dir == 2:  # West
                dx_world, dz_world = dz_ego, -dx_ego
            
            global_x = past_x + dx_world
            global_z = past_z + dz_world
            
            if (global_x == reward_x and global_z == reward_z):
                target_13x13[view_z, view_x] = 1.0
            else:
                target_13x13[view_z, view_x] = 0.0
    
    return target_13x13


def _train_ae_on_batch(model, optimizer, loss_fn, inputs, targets, device):
    """Train autoencoder on batch of trajectory data"""
    input_batch = np.stack([inp[np.newaxis, ..., np.newaxis] for inp in inputs])
    target_batch = np.stack([tgt[np.newaxis, ..., np.newaxis] for tgt in targets])
    
    input_tensor = torch.tensor(input_batch, dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    target_tensor = torch.tensor(target_batch, dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output, target_tensor)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def run_episode(env, agent, cube_model, device, transform, pos_mean, pos_std,
                ae_model, optimizer, loss_fn, vision_device,
                current_task, epsilon, max_steps, 
                training=True, episode_num=0, total_training_episodes=0):
    """
    Run a single episode. 
    If training=True, update SR and vision model.
    If training=False (evaluation), only run inference.
    """
    obs, info = env.reset()
    agent.reset()
    
    step = 0
    episode_reward = 0
    episode_task_reward = 0
    episode_cubes = 0
    ae_triggers_this_episode = 0
    
    # Pass task to environment
    env.set_task(current_task)
    
    # Initialize first action
    current_state = agent.get_state_index()
    current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon if training else 0.05)
    
    # Reset maps
    agent.true_reward_map = np.zeros((agent.grid_size, agent.grid_size))
    agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
    
    trajectory_buffer = deque(maxlen=10)
    
    while step < max_steps:
        agent_pos = agent._get_agent_pos_from_env()
        
        # Detection
        detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
        detected_objects = detection_result['detected_objects']
        positions = detection_result['positions']
        
        # Update feature map
        agent.update_feature_map(detected_objects, positions)
        
        # Compose reward map based on current task
        agent.compose_reward_map(current_task)
        
        # Compute WVF
        agent.compute_wvf()
        
        if detected_objects:
            episode_cubes += len(detected_objects)
        
        # Extract positions for egocentric observation
        goal_pos_red_box = positions.get('red_box')
        goal_pos_blue_box = positions.get('blue_box')
        goal_pos_red_sphere = positions.get('red_sphere')
        goal_pos_blue_sphere = positions.get('blue_sphere')
        
        if goal_pos_red_box:
            goal_pos_red_box = (int(round(goal_pos_red_box[0])), int(round(goal_pos_red_box[1])))
        if goal_pos_blue_box:
            goal_pos_blue_box = (int(round(goal_pos_blue_box[0])), int(round(goal_pos_blue_box[1])))
        if goal_pos_red_sphere:
            goal_pos_red_sphere = (int(round(goal_pos_red_sphere[0])), int(round(goal_pos_red_sphere[1])))
        if goal_pos_blue_sphere:
            goal_pos_blue_sphere = (int(round(goal_pos_blue_sphere[0])), int(round(goal_pos_blue_sphere[1])))
        
        ego_obs = agent.create_egocentric_observation(
            goal_pos_red_box=goal_pos_red_box,
            goal_pos_blue_box=goal_pos_blue_box,
            goal_pos_red_sphere=goal_pos_red_sphere,
            goal_pos_blue_sphere=goal_pos_blue_sphere,
            matrix_size=13
        )
        
        step_info = {
            'agent_view': ego_obs.copy(),
            'agent_pos': tuple(agent._get_agent_pos_from_env()),
            'agent_dir': agent._get_agent_dir_from_env(),
            'normalized_grid': ego_obs.copy()
        }
        trajectory_buffer.append(step_info)
        
        # Step environment
        obs, env_reward, terminated, truncated, info = env.step(current_action)
        step += 1
        
        # Check task satisfaction
        task_satisfied = check_task_satisfaction(info, current_task)
        
        if task_satisfied:
            task_reward = env_reward
            episode_task_reward += task_reward
        else:
            task_reward = 0
        
        episode_reward += env_reward
        
        # Get next state/action
        next_state = agent.get_state_index()
        next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon if training else 0.05)
        done = terminated or truncated
        
        # Update SR only during training
        if training:
            agent.update_sr(current_state, current_action, next_state, next_action, done)
        
        # Vision model processing
        agent_position = agent._get_agent_pos_from_env()
        agent_view = ego_obs
        
        if done:
            x, z = agent_position
            agent_view[12, 6] = 1.0
        
        input_grid = agent_view[np.newaxis, ..., np.newaxis]
        
        with torch.no_grad():
            ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(vision_device)
            predicted_reward_map_tensor = ae_model(ae_input_tensor)
            predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()
        
        agent.visited_positions[agent_position[1], agent_position[0]] = True
        
        # Vision model training only during training phase
        if training and done and step < max_steps:
            agent.true_reward_map[agent_position[1], agent_position[0]] = 1
            
            if len(trajectory_buffer) > 0:
                batch_inputs = []
                batch_targets = []
                
                for past_step in trajectory_buffer:
                    reward_global_pos = agent_position
                    past_target_13x13 = _create_target_view_with_reward(
                        past_step['agent_pos'],
                        past_step['agent_dir'],
                        reward_global_pos,
                        agent.true_reward_map
                    )
                    batch_inputs.append(past_step['normalized_grid'])
                    batch_targets.append(past_target_13x13)
                
                current_target_13x13 = _create_target_view_with_reward(
                    tuple(agent._get_agent_pos_from_env()),
                    agent._get_agent_dir_from_env(),
                    agent_position,
                    agent.true_reward_map
                )
                batch_inputs.append(ego_obs)
                batch_targets.append(current_target_13x13)
                
                _train_ae_on_batch(ae_model, optimizer, loss_fn, batch_inputs, batch_targets, vision_device)
        
        # Map predicted reward to global map
        agent_x, agent_z = agent_position
        ego_center_x = 6
        ego_center_z = 12
        agent_dir = agent._get_agent_dir_from_env()
        
        for view_z in range(13):
            for view_x in range(13):
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                if agent_dir == 3:
                    dx_world, dz_world = dx_ego, dz_ego
                elif agent_dir == 0:
                    dx_world, dz_world = -dz_ego, dx_ego
                elif agent_dir == 1:
                    dx_world, dz_world = -dx_ego, -dz_ego
                elif agent_dir == 2:
                    dx_world, dz_world = dz_ego, -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                    if not agent.visited_positions[global_z, global_x]:
                        predicted_value = predicted_reward_map_2d[view_z, view_x]
                        agent.true_reward_map[global_z, global_x] = predicted_value
        
        # Extract target from true reward map
        target_13x13 = np.zeros((13, 13), dtype=np.float32)
        
        for view_z in range(13):
            for view_x in range(13):
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                if agent_dir == 3:
                    dx_world, dz_world = dx_ego, dz_ego
                elif agent_dir == 0:
                    dx_world, dz_world = -dz_ego, dx_ego
                elif agent_dir == 1:
                    dx_world, dz_world = -dx_ego, -dz_ego
                elif agent_dir == 2:
                    dx_world, dz_world = dz_ego, -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                    target_13x13[view_z, view_x] = agent.true_reward_map[global_z, global_x]
                else:
                    target_13x13[view_z, view_x] = 0.0
        
        # Trigger training on prediction error (only during training)
        if training:
            view_error = np.abs(predicted_reward_map_2d - target_13x13)
            max_error = np.max(view_error)
            mean_error = np.mean(view_error)
            
            if max_error > 0.05 or mean_error > 0.01:
                ae_triggers_this_episode += 1
                target_tensor = torch.tensor(target_13x13[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(vision_device)
                
                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
        
        current_state = next_state
        current_action = next_action
        
        if terminated or truncated:
            break
    
    return {
        'episode_reward': episode_reward,
        'task_reward': episode_task_reward,
        'steps': step,
        'cubes_detected': episode_cubes,
        'ae_triggers': ae_triggers_this_episode,
        'task_satisfied': episode_task_reward > 0
    }


def run_training_phase(env, agent, cube_model, device, transform, pos_mean, pos_std,
                       ae_model, optimizer, loss_fn, vision_device,
                       episodes_per_task=2000, max_steps_per_episode=200):
    """Run training on simple tasks sequentially"""
    
    print("\n" + "="*60)
    print("TRAINING PHASE: Simple Tasks (Sequential)")
    print("="*60)
    
    tasks, eps_per_task, total_episodes = create_training_schedule(episodes_per_task)
    
    print(f"Training schedule: {len(tasks)} simple tasks, {eps_per_task} episodes each")
    print(f"Total training episodes: {total_episodes}")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task['name']} (features: {task['features']})")
    print()
    
    # Tracking
    episode_rewards = []
    task_specific_rewards = []
    episode_lengths = []
    task_rewards_log = []
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    total_cubes = 0
    
    for episode in tqdm(range(total_episodes), desc="Training (Simple Tasks)"):
        # Determine current task
        task_idx = episode // eps_per_task
        current_task = tasks[min(task_idx, len(tasks) - 1)]
        
        result = run_episode(
            env, agent, cube_model, device, transform, pos_mean, pos_std,
            ae_model, optimizer, loss_fn, vision_device,
            current_task, epsilon, max_steps_per_episode,
            training=True, episode_num=episode, total_training_episodes=total_episodes
        )
        
        episode_rewards.append(result['episode_reward'])
        task_specific_rewards.append(result['task_reward'])
        episode_lengths.append(result['steps'])
        task_rewards_log.append((episode, result['task_reward'], result['episode_reward'], current_task['name']))
        
        total_steps += result['steps']
        total_cubes += result['cubes_detected']
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Visualizations at task boundaries and key points
        if episode % 500 == 0 or episode == total_episodes - 1:
            agent_x, agent_z = agent._get_agent_pos_from_env()
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            for idx, (feature_name, feature_map) in enumerate(agent.feature_map.items()):
                row = idx // 3
                col = idx % 3
                im = axes[row, col].imshow(feature_map, cmap='viridis', origin='lower')
                axes[row, col].set_title(f'Feature Map: {feature_name}')
                axes[row, col].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            
            im = axes[1, 1].imshow(agent.composed_reward_map, cmap='viridis', origin='lower')
            axes[1, 1].set_title(f'Composed Map: {current_task["name"]}')
            axes[1, 1].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            im = axes[1, 2].imshow(agent.wvf, cmap='hot', origin='lower')
            axes[1, 2].set_title(f'WVF (Task: {current_task["name"]})')
            axes[1, 2].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f"training_feature_maps/ep{episode}.png"), dpi=150)
            plt.close()
        
        # Log progress at task boundaries
        if (episode + 1) % eps_per_task == 0:
            task_name = current_task['name']
            recent_rewards = task_specific_rewards[-100:] if len(task_specific_rewards) >= 100 else task_specific_rewards
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"\n  ✓ Completed training on '{task_name}' | Avg task reward (last 100): {avg_reward:.3f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total cubes detected: {total_cubes}")
    print(f"  Final epsilon: {epsilon:.4f}")
    
    # Plot training rewards
    plot_task_rewards(task_rewards_log, tasks, eps_per_task, total_episodes, phase="training")
    
    return {
        'rewards': episode_rewards,
        'task_rewards': task_specific_rewards,
        'lengths': episode_lengths,
        'tasks': tasks,
        'episodes_per_task': eps_per_task,
        'final_epsilon': epsilon,
        'task_rewards_log': task_rewards_log
    }


def run_evaluation_phase(env, agent, cube_model, device, transform, pos_mean, pos_std,
                         ae_model, optimizer, loss_fn, vision_device,
                         episodes_per_task=500, max_steps_per_episode=200):
    """Evaluate zero-shot generalization on compositional tasks (no learning)"""
    
    print("\n" + "="*60)
    print("EVALUATION PHASE: Compositional Tasks (Zero-Shot)")
    print("="*60)
    
    tasks, eps_per_task, total_episodes = create_evaluation_schedule(episodes_per_task)
    
    print(f"Evaluation schedule: {len(tasks)} compositional tasks, {eps_per_task} episodes each")
    print(f"Total evaluation episodes: {total_episodes}")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task['name']} (features: {task['features']})")
    print()
    
    # Tracking per task
    results_by_task = {task['name']: {'rewards': [], 'task_rewards': [], 'lengths': [], 'successes': 0} 
                       for task in tasks}
    
    episode_rewards = []
    task_specific_rewards = []
    episode_lengths = []
    task_rewards_log = []
    
    # Fixed low epsilon for evaluation (exploitation)
    eval_epsilon = 0.05
    
    for episode in tqdm(range(total_episodes), desc="Evaluation (Compositional Tasks)"):
        task_idx = episode // eps_per_task
        current_task = tasks[min(task_idx, len(tasks) - 1)]
        
        result = run_episode(
            env, agent, cube_model, device, transform, pos_mean, pos_std,
            ae_model, optimizer, loss_fn, vision_device,
            current_task, eval_epsilon, max_steps_per_episode,
            training=False  # NO LEARNING during evaluation
        )
        
        task_name = current_task['name']
        results_by_task[task_name]['rewards'].append(result['episode_reward'])
        results_by_task[task_name]['task_rewards'].append(result['task_reward'])
        results_by_task[task_name]['lengths'].append(result['steps'])
        if result['task_satisfied']:
            results_by_task[task_name]['successes'] += 1
        
        episode_rewards.append(result['episode_reward'])
        task_specific_rewards.append(result['task_reward'])
        episode_lengths.append(result['steps'])
        task_rewards_log.append((episode, result['task_reward'], result['episode_reward'], task_name))
        
        # Visualizations
        if episode % 200 == 0 or episode == total_episodes - 1:
            agent_x, agent_z = agent._get_agent_pos_from_env()
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            for idx, (feature_name, feature_map) in enumerate(agent.feature_map.items()):
                row = idx // 3
                col = idx % 3
                im = axes[row, col].imshow(feature_map, cmap='viridis', origin='lower')
                axes[row, col].set_title(f'Feature Map: {feature_name}')
                axes[row, col].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            
            im = axes[1, 1].imshow(agent.composed_reward_map, cmap='viridis', origin='lower')
            axes[1, 1].set_title(f'EVAL Composed: {current_task["name"]}')
            axes[1, 1].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            im = axes[1, 2].imshow(agent.wvf, cmap='hot', origin='lower')
            axes[1, 2].set_title(f'EVAL WVF: {current_task["name"]}')
            axes[1, 2].plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f"eval_feature_maps/ep{episode}.png"), dpi=150)
            plt.close()
    
    # Print evaluation summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Zero-Shot Compositional Generalization)")
    print("="*60)
    
    for task_name, data in results_by_task.items():
        n_episodes = len(data['rewards'])
        success_rate = data['successes'] / n_episodes * 100 if n_episodes > 0 else 0
        avg_task_reward = np.mean(data['task_rewards']) if data['task_rewards'] else 0
        avg_length = np.mean(data['lengths']) if data['lengths'] else 0
        
        print(f"\n  {task_name}:")
        print(f"    Success Rate: {success_rate:.1f}% ({data['successes']}/{n_episodes})")
        print(f"    Avg Task Reward: {avg_task_reward:.3f}")
        print(f"    Avg Episode Length: {avg_length:.1f} steps")
    
    # Plot evaluation rewards
    plot_task_rewards(task_rewards_log, tasks, eps_per_task, total_episodes, phase="evaluation")
    
    # Create summary bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    task_names = list(results_by_task.keys())
    success_rates = [results_by_task[t]['successes'] / len(results_by_task[t]['rewards']) * 100 
                     for t in task_names]
    avg_task_rewards = [np.mean(results_by_task[t]['task_rewards']) for t in task_names]
    avg_lengths = [np.mean(results_by_task[t]['lengths']) for t in task_names]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    axes[0].bar(task_names, success_rates, color=colors)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Zero-Shot Success Rate by Task')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center')
    
    axes[1].bar(task_names, avg_task_rewards, color=colors)
    axes[1].set_ylabel('Average Task Reward')
    axes[1].set_title('Average Task Reward by Task')
    
    axes[2].bar(task_names, avg_lengths, color=colors)
    axes[2].set_ylabel('Average Episode Length')
    axes[2].set_title('Average Steps to Goal by Task')
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(generate_save_path('evaluation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Evaluation summary plot saved")
    
    return {
        'rewards': episode_rewards,
        'task_rewards': task_specific_rewards,
        'lengths': episode_lengths,
        'tasks': tasks,
        'episodes_per_task': eps_per_task,
        'results_by_task': results_by_task,
        'task_rewards_log': task_rewards_log
    }


def run_successor_agent(env, agent, training_episodes_per_task=2000, 
                        eval_episodes_per_task=500, max_steps_per_episode=200):
    """Main entry point: Train on simple tasks, evaluate on compositional tasks"""
    
    print("\n" + "="*70)
    print("SUCCESSOR REPRESENTATION AGENT")
    print("Train: Simple Tasks → Evaluate: Compositional Tasks (Zero-Shot)")
    print("="*70)
    
    # Load cube detector
    print("\nLoading cube detector model...")
    cube_model, device, pos_mean, pos_std = load_cube_detector('models/advanced_cube_detector.pth', force_cpu=False)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Vision Model
    print("Loading 2D vision model...")
    vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.size, env.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(vision_device)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # ==================== TRAINING PHASE ====================
    training_results = run_training_phase(
        env, agent, cube_model, device, transform, pos_mean, pos_std,
        ae_model, optimizer, loss_fn, vision_device,
        episodes_per_task=training_episodes_per_task,
        max_steps_per_episode=max_steps_per_episode
    )
    
    # ==================== EVALUATION PHASE ====================
    eval_results = run_evaluation_phase(
        env, agent, cube_model, device, transform, pos_mean, pos_std,
        ae_model, optimizer, loss_fn, vision_device,
        episodes_per_task=eval_episodes_per_task,
        max_steps_per_episode=max_steps_per_episode
    )
    
    return {
        'training': training_results,
        'evaluation': eval_results,
        'algorithm': 'Compositional Successor Agent (Train Simple → Eval Compositional)'
    }


def _find_convergence_episode(all_rewards, window):
    """Find approximate convergence episode"""
    mean_rewards = np.mean(all_rewards, axis=0)
    smoothed = pd.Series(mean_rewards).rolling(window).mean()
    
    if len(smoothed) < window * 2:
        return len(smoothed)
    
    slopes = np.diff(smoothed[window:])
    convergence_threshold = 0.001
    
    for i, slope in enumerate(slopes):
        if abs(slope) < convergence_threshold:
            return i + window
    
    return len(smoothed)


if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")
    
    # Create agent
    agent = SuccessorAgent(env)
    
    all_results = {}
    window = 100
    
    # Run training and evaluation
    results = run_successor_agent(
        env, 
        agent, 
        training_episodes_per_task=2000,  # 4 simple tasks × 2000 = 8000 training episodes
        eval_episodes_per_task=500,        # 4 compositional tasks × 500 = 2000 eval episodes
        max_steps_per_episode=200 
    )
    
    # Store results
    all_results['Compositional SR Agent'] = [results]
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create comparison plots for training
    training_data = results['training']
    eval_data = results['evaluation']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training learning curve
    ax1 = axes[0, 0]
    train_rewards = training_data['rewards']
    train_smooth = pd.Series(train_rewards).rolling(window).mean()
    ax1.plot(train_smooth, label='Training Reward', linewidth=2, color='blue')
    
    # Add task boundaries
    eps_per_task = training_data['episodes_per_task']
    for i in range(1, len(training_data['tasks'])):
        ax1.axvline(x=i * eps_per_task, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Phase: Simple Tasks (Sequential)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Evaluation performance by task
    ax2 = axes[0, 1]
    task_names = list(eval_data['results_by_task'].keys())
    success_rates = [eval_data['results_by_task'][t]['successes'] / 
                     len(eval_data['results_by_task'][t]['rewards']) * 100 
                     for t in task_names]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    bars = ax2.bar(task_names, success_rates, color=colors)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Zero-Shot Generalization: Compositional Tasks')
    ax2.set_ylim(0, 100)
    for bar, rate in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Episode lengths comparison
    ax3 = axes[1, 0]
    train_lengths = pd.Series(training_data['lengths']).rolling(window).mean()
    ax3.plot(train_lengths, label='Training', linewidth=2, color='blue')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length During Training')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    train_final_reward = np.mean(train_rewards[-500:])
    overall_success = np.mean(success_rates)
    
    summary_text = f"""
    EXPERIMENT SUMMARY
    ==================
    
    TRAINING PHASE (Simple Tasks):
    - Tasks: blue, red, box, sphere
    - Episodes per task: {eps_per_task}
    - Total training episodes: {len(train_rewards)}
    - Final avg reward (last 500): {train_final_reward:.3f}
    
    EVALUATION PHASE (Compositional Tasks):
    - Tasks: blue_sphere, red_sphere, blue_box, red_box
    - Episodes per task: {eval_data['episodes_per_task']}
    - Total eval episodes: {len(eval_data['rewards'])}
    
    ZERO-SHOT GENERALIZATION RESULTS:
    - blue_sphere: {success_rates[0]:.1f}%
    - red_sphere: {success_rates[1]:.1f}%
    - blue_box: {success_rates[2]:.1f}%
    - red_box: {success_rates[3]:.1f}%
    - Overall: {overall_success:.1f}%
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = generate_save_path("experiment_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Experiment summary saved to: {save_path}")