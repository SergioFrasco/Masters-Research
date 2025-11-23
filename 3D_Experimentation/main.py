import os
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib to use non-GUI backend

import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot

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
import matplotlib.pyplot as plt  # This must come AFTER matplotlib.use('Agg')
from collections import deque
import gc
import pandas as pd
from train_vision import CubeDetector

# class CubeDetector(nn.Module):
#     """Lightweight CNN for cube detection using MobileNetV2"""
#     def __init__(self, pretrained=False):
#         super(CubeDetector, self).__init__()
#         # Use MobileNetV2 as backbone
#         self.backbone = models.mobilenet_v2(pretrained=pretrained)
#         # Replace final classifier
#         self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
    
#     def forward(self, x):
#         return self.backbone(x)

def plot_task_rewards(task_rewards, tasks, episodes_per_task, max_episodes):
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
    ax1.set_title('Reward Over Episodes with Task Composition', fontsize=14, fontweight='bold')
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
    save_path = generate_save_path('task_compositional_rewards.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Task reward plot saved: {save_path}")

def create_task_schedule(total_episodes):
    """Create interleaved simple and compositional tasks"""
    
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
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
        # Old format - just state dict
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

def plot_wvf(wvf, episode, grid_size, maps_per_row=10):
    """Plot all world value functions in a grid"""
    num_maps = wvf.shape[0]  # state_size
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    im = None
    for idx in range(num_maps):
        ax = axes[idx]
        im = ax.imshow(wvf[idx], cmap='viridis')
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide any unused subplots
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
    
    ego_center_x, ego_center_z = 6, 12  # Agent position in 13x13 view
    past_x, past_z = past_agent_pos
    reward_x, reward_z = reward_pos
    
    for view_z in range(13):
        for view_x in range(13):
            # Calculate offset from agent's position in ego view
            dx_ego = view_x - ego_center_x
            dz_ego = view_z - ego_center_z
            
            # Rotate based on past agent direction
            if past_agent_dir == 3:  # North
                dx_world, dz_world = dx_ego, dz_ego
            elif past_agent_dir == 0:  # East
                dx_world, dz_world = -dz_ego, dx_ego
            elif past_agent_dir == 1:  # South
                dx_world, dz_world = -dx_ego, -dz_ego
            elif past_agent_dir == 2:  # West
                dx_world, dz_world = dz_ego, -dx_ego
            
            # Calculate global coordinates for this view cell
            global_x = past_x + dx_world
            global_z = past_z + dz_world
            
            # Check if this cell contains the reward
            if (global_x == reward_x and global_z == reward_z):
                target_13x13[view_z, view_x] = 1.0
            else:
                target_13x13[view_z, view_x] = 0.0
    
    return target_13x13

def _train_ae_on_batch(model, optimizer, loss_fn, inputs, targets, device):
    """Train autoencoder on batch of trajectory data"""
    # print("Done: Batch training triggered")
    # Convert to tensors and stack
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

    
def run_successor_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    print("\n=== SUCCESSOR REPRESENTATION AGENT WITH COMPOSITIONAL TASKS ===")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    # Create task schedule
    tasks, episodes_per_task = create_task_schedule(max_episodes)
    print(f"Task schedule created: {len(tasks)} tasks, {episodes_per_task} episodes each")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task['name']} ({task['type']}) - features: {task['features']}")
    print()
    
    print("Loading cube detector model...")
    cube_model, device, pos_mean, pos_std = load_cube_detector('models/advanced_cube_detector.pth', force_cpu=False)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Vision Model from 2D (keep intact)
    print("Loading 2D vision model...")
    vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.size, env.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(vision_device)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Tracking
    ae_triggers_per_episode = []
    task_rewards = []  # Store (episode, task_reward, env_reward) tuples
    
    obs, info = env.reset()
    agent.reset()

    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    total_cubes_detected = 0
    
    episode_rewards = []
    episode_lengths = []
    task_specific_rewards = []  #Track rewards that satisfy task
    
    for episode in tqdm(range(max_episodes), desc="Training 3D Successor Agent"):
        step = 0
        episode_reward = 0
        episode_task_reward = 0  #Reward for correct task completion
        episode_cubes = 0
        ae_triggers_this_episode = 0
        
        # Determine current task
        task_idx = episode // episodes_per_task
        current_task = tasks[min(task_idx, len(tasks) - 1)]
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

        # Reset maps for new episode 
        agent.true_reward_map = np.zeros((agent.grid_size, agent.grid_size))
        agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
        
        # Memory to train vision on
        trajectory_buffer = deque(maxlen=10)
        
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

            # Extract positions for egocentric observation (vision model - keep intact)
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
            
            # Create egocentric observation (for vision model)
            ego_obs = agent.create_egocentric_observation(
                goal_pos_red_box=goal_pos_red_box,
                goal_pos_blue_box=goal_pos_blue_box,
                goal_pos_red_sphere=goal_pos_red_sphere,
                goal_pos_blue_sphere=goal_pos_blue_sphere,
                matrix_size=13
            )

            # Store step info for vision training
            step_info = {
                'agent_view': ego_obs.copy(),
                'agent_pos': tuple(agent._get_agent_pos_from_env()),
                'agent_dir': agent._get_agent_dir_from_env(),
                'normalized_grid': ego_obs.copy()
            }
            trajectory_buffer.append(step_info)
    
            # ========== STEP ENVIRONMENT ==========
            obs, env_reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1

            # Check task satisfaction using info from environment
            task_satisfied = check_task_satisfaction(info, current_task)
            
            # Filter reward based on task
            if task_satisfied:
                task_reward = env_reward
                episode_task_reward += task_reward
            else:
                task_reward = 0
            
            episode_reward += env_reward  # Still track total env reward
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (SARSA)
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            done = terminated or truncated
            
            # Update SR matrix
            td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)

            # ============================= VISION MODEL (KEEP INTACT) ====================================
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

            # Learning Signal - Batch training when goal is reached
            if done and step < max_steps_per_episode:
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
                        dx_world = dx_ego
                        dz_world = dz_ego
                    elif agent_dir == 0:
                        dx_world = -dz_ego
                        dz_world = dx_ego
                    elif agent_dir == 1:
                        dx_world = -dx_ego
                        dz_world = -dz_ego
                    elif agent_dir == 2:
                        dx_world = dz_ego
                        dz_world = -dx_ego
                    
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
                        dx_world = dx_ego
                        dz_world = dz_ego
                    elif agent_dir == 0:
                        dx_world = -dz_ego
                        dz_world = dx_ego
                    elif agent_dir == 1:
                        dx_world = -dx_ego
                        dz_world = -dz_ego
                    elif agent_dir == 2:
                        dx_world = dz_ego
                        dz_world = -dx_ego
                    
                    global_x = agent_x + dx_world
                    global_z = agent_z + dz_world
                    
                    if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                        target_13x13[view_z, view_x] = agent.true_reward_map[global_z, global_x]
                    else:
                        target_13x13[view_z, view_x] = 0.0

            # Check for prediction errors and trigger training if needed
            trigger_ae_training = False
            view_error = np.abs(predicted_reward_map_2d - target_13x13)
            max_error = np.max(view_error)
            mean_error = np.mean(view_error)

            if max_error > 0.05 or mean_error > 0.01:
                trigger_ae_training = True

            if trigger_ae_training:
                ae_triggers_this_episode += 1 
                target_tensor = torch.tensor(target_13x13[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(vision_device)

                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                step_loss = loss.item()
            # ============================= END VISION MODEL ====================================
            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            if terminated or truncated:
                break
        
        # Episode ended
        ae_triggers_per_episode.append(ae_triggers_this_episode)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        task_specific_rewards.append(episode_task_reward)
        task_rewards.append((episode, episode_task_reward, episode_reward, current_task['name']))
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Create ground truth reward space
        ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)

        box_red_pos = env.box_red.pos
        box_blue_pos = env.box_blue.pos

        red_x = int(round(box_red_pos[0]))
        red_z = int(round(box_red_pos[2]))
        blue_x = int(round(box_blue_pos[0]))
        blue_z = int(round(box_blue_pos[2]))

        if 0 <= red_x < env.size and 0 <= red_z < env.size:
            ground_truth_reward_space[red_z, red_x] = 1
        if 0 <= blue_x < env.size and 0 <= blue_z < env.size:
            ground_truth_reward_space[blue_z, blue_x] = 1

        # Generate visualizations occasionally
        if episode % 100 == 0 or episode == max_episodes - 1 or episode == 0:
            # Save feature maps
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

            # Vision plots (keep existing)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            all_values = np.concatenate([
                predicted_reward_map_2d.flatten(),
                target_13x13.flatten(),
                agent.true_reward_map.flatten(),
                ground_truth_reward_space.flatten()
            ])
            vmin = np.min(all_values)
            vmax = np.max(all_values)

            im1 = ax1.imshow(predicted_reward_map_2d, cmap='viridis', vmin=vmin, vmax=vmax)
            ax1.set_title(f'Predicted 13x13 View - Ep{episode} Step{step}')
            ax1.plot(6, 12, 'ro', markersize=8, label='Agent')
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            im2 = ax2.imshow(target_13x13, cmap='viridis', vmin=vmin, vmax=vmax)
            ax2.set_title(f'Target 13x13 View (Ground Truth)')
            ax2.plot(6, 12, 'ro', markersize=8, label='Agent')
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            im3 = ax3.imshow(agent.true_reward_map, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax3.set_title(f'True 10x10 Map - Agent at ({agent_x},{agent_z})')
            ax3.plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
            plt.colorbar(im3, ax=ax3, fraction=0.046)

            im4 = ax4.imshow(ground_truth_reward_space, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax4.set_title('Ground Truth Reward Space')
            plt.colorbar(im4, ax=ax4, fraction=0.046)

            plt.tight_layout()
            plt.savefig(generate_save_path(f"vision_plots/maps_ep{episode}_step{step}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Reset environment for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Completed {max_episodes} episodes")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total cubes detected: {total_cubes_detected}")
    
    # Create task-based reward plot
    plot_task_rewards(task_rewards, tasks, episodes_per_task, max_episodes)

    return {
        "rewards": episode_rewards,
        "task_rewards": task_specific_rewards,
        "lengths": episode_lengths,
        "final_epsilon": epsilon,
        "algorithm": "Compositional Successor Agent",
        "tasks": tasks,
        "episodes_per_task": episodes_per_task
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
    # env = DiscreteMiniWorldWrapper(size=10, render_mode = "human")
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array") # For Image Capture
    # env = DiscreteMiniWorldWrapper(size=10)
    
    # create agent
    agent = SuccessorAgent(env)
    
    all_results = {}
    window=100

    # Run training with limits
    successor_results = run_successor_agent(
        env, 
        agent, 
        max_episodes=5000,        
        max_steps_per_episode=200 
    )

    # Store results
    algorithms = ['3D Masters Successor']
    results_list = [successor_results]
    
    for alg, result in zip(algorithms, results_list):
        if alg not in all_results:
            all_results[alg] = []
        all_results[alg].append(result)

    # Force cleanup between seeds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = all_results

    """Analyze and plot comparison results"""
    if not results:
        print("No results to analyze. Run experiments first.")
        

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  
    # Plot 1: Learning curves (rewards)
    ax1 = axes[0, 0]
    for alg_name, runs in results.items():
        all_rewards = np.array([run["rewards"] for run in runs])
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        # Rolling average
        mean_smooth = pd.Series(mean_rewards).rolling(window).mean()
        std_smooth = pd.Series(std_rewards).rolling(window).mean()

        x = range(len(mean_smooth))
        ax1.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
        ax1.fill_between(
            x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Learning Curves (Rewards)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Episode lengths
    ax2 = axes[0, 1]
    for alg_name, runs in results.items():
        all_lengths = np.array([run["lengths"] for run in runs])
        mean_lengths = np.mean(all_lengths, axis=0)
        std_lengths = np.std(all_lengths, axis=0)

        mean_smooth = pd.Series(mean_lengths).rolling(window).mean()
        std_smooth = pd.Series(std_lengths).rolling(window).mean()

        x = range(len(mean_smooth))
        ax2.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
        ax2.fill_between(
            x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
        )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (Steps)")
    ax2.set_title("Learning Efficiency (Steps to Goal)")
    ax2.legend()
    ax2.grid(True)

    # Plot 4: Final performance comparison (last 100 episodes)
    ax4 = axes[1, 0]
    final_rewards = {}
    for alg_name, runs in results.items():
        final_100 = []
        for run in runs:
            final_100.extend(run["rewards"][-100:])  # Last 100 episodes
        final_rewards[alg_name] = final_100

    ax4.boxplot(final_rewards.values(), labels=final_rewards.keys())
    ax4.set_ylabel("Reward")
    ax4.set_title("Final Performance (Last 100 Episodes)")
    ax4.grid(True)

    # Plot 5: Summary statistics
    ax5 = axes[1, 1]
    summary_data = []
    for alg_name, runs in results.items():
        all_rewards = np.array([run["rewards"] for run in runs])
        final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
        convergence_episode = _find_convergence_episode(all_rewards, window)

        summary_data.append({
            "Algorithm": alg_name,
            "Final Performance": f"{final_performance:.3f}",
            "Convergence Episode": convergence_episode,
        })

    summary_df = pd.DataFrame(summary_data)
    ax5.axis("tight")
    ax5.axis("off")
    table = ax5.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax5.set_title("Summary Statistics")

    plt.tight_layout()
    save_path = generate_save_path("experiment_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {save_path}")

    # Save numerical results
    # self.save_results()

    # return summary_df
 