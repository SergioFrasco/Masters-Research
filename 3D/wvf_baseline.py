import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import WVFAgent
from models import WVF_CNN
from tqdm import tqdm
import math
from utils import generate_save_path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from models import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gc
import pandas as pd
from train_advanced_cube_detector2 import CubeDetector


def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model"""
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode to avoid CUDA compatibility issues")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old and new checkpoint formats
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
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    predicted_class_idx = None
    confidence = None
    regression_values = None
    
    with torch.no_grad():
        model_output = model(img_tensor)
        
        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            classification_output, regression_output = model_output
            probs = torch.softmax(classification_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
            regression_values = regression_output.squeeze().cpu().numpy()
            
        elif isinstance(model_output, dict):
            classification_output = model_output.get('classification', None)
            regression_output = model_output.get('regression', None)
            
            if classification_output is not None:
                probs = torch.softmax(classification_output, dim=1)
                predicted_class_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class_idx].item()
            
            if regression_output is not None:
                regression_values = regression_output.squeeze().cpu().numpy()
        
        else:
            probs = torch.softmax(model_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
    
    if regression_values is not None:
        regression_values = regression_values * pos_std + pos_mean
    
    CLASS_NAMES = ['None', 'Red', 'Blue', 'Both']
    label = CLASS_NAMES[predicted_class_idx] if predicted_class_idx is not None else "Unknown"
    
    return {
        "label": label,
        "confidence": confidence,
        "regression": regression_values
    }


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
            
            if past_agent_dir == 3:
                dx_world, dz_world = dx_ego, dz_ego
            elif past_agent_dir == 0:
                dx_world, dz_world = -dz_ego, dx_ego
            elif past_agent_dir == 1:
                dx_world, dz_world = -dx_ego, -dz_ego
            elif past_agent_dir == 2:
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


def run_wvf_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """Run WVF agent with MLP-based value function learning"""
    print("\n=== WVF MLP BASELINE (Nangue Tasse 2020) ===")
    print("Agent uses MLP to learn World Value Functions directly from reward maps")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    # Load cube detector
    print("Loading cube detector model...")
    cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
        'models/advanced_cube_detector.pth', force_cpu=False
    )
    
    # Define transform for cube detector
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Vision model (autoencoder for reward prediction)
    print("Loading 2D vision model...")
    vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.size, env.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(vision_device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    ae_loss_fn = nn.MSELoss()
    
    # Tracking
    ae_triggers_per_episode = []
    
    obs, info = env.reset()
    agent.reset()
    
    episode_rewards = []
    episode_lengths = []
    wvf_losses = []
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    total_cubes_detected = 0
    
    for episode in tqdm(range(max_episodes), desc="Training WVF Agent"):
        step = 0
        episode_reward = 0
        episode_cubes = 0
        ae_triggers_this_episode = 0
        episode_wvf_losses = []
        
        # Initialize episode
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        
        # Reset maps for new episode
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
        
        # Memory for vision model training
        trajectory_buffer = deque(maxlen=10)
        
        # Take initial step
        obs, reward, terminated, truncated, info = env.step(current_action)
        current_state_idx = agent.get_state_index()
        current_exp = [current_state_idx, current_action, None, None, None]
        
        while step < max_steps_per_episode:
            agent_pos = agent._get_agent_pos_from_env()
            
            # Detect cubes BEFORE step
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            label = detection_result['label']
            confidence = detection_result['confidence']
            regression_values = detection_result['regression']
            
            if regression_values is not None:
                regression_values = np.round(regression_values).astype(int)
                rx, rz, bx, bz = regression_values
            
            # Process detection
            if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
                if label == 'Red' or label == 'Blue':
                    episode_cubes += 1
                    total_cubes_detected += 1
                else:
                    episode_cubes += 2
                    total_cubes_detected += 2
                
                # Convert coordinates
                goal_pos_red = None
                goal_pos_blue = None
                if label == 'Red':
                    goal_pos_red = (-rz, rx)
                elif label == 'Blue':
                    goal_pos_blue = (-bz, bx)
                elif label == 'Both':
                    goal_pos_red = (-rz, rx)
                    goal_pos_blue = (-bz, bx)
                
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=goal_pos_red,
                    goal_pos_blue=goal_pos_blue,
                    matrix_size=13
                )
            else:
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=None,
                    goal_pos_blue=None,
                    matrix_size=13
                )
            
            # Store step info
            step_info = {
                'agent_view': ego_obs.copy(),
                'agent_pos': tuple(agent._get_agent_pos_from_env()),
                'agent_dir': agent._get_agent_dir_from_env(),
                'normalized_grid': ego_obs.copy()
            }
            trajectory_buffer.append(step_info)
            
            # Select next action
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(next_action)
            step += 1
            total_steps += 1
            episode_reward += reward
            
            # Detect cubes AFTER step
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            label = detection_result['label']
            confidence = detection_result['confidence']
            regression_values = detection_result['regression']
            
            if regression_values is not None:
                regression_values = np.round(regression_values).astype(int)
                rx, rz, bx, bz = regression_values
            
            if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
                if label == 'Red' or label == 'Blue':
                    episode_cubes += 1
                    total_cubes_detected += 1
                else:
                    episode_cubes += 2
                    total_cubes_detected += 2
                
                goal_pos_red = None
                goal_pos_blue = None
                if label == 'Red':
                    goal_pos_red = (-rz, rx)
                elif label == 'Blue':
                    goal_pos_blue = (-bz, bx)
                elif label == 'Both':
                    goal_pos_red = (-rz, rx)
                    goal_pos_blue = (-bz, bx)
                
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=goal_pos_red,
                    goal_pos_blue=goal_pos_blue,
                    matrix_size=13
                )
            else:
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=None,
                    goal_pos_blue=None,
                    matrix_size=13
                )
            
            # Get next state
            next_state = agent.get_state_index()
            done = terminated or truncated
            
            # Complete experience
            current_exp[2] = next_state
            current_exp[3] = reward
            current_exp[4] = done
            
            # ============ WVF MLP UPDATE ============
            wvf_loss = agent.update(current_exp)
            episode_wvf_losses.append(wvf_loss)
            
            # ============ VISION MODEL UPDATE ============
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
            
            # Train vision model on batch when goal reached
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
                    
                    _train_ae_on_batch(ae_model, ae_optimizer, ae_loss_fn, 
                                      batch_inputs, batch_targets, vision_device)
            
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
            
            # Extract target 13x13 from true reward map
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
            
            # Check for prediction errors and trigger AE training
            view_error = np.abs(predicted_reward_map_2d - target_13x13)
            max_error = np.max(view_error)
            mean_error = np.mean(view_error)
            
            if max_error > 0.05 or mean_error > 0.01:
                ae_triggers_this_episode += 1
                target_tensor = torch.tensor(target_13x13[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(vision_device)
                
                ae_model.train()
                ae_optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = ae_loss_fn(output, target_tensor)
                loss.backward()
                ae_optimizer.step()
            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            if done:
                break
            else:
                current_exp = [current_state, current_action, None, None, None]
        
        # Episode ended
        ae_triggers_per_episode.append(ae_triggers_this_episode)
        
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
        
        # Visualizations
        if episode % 250 == 0 or episode == max_episodes - 1:
            # Get current Q-values for visualization
            agent.update_q_values()
            q_values = agent.current_q_values
            
            # Plot Q-values for each action
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            action_names = ['Turn Left', 'Turn Right', 'Move Forward']
            
            for a in range(3):
                im = axes[a].imshow(q_values[:, :, a], cmap='viridis')
                axes[a].set_title(f'{action_names[a]} Q-values (Episode {episode})')
                axes[a].plot(agent_x, agent_z, 'ro', markersize=8)
                plt.colorbar(im, ax=axes[a])
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f'wvf_qvalues/qvalues_ep{episode}.png'), dpi=150)
            plt.close()
            
            # Vision plots
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
            ax2.set_title(f'Target 13x13 View')
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
            plt.savefig(generate_save_path(f"vision_plots/maps_ep{episode}_step{step}.png"), dpi=150)
            plt.close()
            
            # Plot WVF loss
            if len(wvf_losses) > 0:
                plt.figure(figsize=(10, 5))
                window_size = 50
                if len(wvf_losses) >= window_size:
                    smoothed = np.convolve(wvf_losses, np.ones(window_size)/window_size, mode='valid')
                    smooth_episodes = range(window_size//2, len(wvf_losses) - window_size//2 + 1)
                else:
                    smoothed = wvf_losses
                    smooth_episodes = range(len(wvf_losses))
                
                plt.plot(wvf_losses, alpha=0.3, label='Raw loss')
                if len(wvf_losses) >= window_size:
                    plt.plot(smooth_episodes, smoothed, color='red', linewidth=2, 
                            label=f'Smoothed (window={window_size})')
                
                plt.xlabel('Episode')
                plt.ylabel('WVF MLP Loss')
                plt.title(f'WVF Learning Progress (up to ep {episode})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(generate_save_path(f'wvf_loss/loss_up_to_ep_{episode}.png'))
                plt.close()
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        if len(episode_wvf_losses) > 0:
            wvf_losses.append(np.mean(episode_wvf_losses))
        
        # Reset for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Completed {max_episodes} episodes")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total cubes detected: {total_cubes_detected}")
    print(f"✓ Final WVF loss: {wvf_losses[-1]:.4f}" if wvf_losses else "N/A")
    
    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "wvf_losses": wvf_losses,
        "final_epsilon": epsilon,
        "algorithm": "WVF MLP Baseline",
    }


if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create WVF MLP model
    wvf_model = WVF_CNN(grid_size=env.size, num_actions=3, hidden_channels=64).to(device)
    
    # Create optimizer
    # FIXED: Lower learning rate to prevent explosion (was 0.001)
    wvf_optimizer = optim.Adam(wvf_model.parameters(), lr=0.0001)
    
    # Create agent
    agent = WVFAgent(
        env=env,
        wvf_model=wvf_model,
        optimizer=wvf_optimizer,
        learning_rate=0.0001,  # FIXED: Lowered for stability
        gamma=0.99,
        device=device
    )
    
    # Run training
    results = run_wvf_agent(
        env,
        agent,
        max_episodes=3000,
        max_steps_per_episode=200
    )
    
    # Plot final results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rewards
    window = 100
    mean_smooth = pd.Series(results["rewards"]).rolling(window).mean()
    axes[0].plot(results["rewards"], alpha=0.3, label='Raw')
    axes[0].plot(mean_smooth, linewidth=2, label='Smoothed')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].grid(True)
    
    # Lengths
    mean_smooth = pd.Series(results["lengths"]).rolling(window).mean()
    axes[1].plot(results["lengths"], alpha=0.3, label='Raw')
    axes[1].plot(mean_smooth, linewidth=2, label='Smoothed')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Episode Lengths")
    axes[1].legend()
    axes[1].grid(True)
    
    # WVF Loss
    axes[2].plot(results["wvf_losses"])
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("WVF MLP Loss")
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = generate_save_path("wvf_baseline_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Results saved to: {save_path}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()