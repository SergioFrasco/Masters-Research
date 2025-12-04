# Egocentric Observation (13×13)     Allocentric Reward Map (10×10)
#         ↓                                    ↓
#    Flattened (169)                   Extract goal positions [(x,z), ...]
#         ↓                                    ↓
#    + position (2)                    Normalize each goal
#    + direction (4)                          ↓
#         ↓                            Goal (2) ← one at a time
#         ↓                                    ↓
#         └──────────────┬─────────────────────┘
#                        ↓
#                  Concatenate
#                        ↓
#                 Network Input (177)
#                        ↓
#                  Q(s, g, a)

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import miniworld
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import WVFAgent
from models import WVF_MLP
from tqdm import tqdm
from utils import generate_save_path
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
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
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
    """Run cube detection"""
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
        model_output = model(img_tensor)
        
        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            classification_output, regression_output = model_output
            probs = torch.softmax(classification_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
            regression_values = regression_output.squeeze().cpu().numpy()
        else:
            probs = torch.softmax(model_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
            regression_values = None
    
    if regression_values is not None:
        regression_values = regression_values * pos_std + pos_mean
    
    CLASS_NAMES = ['None', 'Red', 'Blue', 'Both']
    label = CLASS_NAMES[predicted_class_idx]
    
    return {"label": label, "confidence": confidence, "regression": regression_values}


def run_wvf_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """Run WVF agent training"""
    print("\n=== WVF BASELINE (Nangue Tasse) ===")
    print("Goal (x,y) is INPUT to network")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    # Load cube detector
    print("Loading cube detector model...")
    cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
        'models/advanced_cube_detector.pth', force_cpu=False
    )
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Vision model (autoencoder)
    print("Loading 2D vision model...")
    vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(input_channels=1).to(vision_device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    ae_loss_fn = nn.MSELoss()
    
    obs, info = env.reset()
    agent.reset()
    
    episode_rewards = []
    episode_lengths = []
    wvf_losses = []
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_steps = 0
    
    for episode in tqdm(range(max_episodes), desc="Training WVF Agent"):
        step = 0
        episode_reward = 0
        episode_wvf_losses = []
        
        # Reset maps
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
        
        # Get initial observation
        detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
        ego_obs = _get_ego_obs_from_detection(agent, detection_result)
        agent.set_ego_observation(ego_obs)
        
        # Store current state features
        current_state_features = agent._get_state_features()
        
        while step < max_steps_per_episode:
            # Update reward map from egocentric observation
            _update_reward_map_from_ego(agent, ego_obs)
            
            # Select action
            action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_steps += 1
            episode_reward += reward
            done = terminated or truncated
            
            # Get new observation
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            next_ego_obs = _get_ego_obs_from_detection(agent, detection_result)
            agent.set_ego_observation(next_ego_obs)
            
            # Store next state features
            next_state_features = agent._get_state_features()
            
            # Create experience tuple
            experience = [
                current_state_features,
                action,
                next_state_features,
                reward,
                done
            ]
            
            # Update WVF for all detected goals
            wvf_loss = agent.update_for_all_goals(experience)
            if wvf_loss > 0:
                episode_wvf_losses.append(wvf_loss)
            
            # Update reward map with new detections
            _update_reward_map_from_ego(agent, next_ego_obs)
            
            # Move to next step
            current_state_features = next_state_features
            ego_obs = next_ego_obs
            
            if done:
                break
        
        # Episode stats
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        if len(episode_wvf_losses) > 0:
            wvf_losses.append(np.mean(episode_wvf_losses))
        
        # Visualization every 250 episodes
        if episode % 250 == 0 or episode == max_episodes - 1:
            _visualize_progress(agent, env, episode, step, wvf_losses)
        
        # Reset for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Final WVF loss: {wvf_losses[-1]:.4f}" if wvf_losses else "N/A")
    
    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "wvf_losses": wvf_losses,
        "final_epsilon": epsilon,
    }


def _get_ego_obs_from_detection(agent, detection_result):
    """Convert cube detection to egocentric observation"""
    label = detection_result['label']
    confidence = detection_result['confidence']
    regression_values = detection_result['regression']
    
    goal_pos_red = None
    goal_pos_blue = None
    
    if regression_values is not None and label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
        regression_values = np.round(regression_values).astype(int)
        rx, rz, bx, bz = regression_values
        
        if label == 'Red':
            goal_pos_red = (-rz, rx)
        elif label == 'Blue':
            goal_pos_blue = (-bz, bx)
        elif label == 'Both':
            goal_pos_red = (-rz, rx)
            goal_pos_blue = (-bz, bx)
    
    return agent.create_egocentric_observation(
        goal_pos_red=goal_pos_red,
        goal_pos_blue=goal_pos_blue,
        matrix_size=13
    )


def _update_reward_map_from_ego(agent, ego_obs):
    """
    Update the allocentric reward map from egocentric observation.
    This provides the 'goals' for the network.
    """
    agent_x, agent_z = agent._get_agent_pos_from_env()
    agent_dir = agent._get_agent_dir_from_env()
    
    ego_center_x = 6
    ego_center_z = 12
    
    for view_z in range(13):
        for view_x in range(13):
            if ego_obs[view_z, view_x] > 0.5:  # Goal detected
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                # Convert egocentric to world coordinates
                if agent_dir == 3:  # North
                    dx_world, dz_world = dx_ego, dz_ego
                elif agent_dir == 0:  # East
                    dx_world, dz_world = -dz_ego, dx_ego
                elif agent_dir == 1:  # South
                    dx_world, dz_world = -dx_ego, -dz_ego
                elif agent_dir == 2:  # West
                    dx_world, dz_world = dz_ego, -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.grid_size and 0 <= global_z < agent.grid_size:
                    agent.true_reward_map[global_z, global_x] = 1.0


def _visualize_progress(agent, env, episode, step, wvf_losses):
    """Save visualization plots"""
    # Get Q-values for all goals
    agent.update_q_values()
    q_values = agent.current_q_values
    
    if q_values is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        action_names = ['Turn Left', 'Turn Right', 'Move Forward']
        
        for a in range(3):
            im = axes[a].imshow(q_values[:, :, a], cmap='viridis')
            axes[a].set_title(f'{action_names[a]} Q-values (Episode {episode})')
            plt.colorbar(im, ax=axes[a])
        
        plt.tight_layout()
        plt.savefig(generate_save_path(f'wvf_qvalues/qvalues_ep{episode}.png'), dpi=150)
        plt.close()
    
    # Plot reward map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axes[0].imshow(agent.true_reward_map, cmap='viridis', origin='lower')
    axes[0].set_title(f'Detected Reward Map (Episode {episode})')
    plt.colorbar(im1, ax=axes[0])
    
    # Ground truth
    ground_truth = np.zeros((env.size, env.size))
    red_x = int(round(env.box_red.pos[0]))
    red_z = int(round(env.box_red.pos[2]))
    blue_x = int(round(env.box_blue.pos[0]))
    blue_z = int(round(env.box_blue.pos[2]))
    
    if 0 <= red_x < env.size and 0 <= red_z < env.size:
        ground_truth[red_z, red_x] = 1
    if 0 <= blue_x < env.size and 0 <= blue_z < env.size:
        ground_truth[blue_z, blue_x] = 1
    
    im2 = axes[1].imshow(ground_truth, cmap='viridis', origin='lower')
    axes[1].set_title('Ground Truth Reward')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'vision_plots/reward_map_ep{episode}.png'), dpi=150)
    plt.close()
    
    # Loss plot
    if len(wvf_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(wvf_losses, alpha=0.5, label='Raw')
        
        window = min(50, len(wvf_losses))
        if len(wvf_losses) >= window:
            smoothed = np.convolve(wvf_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(wvf_losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
        
        plt.xlabel('Episode')
        plt.ylabel('WVF Loss')
        plt.title(f'WVF Learning (up to episode {episode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(generate_save_path(f'wvf_loss/loss_ep{episode}.png'), dpi=150)
        plt.close()


if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create WVF model (Q-network)
    # State dim = 13*13 (ego_obs) + 2 (position) + 4 (direction) = 175
    wvf_model = WVF_MLP(
        state_dim=175,
        num_actions=3,
        hidden_dim=128
    ).to(device)
    
    # Create target network (copy of Q-network)
    target_model = WVF_MLP(
        state_dim=175,
        num_actions=3,
        hidden_dim=128
    ).to(device)
    
    # Optimizer with lower learning rate to prevent loss explosion
    wvf_optimizer = optim.Adam(wvf_model.parameters(), lr=0.0005)
    
    # Create agent with target network
    agent = WVFAgent(
        env=env,
        wvf_model=wvf_model,
        target_model=target_model,
        optimizer=wvf_optimizer,
        gamma=0.99,
        device=device,
        grid_size=env.size,
        target_update_freq=100
    )
    
    # Run training
    results = run_wvf_agent(
        env,
        agent,
        max_episodes=2000,
        max_steps_per_episode=200
    )
    
    # Plot final results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    window = 100
    
    # Rewards
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
    
    # Loss
    axes[2].plot(results["wvf_losses"])
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("WVF Loss")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(generate_save_path("wvf_baseline_results.png"), dpi=300)
    print(f"Results saved!")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()