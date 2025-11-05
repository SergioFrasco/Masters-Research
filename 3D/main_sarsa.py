import os
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')

# NOW import everything else
import gymnasium as gym
import miniworld

from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
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
from train_advanced_cube_detector2 import CubeDetector

# Rest of your code stays the same...

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
    
    # Initialize output variables
    predicted_class_idx = None
    confidence = None
    regression_values = None
    
    # Inference
    with torch.no_grad():
        model_output = model(img_tensor)
        
        # Handle model output structure
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
    
    # Denormalize regression values
    if regression_values is not None:
        regression_values = regression_values * pos_std + pos_mean
    
    # Decode class index to label
    CLASS_NAMES = ['None', 'Red', 'Blue', 'Both']
    label = CLASS_NAMES[predicted_class_idx] if predicted_class_idx is not None else "Unknown"
    
    return {
        "label": label,
        "confidence": confidence,
        "regression": regression_values
    }

# def compose_wvf(agent, reward_map):
#     """Compose world value functions from SR and reward map"""
#     grid_size = agent.grid_size
#     state_size = agent.state_size
    
#     # Initialize reward maps for each state
#     reward_maps = np.zeros((state_size, grid_size, grid_size))
    
#     # Fill reward maps based on threshold
#     for z in range(grid_size):
#         for x in range(grid_size):
#             curr_reward = reward_map[z, x]
#             idx = z * grid_size + x
#             # Threshold
#             if reward_map[z, x] >= 0.5:
#                 reward_maps[idx, z, x] = curr_reward
    
#     MOVE_FORWARD = 2
#     M_forward = agent.M[MOVE_FORWARD, :, :]
    
#     # Flatten reward maps
#     R_flat_all = reward_maps.reshape(state_size, -1)
    
#     # Compute WVF: V = M @ R^T
#     V_all = M_forward @ R_flat_all.T
    
#     # Reshape back to grid
#     wvf = V_all.T.reshape(state_size, grid_size, grid_size)
    
#     return wvf

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
    """Run with random agent that learns SR and detects cubes"""
    print("\n=== SUCCESSOR REPRESENTATION AGENT MODE WITH CUBE DETECTION ===")
    print("Agent will take random actions and learn SR matrix")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    print("Loading cube detector model...")
    # 1. Load the model
    cube_model, device, pos_mean, pos_std = load_cube_detector('models/advanced_cube_detector.pth', force_cpu=False)
    
    # 2. Define transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Vision Model from 2D
    print("Loading 2D vision model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.size, env.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Initialize reward map
    reward_map = np.zeros((env.size, env.size))

    # Tracking ae triggers
    ae_triggers_per_episode = [] 
    
    obs, info = env.reset()
    agent.reset()

    total_reward = 0
    steps = 0
    episode_rewards = []
    episode_lengths = []
    epsilon = 1
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    episode = 0
    total_steps = 0
    total_cubes_detected = 0
    
    for episode in tqdm(range(max_episodes), desc="Training 3D Successor Agent"):
        step = 0
        episode_reward = 0
        episode_cubes = 0
        ae_triggers_this_episode = 0
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        reward_map = np.zeros((env.size, env.size))

        # Reset maps for new episode 
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
        
        # Memory to train vision on
        trajectory_buffer = deque(maxlen=10)
        trajectory = []

        obs, reward, terminated, truncated, info = env.step(current_action)
        current_state_idx = agent.get_state_index()
        current_exp = [current_state_idx, current_action, None, None, None]
        
        while step < max_steps_per_episode:
            
            agent_pos = agent._get_agent_pos_from_env()
            trajectory.append((agent_pos[0], agent_pos[1], current_action))

            # First (before step) Detect type of cube in the observation, as well as position regression - marked rx, rz, bx, bz
            # where +x is forward, +z is right from agent perspective
            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            
            label = detection_result['label']
            confidence = detection_result['confidence']
            regression_values = detection_result['regression'] # rx, rz, bx, bz
            regression_values = np.round(regression_values).astype(int) # Round to nearest int for positions
            rx, rz, bx, bz = regression_values  # forward, right for each color
            
            # print(f"Detected: {label} with confidence {confidence:.3f}")
            # if regression_values is not None:
                # print(f"Positions: {regression_values}")


            if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
                # Update counters
                if label == 'Red' or label == 'Blue':
                    episode_cubes += 1
                    total_cubes_detected += 1
                else:
                    episode_cubes += 2
                    total_cubes_detected += 2

                # We have:
                # pos x is forward
                # pos z is right

                # We need:
                # pos x is right
                # pos z is south

                # so swap x and z and make x negative
                
                # Check goal position from model
                if label == 'Red':
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = None
                elif label == 'Blue':
                    goal_pos_blue = (-bz, bx)
                    goal_pos_red = None
                elif label == 'Both': 
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = (-bz, bx)

                # Create egocentric observation matrix
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=goal_pos_red,
                    goal_pos_blue=goal_pos_blue,
                    matrix_size=13
                )
            
            else:
                # No cube detected - create empty egocentric observation
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=None,
                    goal_pos_blue=None,
                    matrix_size=13
                )

            # Store step info BEFORE taking action
            step_info = {
                'agent_view': ego_obs.copy(),
                'agent_pos': tuple(agent._get_agent_pos_from_env()),
                'agent_dir': agent._get_agent_dir_from_env(),
                'normalized_grid': ego_obs.copy()
            }
            trajectory_buffer.append(step_info)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1
            episode_reward += reward

            # Save the frame - ensure render mode is "rgb_array"
            # frame = env.render()
            # if frame is not None:
            #     if isinstance(frame, np.ndarray):
            #         img = Image.fromarray(frame)
            #     else:
            #         img = frame
                
            #     # Save RGB frame
            #     save_frame_path = generate_save_path(f'frame_ep{episode:03d}_step{step:03d}.png')
            #     img.save(save_frame_path)
            #     print(f"  Saved frame: {save_frame_path}")

            # time.sleep(10) 

            # Second (after step) Detect type of cube in the observation, as well as position regression - marked rx, rz, bx, bz
            # where +x is forward, +z is right from agent perspective
            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            
            label = detection_result['label']
            confidence = detection_result['confidence']
            regression_values = detection_result['regression'] # rx, rz, bx, bz
            regression_values = np.round(regression_values).astype(int) # Round to nearest int for positions
            rx, rz, bx, bz = regression_values  # forward, right for each color
            
            # print(f"Detected: {label} with confidence {confidence:.3f}")
            # if regression_values is not None:
            #     print(f"Positions: {regression_values}")

            if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
                # Update counters
                if label == 'Red' or label == 'Blue':
                    episode_cubes += 1
                    total_cubes_detected += 1
                else:
                    episode_cubes += 2
                    total_cubes_detected += 2

                # We have:
                # pos x is forward
                # pos z is right

                # We need:
                # pos x is right
                # pos z is south

                # so swap x and z and make x negative
                
                # Check goal position from model
                if label == 'Red':
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = None
                elif label == 'Blue':
                    goal_pos_blue = (-bz, bx)
                    goal_pos_red = None
                elif label == 'Both': 
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = (-bz, bx)

                # Create egocentric observation matrix
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=goal_pos_red,
                    goal_pos_blue=goal_pos_blue,
                    matrix_size=13
                )      
            else:
                # No cube detected - create empty egocentric observation
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=None,
                    goal_pos_blue=None,
                    matrix_size=13
                )
                    
            # print(reward_map)
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (SARSA)
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            done = terminated or truncated
            
            # Update SR matrix with current and next action
            # td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)

            current_exp[2] = next_state
            current_exp[3] = reward
            current_exp[4] = done
        
            if done:
                # Terminal state - update without next experience
                agent.update(current_exp, next_exp=None)
            else:
                # Non-terminal - create next_exp and update
                next_exp = [next_state, next_action, None, None, None]
                agent.update(current_exp, next_exp)


            # ============================= VISION MODEL ====================================
                
            # # Get current agent position
            agent_position = agent._get_agent_pos_from_env()

            # # Get the agent's 13x13 view from observation
            agent_view = ego_obs

            # # If agent is on goal, force the agent's position in view to show reward
            if done:
                x,z = agent_position
                agent_view[12, 6] = 1.0  # Agent position in egocentric view

            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = agent_view[np.newaxis, ..., np.newaxis] 

            with torch.no_grad():
                ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                predicted_reward_map_tensor = ae_model(ae_input_tensor)
                predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()

            # Mark position as visited (using path integration)
            agent.visited_positions[agent_position[1], agent_position[0]] = True

            # Learning Signal - Batch training when goal is reached
            if done and step < max_steps_per_episode:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 1

                if len(trajectory_buffer) > 0:
                    batch_inputs = []
                    batch_targets = []
                    
                    # Include all steps from trajectory buffer (past steps)
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
                    
                    # ALSO include the current step (when agent is on goal)
                    current_target_13x13 =_create_target_view_with_reward(
                        tuple(agent._get_agent_pos_from_env()),
                        agent._get_agent_dir_from_env(),
                        agent_position,
                        agent.true_reward_map
                    )
                    
                    batch_inputs.append(ego_obs)
                    batch_targets.append(current_target_13x13)
                    
                    # Train autoencoder on batch
                    _train_ae_on_batch(ae_model, optimizer, loss_fn, batch_inputs, batch_targets, device)

            # Map the 13x13 predicted reward map to the 10x10 global map
            agent_x, agent_z = agent_position
            ego_center_x = 6
            ego_center_z = 12
            agent_dir = agent._get_agent_dir_from_env()
            
            for view_z in range(13):
                for view_x in range(13):
                    dx_ego = view_x - ego_center_x
                    dz_ego = view_z - ego_center_z
                    
                    if agent_dir == 3:  # North
                        dx_world = dx_ego
                        dz_world = dz_ego
                    elif agent_dir == 0:  # East
                        dx_world = -dz_ego
                        dz_world = dx_ego
                    elif agent_dir == 1:  # South
                        dx_world = -dx_ego
                        dz_world = -dz_ego
                    elif agent_dir == 2:  # West
                        dx_world = dz_ego
                        dz_world = -dx_ego
                    
                    global_x = agent_x + dx_world
                    global_z = agent_z + dz_world
                    
                    if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                        if not agent.visited_positions[global_z, global_x]:
                            predicted_value = predicted_reward_map_2d[view_z, view_x]
                            agent.true_reward_map[global_z, global_x] = predicted_value

            # Extract the 13x13 target from the true reward map
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

            # # Check for prediction errors and trigger training if needed
            trigger_ae_training = False
            view_error = np.abs(predicted_reward_map_2d - target_13x13)
            max_error = np.max(view_error)
            mean_error = np.mean(view_error)

            if max_error > 0.05 or mean_error > 0.01:
                trigger_ae_training = True

            if trigger_ae_training:
                ae_triggers_this_episode += 1 
                target_tensor = torch.tensor(target_13x13[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)

                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                step_loss = loss.item()

            # Update reward maps
            agent.reward_maps.fill(0)

            for y in range(agent.grid_size):
                for x in range(agent.grid_size):
                    curr_reward = agent.true_reward_map[y, x]
                    idx = y * agent.grid_size + x
                    if agent.true_reward_map[y, x] >= 0.25:
                        agent.reward_maps[idx, y, x] = curr_reward

            MOVE_FORWARD = 2
            M_forward = agent.M[MOVE_FORWARD, :, :]
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
            V_all = M_forward @ R_flat_all.T
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            total_reward += reward
            steps += 1
            
            # ============================= EPISODE END CHECK =============================
            if done:
                break
            else:
                current_exp = next_exp
                current_action = next_action

        
        # Episode ended 
        episode += 1
        # print(f"\n=== Episode {episode}/{max_episodes} ended after {step} steps! ===")
        # print(f"Episode reward: {episode_reward:.2f}")
        # print(f"Cubes detected this episode: {episode_cubes}")
        # print(f"Total cubes detected: {total_cubes_detected}")
        # print(f"Reward map sum: {reward_map.sum()}")
        # print(f"SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
        # print(f"Total steps so far: {total_steps}")
        
        # Compose and plot WVF every 50 episodes or on last episode
        # if episode % 1000 == 0 or episode == max_episodes:
        #     if reward_map.sum() > 0:  # Only if we've detected rewards
        #         wvf = compose_wvf(agent, reward_map)
        #         plot_wvf(wvf, episode, agent.grid_size)
        #     plot_sr_matrix(agent, episode)

        ae_triggers_per_episode.append(ae_triggers_this_episode)

        # Create ground truth reward space
        ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)

        box_red_pos = env.box_red.pos
        box_blue_pos = env.box_blue.pos

        # Convert to grid coordinates
        red_x = int(round(box_red_pos[0]))
        red_z = int(round(box_red_pos[2]))
        blue_x = int(round(box_blue_pos[0]))
        blue_z = int(round(box_blue_pos[2]))

        # Mark both boxes in ground truth reward space
        if 0 <= red_x < env.size and 0 <= red_z < env.size:
            ground_truth_reward_space[red_z, red_x] = 1
        if 0 <= blue_x < env.size and 0 <= blue_z < env.size:
            ground_truth_reward_space[blue_z, blue_x] = 1

        # Generate visualizations occasionally
        if episode % 250 == 0 or episode == max_episodes or episode == 0:

            # # Right before creating ground_truth_reward_space
            # print(f"\n=== Debug Info for Episode {episode} ===")
            # print(f"Agent position: ({agent_x}, {agent_z})")
            # print(f"Agent raw pos: {env.agent.pos}")

            # box_red_pos = env.box_red.pos
            # box_blue_pos = env.box_blue.pos
            # print(f"Red box raw pos: {box_red_pos}")
            # print(f"Blue box raw pos: {box_blue_pos}")

            # # Convert to grid coordinates
            # red_x = int(round(box_red_pos[0]))
            # red_z = int(round(box_red_pos[2]))
            # blue_x = int(round(box_blue_pos[0]))
            # blue_z = int(round(box_blue_pos[2]))

            # print(f"Red box grid: ({red_x}, {red_z})")
            # print(f"Blue box grid: ({blue_x}, {blue_z})")

            # # Check what's in true_reward_map at those locations
            # print(f"true_reward_map[{red_z}, {red_x}] = {agent.true_reward_map[red_z, red_x]}")
            # print(f"true_reward_map[{blue_z}, {blue_x}] = {agent.true_reward_map[blue_z, blue_x]}")
            # print(f"ground_truth[{red_z}, {red_x}] = {ground_truth_reward_space[red_z, red_x]}")
            # print(f"ground_truth[{blue_z}, {blue_x}] = {ground_truth_reward_space[blue_z, blue_x]}")

            save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))

            # Saving the Move Forward SR
            forward_M = agent.M[MOVE_FORWARD, :, :]

            plt.figure(figsize=(6, 5))
            im = plt.imshow(forward_M, cmap='hot')
            plt.title(f"Forward SR Matrix (Episode {episode})")
            plt.colorbar(im, label="SR Value")
            plt.tight_layout()
            plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
            plt.close()

            box_red_pos = env.box_red.pos
            box_blue_pos = env.box_blue.pos


            # Create vision plots
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
            ax2.set_title(f'Target 7x7 View (Ground Truth)')
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

            # Plot AE triggers over episodes
            plt.figure(figsize=(10, 5))
            
            window_size = 50
            if len(ae_triggers_per_episode) >= window_size:
                smoothed_triggers = np.convolve(ae_triggers_per_episode, np.ones(window_size)/window_size,  mode='valid')
                smooth_episodes = range(window_size//2, len(ae_triggers_per_episode) - window_size//2 + 1)
            else:
                smoothed_triggers = ae_triggers_per_episode
                smooth_episodes = range(len(ae_triggers_per_episode))
            
            plt.plot(ae_triggers_per_episode, alpha=0.3, label='Raw triggers per episode')
            if len(ae_triggers_per_episode) >= window_size:
                plt.plot(smooth_episodes, smoothed_triggers, color='red', linewidth=2, 
                        label=f'Smoothed (window={window_size})')
            
            plt.xlabel('Episode')
            plt.ylabel('Number of AE Training Triggers')
            plt.title(f'AE Training Frequency Over Episodes (up to ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'ae_triggers/triggers_up_to_ep_{episode}.png'))
            plt.close()
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        
        # Reset environment for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Completed {episode} episodes")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total cubes detected: {total_cubes_detected}")
    print(f"✓ Final SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")

    return {
    "rewards": episode_rewards,
    "lengths": episode_lengths,
    "final_epsilon": epsilon,
    "algorithm": "Masters Successor w/ Path Integration",
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
    # env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array") # For Image Capture
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)
    
    # create agent
    agent = RandomAgentWithSR(env)
    
    all_results = {}
    window=100

    # Run training with limits
    successor_results = run_successor_agent(
        env, 
        agent, 
        max_episodes=10000,        
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
 