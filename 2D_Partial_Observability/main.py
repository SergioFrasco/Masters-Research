import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import matplotlib.pyplot as plt

import absl.logging
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import glob

from minigrid.core.world_object import Goal, Wall
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from tqdm import tqdm
from env import SimpleEnv
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
from utils import create_video_from_images, get_latest_run_dir
from agents import SuccessorAgent
from models import Autoencoder
from models import Autoencoder2
    
# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)
sys.path.append(".")
    
def evaluate_goal_state_values(agent, env, episode, log_file_path):
    """
    Evaluate state values around the agent when it reaches a goal.
    Compare goal state value with neighboring states and log the result.
    """
    current_pos = env.unwrapped.agent_pos
    x, y = current_pos
    
    # Get the current state index (goal state)
    goal_state_idx = agent.get_state_index(None)  # We don't need obs for this
    
    # Define neighboring positions (left, right, up, down)
    neighbors = [
        (x - 1, y),  # left
        (x + 1, y),  # right
        (x, y - 1),  # up
        (x, y + 1),  # down
    ]
    
    # Get the maximum WVF value at the goal state across all reward maps
    goal_max_value = np.max(agent.wvf[:, y, x])
    
    # Check neighboring states
    neighbor_values = []
    for nx, ny in neighbors:
        # Check if neighbor is within bounds and not a wall
        if (0 <= nx < agent.grid_size and 0 <= ny < agent.grid_size):
            # Check if it's a valid position (not a wall)
            cell = env.unwrapped.grid.get(nx, ny)
            from minigrid.core.world_object import Wall
            if cell is None or not isinstance(cell, Wall):
                # Get max WVF value at this neighbor
                neighbor_max_value = np.max(agent.wvf[:, ny, nx])
                neighbor_values.append(neighbor_max_value)
    
    # Determine if goal state has the highest value
    if len(neighbor_values) == 0:
        # No valid neighbors (shouldn't happen in normal grids)
        result = f"Episode {episode}: No valid neighbors to compare\n"
    else:
        max_neighbor_value = max(neighbor_values)
        if goal_max_value >= max_neighbor_value:
            result = f"Episode {episode}: Agent chose to stay at goal.\n"
        else:
            result = f"Episode {episode}: Agent did not choose to stay in the same position.\n"
    
    # Write to log file
    with open(log_file_path, 'a') as f:
        f.write(result)
    
    return result.strip()  # Return without newline for potential printing

def extract_goal_map(obs, tile_size=8):
    """
    Convert RGB observation back to grid representation for AE input
    """
    if isinstance(obs, dict) and 'image' in obs:
        rgb_img = obs['image']
    else:
        rgb_img = obs
    
    # rgb_img shape should be (56, 56, 3) for 7x7 grid with tile_size=8
    height, width, channels = rgb_img.shape
    grid_height = height // tile_size  # Should be 7
    grid_width = width // tile_size    # Should be 7
    
    # Convert back to grid representation by downsampling
    goal_map = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    for i in range(grid_height):
        for j in range(grid_width):
            # Extract the tile
            tile = rgb_img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            
            # Simple goal detection - you may need to adjust this based on your goal colors
            # For example, if goals are green, check for high green values
            if channels >= 3:
                # Check if this tile contains a goal (adjust color thresholds as needed)
                mean_colors = np.mean(tile, axis=(0, 1))
                
                # Example: goals might be bright green (high green, low red/blue)
                if mean_colors[1] > 200 and mean_colors[0] < 100 and mean_colors[2] < 100:
                    goal_map[i, j] = 1.0
                else:
                    goal_map[i, j] = 0.0
            else:
                # Grayscale case
                mean_intensity = np.mean(tile)
                goal_map[i, j] = 1.0 if mean_intensity > 200 else 0.0
    
    return goal_map[..., np.newaxis]  # Add channel dimension: (7, 7, 1)


def train_successor_agent(agent, env, episodes=1000, ae_model=None, max_steps=200, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999, train_vision_threshold=0.1, device='cpu', optimizer=None, loss_fn=None):
    """
    Training loop with egocentric view for autoencoder
    """
    episode_rewards = []
    ae_triggers_per_episode = []
    epsilon = epsilon_start
    step_counts = []

    # Define egocentric view parameters
    VIEW_SIZE = 7  # Agent sees 7x7 grid around itself
    AGENT_POS_IN_VIEW = (VIEW_SIZE // 2, VIEW_SIZE - 1)  # Agent at bottom-center of view
    
    # Encoding values for egocentric view
    EMPTY_SPACE = 0.0
    REWARD = 10.0
    OUT_OF_BOUNDS = 0.0  # or 8.0, you can experiment
    WALL = 0.0  # distinguishable from empty space

    # Tracking where the agent is going
    state_occupancy = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=np.int32)

    # Create log file for goal state evaluations
    log_file_path = generate_save_path("goal_state_evaluations.txt")
    with open(log_file_path, 'w') as f:
        f.write("Goal State Value Evaluations\n")
        f.write("="*50 + "\n")

    for episode in tqdm(range(episodes), "Training Successor Agent"):
        plt.close('all')
        obs, info = env.reset()

        try:
            initial_agent_pos = info["agent_pos"]
            initial_agent_dir = info["agent_dir"]
        except KeyError:
            initial_agent_pos = env.unwrapped.agent_pos
            initial_agent_dir = env.unwrapped.agent_dir

        total_reward = 0
        step_count = 0

        agent.estimated_pos = np.array(initial_agent_pos)
        agent.estimated_dir = initial_agent_dir

        # Reset for new episode
        agent.true_reward_map = np.zeros((env.unwrapped.size, env.unwrapped.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        agent.global_map = np.zeros((agent.grid_size, agent.grid_size))

        ae_trigger_count_this_episode = 0
        agent.visited_positions = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=bool)
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]

        # Update global map with first observation
        observed_positions, observed_values = agent.extract_local_observation_info(obs)
        for (gx, gy), val in zip(observed_positions, observed_values):
            agent.global_map[gy, gx] = val

        for step in range(max_steps):
            # agent_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
            # agent_dir = env.unwrapped.agent_dir
            # # Double checking estimated position
            # print("Exact Pose: ", agent_pos, agent_dir)
            # print("Estimates Pose: ", agent.estimated_pos, agent.estimated_dir)

            # Take action and observe result
            obs, reward, done, _, _ = env.step(current_action)
            
            # Update agent's position estimate
            agent.update_position_estimate(current_action)
            next_state_idx = agent.get_state_index(obs)

            # Get current agent position for tracking
            agent_pos = tuple(env.unwrapped.agent_pos)
            state_occupancy[agent_pos[0], agent_pos[1]] += 1
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx
            current_exp[3] = reward
            current_exp[4] = done
            
            # Get next action
            if step == 0 or episode < 1:
                next_action = agent.sample_random_action(obs, epsilon=epsilon)
            else:
                next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                
            # Create next experience tuple
            next_exp = [next_state_idx, next_action, None, None, None]
            
            # Update agent
            error_w, error_sr = agent.update(current_exp, None if done else next_exp)
            
            total_reward += reward
            step_count += 1
            
            # Prepare for next step
            current_exp = next_exp
            current_action = next_action
            
            # ============== EGOCENTRIC VISION MODEL ==============
            
            # 1. CREATE EGOCENTRIC VIEW INPUT
            egocentric_input = create_egocentric_view(env, agent.estimated_pos, agent.estimated_dir, VIEW_SIZE, EMPTY_SPACE, REWARD, OUT_OF_BOUNDS, WALL, done=done)
            
            # 2. GET AE PREDICTION
            input_tensor = torch.tensor(egocentric_input[np.newaxis, ..., np.newaxis], 
                                      dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            
            with torch.no_grad():
                predicted_ego_view = ae_model(input_tensor)
                predicted_ego_view_2d = predicted_ego_view.squeeze().cpu().numpy()

            # 3. UPDATE TRUE REWARD MAP WITH PREDICTION
            # Mark current position with ground truth
            agent.visited_positions[agent_pos[0], agent_pos[1]] = True
            if done and step < max_steps:
                agent.true_reward_map[agent_pos[0], agent_pos[1]] = 1.0
            else:
                agent.true_reward_map[agent_pos[0], agent_pos[1]] = 0.0

            # Update true_reward_map only for visible cells using prediction
            visible_global_positions = get_visible_global_positions(
                agent.estimated_pos, agent.estimated_dir, VIEW_SIZE, env.unwrapped.size
            )
            
            update_true_reward_map_from_egocentric_prediction(
                agent, predicted_ego_view_2d, visible_global_positions, 
                VIEW_SIZE, learning_rate=1.0
            )

            # 4. CREATE EGOCENTRIC TARGET FROM TRUE REWARD MAP
            egocentric_target = create_egocentric_target_from_true_map(
                agent.true_reward_map, agent.estimated_pos, agent.estimated_dir, 
                VIEW_SIZE, OUT_OF_BOUNDS
            )

            # 5. DECIDE WHETHER TO TRAIN AE
            center_x = VIEW_SIZE // 2
            agent_ego_x = center_x
            agent_ego_y = VIEW_SIZE - 1
            trigger_ae_training = False
            if abs(predicted_ego_view_2d[agent_ego_y, agent_ego_x] - egocentric_target[agent_ego_y, agent_ego_x]) > train_vision_threshold:
                ae_trigger_count_this_episode += 1
                trigger_ae_training = True
            
            if trigger_ae_training:
                ae_trigger_count_this_episode += 1
                
                # Train autoencoder
                target_tensor = torch.tensor(egocentric_target[np.newaxis, ..., np.newaxis], 
                                           dtype=torch.float32).permute(0, 3, 1, 2).to(device)

                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()

            # 6. UPDATE WVF (same as before)
            # Update global map for other processing
            observed_positions, observed_values = agent.extract_local_observation_info(obs)
            for (gx, gy), val in zip(observed_positions, observed_values):
                agent.global_map[gy, gx] = val

            # Update reward maps and WVF
            agent.reward_maps.fill(0)
            for y in range(agent.grid_size):
                for x in range(agent.grid_size):
                    reward = agent.true_reward_map[y, x]
                    idx = y * agent.grid_size + x
                    reward_threshold = 0.5
                    if reward > reward_threshold:
                        agent.reward_maps[idx, y, x] = 1
                    else:
                        agent.reward_maps[idx, y, x] = 0

            # Compute WVF
            M_flat = np.mean(agent.M, axis=0)
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
            V_all = M_flat @ R_flat_all.T
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

            if done:
                if episode % 100 == 0:
                    evaluate_goal_state_values(agent, env, episode, log_file_path)
                step_counts.append(step)
                break
                
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        ae_triggers_per_episode.append(ae_trigger_count_this_episode)

        # Generate visualizations occasionally
        if episode % 100 == 0:
            save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))
            
            # Save egocentric view visualization
            save_egocentric_view_visualization(
                egocentric_input, predicted_ego_view_2d, egocentric_target, 
                episode, step
            )

    # Save model and create visualizations (same as before)
    torch.save(ae_model.state_dict(), generate_save_path('vision_model.pth'))
    
    window = 20
    rolling = pd.Series(ae_triggers_per_episode).rolling(window).mean()

    # Make a video of how the SR changes over time
    latest_run = get_latest_run_dir()  # e.g., results/current/2025-05-27/run_3
    sr_folder = os.path.join(latest_run, "SR")
    video_path = os.path.join(latest_run, "sr_video.avi")
    create_video_from_images(sr_folder, video_path, fps=5, sort_numerically=True)

    # Plotting number of AE Training Triggers
    plt.figure(figsize=(10, 5))
    plt.plot(rolling, label=f'Rolling Avg (window={window})', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Number of AE Training Triggers')
    plt.title('Autoencoder Training Triggers per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(generate_save_path("ae_training_triggers.png"))

    # Plotting state occupancy
    plt.figure(figsize=(6, 6))
    plt.imshow(state_occupancy, cmap='Blues', interpolation='nearest')
    plt.title('State Occupancy Heatmap')
    plt.colorbar(label='Number of Visits')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(generate_save_path("state_occupancy_heatmap.png"))
    plt.close()

    # Plotting state starting state occupancy
    # plt.figure(figsize=(6, 6))
    # plt.imshow(agent_starting_positions, cmap='Blues', interpolation='nearest')
    # plt.title('Starting State Occupancy Heatmap')
    # plt.colorbar(label='Number of starts')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig(generate_save_path("starting_state_occupancy_heatmap.png"))
    # plt.close()


    # Plotting the number of reward occurences in each state
    # plt.figure(figsize=(6, 6))
    # plt.imshow(reward_occurence_map, cmap='hot', interpolation='nearest')
    # plt.title('Reward Occurrence Map (Heatmap)')
    # plt.colorbar(label='Times Reward Observed')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig(generate_save_path("reward_occurences.png"))
    # plt.close()

    # Plotting the number of steps taken in each episode - See if its learning
    rolling = pd.Series(step_counts).rolling(window).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(rolling, label=f'Rolling Avg Step Count (window={window})', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Taken')
    plt.title('Steps taken per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(generate_save_path("step_count.png"))
    return episode_rewards

    
    return episode_rewards


# ============== HELPER FUNCTIONS ==============
def create_egocentric_view(env, agent_pos, agent_dir, view_size, empty_val, reward_val, oob_val, wall_val, done):
    """
    Create egocentric view with agent at bottom-center facing up
    
    Args:
        env: MiniGrid environment
        agent_pos: Current agent position (x, y)
        agent_dir: Current agent direction (0=right, 1=down, 2=left, 3=up)
        view_size: Size of the egocentric view (e.g., 7 for 7x7)
        empty_val, reward_val, oob_val, wall_val: Encoding values
    
    Returns:
        egocentric_view: 2D array of shape (view_size, view_size)
    """
    egocentric_view = np.full((view_size, view_size), oob_val)
    
    agent_x, agent_y = agent_pos

    # Agent is at bottom center, facing up in egocentric coordinates
    center_x = view_size // 2
    agent_ego_x = center_x
    agent_ego_y = view_size - 1
    
    # Iterate through egocentric view positions
    for ego_y in range(view_size):
        for ego_x in range(view_size):
            # Convert egocentric coordinates to global coordinates
            global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, agent_x, agent_y, agent_dir, view_size)
            
            # # Special case: if this is the agent's position, encode as empty
            # if global_x == agent_x and global_y == agent_y:
            #     egocentric_view[ego_y, ego_x] = empty_val
            #     continue

            # Check if global position is within environment bounds
            if 0 <= global_x < env.unwrapped.size and 0 <= global_y < env.unwrapped.size:
                cell = env.unwrapped.grid.get(global_x, global_y)
                
                if cell is None:
                    egocentric_view[ego_y, ego_x] = empty_val
                elif cell.type == 'wall':
                    egocentric_view[ego_y, ego_x] = wall_val
                elif cell.type == 'goal':
                    egocentric_view[ego_y, ego_x] = reward_val
                else:
                    egocentric_view[ego_y, ego_x] = empty_val
            # else: keep out_of_bounds value

            # Specifically check agents position for reward
            if ego_y == agent_ego_x and ego_x == agent_ego_y:
                if done:
                    egocentric_view[ego_y, ego_x] = reward_val
                else:
                    egocentric_view[ego_y, ego_x] = 0

    return egocentric_view


def egocentric_to_global_coords(ego_x, ego_y, agent_x, agent_y, agent_dir, view_size):
    """
    Convert egocentric coordinates to global coordinates
    
    In egocentric view: agent is at (view_size//2, view_size-1) facing up
    """
    center_x = view_size // 2
    agent_ego_y = view_size - 1
    
    # Relative position in egocentric frame (agent facing up)
    rel_x = ego_x - center_x
    rel_y = agent_ego_y - ego_y  # Positive y goes "forward" (up in ego frame)
    
    # Rotate based on agent's actual direction
    if agent_dir == 0:  # facing right
        global_offset_x, global_offset_y = rel_y, -rel_x
    elif agent_dir == 1:  # facing down  
        global_offset_x, global_offset_y = rel_x, rel_y
    elif agent_dir == 2:  # facing left
        global_offset_x, global_offset_y = -rel_y, rel_x
    else:  # agent_dir == 3, facing up
        global_offset_x, global_offset_y = -rel_x, -rel_y
    
    return agent_x + global_offset_x, agent_y + global_offset_y


def get_visible_global_positions(agent_pos, agent_dir, view_size, env_size):
    """
    Get all global positions that are visible in the current egocentric view
    """
    visible_positions = []
    agent_x, agent_y = agent_pos
    
    for ego_y in range(view_size):
        for ego_x in range(view_size):
            global_x, global_y = egocentric_to_global_coords(
                ego_x, ego_y, agent_x, agent_y, agent_dir, view_size
            )
            
            if 0 <= global_x < env_size and 0 <= global_y < env_size:
                visible_positions.append((global_x, global_y, ego_x, ego_y))
    
    return visible_positions


def update_true_reward_map_from_egocentric_prediction(agent, predicted_ego_view, 
                                                    visible_positions, view_size, 
                                                    learning_rate=1.0):
    """
    Update true_reward_map using predictions from egocentric view
    Only updates cells that are currently visible and not previously visited
    """
    for global_x, global_y, ego_x, ego_y in visible_positions:
        # Skip already visited positions to preserve ground truth
        if not agent.visited_positions[global_x, global_y]:
            predicted_value = predicted_ego_view[ego_y, ego_x]
            
            if predicted_value > 0.001:  # Only update if prediction is significant
                # Blend with existing value using learning rate
                current_value = agent.true_reward_map[global_y, global_x]
                agent.true_reward_map[global_y, global_x] = (
                    (1 - learning_rate) * current_value + 
                    learning_rate * predicted_value
                )
            else:
                # Set to zero if prediction is very low
                agent.true_reward_map[global_y, global_x] = 0.0


def create_egocentric_target_from_true_map(true_reward_map, agent_pos, agent_dir, 
                                         view_size, oob_val=0.0):
    """
    Create egocentric target from the current true_reward_map
    This serves as the training target for the autoencoder
    """
    egocentric_target = np.full((view_size, view_size), oob_val)
    agent_x, agent_y = agent_pos
    env_size = true_reward_map.shape[0]  # Assuming square grid
    
    for ego_y in range(view_size):
        for ego_x in range(view_size):
            global_x, global_y = egocentric_to_global_coords(
                ego_x, ego_y, agent_x, agent_y, agent_dir, view_size
            )
            
            if 0 <= global_x < env_size and 0 <= global_y < env_size:
                egocentric_target[ego_y, ego_x] = true_reward_map[global_y, global_x]
            # else: keep out_of_bounds value
    
    return egocentric_target


def save_egocentric_view_visualization(input_view, prediction, target, episode, step):
    """
    Save visualization of egocentric views for debugging
    Shows agent position as a red dot in the center bottom
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Agent position in egocentric coordinates (bottom center)
    view_size = input_view.shape[0]
    agent_ego_x = view_size // 2
    agent_ego_y = view_size - 1
    
    # Plot input view
    im0 = axes[0].imshow(input_view, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Input View')
    axes[0].axis('off')
    # Add red dot for agent position
    axes[0].plot(agent_ego_x, agent_ego_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot prediction
    im1 = axes[1].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('AE Prediction')
    axes[1].axis('off')
    # Add red dot for agent position
    axes[1].plot(agent_ego_x, agent_ego_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot target
    im2 = axes[2].imshow(target, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Target')
    axes[2].axis('off')
    # Add red dot for agent position
    axes[2].plot(agent_ego_x, agent_ego_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(generate_save_path(f"egocentric_views/episode_{episode}_step_{step}.png"))
    plt.close()

def main():

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup partially observable image-based env
    env = RGBImgPartialObsWrapper(SimpleEnv(size=10))  
    env = ImgObsWrapper(env)  # Optional:if I want only the image in obs
    # print(env)

    # Pass the true environment (not the wrapper) to agent
    agent = SuccessorAgent(env.unwrapped)

    # Setup vision system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.unwrapped.size, env.unwrapped.size, 1)
    ae_model = Autoencoder2(input_channels=input_shape[-1]).to(device)

    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train agent
    rewards = train_successor_agent(agent, env, ae_model=ae_model, optimizer=optimizer, device=device, loss_fn = loss_fn)

    # Convert to pandas Series for rolling average
    rewards_series = pd.Series(rewards)
    rolling_window = 20  # You can adjust this value
    smoothed_rewards = rewards_series.rolling(rolling_window).mean()

    # Plot rewards over episodes with rolling average
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    plt.plot(smoothed_rewards, label=f'Rolling Avg (window={rolling_window})', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(generate_save_path("rewards_over_episodes.png"))


if __name__ == "__main__":
    main()

