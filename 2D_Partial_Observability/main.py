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

COLOR_TO_OBJECT = {
    (0, 255, 0): "goal",      # Green square
    (0, 0, 0): "empty",       # Floor/black
    (255, 0, 0): "lava",      # Red
    (100, 100, 100): "wall",  # Gray

}
            
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

def extract_goal_map(obs):
    """
    Turn an RGB observation into a binary map:
    1 if goal, 0 otherwise.
    """
    h, w, c = obs.shape
    goal_map = np.zeros((h, w), dtype=np.float32)
    
    # Convert each pixel triplet into object type
    for y in range(h):
        for x in range(w):
            rgb = tuple(obs[y, x, :3])  # take RGB triplet
            obj = COLOR_TO_OBJECT.get(rgb, "other")
            if obj == "goal":
                goal_map[y, x] = 1.0  # goal = 1
            else:
                goal_map[y, x] = 0.0  # everything else = 0

    return goal_map[..., np.newaxis]  # keep channel for AE input

def train_successor_agent(agent, env, episodes = 1000, ae_model=None, max_steps=200, epsilon_start=1.0, epsilon_end=0.5, epsilon_decay=0.9995, train_vision_threshold=0.1, device = 'cpu', optimizer = None, loss_fn = None):
    """
    Training loop for SuccessorAgent in MiniGrid environment with vision model integration, SR tracking, and WVF formation
    """
    episode_rewards = []
    ae_triggers_per_episode = []
    epsilon = epsilon_start
    step_counts = []

    # Tracking where the agent is going
    state_occupancy = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=np.int32)

    # Create log file for goal state evaluations
    log_file_path = generate_save_path("goal_state_evaluations.txt")
    # Clear the file at the start
    with open(log_file_path, 'w') as f:
        f.write("Goal State Value Evaluations\n")
        f.write("="*50 + "\n")

    # Tracking where the agent is starting
    # agent_starting_positions = np.zeros((env.size, env.size), dtype=np.int32)

    # Tracking where rewards are occuring
    # reward_occurence_map = np.zeros((env.size,env.size), dtype = np.int32)
    
    for episode in tqdm(range(episodes), "Training Successor Agent"):
        plt.close('all')  # to close all open figures and save memory
        obs, info = env.reset()

        try:
            initial_agent_pos = info["agent_pos"]
            initial_agent_dir = info["agent_dir"]
        except KeyError:
            # Fallback to getting directly from environment
            initial_agent_pos = env.unwrapped.agent_pos
            initial_agent_dir = env.unwrapped.agent_dir

        total_reward = 0
        step_count = 0

        plt.close('all')  # to close all open figures and save memory
        # obs, _ = env.reset()
        # total_reward = 0
        # step_count = 0

        agent.estimated_pos = np.array(initial_agent_pos)  # Start position 
        agent.estimated_dir = initial_agent_dir  # Start direction 

        # agent_pos = tuple(env.agent_pos)  
        # agent_starting_positions[agent_pos[1], agent_pos[0]] += 1

        # For every new episode, reset the reward map, and WVF, the SR stays consistent with the environment i.e doesn't reset
        agent.true_reward_map = np.zeros((env.unwrapped.size, env.unwrapped.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        # Reset the "Global map" becuase every episode will be a new input into the AE (different reward space)
        agent.global_map = np.zeros((agent.grid_size, agent.grid_size))

        # Counter
        ae_trigger_count_this_episode = 0
        
        # Reset visited positions for new episode
        agent.visited_positions = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=bool)
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]

        # Update global map  with very first look at the world
        observed_positions, observed_values  = agent.extract_local_observation_info(obs)  # extract agent's local observation info
        for (gx, gy), val in zip(observed_positions, observed_values):
                agent.global_map[gy, gx] = val

        for step in range(max_steps):
            # Check if path integration works
            # print("Agents Position: ", agent.estimated_pos)
            # print("Agents Direction: ", agent.estimated_dir)

            # Take action and observe result
            obs, reward, done, _, _ = env.step(current_action)
            

            # Update agents position estimate for Partial Observability
            agent.update_position_estimate(current_action)

            next_state_idx = agent.get_state_index(obs)

            # This confirms what the agent sees is correct
            # Egocentric view at every step
            # plt.imshow(obs)  # obs is now the partial egocentric RGB image
            # plt.title("Agent's Egocentric Observation")
            # plt.axis('off')
            # plt.show()
            # plt.savefig(generate_save_path(f"egocentric_view/step_{step}"))  

            # This confirms the conversion from egocentric view to global is correct, however the agents vision can ONLY NOT see behind it, it sees everywhere else
            observed_positions, observed_values  = agent.extract_local_observation_info(obs)  # extract agent's local observation info
            # Plot
            # plt.figure(figsize=(6, 6))
            # plt.imshow(agent.global_map, cmap="gray")  # or your actual map array
            # for gx, gy in observed_positions:
            #     plt.scatter(gx, gy, c="red", s=100, edgecolors="black")
            # plt.title(f"Episode {episode} - Agent View Overlay")
            # plt.savefig(generate_save_path(f"global_map/step_{step}"))

            # print(agent.global_map)

            # This confirms both observations and rewards get converted to global correctly
            # to use this plot uncomment the plt in the for loop as well
            # Plot
            # plt.figure(figsize=(6, 6))
            # plt.imshow(agent.global_map, cmap="gray")  # or your actual map array
            for (gx, gy), val in zip(observed_positions, observed_values):
                agent.global_map[gy, gx] = val
                # plt.scatter(gx, gy, c="red" if val > 0 else "blue", s=100, edgecolors="black")
                
            # plt.title(f"Episode {episode} - Agent View Overlay")
            # plt.savefig(generate_save_path(f"global_map/step_{step}"))

            # Tracking where the agent is going
            agent_pos = tuple(env.unwrapped.agent_pos)  
            state_occupancy[agent_pos[0], agent_pos[1]] += 1  
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx  # next state
            current_exp[3] = reward          # reward
            current_exp[4] = done            # done flag
            
            # Here we need to sample from WVF.
            # 1. Build the WVF for this moment in time
            # 2. Choose the Max one of these Maps
            # 3. Sample an Action from this map with decaying epsilon probability

            # Get next action, for the first step just use a q-learned action as the WVF is only setup after the first step, thereafter use WVF
            # Also checks if we actually have a max map. ie if we're not cofident in our WVF we sample a q-learned action
            # CHANGED - introducing a warmup period for the SR
            if step == 0 or episode < 1:
                # print("Normal Action Taken")
                next_action = agent.sample_random_action(obs, epsilon=epsilon)
            
            # Sample an action from the WVF
            else:
                # if print_flag:
                    #   print("First WVF Action Taken")
                #     print_flag = False
                # Sample an action from the max WVF
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
            
            # ------------------ Vision model -------------------

            # Update the agent's true_reward_map based on current observation
            agent_position = tuple(env.unwrapped.agent_pos)
            
            # Extract the agent's local observation window from the environment
            observed_positions, observed_values = agent.extract_local_observation_info(obs)

            # Create a local observation window 
            # First, determine the observation window size from the agent's view
            obs_window_size = 7  # Adjust based on my actual observation window size

            # Create the input for the AE using only the observed window
            # Convert the RGB observation to a format suitable for the AE
            input_grid = extract_goal_map(obs)[np.newaxis, ...]
            
            # Get the predicted reward map from the AE
            # predicted_reward_map = ae_model.predict(input_grid, verbose=0)
            # predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]
            # Get prediction for the observation window only
            with torch.no_grad():
                ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                predicted_window_tensor = ae_model(ae_input_tensor)  # Predicts the obs window
                predicted_window_2d = predicted_window_tensor.squeeze().cpu().numpy()  # (obs_window_size, obs_window_size)

            # Create target for the observation window from true_reward_map
            # TODO Look to change this to agent estimated pos
            agent_x, agent_y = env.unwrapped.agent_pos
            half_window = obs_window_size // 2

            start_x = max(0, agent_x - half_window)
            end_x = min(agent.grid_size, agent_x + half_window + 1)
            start_y = max(0, agent_y - half_window)
            end_y = min(agent.grid_size, agent_y + half_window + 1)

            # Extract the target window from true_reward_map
            target_window = agent.true_reward_map[start_y:end_y, start_x:end_x]

            # Pad target window to match input size if needed
            padded_target = np.zeros((obs_window_size, obs_window_size))
            actual_h, actual_w = target_window.shape
            pad_h = (obs_window_size - actual_h) // 2
            pad_w = (obs_window_size - actual_w) // 2
            padded_target[pad_h:pad_h+actual_h, pad_w:pad_w+actual_w] = target_window

            # Give the vision model output as a perfect reward map
            # predicted_reward_map_2d = grid[..., 0].copy()
            # predicted_reward_map_2d[object_layer == 2] = 0.0   # Wall 
            # predicted_reward_map_2d[object_layer == 1] = 0.0   # Open space
            # predicted_reward_map_2d[object_layer == 8] = 1.0 


            # Mark position as visited
            agent.visited_positions[agent_position[0], agent_position[1]] = True

            # Learning Signal
            if done and step < max_steps:
                agent.true_reward_map[agent_position[0], agent_position[1]] = 1
            else:
                agent.true_reward_map[agent_position[0], agent_position[1]] = 0


            # NEW CODE:
            # Update true_reward_map only within the observation window
            true_map_learning_rate = 1  

            # Map the predicted window back to global coordinates
            for local_y in range(obs_window_size):
                for local_x in range(obs_window_size):
                    # Convert local coordinates to global coordinates
                    global_x = start_x + local_x - pad_w
                    global_y = start_y + local_y - pad_h
                    
                    # Check if global coordinates are valid and within the actual observed area
                    if (0 <= global_x < agent.grid_size and 
                        0 <= global_y < agent.grid_size and
                        local_y >= pad_h and local_y < pad_h + actual_h and
                        local_x >= pad_w and local_x < pad_w + actual_w):
                        
                        predicted_value = predicted_window_2d[local_y, local_x]
                        
                        # Only update if agent hasn't been to this position (maintaining ground truth priority)
                        if not agent.visited_positions[global_x, global_y]:
                            if predicted_value > 0.001:
                                # Blend prediction with current value
                                agent.true_reward_map[global_y, global_x] = (
                                    (1 - true_map_learning_rate) * agent.true_reward_map[global_y, global_x] + 
                                    true_map_learning_rate * predicted_value
                                )
                            else:
                                # Fade toward zero
                                agent.true_reward_map[global_y, global_x] *= (1 - true_map_learning_rate)

            trigger_ae_training = False
    
            # Check if prediction error in the current agent position exceeds threshold
            agent_local_x = pad_w + (agent_x - start_x)
            agent_local_y = pad_h + (agent_y - start_y)

            if (0 <= agent_local_x < obs_window_size and 0 <= agent_local_y < obs_window_size):
                predicted_at_agent = predicted_window_2d[agent_local_y, agent_local_x]
                true_at_agent = agent.true_reward_map[agent_y, agent_x]
                
                if abs(predicted_at_agent - true_at_agent) > train_vision_threshold:
                    ae_trigger_count_this_episode += 1
                    trigger_ae_training = True
                
            # we then look to train the AE on this single step, where the input is the image from the environment and the loss propagation
            # is between this input image and the agents true_reward_map.
            
            # Train the AE
            if trigger_ae_training:
                # Use only the observation window as target
                target_tensor = torch.tensor(padded_target[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)  # (1, 1, obs_window_size, obs_window_size)

                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                step_loss = loss.item()
            
            # Update the agents WVF with the SR and predicted true reward map
            # Decompose the reward map into individual reward maps for each goal
            # Update per-state reward maps from true_reward_map
            agent.reward_maps.fill(0)  # Reset all maps to zero

            for y in range(agent.grid_size):
                for x in range(agent.grid_size):
                    reward = agent.true_reward_map[y, x]
                    idx = y * agent.grid_size + x
                    reward_threshold = 0.5
                    if reward > reward_threshold:
                        # changed from = reward to 1
                        agent.reward_maps[idx, y, x] = 1
                    else:
                        agent.reward_maps[idx, y, x] = 0 

            # Average the successor representation across actions
            M_flat = np.mean(agent.M, axis=0)  # shape: (100, 100)

            # Flatten reward maps: (100, 10, 10) -> (100, 100)
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)  # shape: (100, 100)

            # Compute value functions: (100, 100) @ (100, 100).T --> (100, 100)
            V_all = M_flat @ R_flat_all.T  # shape: (100, 100), each column is V for a reward map

            # Reshape to (100, 10, 10) to match original spatial layout
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

            # Reward found, next episode
            if done:

                 # BEFORE breaking, evaluate the goal state values
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
        if episode % 50 == 0:
            # save_all_reward_maps(agent, save_path=generate_save_path(f"reward_maps_episode_{episode}"))
            save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))

            # Saving the SR
            # Averaged SR matrix: shape (state_size, state_size)
            averaged_M = np.mean(agent.M, axis=0)

            # Create a figure
            plt.figure(figsize=(6, 5))
            im = plt.imshow(averaged_M, cmap='hot')
            plt.title(f"Averaged SR Matrix (Episode {episode})")
            plt.colorbar(im, label="SR Value")  # Add colorbar
            plt.tight_layout()
            plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
            plt.close()  # Close the figure to free memory

            save_env_map_pred(agent = agent, normalized_grid = agent.global_map, predicted_reward_map_2d = predicted_window_2d, episode = episode, save_path=generate_save_path(f"predictions/episode_{episode}"))
        
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
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)

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

