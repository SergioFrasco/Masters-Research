import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf
import math
import pandas as pd
import glob

from minigrid.core.world_object import Goal, Wall
from tqdm import tqdm
from env import SimpleEnv, data_collector
from models import build_autoencoder, focal_mse_loss, load_trained_autoencoder, weighted_focal_mse_loss
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
from utils import create_video_from_images, get_latest_run_dir
from models.construct_sr import constructSR
from agents import SuccessorAgent

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable GPU if not needed
tf.config.set_visible_devices([], "GPU")

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)
sys.path.append(".")
    
# epsilon decay = 0.995 before
# 0.999 better

def train_successor_agent(agent, env, episodes = 3001, ae_model=None, max_steps=200, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999, train_vision_threshold=0.1):
    """
    Training loop for SuccessorAgent in MiniGrid environment with vision model integration, SR tracking, and WVF formation
    """
    episode_rewards = []
    ae_triggers_per_episode = []
    epsilon = epsilon_start
    step_counts = []

    # Tracking where the agent is going
    state_occupancy = np.zeros((env.size, env.size), dtype=np.int32)

    # Tracking where rewards are occuring
    reward_occurence_map = np.zeros((env.size,env.size), dtype = np.int32)

    print_flag = True
    
    for episode in tqdm(range(episodes), "Training Successor Agent"):
        plt.close('all')  # to close all open figures and save memory
        obs = env.reset()
        total_reward = 0
        step_count = 0

        # For every new episode, reset the reward map, and WVF, the SR stays consistent with the environment i.e doesn't reset
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        ae_trigger_count_this_episode = 0
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]
        
        for step in range(max_steps):
            # Take action and observe result
            obs, reward, done, _, _ = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)

            # Tracking where the agent is going
            agent_pos = tuple(env.agent_pos)  # (x, y)
            state_occupancy[agent_pos[1], agent_pos[0]] += 1  # (row, col) = (y, x)
            agent.update_visit_counts(env.agent_pos) # for the exploration bonus
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx  # next state
            current_exp[3] = reward          # reward
            current_exp[4] = done            # done flag
            
            # Here we need to sample from WVF.
            # 1. Build the WVF for this moment in time
            # 2. Choose the Max one of these Maps
            # 3. Sample an Action from this map with decaying epsilon probability
            
            # CHANGED - I think this may be causing the agent to learn to move to only 1 position
            # # Get the map with the single highest max value
            # max_vals = agent.wvf.max(axis=(1, 2))  # shape: (100,)
            # best_map_index = np.argmax(max_vals)   # index of the map with the highest max value
            # chosen_map = agent.wvf[best_map_index]  # shape: (10, 10)

            # reward_threshold = 0.5
            # agent_y, agent_x = env.agent_pos  # assuming (y, x) format

            # # Step 1: Create a mask of where values exceed the threshold
            # exceeds_threshold = agent.wvf > reward_threshold  # shape: (num_maps, H, W)

            # # Step 2: Compute distance of each exceeding point to the agent
            # H, W = agent.wvf.shape[1:]
            # y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            # distances = np.sqrt((y_coords - agent_y)**2 + (x_coords - agent_x)**2)  # shape: (H, W)

            # # Broadcast distances to all maps
            # distances_broadcasted = np.broadcast_to(distances, agent.wvf.shape)  # (num_maps, H, W)

            # # Step 3: Mask distances where threshold not exceeded
            # masked_distances = np.where(exceeds_threshold, distances_broadcasted, np.inf)

            # # Step 4: Get minimum distance for each map
            # min_dist_per_map = masked_distances.min(axis=(1, 2))  # shape: (num_maps,)

            # # Step 5: Only consider maps where at least one value exceeded the threshold
            # valid_maps = np.any(exceeds_threshold, axis=(1, 2))  # shape: (num_maps,)
            # valid_dists = np.where(valid_maps, min_dist_per_map, np.inf)

            # # Step 6: Choose the map with the smallest such distance
            # best_map_index = np.argmin(valid_dists)
            # chosen_map = agent.wvf[best_map_index]

            
            # Random actions for the firt N episodes to check if SR improves
            # warmup_episodes = 100
            # if episode < warmup_episodes:
            #     next_action = agent.sample_random_action(obs, epsilon=epsilon)
            # else:
            #     # Get reward map from 
            #     if print_flag:
            #         print("First WVF Action Taken")
            #         print_flag = False
                
            #     next_action = agent.sample_action_with_wvf(obs, chosen_reward_map=chosen_map, epsilon=epsilon)

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
            
            # ------------------Vision model----------------
            # Update the agent's true_reward_map based on current observation
            agent_position = tuple(env.agent_pos)
            # print(agent_position)

            # Get the current environment grid
            grid = env.grid.encode()
            normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)  # Shape: (H, W)

            # Setting up input for the AE to obtain it's prediction of the space
            # Object types are in grid[..., 0]
            object_layer = grid[..., 0]
            normalized_grid[object_layer == 2] = 0.0   # Wall 
            normalized_grid[object_layer == 1] = 0.0   # Open space
            normalized_grid[object_layer == 8] = 1.0   # Reward (e.g. goal object)

            # Check reward Occurence
            # Iterate over the grid and increment reward occurrence where object type == 8 (goal)
            for y in range(env.height):
                for x in range(env.width):
                    if object_layer[y, x] == 8:
                        reward_occurence_map[y, x] += 1
            
            # Rotate the grid to match render_mode = human 
            # normalized_grid = np.flipud(normalized_grid)
            # normalized_grid = np.rot90(normalized_grid, k=-1)
            
            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
            
            # CHANGED - commented this out to see if perfect reward map trains the SR correctly
            # Get the predicted reward map from the AE
            # predicted_reward_map = ae_model.predict(input_grid, verbose=0)
            # predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]
            predicted_reward_map_2d = grid[..., 0]
            predicted_reward_map_2d[object_layer == 2] = 0.0   # Wall 
            predicted_reward_map_2d[object_layer == 1] = 0.0   # Open space
            predicted_reward_map_2d[object_layer == 8] = 1.0 

            # predicted_reward_map_2d = np.flipud(predicted_reward_map_2d)
            # predicted_reward_map_2d = np.rot90(predicted_reward_map_2d, k=-1)
            # predicted_reward_map_2d = np.rot90(normalized_grid, k=-1)

            
            # Update the rest of the true_reward_map with AE predictions
            for y in range(agent.true_reward_map.shape[0]):
                for x in range(agent.true_reward_map.shape[1]):
                    if (x, y) != agent_position:  # Skip the reward position
                        # Get the predicted value for this position from the AE
                        predicted_value = predicted_reward_map_2d[y, x]
                        agent.true_reward_map[y, x] = predicted_value

            # Learning Signal
            if done:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 1
            else:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 0


            trigger_ae_training = False
            if abs(predicted_reward_map_2d[agent_position[1], agent_position[0]] - agent.true_reward_map[agent_position[1], agent_position[0]]) > train_vision_threshold:
                ae_trigger_count_this_episode += 1
                trigger_ae_training = True
                
            
            # we then look to train the AE on this single step, where the input is the image from the environment and the loss propagation
            # is between this input image and the agents true_reward_map.
            if trigger_ae_training:
                # print("AE Training Triggered")
                target = agent.true_reward_map[np.newaxis, ..., np.newaxis]

                # Train the model for a single step
                history = ae_model.fit(
                    input_grid,       # Input: current environment grid 
                    target,           # Target: agent's true_reward_map
                    epochs=1,         # Just one training step
                    batch_size=1,     # Single sample
                    verbose=0         # Suppress output for cleaner logs
                )
                
                # Track training loss
                step_loss = history.history['loss'][0]
                # print(f"Vision model training loss: {step_loss:.4f}")
            
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
                        agent.reward_maps[idx, y, x] = reward
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
                step_counts.append(step)
                break
                
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        ae_triggers_per_episode.append(ae_trigger_count_this_episode)

        # Generate visualizations occasionally
        if episode % 100 == 0:
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
            plt.savefig(generate_save_path(f'SR/averaged_M_{episode}.png'))
            plt.close()  # Close the figure to free memory

        #     save_env_map_pred(agent = agent, normalized_grid = normalized_grid, predicted_reward_map_2d = predicted_reward_map_2d, episode = episode, save_path=generate_save_path(f"episode_{episode}"))
        
    ae_model.save(generate_save_path('vision_model.h5'))
    
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

    # Plotting the number of reward occurences in each state
    plt.figure(figsize=(6, 6))
    plt.imshow(reward_occurence_map, cmap='hot', interpolation='nearest')
    plt.title('Reward Occurrence Map (Heatmap)')
    plt.colorbar(label='Times Reward Observed')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(generate_save_path("reward_occurences.png"))
    plt.close()

    # Plotting the number of steps taken in each episode - See if its learning
    rolling = pd.Series(step_counts).rolling(window).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(rolling, label=f'Rolling Avg Step Count (window={window})', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Taken')
    plt.title('Steos taken per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(generate_save_path("step_count.png"))
    return episode_rewards


def main():
    # Setup the environment
    # env = SimpleEnv(size=10, render_mode = "human")
    env = SimpleEnv(size=10)

    # Setup the agent
    agent = SuccessorAgent(env)

    # Setup the agents Vision System
    input_shape = (env.size, env.size, 1)  
    ae_model = build_autoencoder(input_shape)

    # CHANGED to quadratic loss
    # ae_model.compile(optimizer='adam', loss=focal_mse_loss)
    ae_model.compile(optimizer='adam', loss='mse')

    # Train the agent
    rewards = train_successor_agent(agent, env, ae_model = ae_model) 

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

