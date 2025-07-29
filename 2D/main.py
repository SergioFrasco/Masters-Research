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
    
def evaluate_goal_state_values(agent, env, episode, log_file_path):
    """
    Evaluate state values around the agent when it reaches a goal.
    Compare goal state value with neighboring states and log the result.
    """
    current_pos = env.agent_pos
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
            cell = env.grid.get(nx, ny)
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

def train_successor_agent(agent, env, episodes = 1201, ae_model=None, max_steps=200, epsilon_start=1.0, epsilon_end=0.5, epsilon_decay=0.995, train_vision_threshold=0.1):
    """
    Training loop for SuccessorAgent in MiniGrid environment with vision model integration, SR tracking, and WVF formation
    """
    episode_rewards = []
    ae_triggers_per_episode = []
    epsilon = epsilon_start
    step_counts = []

    # Tracking where the agent is going
    state_occupancy = np.zeros((env.size, env.size), dtype=np.int32)

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
        obs = env.reset()
        total_reward = 0
        step_count = 0

        # agent_pos = tuple(env.agent_pos)  
        # agent_starting_positions[agent_pos[1], agent_pos[0]] += 1

        # For every new episode, reset the reward map, and WVF, the SR stays consistent with the environment i.e doesn't reset
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        ae_trigger_count_this_episode = 0

        # Reset visited positions for new episode
        agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]

        # Check reward Occurence
        # Iterate over the grid and increment reward occurrence where object type == 8 (goal)
        # grid1 = env.grid.encode()
        # object_layer1 = grid1[..., 0]
        # for y in range(env.height):
        #     for x in range(env.width):
        #         if object_layer1[y, x] == 8:
        #             reward_occurence_map[y, x] += 1

        for step in range(max_steps):
            # Take action and observe result
            obs, reward, done, _, _ = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)

            # Tracking where the agent is going
            agent_pos = tuple(env.agent_pos)  
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
            agent_position = tuple(env.agent_pos)

            # Get the current environment grid
            grid = env.grid.encode()
            normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)  # Shape: (H, W)

            # Setting up input for the AE to obtain it's prediction of the space
            # Object types are in grid[..., 0]
            object_layer = grid[..., 0]
            normalized_grid[object_layer == 2] = 0.0   # Wall 
            normalized_grid[object_layer == 1] = 0.0   # Open space
            normalized_grid[object_layer == 8] = 1.0   # Reward (e.g. goal object)
            
            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
            
            # Get the predicted reward map from the AE
            predicted_reward_map = ae_model.predict(input_grid, verbose=0)
            predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]

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

            # Update the rest of the true_reward_map with AE predictions
            for y in range(agent.true_reward_map.shape[0]):
                for x in range(agent.true_reward_map.shape[1]):
                    # Skip visited positions - don't override ground truth
                    if not agent.visited_positions[y, x]:
                        # Get the predicted value for this position from the AE
                        predicted_value = predicted_reward_map_2d[y, x]
                        if predicted_value > 0.001:
                            agent.true_reward_map[y, x] = predicted_value
                        else:
                            agent.true_reward_map[y, x] = 0

            trigger_ae_training = False
            if abs(predicted_reward_map_2d[agent_position[0], agent_position[1]] - agent.true_reward_map[agent_position[0], agent_position[1]]) > train_vision_threshold:
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
        if episode % 250 == 0:
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

            save_env_map_pred(agent = agent, normalized_grid = normalized_grid, predicted_reward_map_2d = predicted_reward_map_2d, episode = episode, save_path=generate_save_path(f"predictions/episode_{episode}"))
        
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

