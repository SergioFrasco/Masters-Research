import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgent
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper


 
def run_successor_experiment(self, episodes=5000, max_steps=10, seed=20):
    """Run Master agent experiment"""
    
    np.random.seed(seed)
    env = SimpleEnv(size=10, render_mode = "human")
    # env = SimpleEnv(size=10)
    # env = ViewSizeWrapper(env, agent_view_size=7) 

    agent = SuccessorAgent(env.unwrapped)

    # Setup torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.unwrapped.size, env.unwrapped.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()


    # Setup vision model
    # input_shape = (env.size, env.size, 1)
    # ae_model = build_autoencoder(input_shape)
    # ae_model.compile(optimizer="adam", loss="mse")

    episode_rewards = []
    episode_lengths = []
    epsilon = 1
    epsilon_end = 0.05
    epsilon_decay = 0.9995

    for episode in tqdm(range(episodes), desc=f"Masters Successor (seed {seed})"):
        obs = env.reset()
        total_reward = 0
        steps = 0
        trajectory = []  # Track trajectory for failure analysis
        

        # Reset for new episode 
        agent.true_reward_map = np.zeros((env.unwrapped.size, env.unwrapped.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        agent.visited_positions = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=bool)

        current_state_idx = agent.get_state_index(obs)
        
        # === MOVED: Initial vision update BEFORE first action selection ===
        # agent_position = tuple(env.unwrapped.agent_pos)
        # agent_dir = env.unwrapped.agent_dir
        
        # # Create initial egocentric view
        # egocentric_input = create_egocentric_view(env, agent_position, agent_dir, VIEW_SIZE, 
        #                                         EMPTY_SPACE, REWARD, OUT_OF_BOUNDS, WALL, done=False)
        
        # # Get perfect vision prediction
        # predicted_ego_view_2d = np.zeros((VIEW_SIZE, VIEW_SIZE))
        # for ego_y in range(VIEW_SIZE):
        #     for ego_x in range(VIEW_SIZE):
        #         global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, agent_position[0], 
        #                                                         agent_position[1], agent_dir, VIEW_SIZE)
                
        #         if 0 <= global_x < env.unwrapped.size and 0 <= global_y < env.unwrapped.size:
        #             cell = env.unwrapped.grid.get(global_x, global_y)
        #             if cell is not None and cell.type == 'goal':
        #                 predicted_ego_view_2d[ego_y, ego_x] = 1.0
        #             else:
        #                 predicted_ego_view_2d[ego_y, ego_x] = 0.0
        #         else:
        #             predicted_ego_view_2d[ego_y, ego_x] = 0.0
        
        # # Update initial reward map
        # agent.visited_positions[agent_position[0], agent_position[1]] = True
        # agent.true_reward_map[agent_position[0], agent_position[1]] = 0.0  # Starting position has no reward
        
        # visible_global_positions = get_visible_global_positions(agent_position, agent_dir, VIEW_SIZE, env.unwrapped.size)
        # update_true_reward_map_from_egocentric_prediction(agent, predicted_ego_view_2d, visible_global_positions, 
        #                                                 VIEW_SIZE, agent_position=agent_position, agent_dir=agent_dir, learning_rate=1.0, env=env, episode=episode, step=0)
        
        # # Update initial WVF
        # agent.reward_maps.fill(0)
        # for y in range(agent.grid_size):
        #     for x in range(agent.grid_size):
        #         reward = agent.true_reward_map[y, x]
        #         idx = y * agent.grid_size + x
        #         reward_threshold = 0.5
        #         if reward > reward_threshold:
        #             agent.reward_maps[idx, y, x] = 1
        #         else:
        #             agent.reward_maps[idx, y, x] = 0
        
        # M_flat = np.mean(agent.M, axis=0)
        # R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
        # V_all = M_flat @ R_flat_all.T
        # agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)


        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]

        for step in range(max_steps):
            # Record position and action for trajectory
            agent_pos = tuple(env.unwrapped.agent_pos)
            trajectory.append((agent_pos[0], agent_pos[1], current_action))
            
            obs, reward, done, _, _ = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)

            # Plot just this step's egocentric view
            agent_view = obs['image']

            # Extract object types (first channel)
            object_types = agent_view[:, :, 0]

            plt.figure(figsize=(6, 6))
            plt.imshow(object_types, cmap='tab10', vmin=0, vmax=10)
            plt.title(f'Step {step} - Object Types')
            plt.axis('off')

            # Add colorbar with labels
            cbar = plt.colorbar(ticks=[0, 1, 2, 5, 8])
            cbar.ax.set_yticklabels(['Unseen', 'Empty', 'Wall', 'Goal', 'Agent'])

            plt.savefig(f'step_{step}.png', bbox_inches='tight', dpi=100)
            plt.close()

            # Complete experience
            current_exp[2] = next_state_idx
            current_exp[3] = reward
            current_exp[4] = done

            # Choose next action
            if step == 0 or episode < 1:  # Warmup period
                next_action = agent.sample_random_action(obs, epsilon=epsilon)
            else:
                next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

            next_exp = [next_state_idx, next_action, None, None, None]

            # Update agent
            agent.update(current_exp, None if done else next_exp)

            # Vision Model
            # Update the agent's true_reward_map based on current observation
            agent_position = tuple(env.unwrapped.agent_pos)

            # Get the current environment grid
            grid = env.unwrapped.grid.encode()
            normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)  # Shape: (H, W)

            # Setting up input for the AE to obtain it's prediction of the space
            object_layer = grid[..., 0]
            normalized_grid[object_layer == 2] = 0.0  # Wall
            normalized_grid[object_layer == 1] = 0.0  # Open space
            normalized_grid[object_layer == 8] = 1.0  # Reward (e.g. goal object)

            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

            # Get the predicted reward map from the AE
            with torch.no_grad():
                ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)
                predicted_reward_map_tensor = ae_model(ae_input_tensor)  # (1, 1, H, W)
                predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()  # (H, W)

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
                    if not agent.visited_positions[y, x]:
                        predicted_value = predicted_reward_map_2d[y, x]
                        if predicted_value > 0.001:
                            agent.true_reward_map[y, x] = predicted_value
                        else:
                            agent.true_reward_map[y, x] = 0

            # Train the vision model
            trigger_ae_training = False
            train_vision_threshold = 0.1
            if (abs(predicted_reward_map_2d[agent_position[0], agent_position[1]]- agent.true_reward_map[agent_position[0], agent_position[1]])> train_vision_threshold):
                trigger_ae_training = True

            if trigger_ae_training:
                target_tensor = torch.tensor(agent.true_reward_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)

                ae_model.train()
                optimizer.zero_grad()
                output = ae_model(ae_input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                step_loss = loss.item()


            agent.reward_maps.fill(0)  # Reset all maps to zero

            for y in range(agent.grid_size):
                for x in range(agent.grid_size):
                    curr_reward = agent.true_reward_map[y, x]
                    idx = y * agent.grid_size + x
                    reward_threshold = 0.5
                    if curr_reward > reward_threshold:
                        agent.reward_maps[idx, y, x] = 1
                    else:
                        agent.reward_maps[idx, y, x] = 0

            # Update agent WVF
            M_flat = np.mean(agent.M, axis=0)
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
            V_all = M_flat @ R_flat_all.T
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

            total_reward += reward
            steps += 1
            current_exp = next_exp
            current_action = next_action

            if done:
                break

        # Check for failure in last 100 episodes and save trajectory plot
        if episode >= episodes - 100 and not done:
            self.plot_and_save_trajectory("Masters Successor", episode, trajectory, env.unwrapped.size, seed)

            # Generate visualizations occasionally
        if episode % 200 == 0:
            save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))
            
            # Save egocentric view visualization using exact position
            # save_egocentric_view_visualization(
            #     egocentric_input,
            #     predicted_ego_view_2d,
            #     egocentric_target,
            #     episode,
            #     step,
            #     agent_pos=agent_pos,  # Use exact position
            #     agent_dir=agent_dir,  # Use exact direction
            #     env_size=env.unwrapped.size
            # )

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
    
            # Plot the actual environment layout with viridis
            env_grid = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=float)
            
            # Iterate through the grid to identify objects
            for x in range(env.unwrapped.size):
                for y in range(env.unwrapped.size):
                    cell = env.unwrapped.grid.get(x, y)
                    if cell is None:
                        env_grid[y, x] = 0.0  # Empty space - dark purple
                    elif cell.type == 'wall':
                        env_grid[y, x] = 0.3  # Wall - dark blue/green
                    elif cell.type == 'goal':
                        env_grid[y, x] = 1.0  # Goal/Reward - bright yellow
            
            # Mark current agent position using exact position
            agent_x, agent_y = env.unwrapped.agent_pos
            env_grid[agent_y, agent_x] = 0.6  # Agent - teal/green
            
            # Create the plot
            plt.figure(figsize=(6, 6))
            im = plt.imshow(env_grid, cmap='viridis', vmin=0, vmax=1)
            plt.title(f"Environment Ground Truth (Episode {episode})")
            
            # Add grid lines for clarity
            plt.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
            plt.xticks(range(env.unwrapped.size))
            plt.yticks(range(env.unwrapped.size))
            
            # Add colorbar with labels
            cbar = plt.colorbar(im, label="Object Type", ticks=[0, 0.3, 0.6, 1.0])
            cbar.ax.set_yticklabels(['Empty', 'Wall', 'Agent', 'Goal'])
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f'environment/ground_truth_episode_{episode}.png'))
            plt.close()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "final_epsilon": epsilon,
            "algorithm": "Masters Successor",
        }
 
def run_comparison_experiment(self, episodes=5000):
    """Run comparison between all agents across multiple seeds"""
    all_results = {}
    
    for seed in range(self.num_seeds):
        print(f"\n=== Running experiments with seed {seed} ===")

        # Run Masters successor
        successor_results = self.run_successor_experiment(episodes=episodes, seed=seed)
        
        # Store results
        algorithms = ['Masters Successor']
        results_list = [successor_results]
        
        for alg, result in zip(algorithms, results_list):
            if alg not in all_results:
                all_results[alg] = []
            all_results[alg].append(result)

        # Force cleanup between seeds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    self.results = all_results
    return all_results


def main():
    """Run the experiment comparison"""
    print("Starting baseline comparison experiment...")

    # Run experiments
    results = run_comparison_experiment(episodes=1)

    # Analyze and plot results
    summary = analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()