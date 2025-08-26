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
from utils.coordinate_system import CoordinateSystem
import json
import time
import gc
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper



# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class PartiallyObservableSuccessorAgent(SuccessorAgent):
    """
    Extended Successor Agent that builds a map from partial observations.
    """
    def __init__(self, env, view_size=7):
        super().__init__(env)
        self.view_size = view_size
        
        # STEP 1: Initialize learned map and confidence tracking
        # learned_map: Accumulates predicted reward probabilities over time
        # confidence_map: Tracks how many times each cell has been observed
        self.learned_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
    def egocentric_to_global_coords(self, local_x, local_y, agent_x, agent_y, agent_dir):
        """Use standardized coordinate transformation"""
        agent_pos = (agent_x, agent_y)
        global_pos = CoordinateSystem.egocentric_to_global(
            local_x, local_y, agent_pos, agent_dir, self.view_size
        )
        return global_pos[0], global_pos[1] 


    # Update array access patterns throughout:
    def update_learned_map_from_partial_view(self, predicted_partial_2d, agent_x, agent_y, agent_dir):
        for local_y in range(self.view_size):
            for local_x in range(self.view_size):
                global_x, global_y = self.egocentric_to_global_coords(
                    local_x, local_y, agent_x, agent_y, agent_dir
                )
                
                if CoordinateSystem.is_valid_position((global_x, global_y), self.grid_size):
                    # Use standardized array access
                    old_value = CoordinateSystem.array_get(self.learned_map, (global_x, global_y))
                    new_value = predicted_partial_2d[local_y, local_x]
                    update_learning_rate = 1
                    new_learned_value = old_value * (1 - update_learning_rate) + new_value * update_learning_rate
                    CoordinateSystem.array_set(self.learned_map, (global_x, global_y), new_learned_value)


class ExperimentRunner:
    """Handles running experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def plot_and_save_trajectory(self, agent_name, episode, trajectory, env_size, seed):
        """Plot and save the agent's trajectory for failed episodes"""
        print(f"Agent {agent_name} failed: plotting trajectory")
        
        # Create a grid to visualize the path
        grid = np.zeros((env_size, env_size), dtype=str)
        grid[:] = '.'  # Empty spaces
        
        # Mark the trajectory
        for i, (x, y, action) in enumerate(trajectory):
            # FIXED: Use [y, x] for grid indexing
            if i == 0:
                grid[y, x] = 'S'  # Start
            elif i == len(trajectory) - 1:
                grid[y, x] = 'E'  # End
            else:
                # Use action arrows
                action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
                grid[y, x] = action_symbols.get(action, str(i % 10))
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a numerical grid for visualization
        visual_grid = np.zeros((env_size, env_size))
        color_map = {'S': 1, 'E': 2, '↑': 3, '→': 4, '↓': 5, '←': 6, '.': 0}
        
        for i in range(env_size):
            for j in range(env_size):
                visual_grid[i, j] = color_map.get(grid[i, j], 0)
        
        # Plot the grid
        im = ax.imshow(visual_grid, cmap='tab10', alpha=0.8)
        
        # Add text annotations
        for i in range(env_size):
            for j in range(env_size):
                ax.text(j, i, grid[i, j], ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
        
        # Customize the plot
        ax.set_title(f'{agent_name} Trajectory - Episode {episode}\nPath length: {len(trajectory)} steps', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='tab:blue', label='Start (S)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:orange', label='End (E)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:green', label='Up (↑)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:red', label='Right (→)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:purple', label='Down (↓)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:brown', label='Left (←)')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Generate filename and save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{agent_name.replace(' ', '_').lower()}_episode_{episode}_seed_{seed}_{timestamp}.png"
        save_path = generate_save_path(filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Trajectory plot saved to: {save_path}")

    
    def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run Partially Observable Successor Agent experiment"""
        
        np.random.seed(seed)
        env = SimpleEnv(size=10)
        view_size = 7
        # env = ViewSizeWrapper(env, agent_view_size=view_size)  # 7x7 partial view
        
        # STEP 4: Initialize the partially observable agent and vision model
        agent = PartiallyObservableSuccessorAgent(env.unwrapped, view_size=view_size)

        # Setup torch with partial observable autoencoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_model = Autoencoder(input_channels=1).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        episode_rewards = []
        episode_lengths = []
        epsilon = 1
        epsilon_end = 0.05
        epsilon_decay = 0.9995

        for episode in tqdm(range(episodes), desc=f"Partial Observable Successor (seed {seed})"):
            obs = env.reset()
            total_reward = 0
            steps = 0
            trajectory = []  # Track trajectory for failure analysis

            # Reset for new episode - but keep learned_map to accumulate knowledge!
            agent.true_reward_map = np.zeros((env.unwrapped.size, env.unwrapped.size))
            agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
            agent.visited_positions = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=bool)

            current_state_idx = agent.get_state_index(obs)
            current_action = agent.sample_random_action(obs, epsilon=epsilon)
            current_exp = [current_state_idx, current_action, None, None, None]

            for step in range(max_steps):
                # Record position and action for trajectory
                agent_pos = tuple(env.unwrapped.agent_pos)
                agent_dir = env.unwrapped.agent_dir
                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                obs, reward, done, _, _ = env.step(current_action)
                next_state_idx = agent.get_state_index(obs)

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

                # ========== VISION MODEL WITH PARTIAL OBSERVABILITY ==========
                agent_position = tuple(env.unwrapped.agent_pos)
                # STEP 5: Process the partial observation (7x7 view)
                partial_obs = obs['image']  # 7x7x3 observation from ViewSizeWrapper
                
                # STEP 5a: Normalize the partial observation for the autoencoder
                normalized_partial = np.zeros((partial_obs.shape[0], partial_obs.shape[1]), dtype=np.float32)
                object_layer = partial_obs[..., 0]  # First channel contains object types
                
                # Map object types to reward probabilities
                normalized_partial[object_layer == 2] = 0.0  # Wall
                normalized_partial[object_layer == 1] = 0.0  # Open space  
                normalized_partial[object_layer == 8] = 1.0  # Goal/Reward - this is what we want to predict!
                
                # STEP 5b: Prepare input for autoencoder (add batch and channel dimensions)
                input_partial = normalized_partial[np.newaxis, ..., np.newaxis]  # (1, 7, 7, 1)
                
                # STEP 6: Get prediction from the vision model
                # with torch.no_grad():
                #     ae_input_tensor = torch.tensor(input_partial, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                #     predicted_partial = ae_model(ae_input_tensor)  # (1, 1, 7, 7)
                #     predicted_partial_2d = predicted_partial.squeeze().cpu().numpy()  # (7, 7)
                

                # STEP 7: Update the global learned map with the partial prediction
                # agent_x, agent_y = agent_position
                # agent.update_learned_map_from_partial_view(predicted_partial_2d, agent_x, agent_y, agent_dir)

                # Use perfect vision, instead of passing predicted_partial_2d here, pass normalized_partial
                # STEP 7: Update the global learned map with the partial prediction

                # FOR PERFECT VISION: Skip the learned map update entirely, or update it with perfect ground truth
                # Comment out this line to stop accumulating imperfect information:
                # agent.update_learned_map_from_partial_view(normalized_partial, agent_x, agent_y, agent_dir)

                # Instead, build perfect learned map directly from environment
                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        cell = env.unwrapped.grid.get(x, y)
                        if cell is not None and cell.type == 'goal':
                            agent.learned_map[y, x] = 1.0  # Perfect knowledge of goal locations
                        else:
                            agent.learned_map[y, x] = 0.0   # Perfect knowledge of non-goal locations
                                
                # FIXED: Mark current position as visited using [y, x]
                CoordinateSystem.array_set(agent.visited_positions, agent_position, True)
                
                # STEP 8: Update true reward map at current position (ground truth for training)
                # FIXED: Use [y, x] for array indexing
                if done and step < max_steps:
                    CoordinateSystem.array_set(agent.true_reward_map, agent_position, 1.0) # Found Reward
                else:
                    CoordinateSystem.array_set(agent.true_reward_map, agent_position, 0.0)  # No reward here
                
                # STEP 9: Create training target for the partial view
                # This combines ground truth (visited positions) with predictions (unvisited)
                # partial_target = np.zeros((view_size, view_size), dtype=np.float32)
                
                # for local_y in range(view_size):
                #     for local_x in range(view_size):
                #         # Convert to global coordinates
                #         global_x, global_y = agent.egocentric_to_global_coords(local_x, local_y, agent_x, agent_y, agent_dir)
                        
                #         if 0 <= global_x < agent.grid_size and 0 <= global_y < agent.grid_size:
                #             # FIXED: Use [y, x] for array indexing
                #             # Use true reward if we've visited this position
                #             if agent.visited_positions[global_y, global_x]:
                #                 partial_target[local_y, local_x] = agent.true_reward_map[global_y, global_x]
                #             else:
                #                 # Use predicted value for unvisited areas (self-supervised learning)
                #                 partial_target[local_y, local_x] = agent.learned_map[global_y, global_x]

                # STEP 9: Create training target for the partial view with PERFECT VISION
                # This should show exactly what the agent sees in the 7x7 view from ground truth
                partial_target = np.zeros((view_size, view_size), dtype=np.float32)

                for local_y in range(view_size):
                    for local_x in range(view_size):
                        # Convert to global coordinates
                        global_x, global_y = agent.egocentric_to_global_coords(local_x, local_y, agent_position[0], agent_position[1], agent_dir)
                        
                        if 0 <= global_x < agent.grid_size and 0 <= global_y < agent.grid_size:
                            # PERFECT VISION: Read directly from the environment ground truth
                            cell = env.unwrapped.grid.get(global_x, global_y)
                            
                            if cell is None:
                                partial_target[local_y, local_x] = 0.0  # Empty space
                            elif cell.type == 'wall':
                                partial_target[local_y, local_x] = 0.0  # Wall (no reward)
                            elif cell.type == 'goal':
                                partial_target[local_y, local_x] = 1.0  # Goal/Reward
                            else:
                                partial_target[local_y, local_x] = 0.0  # Other objects (no reward)
                        else:
                            # Outside grid bounds
                            partial_target[local_y, local_x] = 0.0
                
                # STEP 10: Train the vision model when prediction error is high
                # Calculate prediction error at agent's current position (center of 7x7 view)
                center_idx = view_size // 2
                # Perfect vision - uncomment this when done debugging
                # prediction_error = abs(predicted_partial_2d[center_idx, center_idx] - agent.true_reward_map[agent_y, agent_x])
                
                # train_vision_threshold = 0.1
                # if prediction_error > train_vision_threshold:
                #     # STEP 10a: Prepare target tensor
                #     target_tensor = torch.tensor(partial_target[np.newaxis, ..., np.newaxis], dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (1, 1, 7, 7)
                    
                #     # STEP 10b: Train the autoencoder
                #     ae_model.train()
                #     optimizer.zero_grad()
                #     output = ae_model(ae_input_tensor)
                #     loss = loss_fn(output, target_tensor)
                #     loss.backward()
                #     optimizer.step()
                    
                #     step_loss = loss.item()
                
                # STEP 11: Update reward maps for successor representation
                # Now using the learned map instead of true reward map
                agent.reward_maps.fill(0)  # Reset all maps to zero
                
                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        # Use learned map which is built from partial observations
                        curr_reward = agent.learned_map[y, x]
                        
                        idx = y * agent.grid_size + x
                        reward_threshold = 0.5
                        if curr_reward > reward_threshold:
                            agent.reward_maps[idx, y, x] = 1 
                        else:
                            agent.reward_maps[idx, y, x] = 0

                # STEP 12: Update agent's value function using the learned reward maps
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
            # if episode >= episodes - 100 and not done:
            #     self.plot_and_save_trajectory("Partial Observable Successor", episode, trajectory, env.unwrapped.size, seed)

            # Generate visualizations occasionally
            if episode % 1000 == 0:
                # Save value functions
                save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))
                
                # STEP 13: Visualize the learned map (what the agent has learned about rewards)
                plt.figure(figsize=(12, 5))
                
                # Plot 1: Learned reward map
                plt.subplot(1, 3, 1)
                im1 = plt.imshow(agent.learned_map, cmap='hot', vmin=0, vmax=1)
                plt.title(f"Learned Reward Map (Episode {episode})")
                plt.colorbar(im1, label="Predicted Reward Probability")
                
                plt.tight_layout()
                plt.savefig(generate_save_path(f'learned_maps/maps_episode_{episode}.png'))
                plt.close()
                
                # Visualize averaged SR matrix
                averaged_M = np.mean(agent.M, axis=0)
                plt.figure(figsize=(6, 5))
                im = plt.imshow(averaged_M, cmap='hot')
                plt.title(f"Averaged SR Matrix (Episode {episode})")
                plt.colorbar(im, label="SR Value")
                plt.tight_layout()
                plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
                plt.close()
        
                # Plot the actual environment layout (ground truth for comparison)
                env_grid = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=float)
                
                for x in range(env.unwrapped.size):
                    for y in range(env.unwrapped.size):
                        cell = env.unwrapped.grid.get(x, y)
                        # FIXED: Use [y, x] for array indexing
                        if cell is None:
                            env_grid[y, x] = 0.0  # Empty space
                        elif cell.type == 'wall':
                            env_grid[y, x] = 0.3  # Wall
                        elif cell.type == 'goal':
                            env_grid[y, x] = 1.0  # Goal/Reward
                
                # Mark current agent position
                agent_x, agent_y = env.unwrapped.agent_pos
                # FIXED: Use [y, x] for array indexing
                env_grid[agent_y, agent_x] = 0.6  # Agent
                
                plt.figure(figsize=(6, 6))
                im = plt.imshow(env_grid, cmap='viridis', vmin=0, vmax=1)
                plt.title(f"Environment Ground Truth (Episode {episode})")
                plt.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                plt.xticks(range(env.unwrapped.size))
                plt.yticks(range(env.unwrapped.size))
                cbar = plt.colorbar(im, label="Object Type", ticks=[0, 0.3, 0.6, 1.0])
                cbar.ax.set_yticklabels(['Empty', 'Wall', 'Agent', 'Goal'])
                plt.tight_layout()
                plt.savefig(generate_save_path(f'environment/ground_truth_episode_{episode}.png'))
                plt.close()

               # ============== DEBUGGING PLOTS FOR EVERY STEP ==============
                if env is not None and episode is not None and step is not None:
                    
                    visible_positions = get_visible_global_positions(agent_position, agent_dir, view_size, 10)

                    # Changed from 1x3 to 2x2 to accommodate the learned map plot
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # === PLOT 1: GROUND TRUTH ENVIRONMENT ===
                    env_grid = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=float)
                    
                    # Build ground truth environment grid
                    for x in range(env.unwrapped.size):
                        for y in range(env.unwrapped.size):
                            cell = env.unwrapped.grid.get(x, y)
                            # FIXED: Use [y, x] for grid array indexing
                            if cell is None:
                                env_grid[y, x] = 0.0  # Empty space
                            elif cell.type == 'wall':
                                env_grid[y, x] = 0.5  # Wall
                            elif cell.type == 'goal':
                                env_grid[y, x] = 1.0  # Goal/Reward
                            else:
                                env_grid[y, x] = 0.2  # Other objects
                    
                    # Plot ground truth environment
                    im1 = axes[0,0].imshow(env_grid, cmap='viridis', vmin=0, vmax=1)
                    axes[0,0].set_title(f'Ground Truth Environment\nEpisode {episode}, Step {step}')
                    
                    # Mark agent position and direction on ground truth
                    agent_x, agent_y = int(agent_position[0]), int(agent_position[1])
                    # When plotting with matplotlib, x is column, y is row
                    axes[0,0].plot(agent_x, agent_y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
                    
                    # Draw arrow showing agent direction
                    dx, dy = 0, 0
                    arrow_length = 0.4
                    if agent_dir == 0:  # right
                        dx = arrow_length
                    elif agent_dir == 1:  # down
                        dy = arrow_length
                    elif agent_dir == 2:  # left
                        dx = -arrow_length
                    elif agent_dir == 3:  # up
                        dy = -arrow_length
                    
                    axes[0,0].arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.1, 
                                fc='red', ec='white', linewidth=2)
                    
                    # Add grid and labels for ground truth
                    axes[0,0].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                    axes[0,0].set_xticks(range(env.unwrapped.size))
                    axes[0,0].set_yticks(range(env.unwrapped.size))
                    axes[0,0].set_xlabel('X')
                    axes[0,0].set_ylabel('Y')
                    
                    # Add colorbar with labels for ground truth
                    cbar1 = fig.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
                    cbar1.set_label('Object Type')
                    
                    # === PLOT 2: TRUE REWARD MAP ===
                    im2 = axes[0,1].imshow(agent.true_reward_map, cmap='hot', vmin=0, vmax=1)
                    axes[0,1].set_title(f'True Reward Map (Agent\'s Belief)\nEpisode {episode}, Step {step}')
                    
                    # Mark agent position and direction on true reward map
                    axes[0,1].plot(agent_x, agent_y, 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
                    axes[0,1].arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.1, 
                                fc='cyan', ec='white', linewidth=2)
                    
                    # Highlight visited positions with small markers
                    # FIXED: np.where returns (row_indices, col_indices) which is (y, x)
                    visited_y, visited_x = np.where(agent.visited_positions)
                    axes[0,1].scatter(visited_x, visited_y, c='lime', s=20, marker='s', alpha=0.7, 
                                edgecolors='black', linewidths=0.5, label='Visited')
                    
                    # Highlight currently visible positions
                    visible_x = [pos[0] for pos in visible_positions]
                    visible_y = [pos[1] for pos in visible_positions]
                    axes[0,1].scatter(visible_x, visible_y, c='yellow', s=30, marker='o', alpha=0.8,
                                edgecolors='black', linewidths=1, label='Currently Visible')
                    
                    # Add grid and labels for true reward map
                    axes[0,1].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                    axes[0,1].set_xticks(range(env.unwrapped.size))
                    axes[0,1].set_yticks(range(env.unwrapped.size))
                    axes[0,1].set_xlabel('X')
                    axes[0,1].set_ylabel('Y')
                    axes[0,1].legend(loc='upper right', bbox_to_anchor=(1, 1))
                    
                    # Add colorbar for true reward map
                    cbar2 = fig.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
                    cbar2.set_label('Reward Belief')
                    
                    # === PLOT 3: AUTOENCODER TARGET (7x7 PARTIAL VIEW) ===
                    im3 = axes[1,0].imshow(partial_target, cmap='hot', vmin=0, vmax=1)
                    axes[1,0].set_title(f'AE Target (7x7 View)\nEpisode {episode}, Step {step}')
                    
                    # Mark the center bottom of the view (where agent is)
                    center_idx_x = view_size // 2
                    center_idx_y = view_size - 1
                    axes[1,0].plot(center_idx_x, center_idx_y, 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
                    
                    # Add arrow showing agent direction in the 7x7 view
                    dx_local, dy_local = 0, -arrow_length
                    axes[1,0].arrow(center_idx_x, center_idx_y, dx_local * 0.3, dy_local * 0.3, head_width=0.15, head_length=0.1, fc='cyan', ec='white', linewidth=2)
                    
                    # Add grid and labels for AE target
                    axes[1,0].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                    axes[1,0].set_xticks(range(view_size))
                    axes[1,0].set_yticks(range(view_size))
                    axes[1,0].set_xlabel('Local X (Egocentric)')
                    axes[1,0].set_ylabel('Local Y (Egocentric)')
                    
                    # Add colorbar for AE target
                    cbar3 = fig.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)
                    cbar3.set_label('Target Reward')
                    
                    # Add text annotations showing which pixels in the 7x7 view correspond to visited vs predicted areas
                    visited_count = 0
                    predicted_count = 0
                    for local_y in range(view_size):
                        for local_x in range(view_size):
                            # Convert to global coordinates to check if visited
                            global_x, global_y = agent.egocentric_to_global_coords(local_x, local_y, agent_x, agent_y, agent_dir)
                            if 0 <= global_x < agent.grid_size and 0 <= global_y < agent.grid_size:
                                if agent.visited_positions[global_y, global_x]:
                                    visited_count += 1
                                    # Mark visited positions with small green dots
                                    axes[1,0].plot(local_x, local_y, 'g.', markersize=4, alpha=0.8)
                                else:
                                    predicted_count += 1
                                    # Mark predicted positions with small red dots
                                    axes[1,0].plot(local_x, local_y, 'r.', markersize=4, alpha=0.8)
                    
                    # === PLOT 4: LEARNED MAP (AGENT'S ACCUMULATED KNOWLEDGE) ===
                    im4 = axes[1,1].imshow(agent.learned_map, cmap='hot', vmin=0, vmax=1)
                    axes[1,1].set_title(f'Learned Map (Accumulated)\nEpisode {episode}, Step {step}')
                    
                    # Mark agent position and direction on learned map
                    axes[1,1].plot(agent_x, agent_y, 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
                    axes[1,1].arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.1, 
                                fc='cyan', ec='white', linewidth=2)
                    
                    # Highlight visited positions with small markers
                    axes[1,1].scatter(visited_x, visited_y, c='lime', s=20, marker='s', alpha=0.7, 
                                edgecolors='black', linewidths=0.5, label='Visited')
                    
                    # Highlight currently visible positions
                    axes[1,1].scatter(visible_x, visible_y, c='yellow', s=30, marker='o', alpha=0.8,
                                edgecolors='black', linewidths=1, label='Currently Visible')
                    
                    # Add grid and labels for learned map
                    axes[1,1].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                    axes[1,1].set_xticks(range(env.unwrapped.size))
                    axes[1,1].set_yticks(range(env.unwrapped.size))
                    axes[1,1].set_xlabel('X')
                    axes[1,1].set_ylabel('Y')
                    axes[1,1].legend(loc='upper right', bbox_to_anchor=(1, 1))
                    
                    # Add colorbar for learned map
                    cbar4 = fig.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
                    cbar4.set_label('Learned Reward Prob')
                    
                    # Add text annotations with key information
                    direction_names = ['Right', 'Down', 'Left', 'Up']
                    info_text = f"Agent Dir: {direction_names[agent_dir]}\n"
                    info_text += f"Position: ({agent_x}, {agent_y})\n"
                    info_text += f"Visited Positions: {np.sum(agent.visited_positions)}\n"
                    info_text += f"AE Target - Visited: {visited_count}, Predicted: {predicted_count}\n"
                    info_text += f"Learned Map Max: {np.max(agent.learned_map):.3f}\n"
                    info_text += f"Learned Map Nonzero: {np.count_nonzero(agent.learned_map)}"
                    
                    fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                                        facecolor="lightgray", alpha=0.8))
                    
                    plt.tight_layout()
                    
                    # Save the plot
                    save_path = generate_save_path(f"debug_reward_maps/episode_{episode}_step_{step:03d}.png")
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Also save a summary every 10 steps showing reward retention
                    if step % 10 == 0:
                        # Create a difference plot to show how much the true reward map differs from ground truth
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                        
                        # Create ground truth reward map (1 where goals are, 0 elsewhere)
                        gt_reward_map = np.zeros_like(agent.true_reward_map)
                        for x in range(env.unwrapped.size):
                            for y in range(env.unwrapped.size):
                                cell = env.unwrapped.grid.get(x, y)
                                if cell is not None and cell.type == 'goal':
                                    # FIXED: Use [y, x] for array indexing
                                    gt_reward_map[y, x] = 1.0
                        
                        # Calculate difference (how well agent remembers rewards)
                        difference = np.abs(agent.true_reward_map - gt_reward_map)
                        
                        im = ax.imshow(difference, cmap='Reds', vmin=0, vmax=1)
                        ax.set_title(f'Reward Memory Error\nEpisode {episode}, Step {step}\n(Red = Forgot Reward, Dark = Correct)')
                        
                        # Mark agent
                        ax.plot(agent_x, agent_y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
                        
                        # Add grid
                        ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                        ax.set_xticks(range(env.unwrapped.size))
                        ax.set_yticks(range(env.unwrapped.size))
                        
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Absolute Error')
                        plt.tight_layout()
                        
                        save_path = generate_save_path(f"reward_memory_error/episode_{episode}_step_{step:03d}.png")
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        plt.close()

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Partial Observable Successor",
        }
 
    def run_comparison_experiment(self, episodes=5000):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")

            # Run Partial Observable successor
            successor_results = self.run_successor_experiment(episodes=episodes, seed=seed)
            
            # Store results
            algorithms = ['Partial Observable Successor']
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
    
    def analyze_results(self, window=5):
        """Analyze and plot comparison results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Learning curves (rewards)
        ax1 = axes[0, 0]
        for alg_name, runs in self.results.items():
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
        for alg_name, runs in self.results.items():
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

        # Plot 3: Final performance comparison (last 100 episodes)
        ax3 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])  # Last 100 episodes
            final_rewards[alg_name] = final_100

        ax3.boxplot(final_rewards.values(), labels=final_rewards.keys())
        ax3.set_ylabel("Reward")
        ax3.set_title("Final Performance (Last 100 Episodes)")
        ax3.grid(True)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)

            summary_data.append(
                {
                    "Algorithm": alg_name,
                    "Final Performance": final_performance,
                    "Convergence Episode": convergence_episode,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        ax4.axis("tight")
        ax4.axis("off")
        table = ax4.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax4.set_title("Summary Statistics")

        plt.tight_layout()
        save_path = generate_save_path("experiment_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

        # Save numerical results
        self.save_results()

        return summary_df

    def _find_convergence_episode(self, all_rewards, window):
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

    def save_results(self):
        """Save experimental results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save raw results as JSON
        results_file = generate_save_path(f"experiment_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_run = {
                    "rewards": [float(r) for r in run["rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "final_epsilon": float(run["final_epsilon"]),
                    "algorithm": run["algorithm"],
                }
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")

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


def egocentric_to_global_coords(ego_x, ego_y, agent_x, agent_y, agent_dir, view_size):
    """Use standardized coordinate transformation"""
    agent_pos = (agent_x, agent_y)
    global_pos = CoordinateSystem.egocentric_to_global(
        ego_x, ego_y, agent_pos, agent_dir, view_size
    )
    return global_pos[0], global_pos[1]


def main():
    """Run the experiment comparison"""
    print("Starting PARTIAL OBSERVABLE vision learning experiment...")
    print("=" * 60)
    print("Key changes from fully observable:")
    print("1. Agent only sees 7x7 partial view")
    print("2. Builds up learned map over time through exploration")
    print("3. Vision model learns to predict rewards from local patterns")
    print("4. Confidence tracking and exploration bonuses")
    print("=" * 60)

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=10000)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED!")
    print("Check the results/ folder for:")
    print("- learned_maps/: Visualizations of learned reward maps")
    print("- wvfs/: Value function visualizations")
    print("- sr/: Successor representation matrices")
    print("- environment/: Ground truth comparisons")
    print("=" * 60)


if __name__ == "__main__":
    main()