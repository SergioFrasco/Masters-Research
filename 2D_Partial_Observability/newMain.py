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


# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
            if i == 0:
                grid[x, y] = 'S'  # Start
            elif i == len(trajectory) - 1:
                grid[x, y] = 'E'  # End
            else:
                # Use action arrows
                action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
                grid[x, y] = action_symbols.get(action, str(i % 10))
        
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

    
    def run_successor_experiment(self, episodes=5000, max_steps=300, seed=20):
        """Run Master agent experiment"""

        # Define egocentric view parameters
        VIEW_SIZE = 7  # Agent sees 7x7 grid around itself
        # Encoding values for egocentric view
        EMPTY_SPACE = 0.0
        REWARD = 1.0
        OUT_OF_BOUNDS = 0.0  # or 8.0
        WALL = 0.0  # distinguishable from empty space
        
        np.random.seed(seed)

        env = SimpleEnv(size=10)
        env = ViewSizeWrapper(env, agent_view_size=7)  # 7x7 partial view
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
            agent_position = tuple(env.unwrapped.agent_pos)
            agent_dir = env.unwrapped.agent_dir
            
            # Create initial egocentric view
            egocentric_input = create_egocentric_view(env, agent_position, agent_dir, VIEW_SIZE, 
                                                    EMPTY_SPACE, REWARD, OUT_OF_BOUNDS, WALL, done=False)
            
            # Get perfect vision prediction
            predicted_ego_view_2d = np.zeros((VIEW_SIZE, VIEW_SIZE))
            for ego_y in range(VIEW_SIZE):
                for ego_x in range(VIEW_SIZE):
                    global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, agent_position[0], 
                                                                    agent_position[1], agent_dir, VIEW_SIZE)
                    
                    if 0 <= global_x < env.unwrapped.size and 0 <= global_y < env.unwrapped.size:
                        cell = env.unwrapped.grid.get(global_x, global_y)
                        if cell is not None and cell.type == 'goal':
                            predicted_ego_view_2d[ego_y, ego_x] = 1.0
                        else:
                            predicted_ego_view_2d[ego_y, ego_x] = 0.0
                    else:
                        predicted_ego_view_2d[ego_y, ego_x] = 0.0
            
            # Update initial reward map
            agent.visited_positions[agent_position[0], agent_position[1]] = True
            agent.true_reward_map[agent_position[0], agent_position[1]] = 0.0  # Starting position has no reward
            
            visible_global_positions = get_visible_global_positions(agent_position, agent_dir, VIEW_SIZE, env.unwrapped.size)
            update_true_reward_map_from_egocentric_prediction(agent, predicted_ego_view_2d, visible_global_positions, 
                                                            VIEW_SIZE, agent_position=agent_position, agent_dir=agent_dir, learning_rate=1.0, env=env, episode=episode, step=0)
            
            # Update initial WVF
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
            
            M_flat = np.mean(agent.M, axis=0)
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
            V_all = M_flat @ R_flat_all.T
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)


            current_action = agent.sample_random_action(obs, epsilon=epsilon)
            current_exp = [current_state_idx, current_action, None, None, None]

            for step in range(max_steps):
                # Record position and action for trajectory
                agent_pos = tuple(env.unwrapped.agent_pos)
                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                obs, reward, done, _, _ = env.step(current_action)
                next_state_idx = agent.get_state_index(obs)

                # Complete experience
                current_exp[2] = next_state_idx
                current_exp[3] = reward
                current_exp[4] = done

                # ============== EGOCENTRIC VISION MODEL ==============
                # Update the agent's true_reward_map based on current observation
                agent_position = tuple(env.unwrapped.agent_pos)
                agent_dir = env.unwrapped.agent_dir

                # 1. CREATE EGOCENTRIC VIEW INPUT
                egocentric_input = create_egocentric_view(env, agent_position, agent_dir, VIEW_SIZE, EMPTY_SPACE, REWARD, OUT_OF_BOUNDS, WALL, done=done)

                # 2. GET AE PREDICTION
                # input_tensor = torch.tensor(egocentric_input[np.newaxis, ..., np.newaxis], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                
                # with torch.no_grad():
                #     predicted_ego_view = ae_model(input_tensor)
                #     predicted_ego_view_2d = predicted_ego_view.squeeze().cpu().numpy()

                # 2. GET PERFECT GROUND TRUTH INSTEAD OF AE PREDICTION
                # Create perfect prediction based on actual environment
                predicted_ego_view_2d = np.zeros((VIEW_SIZE, VIEW_SIZE))

                for ego_y in range(VIEW_SIZE):
                    for ego_x in range(VIEW_SIZE):
                        global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, agent_position[0], agent_position[1], agent_dir, VIEW_SIZE)
                        
                        # Check actual environment for ground truth
                        if 0 <= global_x < env.unwrapped.size and 0 <= global_y < env.unwrapped.size:
                            cell = env.unwrapped.grid.get(global_x, global_y)
                            if cell is not None and cell.type == 'goal':
                                predicted_ego_view_2d[ego_y, ego_x] = 1.0  # Perfect prediction
                            else:
                                predicted_ego_view_2d[ego_y, ego_x] = 0.0
                        else:
                            predicted_ego_view_2d[ego_y, ego_x] = 0.0  # Out of bounds

                # # 3. UPDATE TRUE REWARD MAP WITH PREDICTION
                # # Mark current position with ground truth 
                agent.visited_positions[agent_position[0], agent_position[1]] = True

                # # Learning Signal 
                if done and step < max_steps:
                    agent.true_reward_map[agent_position[0], agent_position[1]] = 1.0 
                else:
                    agent.true_reward_map[agent_position[0], agent_position[1]] = 0.0

                # # Update visible global positions using exact position
                visible_global_positions = get_visible_global_positions(agent_position, agent_dir, VIEW_SIZE, env.unwrapped.size)

                update_true_reward_map_from_egocentric_prediction(agent, predicted_ego_view_2d, visible_global_positions, VIEW_SIZE, agent_position = agent_position, agent_dir=agent_dir, learning_rate=1.0, env=env, episode=episode, step=step)

                # 4. CREATE EGOCENTRIC TARGET FROM TRUE REWARD MAP
                egocentric_target = create_egocentric_target_from_true_map(agent.true_reward_map, agent_position, agent_dir, done, VIEW_SIZE, OUT_OF_BOUNDS)


                # 5. DECIDE WHETHER TO TRAIN AE
                # center_x = VIEW_SIZE // 2
                # agent_ego_x = center_x
                # agent_ego_y = VIEW_SIZE - 1

                # # Commented because testing perfect vision
                # # trigger_ae_training = False
                # # if abs(predicted_ego_view_2d[agent_ego_y, agent_ego_x] - egocentric_target[agent_ego_y, agent_ego_x]) > train_vision_threshold:
                # #     ae_trigger_count_this_episode += 1
                # #     trigger_ae_training = True
                
                # # if trigger_ae_training:
                # #     # Train autoencoder
                # #     target_tensor = torch.tensor(egocentric_target[np.newaxis, ..., np.newaxis], dtype=torch.float32).permute(0, 3, 1, 2).to(device)

                # #     ae_model.train()
                # #     optimizer.zero_grad()
                # #     output = ae_model(input_tensor)
                # #     loss = loss_fn(output, target_tensor)
                # #     loss.backward()
                # #     optimizer.step()

                # 6. UPDATE WVF 
                agent.reward_maps.fill(0)

                # Use binary thresholding like reference
                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        reward = agent.true_reward_map[y, x]
                        idx = y * agent.grid_size + x
                        reward_threshold = 0.5
                        if reward > reward_threshold:
                            agent.reward_maps[idx, y, x] = 1
                        else:
                            agent.reward_maps[idx, y, x] = 0

                # # FIX FROM CLAUDE THAT PROBABLY WONT WORK -  Update ALL reward maps with the discovered rewards
                for idx in range(agent.state_size):
                    for y in range(agent.grid_size):
                        for x in range(agent.grid_size):
                            if agent.true_reward_map[y, x] > 0.5:
                                # ALL hypothetical goal locations should know about this reward
                                agent.reward_maps[idx, y, x] = 1.0
                            else:
                                agent.reward_maps[idx, y, x] = 0.0

                # Instead of binary thresholding:
                # for y in range(agent.grid_size):
                #     for x in range(agent.grid_size):
                #         reward = agent.true_reward_map[y, x]
                #         idx = y * agent.grid_size + x
                #         # Use the actual predicted values, not binary
                #         agent.reward_maps[idx, y, x] = reward

                # # Get the current environment grid
                # grid = env.unwrapped.grid.encode()
                # normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)  # Shape: (H, W)

                # # Setting up input for the AE to obtain it's prediction of the space
                # object_layer = grid[..., 0]
                # normalized_grid[object_layer == 2] = 0.0  # Wall
                # normalized_grid[object_layer == 1] = 0.0  # Open space
                # normalized_grid[object_layer == 8] = 1.0  # Reward (e.g. goal object)

                # # Reshape for the autoencoder (add batch and channel dims)
                # input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

                # # Get the predicted reward map from the AE
                # with torch.no_grad():
                #     ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)
                #     predicted_reward_map_tensor = ae_model(ae_input_tensor)  # (1, 1, H, W)
                #     predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()  # (H, W)

                # # Mark position as visited
                # agent.visited_positions[agent_position[0], agent_position[1]] = True

                # # Learning Signal
                # if done and step < max_steps:
                #     agent.true_reward_map[agent_position[0], agent_position[1]] = 1
                # else:
                #     agent.true_reward_map[agent_position[0], agent_position[1]] = 0

                # # Update the rest of the true_reward_map with AE predictions
                # for y in range(agent.true_reward_map.shape[0]):
                #     for x in range(agent.true_reward_map.shape[1]):
                #         if not agent.visited_positions[y, x]:
                #             predicted_value = predicted_reward_map_2d[y, x]
                #             if predicted_value > 0.001:
                #                 agent.true_reward_map[y, x] = predicted_value
                #             else:
                #                 agent.true_reward_map[y, x] = 0

                # # Train the vision model
                # trigger_ae_training = False
                # train_vision_threshold = 0.1
                # if (abs(predicted_reward_map_2d[agent_position[0], agent_position[1]]- agent.true_reward_map[agent_position[0], agent_position[1]])> train_vision_threshold):
                #     trigger_ae_training = True

                # if trigger_ae_training:
                #     target_tensor = torch.tensor(agent.true_reward_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                #     target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)

                #     ae_model.train()
                #     optimizer.zero_grad()
                #     output = ae_model(ae_input_tensor)
                #     loss = loss_fn(output, target_tensor)
                #     loss.backward()
                #     optimizer.step()
                    
                #     step_loss = loss.item()


                # agent.reward_maps.fill(0)  # Reset all maps to zero

                # for y in range(agent.grid_size):
                #     for x in range(agent.grid_size):
                #         curr_reward = agent.true_reward_map[y, x]
                #         idx = y * agent.grid_size + x
                #         reward_threshold = 0.5
                #         if curr_reward > reward_threshold:
                #             agent.reward_maps[idx, y, x] = 1
                #         else:
                #             agent.reward_maps[idx, y, x] = 0

                # Update agent WVF
                M_flat = np.mean(agent.M, axis=0)
                R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
                V_all = M_flat @ R_flat_all.T
                agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

                # Choose next action
                if step == 0 or episode < 1:  # Warmup period
                    next_action = agent.sample_random_action(obs, epsilon=epsilon)
                else:
                    next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

                next_exp = [next_state_idx, next_action, None, None, None]

                # Update agent
                agent.update(current_exp, None if done else next_exp)


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
    
    def analyze_results(self, window=100):
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

# HELPER FUNCTIONS
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
            global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, int(agent_x), int(agent_y), agent_dir, view_size)
            
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
            if ego_x == center_x and ego_y == agent_ego_y:
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
                                                    visible_positions, view_size, agent_position, agent_dir,
                                                    learning_rate=0.2, env=None, episode=None, step=None):
    """
    Update true_reward_map using predictions from egocentric view
    Only updates cells that are currently visible and not previously visited
    
    Added debugging plots to visualize ground truth vs true reward map
    """
    for global_x, global_y, ego_x, ego_y in visible_positions:
        predicted_value = predicted_ego_view[ego_y, ego_x]
        
        # For perfect vision, just set it directly
        if predicted_value > 0.5:  # There's a goal here
            agent.true_reward_map[global_y, global_x] = 1.0
        # Don't overwrite with 0 unless you're sure there's no goal
        elif agent.visited_positions[global_x, global_y]:
            # Only set to 0 if we've been there and know there's no goal
            agent.true_reward_map[global_y, global_x] = 0.0
            
    # ============== DEBUGGING PLOTS FOR EVERY STEP ==============
    # if env is not None and episode is not None and step is not None:
    #     import matplotlib.pyplot as plt
    #     from utils.plotting import generate_save_path
        
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
    #     # === PLOT 1: GROUND TRUTH ENVIRONMENT ===
    #     env_grid = np.zeros((env.unwrapped.size, env.unwrapped.size), dtype=float)
        
    #     # Build ground truth environment grid
    #     for x in range(env.unwrapped.size):
    #         for y in range(env.unwrapped.size):
    #             cell = env.unwrapped.grid.get(x, y)
    #             if cell is None:
    #                 env_grid[y, x] = 0.0  # Empty space
    #             elif cell.type == 'wall':
    #                 env_grid[y, x] = 0.5  # Wall
    #             elif cell.type == 'goal':
    #                 env_grid[y, x] = 1.0  # Goal/Reward
    #             else:
    #                 env_grid[y, x] = 0.2  # Other objects
        
        # # Plot ground truth environment
        # im1 = axes[0].imshow(env_grid, cmap='viridis', vmin=0, vmax=1)
        # axes[0].set_title(f'Ground Truth Environment\nEpisode {episode}, Step {step}')
        
        # # Mark agent position and direction on ground truth
        # agent_x, agent_y = int(agent_position[0]), int(agent_position[1])
        # axes[0].plot(agent_x, agent_y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
        
        # # Draw arrow showing agent direction
        # dx, dy = 0, 0
        # arrow_length = 0.4
        # if agent_dir == 0:  # right
        #     dx = arrow_length
        # elif agent_dir == 1:  # down
        #     dy = arrow_length
        # elif agent_dir == 2:  # left
        #     dx = -arrow_length
        # elif agent_dir == 3:  # up
        #     dy = -arrow_length
        
        # axes[0].arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.1, 
        #              fc='red', ec='white', linewidth=2)
        
        # # Add grid and labels for ground truth
        # axes[0].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
        # axes[0].set_xticks(range(env.unwrapped.size))
        # axes[0].set_yticks(range(env.unwrapped.size))
        # axes[0].set_xlabel('X')
        # axes[0].set_ylabel('Y')
        
        # # Add colorbar with labels for ground truth
        # cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        # cbar1.set_label('Object Type')
        
        # # === PLOT 2: TRUE REWARD MAP ===
        # im2 = axes[1].imshow(agent.true_reward_map, cmap='hot', vmin=0, vmax=1)
        # axes[1].set_title(f'True Reward Map (Agent\'s Belief)\nEpisode {episode}, Step {step}')
        
        # # Mark agent position and direction on true reward map
        # axes[1].plot(agent_x, agent_y, 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
        # axes[1].arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.1, 
        #              fc='cyan', ec='white', linewidth=2)
        
        # # Highlight visited positions with small markers
        # visited_x, visited_y = np.where(agent.visited_positions)
        # axes[1].scatter(visited_x, visited_y, c='lime', s=20, marker='s', alpha=0.7, 
        #                edgecolors='black', linewidths=0.5, label='Visited')
        
        # # Highlight currently visible positions
        # visible_x = [pos[0] for pos in visible_positions]
        # visible_y = [pos[1] for pos in visible_positions]
        # axes[1].scatter(visible_x, visible_y, c='yellow', s=30, marker='o', alpha=0.8,
        #                edgecolors='black', linewidths=1, label='Currently Visible')
        
        # # Add grid and labels for true reward map
        # axes[1].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
        # axes[1].set_xticks(range(env.unwrapped.size))
        # axes[1].set_yticks(range(env.unwrapped.size))
        # axes[1].set_xlabel('X')
        # axes[1].set_ylabel('Y')
        # axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # # Add colorbar for true reward map
        # cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        # cbar2.set_label('Reward Belief')
        
        # # Add text annotations with key information
        # direction_names = ['Right', 'Down', 'Left', 'Up']
        # info_text = f"Agent Dir: {direction_names[agent.estimated_dir]}\n"
        # info_text += f"Position: ({agent_x}, {agent_y})\n"
        # info_text += f"Visited Positions: {np.sum(agent.visited_positions)}\n"
        # info_text += f"Learning Rate: {learning_rate}"
        
        # fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
        #                                                       facecolor="lightgray", alpha=0.8))
        
        # plt.tight_layout()
        
        # # Save the plot
        # save_path = generate_save_path(f"debug_reward_maps/episode_{episode}_step_{step:03d}.png")
        # plt.savefig(save_path, dpi=100, bbox_inches='tight')
        # plt.close()
        
        # # Also save a summary every 10 steps showing reward retention
        # if step % 10 == 0:
        #     # Create a difference plot to show how much the true reward map differs from ground truth
        #     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
        #     # Create ground truth reward map (1 where goals are, 0 elsewhere)
        #     gt_reward_map = np.zeros_like(agent.true_reward_map)
        #     for x in range(env.unwrapped.size):
        #         for y in range(env.unwrapped.size):
        #             cell = env.unwrapped.grid.get(x, y)
        #             if cell is not None and cell.type == 'goal':
        #                 gt_reward_map[y, x] = 1.0
            
        #     # Calculate difference (how well agent remembers rewards)
        #     difference = np.abs(agent.true_reward_map - gt_reward_map)
            
        #     im = ax.imshow(difference, cmap='Reds', vmin=0, vmax=1)
        #     ax.set_title(f'Reward Memory Error\nEpisode {episode}, Step {step}\n(Red = Forgot Reward, Dark = Correct)')
            
        #     # Mark agent
        #     ax.plot(agent_x, agent_y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
            
        #     # Add grid
        #     ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
        #     ax.set_xticks(range(env.unwrapped.size))
        #     ax.set_yticks(range(env.unwrapped.size))
            
        #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Absolute Error')
        #     plt.tight_layout()
            
        #     save_path = generate_save_path(f"reward_memory_error/episode_{episode}_step_{step:03d}.png")
        #     plt.savefig(save_path, dpi=100, bbox_inches='tight')
        #     plt.close()

def create_egocentric_target_from_true_map(true_reward_map, agent_pos, agent_dir, done, view_size, oob_val=0.0):
    """
    Create egocentric target from the current true_reward_map
    This serves as the training target for the autoencoder
    """
    egocentric_target = np.full((view_size, view_size), oob_val)
    agent_x, agent_y = agent_pos
    env_size = true_reward_map.shape[0]  # Assuming square grid
    
    for ego_y in range(view_size):
        for ego_x in range(view_size):
            global_x, global_y = egocentric_to_global_coords(ego_x, ego_y, int(agent_x), int(agent_y), agent_dir, view_size)
            
            if 0 <= global_x < env_size and 0 <= global_y < env_size:
                egocentric_target[ego_y, ego_x] = true_reward_map[global_y, global_x]
            # else: keep out_of_bounds value
            center_x = view_size // 2
            agent_ego_x = center_x
            agent_ego_y = view_size - 1

            # Learning Signal
            if done and ego_y == agent_ego_y and ego_x == agent_ego_x:
                egocentric_target[ego_y, ego_x] = 1
            
            if not done and ego_y == agent_ego_y and ego_x == agent_ego_x:
                egocentric_target[ego_y, ego_x] = 0
    
    return egocentric_target

# =========================FUNCTIONS FOR PLOTTING===========================

def save_egocentric_view_visualization(input_view, prediction, target, episode, step, agent_pos=None, agent_dir=None, env_size=None):
    """
    Save visualization showing how egocentric views map to global coordinates.
    Shows the full environment grid with egocentric values overlaid where visible.
    """
    if agent_pos is None or agent_dir is None or env_size is None:
        print("Warning: Need agent_pos, agent_dir, and env_size for global visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    view_size = input_view.shape[0]
    
    for ax, mat, title in zip(axes, [input_view, prediction, target],
                              ["Input View", "AE Prediction", "Target"]):
        
        # Create empty global grid
        global_grid = np.full((env_size, env_size), np.nan)  # NaN for unseen areas
        
        # Map egocentric view to global coordinates
        for ego_y in range(view_size):
            for ego_x in range(view_size):
                # Convert egocentric to global coordinates
                global_x, global_y = egocentric_to_global_coords(
                    ego_x, ego_y, int(agent_pos[0]), int(agent_pos[1]), 
                    agent_dir, view_size
                )
                
                # If within bounds, place the value
                if 0 <= global_x < env_size and 0 <= global_y < env_size:
                    global_grid[global_y, global_x] = mat[ego_y, ego_x]
        
        # Plot the global grid
        im = ax.imshow(global_grid, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title} (Global View)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Mark agent position and direction
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        ax.plot(agent_x, agent_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Draw arrow showing agent direction
        dx, dy = 0, 0
        if agent_dir == 0:  # right
            dx = 0.4
        elif agent_dir == 1:  # down
            dy = 0.4
        elif agent_dir == 2:  # left
            dx = -0.4
        elif agent_dir == 3:  # up
            dy = -0.4
        
        ax.arrow(agent_x, agent_y, dx, dy, head_width=0.2, head_length=0.15, 
                fc='red', ec='white', linewidth=1)
        
        # Draw grid lines
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(env_size))
        ax.set_yticks(range(env_size))
        
        # Draw the viewing cone/box to show what area is visible
        # Calculate corners of the egocentric view in global coordinates
        corners = []
        for (ego_x, ego_y) in [(0, 0), (view_size-1, 0), 
                               (view_size-1, view_size-1), (0, view_size-1)]:
            global_x, global_y = egocentric_to_global_coords(
                ego_x, ego_y, int(agent_pos[0]), int(agent_pos[1]), 
                agent_dir, view_size
            )
            if 0 <= global_x < env_size and 0 <= global_y < env_size:
                corners.append((global_x, global_y))
        
        # Draw lines connecting the corners to show field of view
        if len(corners) >= 2:
            for i in range(len(corners)):
                x1, y1 = corners[i]
                x2, y2 = corners[(i+1) % len(corners)]
                ax.plot([x1, x2], [y1, y2], 'w--', alpha=0.5, linewidth=1)
        
        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Episode {episode}, Step {step} - Agent at ({agent_x}, {agent_y}) facing {['right', 'down', 'left', 'up'][agent_dir]}")
    plt.tight_layout()
    plt.savefig(generate_save_path(f"egocentric_views/episode_{episode}_step_{step}.png"))
    plt.close()

def main():
    """Run the experiment comparison"""
    print("Starting baseline comparison experiment...")

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=1000)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()