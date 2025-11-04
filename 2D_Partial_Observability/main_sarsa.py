import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgentPartialSARSA  # Import the new path integration agent
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path, getch
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper
from utils.sr_comparison import SRComparator 

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class ExperimentRunner:
    """Handles running experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}
        self.trajectory_buffer_size = 10 

        # Load optimal SR for comparison
        self.optimal_sr = self.load_optimal_sr()
        self.sr_comparator = SRComparator(self.optimal_sr) if self.optimal_sr is not None else None

    def load_optimal_sr(self):
        """Load the pre-computed optimal SR from datasets/"""
        try:
            optimal_sr_path = 'datasets/optimal_sr_10x10_gamma099.npy'
            if os.path.exists(optimal_sr_path):
                optimal_sr = np.load(optimal_sr_path)
                print(f"Loaded optimal SR from {optimal_sr_path}")
                print(f"Optimal SR shape: {optimal_sr.shape}")
                return optimal_sr
            else:
                print(f"Warning: Optimal SR not found at {optimal_sr_path}")
                print("Run the SR generation script first to create it.")
                return None
        except Exception as e:
            print(f"Error loading optimal SR: {e}")
            return None
    
    def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20, manual = False):
        """Run Master agent experiment with path integration"""
        
        np.random.seed(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)

        agent = SuccessorAgentPartialSARSA(env)

        # Setup torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_shape = (env.size, env.size, 1)
        ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        ae_triggers_per_episode = [] 
        episode_rewards = []
        episode_lengths = []
        epsilon = 1
        epsilon_end = 0.05
        epsilon_decay = 0.9998
        # epsilon_decay = 1

        path_integration_errors = []

        for episode in tqdm(range(episodes), desc=f"Masters Successor (seed {seed})"):
            obs, _ = env.reset()
            obs['image'] = obs['image'].T
            
            # Reset agent for new episode
            agent.reset_path_integration()
            agent.initialize_path_integration(obs)

            trajectory_buffer = deque(maxlen=self.trajectory_buffer_size)
            
            total_reward = 0
            steps = 0
            trajectory = []
            ae_triggers_this_episode = 0
            episode_path_errors = 0

            # Reset maps for new episode 
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

            obs, current_action, manual = execute_turns_until_forward(agent, obs, epsilon, env, manual)
            current_state_idx = agent.get_state_index(obs)
            current_exp = [current_state_idx, current_action, None, None, None]
            
            for step in range(max_steps):
                # Record position and action for trajectory (using path integration)
                agent_pos = agent.internal_pos
                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                # Store step info BEFORE taking action
                # Make the normalized grid for step info
                agent_view = obs['image'][0]

                # Convert to channels last for easier processing
                normalized_grid = np.zeros((7, 7), dtype=np.float32)

                # Setting up input for the AE based on agent's partial view
                normalized_grid[agent_view == 2] = 0.0  # Wall
                normalized_grid[agent_view == 1] = 0.0  # Open space  
                normalized_grid[agent_view == 8] = 1.0 

                # Store step info BEFORE taking action
                step_info = {
                    'agent_view': obs['image'][0].copy(),
                    'agent_pos': tuple(agent.internal_pos),
                    'agent_dir': agent.internal_dir,
                    'normalized_grid': normalized_grid.copy()
                }
                trajectory_buffer.append(step_info)

                old_agent_pos = env.agent_pos

                # Take action in environment
                obs, reward, done, _, _ = env.step(current_action)

                update_sr = True
                if tuple(env.agent_pos) == tuple(old_agent_pos):
                    update_sr = False

                # Update internal state based on action taken
                agent.update_internal_state(current_action)
                
                # next_state_idx = agent.get_state_index(obs)
                obs['image'] = obs['image'].T

                # Get the next forward action (after any necessary turns)
                if not done:
                    obs, next_action, manual = execute_turns_until_forward(agent, obs, epsilon, env, manual)
                    next_state_idx = agent.get_state_index(obs)
                else:
                    # If done, we still need a next state for the update
                    next_state_idx = agent.get_state_index(obs)
                    next_action = current_action  # Doesn't matter since episode is done

                # Complete current experience
                current_exp[2] = next_state_idx
                current_exp[3] = reward
                current_exp[4] = done

                # ============================= ACTION SELECTION =============================
                # if manual:
                #     print(f"Episode {episode}, Step {step}")
                #     key = getch().lower()
                    
                #     if key == 'w':
                #         next_action = 2  # forward
                #     elif key == 'a':
                #         next_action = 0  # turn left
                #     elif key == 'd':
                #         next_action = 1  # turn right
                #     elif key == 's':
                #         next_action = 5  # toggle
                #     elif key == 'q':
                #         manual = False
                #         next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                #     elif key == '\r' or key == '\n':
                #         next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                #     else:
                #         next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                # else:
                #     next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

                # ============================= SR UPDATE =============================
                if update_sr:
                    if done:
                        # Terminal state - update without next experience
                        agent.update(current_exp, next_exp=None)
                    else:
                        # Non-terminal - create next_exp and update
                        next_exp = [next_state_idx, next_action, None, None, None]
                        agent.update(current_exp, next_exp)

                # ============================= VISION MODEL ====================================
                
                # Get current agent position (using path integration)
                agent_position = agent.internal_pos

                # Get the agent's 7x7 view from observation
                agent_view = obs['image'][0]

                # Convert to channels last for easier processing
                normalized_grid = np.zeros((7, 7), dtype=np.float32)

                # Setting up input for the AE based on agent's partial view
                normalized_grid[agent_view == 2] = 0.0  # Wall
                normalized_grid[agent_view == 1] = 0.0  # Open space  
                normalized_grid[agent_view == 8] = 1.0  # Goal

                # If agent is on goal, force the agent's position in view to show reward
                if done:
                    normalized_grid[6, 3] = 1.0  # Agent position in egocentric view

                # Reshape for the autoencoder (add batch and channel dims)
                input_grid = normalized_grid[np.newaxis, ..., np.newaxis] 

                with torch.no_grad():
                    ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                    predicted_reward_map_tensor = ae_model(ae_input_tensor)
                    predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()
                    predicted_reward_map_2d = np.clip(predicted_reward_map_2d, 0.0, 1.0) 

                # Mark position as visited (using path integration)
                agent.visited_positions[agent_position[1], agent_position[0]] = True

                # Learning Signal - Batch training when goal is reached
                if done and step < max_steps:
                    agent.true_reward_map[agent_position[1], agent_position[0]] = 1

                    if len(trajectory_buffer) > 0:
                        batch_inputs = []
                        batch_targets = []
                        
                        # Include all steps from trajectory buffer (past steps)
                        for past_step in trajectory_buffer:
                            reward_global_pos = agent_position
                            
                            past_target_7x7 = self._create_target_view_with_reward(
                                past_step['agent_pos'], 
                                past_step['agent_dir'],
                                reward_global_pos,
                                agent.true_reward_map
                            )
                            
                            batch_inputs.append(past_step['normalized_grid'])
                            batch_targets.append(past_target_7x7)
                        
                        # ALSO include the current step (when agent is on goal)
                        current_target_7x7 = self._create_target_view_with_reward(
                            tuple(agent.internal_pos),
                            agent.internal_dir,
                            agent_position,
                            agent.true_reward_map
                        )
                        
                        batch_inputs.append(normalized_grid)
                        batch_targets.append(current_target_7x7)
                        
                        # Train autoencoder on batch
                        self._train_ae_on_batch(ae_model, optimizer, loss_fn, 
                                            batch_inputs, batch_targets, device)

                # Map the 7x7 predicted reward map to the 10x10 global map
                agent_x, agent_y = agent_position
                ego_center_x = 3
                ego_center_y = 6
                agent_dir = agent.internal_dir
                
                for view_y in range(7):
                    for view_x in range(7):
                        dx_ego = view_x - ego_center_x
                        dy_ego = view_y - ego_center_y
                        
                        if agent_dir == 3:  # North
                            dx_world = dx_ego
                            dy_world = dy_ego
                        elif agent_dir == 0:  # East
                            dx_world = -dy_ego
                            dy_world = dx_ego
                        elif agent_dir == 1:  # South
                            dx_world = -dx_ego
                            dy_world = -dy_ego
                        elif agent_dir == 2:  # West
                            dx_world = dy_ego
                            dy_world = -dx_ego
                        
                        global_x = agent_x + dx_world
                        global_y = agent_y + dy_world
                        
                        if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
                            if not agent.visited_positions[global_y, global_x]:
                                predicted_value = predicted_reward_map_2d[view_y, view_x]
                                predicted_value = np.clip(predicted_value, 0.0, 1.0)
                                agent.true_reward_map[global_y, global_x] = predicted_value

                # Extract the 7x7 target from the true reward map
                target_7x7 = np.zeros((7, 7), dtype=np.float32)
                
                for view_y in range(7):
                    for view_x in range(7):
                        dx_ego = view_x - ego_center_x
                        dy_ego = view_y - ego_center_y
                        
                        if agent_dir == 3:
                            dx_world = dx_ego
                            dy_world = dy_ego
                        elif agent_dir == 0:
                            dx_world = -dy_ego
                            dy_world = dx_ego
                        elif agent_dir == 1:
                            dx_world = -dx_ego
                            dy_world = -dy_ego
                        elif agent_dir == 2:
                            dx_world = dy_ego
                            dy_world = -dx_ego
                        
                        global_x = agent_x + dx_world
                        global_y = agent_y + dy_world
                        
                        if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
                            target_7x7[view_y, view_x] = agent.true_reward_map[global_y, global_x]
                        else:
                            target_7x7[view_y, view_x] = 0.0

                # Check for prediction errors and trigger training if needed
                trigger_ae_training = False
                view_error = np.abs(predicted_reward_map_2d - target_7x7)
                max_error = np.max(view_error)
                mean_error = np.mean(view_error)

                if max_error > 0.05 or mean_error > 0.01:
                    trigger_ae_training = True

                if trigger_ae_training:
                    ae_triggers_this_episode += 1 
                    target_tensor = torch.tensor(target_7x7[np.newaxis, ..., np.newaxis], dtype=torch.float32)
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

                # Update agent WVF
                MOVE_FORWARD = 2
                M_forward = agent.M[MOVE_FORWARD, :, :]
                R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
                V_all = M_forward @ R_flat_all.T
                agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

                # Update counters
                total_reward += reward
                steps += 1

                # ============================= EPISODE END CHECK =============================
                if done:
                    break
                else:
                    # Move to next transition (forward action only)
                    current_exp = next_exp
                    current_action = next_action

            # ============================= POST-EPISODE PROCESSING =============================
            ae_triggers_per_episode.append(ae_triggers_this_episode)
            path_integration_errors.append(episode_path_errors)
            
            # Create ground truth reward space
            ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)

            if hasattr(env, 'goal_pos'):
                goal_x, goal_y = env.goal_pos
                ground_truth_reward_space[goal_y, goal_x] = 1.0
            elif hasattr(env, '_goal_pos'):
                goal_x, goal_y = env._goal_pos
                ground_truth_reward_space[goal_y, goal_x] = 1.0
            else:
                if hasattr(env, 'grid'):
                    for y in range(env.size):
                        for x in range(env.size):
                            cell = env.grid.get(x, y)
                            if cell is not None and hasattr(cell, 'type') and cell.type == 'goal':
                                ground_truth_reward_space[y, x] = 1.0
            

            # Generate visualizations occasionally
            if episode % 500 == 0:
                save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))

                MOVE_FORWARD = 2
                # Saving the Move Forward SR
                forward_M = agent.M[MOVE_FORWARD, :, :]

                # ============================= SR COMPARISON =============================
                if self.sr_comparator is not None:
                    print("Attempting SR comparison...")
                    try:
                        # Compare with optimal SR
                        metrics = self.sr_comparator.compare(forward_M, episode)
                        
                        if metrics:
                            print(f"\nSR Comparison Metrics (Episode {episode}):")
                            for key, value in metrics.items():
                                print(f"  {key}: {value:.6f}")
                        else:
                            print("Warning: metrics returned None!")
                        
                        # Visualize comparison
                        print("Creating SR comparison visualization...")
                        self.sr_comparator.visualize_comparison(forward_M, episode)
                        print("SR comparison visualization completed!")
                        
                    except Exception as e:
                        print(f"ERROR in SR comparison: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("WARNING: sr_comparator is None - SR comparison skipped!")
                
                print(f"{'='*60}\n")
                plt.figure(figsize=(6, 5))
                im = plt.imshow(forward_M, cmap='hot')
                plt.title(f"Forward SR Matrix (Episode {episode})")
                plt.colorbar(im, label="SR Value")
                plt.tight_layout()
                plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
                plt.close()

                # Create vision plots
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                all_values = np.concatenate([
                    predicted_reward_map_2d.flatten(),
                    target_7x7.flatten(),
                    agent.true_reward_map.flatten(),
                    ground_truth_reward_space.flatten()
                ])
                vmin = np.min(all_values)
                vmax = np.max(all_values)

                im1 = ax1.imshow(predicted_reward_map_2d, cmap='viridis', vmin=vmin, vmax=vmax)
                ax1.set_title(f'Predicted 7x7 View - Ep{episode} Step{step}')
                ax1.plot(3, 6, 'ro', markersize=8, label='Agent')
                plt.colorbar(im1, ax=ax1, fraction=0.046)

                im2 = ax2.imshow(target_7x7, cmap='viridis', vmin=vmin, vmax=vmax)
                ax2.set_title(f'Target 7x7 View (Ground Truth)')
                ax2.plot(3, 6, 'ro', markersize=8, label='Agent')
                plt.colorbar(im2, ax=ax2, fraction=0.046)

                im3 = ax3.imshow(agent.true_reward_map, cmap='viridis', vmin=vmin, vmax=vmax)
                ax3.set_title(f'True 10x10 Map - Agent at ({agent_x},{agent_y})')
                ax3.plot(agent_x, agent_y, 'ro', markersize=8, label='Agent')
                plt.colorbar(im3, ax=ax3, fraction=0.046)

                im4 = ax4.imshow(ground_truth_reward_space, cmap='viridis', vmin=vmin, vmax=vmax)
                ax4.set_title('Ground Truth Reward Space')
                plt.colorbar(im4, ax=ax4, fraction=0.046)

                plt.tight_layout()
                plt.savefig(generate_save_path(f"vision_plots/maps_ep{episode}_step{step}.png"), dpi=150, bbox_inches='tight')
                plt.close()

                # Plot path integration accuracy
                if episode > 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(path_integration_errors, alpha=0.7, label='Path integration errors per episode')
                    plt.xlabel('Episode')
                    plt.ylabel('Number of Position/Direction Errors')
                    plt.title(f'Path Integration Accuracy (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'path_integration/errors_up_to_ep_{episode}.png'))
                    plt.close()

                # Plot AE triggers over episodes
                plt.figure(figsize=(10, 5))
                
                window_size = 50
                if len(ae_triggers_per_episode) >= window_size:
                    smoothed_triggers = np.convolve(ae_triggers_per_episode, 
                                                np.ones(window_size)/window_size, 
                                                mode='valid')
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
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        # Print final path integration statistics
        total_errors = sum(path_integration_errors)
        print(f"\nPath Integration Summary for seed {seed}:")
        print(f"Total position/direction errors: {total_errors}")
        print(f"Episodes with errors: {sum(1 for x in path_integration_errors if x > 0)}")
        print(f"Average errors per episode: {total_errors / episodes:.4f}")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Masters Successor w/ Path Integration",
            "path_integration_errors": path_integration_errors,
        }
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual = False):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")

            # Run Masters successor with path integration
            successor_results = self.run_successor_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual = manual)
            
            # Store results
            algorithms = ['Masters Successor w/ Path Integration']
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
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Added one more subplot for path integration

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

        # Plot 3: Path Integration Accuracy
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            if "path_integration_errors" in runs[0]:  # Check if this agent has path integration data
                all_errors = np.array([run["path_integration_errors"] for run in runs])
                mean_errors = np.mean(all_errors, axis=0)
                std_errors = np.std(all_errors, axis=0)

                mean_smooth = pd.Series(mean_errors).rolling(window).mean()
                std_smooth = pd.Series(std_errors).rolling(window).mean()

                x = range(len(mean_smooth))
                ax3.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
                ax3.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
                )

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Path Integration Errors")
        ax3.set_title("Path Integration Accuracy")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Final performance comparison (last 100 episodes)
        ax4 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
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
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            
            # Add path integration statistics if available
            total_path_errors = 0
            if "path_integration_errors" in runs[0]:
                total_path_errors = np.sum([np.sum(run["path_integration_errors"]) for run in runs])

            summary_data.append({
                "Algorithm": alg_name,
                "Final Performance": f"{final_performance:.3f}",
                "Convergence Episode": convergence_episode,
                "Total Path Errors": total_path_errors,
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

        # Plot 6: Path integration error distribution
        ax6 = axes[1, 2]
        for alg_name, runs in self.results.items():
            if "path_integration_errors" in runs[0]:
                all_errors_flat = []
                for run in runs:
                    all_errors_flat.extend(run["path_integration_errors"])
                
                # Create histogram of error counts
                unique_errors, counts = np.unique(all_errors_flat, return_counts=True)
                ax6.bar(unique_errors, counts, alpha=0.7, label=alg_name)

        ax6.set_xlabel("Errors per Episode")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Path Integration Error Distribution")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = generate_save_path("experiment_comparison_with_path_integration.png")
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
        results_file = generate_save_path(f"experiment_results_path_integration_{timestamp}.json")

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
                # Add path integration errors if available
                if "path_integration_errors" in run:
                    json_run["path_integration_errors"] = [int(e) for e in run["path_integration_errors"]]
                
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")

    def _create_target_view_with_reward(self, past_agent_pos, past_agent_dir, reward_pos, reward_map):
        """Create 7x7 target view from past agent position showing reward location"""
        target_7x7 = np.zeros((7, 7), dtype=np.float32)
        
        ego_center_x, ego_center_y = 3, 6  # Agent position in 7x7 view
        past_x, past_y = past_agent_pos
        reward_x, reward_y = reward_pos
        
        for view_y in range(7):
            for view_x in range(7):
                # Calculate offset from agent's position in ego view
                dx_ego = view_x - ego_center_x
                dy_ego = view_y - ego_center_y
                
                # Rotate based on past agent direction
                if past_agent_dir == 3:  # North
                    dx_world, dy_world = dx_ego, dy_ego
                elif past_agent_dir == 0:  # East
                    dx_world, dy_world = -dy_ego, dx_ego
                elif past_agent_dir == 1:  # South
                    dx_world, dy_world = -dx_ego, -dy_ego
                elif past_agent_dir == 2:  # West
                    dx_world, dy_world = dy_ego, -dx_ego
                
                # Calculate global coordinates for this view cell
                global_x = past_x + dx_world
                global_y = past_y + dy_world
                
                # Check if this cell contains the reward
                if (global_x == reward_x and global_y == reward_y):
                    target_7x7[view_y, view_x] = 1.0
                else:
                    target_7x7[view_y, view_x] = 0.0
        
        return target_7x7

    def _train_ae_on_batch(self, model, optimizer, loss_fn, inputs, targets, device):
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

def execute_turns_until_forward(agent, obs, epsilon, env, manual=False):
    """
    Keep executing turn actions until a forward action is selected.
    Returns the final observation after all turns and the forward action.
    """
    while True:

        action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        
        # If it's a forward, we're done turning
        if action == 2:  # forward or toggle
            return obs, action, manual
        
        # Execute the turn action
        obs, _, _, _, _ = env.step(action)
        obs['image'] = obs['image'].T
        
        # Update agent's internal state for the turn
        agent.update_internal_state(action)

def main():
    """Run the experiment comparison with path integration"""
    print("Starting baseline comparison experiment with path integration...")

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=10000, max_steps=200, manual = False)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()