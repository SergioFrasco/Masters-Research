import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgentPartial, DQNAgentPartial, LSTM_DQN_Agent  # Import the new path integration agent
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

    def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20, manual = False):
            """Run Master agent experiment with path integration"""
            
            np.random.seed(seed)

            if manual:
                print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
                env = SimpleEnv(size=self.env_size, render_mode='human')
            else:
                env = SimpleEnv(size=self.env_size)

            agent = SuccessorAgentPartial(env)  # Use path integration agent

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
            epsilon_decay = 0.9995

            # Track path integration accuracy
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

                current_state_idx = agent.get_state_index(obs)
                current_action = agent.sample_random_action(obs, epsilon=epsilon)
                current_exp = [current_state_idx, current_action, None, None, None]
                
                for step in range(max_steps):
                    # Record position and action for trajectory (using path integration)
                    agent_pos = agent.internal_pos
                    trajectory.append((agent_pos[0], agent_pos[1], current_action))
                    
                    # Make the normalized grid for step  info
                    agent_view = obs['image'][0]  

                    # Convert to channels last for easier processing
                    normalized_grid = np.zeros((7, 7), dtype=np.float32)

                    # Setting up input for the AE based on agent's partial view
                    normalized_grid[agent_view == 2] = 0.0  # Wall
                    normalized_grid[agent_view == 1] = 0.0  # Open space  
                    normalized_grid[agent_view == 8] = 1.0 

                    step_info = {
                        'agent_view': obs['image'][0].copy(),  # 7x7 view
                        'agent_pos': tuple(agent.internal_pos),
                        'agent_dir': agent.internal_dir,
                        'normalized_grid': normalized_grid.copy()  # The AE input
                    }
                    trajectory_buffer.append(step_info)

                    # Take action in environment
                    obs, reward, done, _, _ = env.step(current_action)
                    
                    # Update internal state based on action taken
                    agent.update_internal_state(current_action)
                    
                    # Verify path integration accuracy 
                    # if episode % 100 == 0:  # Check every 100 episodes
                    #     is_accurate, error_msg = agent.verify_path_integration(obs)
                    #     if not is_accurate:
                    #         episode_path_errors += 1
                    #         if episode_path_errors == 1:  # Print first error of episode
                    #             print(f"Episode {episode}, Step {step}: {error_msg}")

                    next_state_idx = agent.get_state_index(obs)
                    obs['image'] = obs['image'].T

                    # Complete experience
                    current_exp[2] = next_state_idx
                    current_exp[3] = reward
                    current_exp[4] = done

                    if manual:
                        # env.render()
                        print(f"Episode {episode}, Step {step}")
                        # print("W=forward, A=turn left, D=turn right, S=toggle, Q=quit manual, ENTER=auto")
                        
                        key = getch().lower()
                        
                        if key == 'w':
                            next_action = 2  # forward
                        elif key == 'a':
                            next_action = 0  # turn left
                        elif key == 'd':
                            next_action = 1  # turn right
                        elif key == 's':
                            next_action = 5  # toggle
                        elif key == 'q':
                            manual = False
                            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                        elif key == '\r' or key == '\n':  # Enter key
                            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                        else:
                            # Any other key = auto action
                            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                        
                    # Sample actions with WVF
                    else:
                            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

                    next_exp = [next_state_idx, next_action, None, None, None]

                    # Update agent - always pass next_exp since we're not terminating early
                    agent.update(current_exp, next_exp)

                    # ============================= Vision Model ====================================
                
                    # Update the agent's true_reward_map based on current observation
                    agent_position = agent.internal_pos  # Use path integration position

                    # Get the agent's 7x7 view from observation (already processed)
                    agent_view = obs['image'][0]  

                    # Convert to channels last for easier processing
                    normalized_grid = np.zeros((7, 7), dtype=np.float32)

                    # Setting up input for the AE based on agent's partial view
                    normalized_grid[agent_view == 2] = 0.0  # Wall
                    normalized_grid[agent_view == 1] = 0.0  # Open space  
                    normalized_grid[agent_view == 8] = 1.0 

                    # Reshape for the autoencoder (add batch and channel dims)
                    input_grid = normalized_grid[np.newaxis, ..., np.newaxis] 

                    with torch.no_grad():
                        ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                        predicted_reward_map_tensor = ae_model(ae_input_tensor)
                        predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()

                    # Mark position as visited (using path integration)
                    agent.visited_positions[agent_position[1], agent_position[0]] = True  # y,x

                    # Learning Signal
                    if done and step < max_steps:
                        agent.true_reward_map[agent_position[1], agent_position[0]] = 1

                        # Added: Logic for training on past 10 steps (BATCH) when goal is reached

                        if len(trajectory_buffer) > 1:  # Need at least 2 steps
                            batch_inputs = []
                            batch_targets = []
                            
                            for past_step in trajectory_buffer:
                                # Get the reward location in global coordinates
                                reward_global_pos = agent_position  # Current reward location
                                
                                # Create target 7x7 for this past step
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
                                tuple(agent.internal_pos),  # Current position (on goal)
                                agent.internal_dir,         # Current direction
                                agent_position,             # Reward position (same as current position)
                                agent.true_reward_map
                            )
                            
                            batch_inputs.append(normalized_grid)  # Current step's input
                            batch_targets.append(current_target_7x7)  # Current step's target
                            
                            # Train autoencoder on batch
                            self._train_ae_on_batch(ae_model, optimizer, loss_fn, 
                                                batch_inputs, batch_targets, device)

                    else:
                        agent.true_reward_map[agent_position[1], agent_position[0]] = 0

                    # Map the 7x7 predicted reward map to the 10x10 global map
                    agent_x, agent_y = agent_position
                    ego_center_x = 3  # Center column
                    ego_center_y = 6  # Bottom row
                    agent_dir = agent.internal_dir  # Use path integration direction
                    
                    for view_y in range(7):
                        for view_x in range(7):
                            # Calculate offset from agent's position in ego view
                            dx_ego = view_x - ego_center_x  # -3 to 3
                            dy_ego = view_y - ego_center_y  # -6 to 0
                            
                            # Rotate the offset based on agent's direction to get world offsets
                            if agent_dir == 3:  # Facing up (north)
                                dx_world = dx_ego
                                dy_world = dy_ego
                            elif agent_dir == 0:  # Facing right (east)
                                dx_world = -dy_ego
                                dy_world = dx_ego
                            elif agent_dir == 1:  # Facing down (south)
                                dx_world = -dx_ego
                                dy_world = -dy_ego
                            elif agent_dir == 2:  # Facing left (west)
                                dx_world = dy_ego
                                dy_world = -dx_ego
                            
                            # Calculate global coordinates
                            global_x = agent_x + dx_world
                            global_y = agent_y + dy_world
                            
                            # Check if the global coordinates are within the map bounds
                            if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
                                # Only update if position hasn't been visited (to preserve ground truth)
                                if not agent.visited_positions[global_y, global_x]:
                                    predicted_value = predicted_reward_map_2d[view_y, view_x]
                                    agent.true_reward_map[global_y, global_x] = predicted_value
    
                    # Extract the 7x7 target from the true reward map corresponding to agent's view
                    target_7x7 = np.zeros((7, 7), dtype=np.float32)
                    
                    agent_x, agent_y = agent_position
                    agent_dir = agent.internal_dir  # Use path integration direction
                    
                    for view_y in range(7):
                        for view_x in range(7):
                            # Calculate offset from agent's position in ego view
                            dx_ego = view_x - ego_center_x  # -3 to 3
                            dy_ego = view_y - ego_center_y  # -6 to 0
                            
                            # Rotate the offset based on agent's direction to get world offsets
                            if agent_dir == 3:  # Facing up (north)
                                dx_world = dx_ego
                                dy_world = dy_ego
                            elif agent_dir == 0:  # Facing right (east)
                                dx_world = -dy_ego
                                dy_world = dx_ego
                            elif agent_dir == 1:  # Facing down (south)
                                dx_world = -dx_ego
                                dy_world = -dy_ego
                            elif agent_dir == 2:  # Facing left (west)
                                dx_world = dy_ego
                                dy_world = -dx_ego
                            
                            # Calculate global coordinates
                            global_x = agent_x + dx_world
                            global_y = agent_y + dy_world
                            
                            # Extract value from true reward map if within bounds
                            if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
                                target_7x7[view_y, view_x] = agent.true_reward_map[global_y, global_x]
                            else:
                                # Out of bounds positions get 0
                                target_7x7[view_y, view_x] = 0.0

                    trigger_ae_training = False
                    # Check if ANY position in the view has significant error
                    view_error = np.abs(predicted_reward_map_2d - target_7x7)
                    max_error = np.max(view_error)
                    mean_error = np.mean(view_error)

                    if max_error > 0.05 or mean_error > 0.01:
                        trigger_ae_training = True

                    if trigger_ae_training:
                        # print("Training Triggered!")
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
                

                    agent.reward_maps.fill(0)  # Reset all maps to zero

                    for y in range(agent.grid_size):
                        for x in range(agent.grid_size):
                            curr_reward = agent.true_reward_map[y, x]
                            idx = y * agent.grid_size + x
                            if agent.true_reward_map[y, x] >= 0.5: #Threshold
                                agent.reward_maps[idx, y, x] = curr_reward

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

                ae_triggers_per_episode.append(ae_triggers_this_episode)
                path_integration_errors.append(episode_path_errors)
                
                #  Create ground truth reward space based on environment for plotting
                ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)

                # Get the actual goal position from the environment
                # The goal position should be available from the environment
                if hasattr(env, 'goal_pos'):
                    goal_x, goal_y = env.goal_pos
                    ground_truth_reward_space[goal_y, goal_x] = 1.0
                elif hasattr(env, '_goal_pos'):
                    goal_x, goal_y = env._goal_pos
                    ground_truth_reward_space[goal_y, goal_x] = 1.0
                else:
                    # If goal position is not directly accessible, extract from observation
                    # Look for the goal object (usually object type 8) in the full grid
                    if hasattr(env, 'grid'):
                        for y in range(env.size):
                            for x in range(env.size):
                                cell = env.grid.get(x, y)
                                if cell is not None and hasattr(cell, 'type') and cell.type == 'goal':
                                    ground_truth_reward_space[y, x] = 1.0
                                    

                # Generate visualizations occasionally
                if episode % 500 == 0:
                    save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))

                    # Saving the SR
                    averaged_M = np.mean(agent.M, axis=0)

                    # Create a figure
                    plt.figure(figsize=(6, 5))
                    im = plt.imshow(averaged_M, cmap='hot')
                    plt.title(f"Averaged SR Matrix (Episode {episode})")
                    plt.colorbar(im, label="SR Value")
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
                    plt.close()

                    # Create vision plots
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                    # Calculate global vmin and vmax across all arrays
                    all_values = np.concatenate([
                        predicted_reward_map_2d.flatten(),
                        target_7x7.flatten(),
                        agent.true_reward_map.flatten(),
                        ground_truth_reward_space.flatten()
                    ])
                    vmin = np.min(all_values)
                    vmax = np.max(all_values)

                    # Plot predicted 7x7 view (top-left)
                    im1 = ax1.imshow(predicted_reward_map_2d, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax1.set_title(f'Predicted 7x7 View - Ep{episode} Step{step}')
                    ax1.plot(3, 6, 'ro', markersize=8, label='Agent')
                    plt.colorbar(im1, ax=ax1, fraction=0.046)

                    # Plot target 7x7 view (top-right)
                    im2 = ax2.imshow(target_7x7, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax2.set_title(f'Target 7x7 View (Ground Truth)')
                    ax2.plot(3, 6, 'ro', markersize=8, label='Agent')
                    plt.colorbar(im2, ax=ax2, fraction=0.046)

                    # Plot true 10x10 reward map (bottom-left)
                    im3 = ax3.imshow(agent.true_reward_map, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax3.set_title(f'True 10x10 Map - Agent at ({agent_x},{agent_y})')
                    ax3.plot(agent_x, agent_y, 'ro', markersize=8, label='Agent')
                    plt.colorbar(im3, ax=ax3, fraction=0.046)

                    # Plot ground truth reward space (bottom-right)
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
    
    def run_dqn_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run DQN agent experiment with path integration and vision"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)

        # Initialize DQN agent
        agent = DQNAgentPartial(env, 
                               learning_rate=0.001,
                               gamma=0.99,
                               epsilon_start=1.0,
                               epsilon_end=0.05,
                               epsilon_decay=0.9995,
                               memory_size=10000,
                               batch_size=32,
                               target_update_freq=100)

        # Setup vision model (autoencoder)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_shape = (7, 7, 1)  # 7x7 partial view
        ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Tracking variables
        ae_triggers_per_episode = [] 
        episode_rewards = []
        episode_lengths = []
        path_integration_errors = []
        dqn_losses = []

        for episode in tqdm(range(episodes), desc=f"DQN Partial Observable (seed {seed})"):
            obs = env.reset()
            
            
            # Reset agent for new episode
            agent.reset_path_integration()
            agent.initialize_path_integration(obs)
            
            total_reward = 0
            steps = 0
            trajectory = []
            ae_triggers_this_episode = 0
            episode_path_errors = 0
            episode_dqn_losses = []

            # Reset maps for new episode 
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

            # Get initial state for DQN
            # current_obs = obs.copy()
            current_obs = obs
            current_state = agent.get_dqn_state(current_obs)

            # Transpose image for processing
            if 'image' in obs:
                obs['image'] = obs['image'].T

            
            for step in range(max_steps):
                # Record position and action for trajectory (using path integration)
                agent_pos = agent.internal_pos
                
                # Select action using DQN
                if manual:
                    print(f"Episode {episode}, Step {step}")
                    key = getch().lower()
                    
                    if key == 'w':
                        current_action = 2  # forward
                    elif key == 'a':
                        current_action = 0  # turn left
                    elif key == 'd':
                        current_action = 1  # turn right
                    elif key == 's':
                        current_action = 5  # toggle
                    elif key == 'q':
                        manual = False
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                    elif key == '\r' or key == '\n':  # Enter key
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                    else:
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)

                else:
                    current_action = agent.select_action_dqn(current_obs, agent.epsilon)

                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                # Take action in environment
                obs, reward, done, _, _ = env.step(current_action)

                # Transpose image for processing
                if 'image' in obs:
                    obs['image'] = obs['image'].T
                
                # Update internal state based on action taken
                agent.update_internal_state(current_action)
                
                # Verify path integration accuracy periodically
                if episode % 500 == 0:
                    is_accurate, error_msg = agent.verify_path_integration(obs)
                    if not is_accurate:
                        episode_path_errors += 1
                        if episode_path_errors == 1:
                            print(f"Episode {episode}, Step {step}: {error_msg}")

                # Get next state for DQN
                next_obs = obs.copy()
                next_state = agent.get_dqn_state(next_obs)


                # === DQN Training (NEW) ===
                # Store experience in DQN memory
                agent.remember(current_state, current_action, reward, next_state, done)
                
                # Train DQN if enough experiences
                if len(agent.memory) >= agent.batch_size:
                    dqn_loss = agent.train_dqn()
                    episode_dqn_losses.append(dqn_loss)

                # Update for next iteration
                total_reward += reward
                steps += 1
                current_obs = next_obs
                current_state = next_state

                if done:
                    break

            # Decay epsilon
            agent.decay_epsilon()
            
            # Record episode statistics
            ae_triggers_per_episode.append(ae_triggers_this_episode)
            path_integration_errors.append(episode_path_errors)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_dqn_losses:
                dqn_losses.append(np.mean(episode_dqn_losses))
            else:
                dqn_losses.append(0.0)

            # Generate visualizations occasionally
            if episode % 500 == 0:
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
                                    break

                plt.tight_layout()
                plt.savefig(generate_save_path(f"dqn_vision_plots/maps_ep{episode}.png"), dpi=150, bbox_inches='tight')
                plt.close()

                # DQN loss plot
                if len(dqn_losses) > 10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(dqn_losses, alpha=0.7, label='DQN Loss')
                    if len(dqn_losses) >= 50:
                        smoothed_loss = np.convolve(dqn_losses, np.ones(50)/50, mode='valid')
                        plt.plot(range(25, len(dqn_losses) - 24), smoothed_loss, color='red', linewidth=2, label='Smoothed Loss')
                    plt.xlabel('Episode')
                    plt.ylabel('Mean DQN Loss')
                    plt.title(f'DQN Training Loss (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'dqn_loss/loss_up_to_ep_{episode}.png'))
                    plt.close()

                # Path integration accuracy
                if episode > 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(path_integration_errors, alpha=0.7, label='Path integration errors per episode')
                    plt.xlabel('Episode')
                    plt.ylabel('Number of Position/Direction Errors')
                    plt.title(f'Path Integration Accuracy (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'dqn_path_integration/errors_up_to_ep_{episode}.png'))
                    plt.close()

        # Print final statistics
        total_errors = sum(path_integration_errors)
        print(f"\nDQN Path Integration Summary for seed {seed}:")
        print(f"Total position/direction errors: {total_errors}")
        print(f"Episodes with errors: {sum(1 for x in path_integration_errors if x > 0)}")
        print(f"Average errors per episode: {total_errors / episodes:.4f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average DQN loss (final 100 episodes): {np.mean(dqn_losses[-100:]):.6f}")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "DQN with Path Integration & Vision",
            "path_integration_errors": path_integration_errors,
            "dqn_losses": dqn_losses,
            "ae_triggers": ae_triggers_per_episode,
        }
    
    def run_lstm_dqn_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run LSTM-DQN agent experiment"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)

        # Initialize LSTM-DQN agent
        agent = LSTM_DQN_Agent(
            env,
            sequence_length=16,
            frame_stack_k=4,
            lstm_hidden_dim=128, 
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=5000,
            batch_size=8,
            target_update_freq=500
        )

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        lstm_losses = []

        for episode in tqdm(range(episodes), desc=f"LSTM-DQN (seed {seed})"):
            # Reset environment
            obs, _ = env.reset()
            obs["image"] = obs['image'].T
            
            # Reset episode state in agent (frame stack and hidden state)
            agent.reset_episode(obs)
            
            total_reward = 0
            steps = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Update frame stack ONCE per step
                # Extract and push the current frame to the stack
                frame = agent._extract_frame(obs)
                agent.frame_stack.push(frame)
                
                # Now get the stacked state 
                stacked = agent.frame_stack.get_stack()
                stacked = np.array(stacked, dtype=np.float32)
                current_state = torch.FloatTensor(stacked).to(agent.device) / 10.0
                
                # Select action (uses current frame stack, doesn't modify it)
                if manual:
                    print(f"Episode {episode}, Step {step}")
                    key = getch().lower()
                    
                    if key == 'w':
                        action = 2  # forward
                    elif key == 'a':
                        action = 0  # turn left
                    elif key == 'd':
                        action = 1  # turn right
                    elif key == 's':
                        action = 5  # toggle
                    elif key == 'q':
                        manual = False
                        action = agent.select_action(obs)
                    elif key == '\r' or key == '\n':  # Enter key
                        action = agent.select_action(obs)
                    else:
                        action = agent.select_action(obs)
                else:
                    action = agent.select_action(obs)
                
                # Take action in environment
                next_obs, reward, done, _, _ = env.step(action)
                next_obs["image"] = next_obs['image'].T
                
                # Update frame stack with new observation and get next state
                next_frame = agent._extract_frame(next_obs)
                agent.frame_stack.push(next_frame)
                next_stacked = agent.frame_stack.get_stack()
                next_stacked = np.array(next_stacked, dtype=np.float32)
                next_state = torch.FloatTensor(next_stacked).to(agent.device) / 10.0
                
                # Store transition in episode buffer
                agent.store_transition(current_state, action, reward, next_state, done)
                
                # Update counters
                total_reward += reward
                steps += 1
                
                # Move to next observation
                obs = next_obs
                
                if done:
                    break
            
            # Train after episode ends
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                episode_losses.append(loss)
                lstm_losses.append(loss)
            else:
                lstm_losses.append(0.0)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Logging
            if episode % 500 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                avg_loss = np.mean(lstm_losses[-100:]) if len(lstm_losses) >= 100 else np.mean(lstm_losses)
                
                print(f"\nEpisode {episode}")
                print(f"  Avg Reward (last 100): {avg_reward:.3f}")
                print(f"  Avg Length (last 100): {avg_length:.1f}")
                print(f"  Avg Loss (last 100): {avg_loss:.6f}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Replay buffer size: {len(agent.memory)} sequences")
            
            # Visualizations (keeping your existing visualization code)
            if episode % 500 == 0 and episode > 0:
                # Loss plot
                if len(lstm_losses) > 10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(lstm_losses, alpha=0.7, label='LSTM-DQN Loss')
                    if len(lstm_losses) >= 50:
                        smoothed_loss = np.convolve(lstm_losses, np.ones(50)/50, mode='valid')
                        plt.plot(range(25, len(lstm_losses) - 24), smoothed_loss, 
                                color='red', linewidth=2, label='Smoothed Loss')
                    plt.xlabel('Episode')
                    plt.ylabel('Mean Loss')
                    plt.title(f'LSTM-DQN Training Loss (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'lstm_dqn_loss/loss_up_to_ep_{episode}.png'))
                    plt.close()
                
                # Reward plot
                plt.figure(figsize=(10, 5))
                plt.plot(episode_rewards, alpha=0.7)
                if len(episode_rewards) >= 50:
                    smoothed_rewards = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
                    plt.plot(range(25, len(episode_rewards) - 24), smoothed_rewards,
                            color='green', linewidth=2, label='Smoothed')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title(f'LSTM-DQN Learning Curve (up to ep {episode})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(generate_save_path(f'lstm_dqn_rewards/rewards_up_to_ep_{episode}.png'))
                plt.close()
        
        # Print final statistics
        print(f"\nLSTM-DQN Summary for seed {seed}:")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average reward (final 100 episodes): {np.mean(episode_rewards[-100:]):.3f}")
        print(f"Average length (final 100 episodes): {np.mean(episode_lengths[-100:]):.1f}")
        print(f"Average loss (final 100 episodes): {np.mean(lstm_losses[-100:]):.6f}")
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "LSTM-DQN with Frame Stacking",
            "lstm_losses": lstm_losses,
        }
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual = False):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")

            # Run Masters successor with path integration
            successor_results = self.run_successor_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual = manual)

            dqn_results = self.run_dqn_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual = manual)

            dqn_lstm_results = self.run_lstm_dqn_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual = manual)
            
            # Store results
            algorithms = ['Masters Successor', 'DQN', 'LSTM-DQN']
            results_list = [successor_results, dqn_results, dqn_lstm_results]
            
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



def main():
    """Run the experiment comparison with path integration"""
    print("Starting baseline comparison experiment with path integration...")

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=15, num_seeds=2)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=50001, max_steps=100, manual = False)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()