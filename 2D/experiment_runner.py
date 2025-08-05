import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgent, ImprovedVisionOnlyAgent, VisionDQNAgent
from models import build_autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


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

    # def run_qlearning_experiment(self, episodes=5000, max_steps=200, seed=20):
    #     """Run Q-learning baseline experiment"""
    #     np.random.seed(seed)

    #     # avoid circular imports
    #     from agents import QLearningAgent

    #     env = SimpleEnv(size=self.env_size)
    #     agent = QLearningAgent(env)

    #     episode_rewards = []
    #     episode_lengths = []

    #     for episode in tqdm(range(episodes), desc=f"Q-Learning (seed {seed})"):
    #         obs = env.reset()
    #         total_reward = 0
    #         steps = 0

    #         state_idx = agent.get_state_index(obs)

    #         for step in range(max_steps):
    #             action = agent.choose_action(state_idx)
    #             obs, reward, done, _, _ = env.step(action)
    #             next_state_idx = agent.get_state_index(obs)

    #             # Update Q-table
    #             agent.update(state_idx, action, reward, next_state_idx, done)

    #             total_reward += reward
    #             steps += 1
    #             state_idx = next_state_idx

    #             if done:
    #                 break

    #         agent.decay_epsilon()
    #         episode_rewards.append(total_reward)
    #         episode_lengths.append(steps)

    #     return {
    #         "rewards": episode_rewards,
    #         "lengths": episode_lengths,
    #         "final_epsilon": agent.epsilon,
    #         "algorithm": "Q-Learning",
    #     }

    def run_improved_vision_dqn_experiment(self, episodes=5000, max_steps=200, seed=20):
        """
        Run experiment with improved Vision-based DQN agent
        """
        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force CPU usage to avoid GPU memory issues
        device = torch.device("cpu")
        
        env = None
        agent = None
        
        try:
            # Create environment
            env = SimpleEnv(size=self.env_size)
            
            # Create improved agent
            agent = VisionDQNAgent(
                env, 
                learning_rate=0.0005,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.9995,
                memory_size=50000,
                batch_size=32,
                target_update_freq=1000,
                learning_starts=1000,
                train_freq=4
            )
            
            # Force agent to use CPU
            agent.device = device
            agent.q_network = agent.q_network.to(device)
            agent.target_network = agent.target_network.to(device)

            episode_rewards = []
            episode_lengths = []
            training_stats = []

            for episode in tqdm(range(episodes), desc=f"Improved Vision DQN (seed {seed})"):
                try:
                    obs = env.reset()
                    agent.reset_episode()
                    total_reward = 0
                    steps = 0
                    trajectory = []

                    # Get initial state using improved vision
                    current_state = agent.get_vision_state(obs)

                    for step in range(max_steps):
                        # Record position and action for trajectory
                        agent_pos = tuple(env.agent_pos)
                        
                        # Choose action based on improved vision
                        action = agent.get_action(current_state)
                        trajectory.append((agent_pos[0], agent_pos[1], action))
                        
                        # Take action
                        obs, reward, done, _, _ = env.step(action)
                        next_state = agent.get_vision_state(obs)
                        
                        # Store experience
                        agent.remember(current_state, action, reward, next_state, done)
                        
                        # Train the network
                        loss, avg_q = agent.replay()
                        
                        # Step the agent
                        agent.step()
                        
                        total_reward += reward
                        steps += 1
                        current_state = next_state
                        
                        if done:
                            break

                    # Check for failure in last 100 episodes and save trajectory plot
                    if episode >= episodes - 100 and not done:
                        self.plot_and_save_trajectory("Improved Vision DQN", episode, trajectory, env.size, seed)

                    # Decay epsilon
                    agent.decay_epsilon()
                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    
                    # Collect training statistics
                    if episode % 100 == 0:
                        stats = agent.get_stats()
                        training_stats.append({
                            'episode': episode,
                            **stats
                        })
                        
                        print(f"Episode {episode}: "
                            f"Reward={total_reward:.2f}, "
                            f"Steps={steps}, "
                            f"Epsilon={stats['epsilon']:.3f}, "
                            f"Avg Loss={stats['avg_loss']:.4f}")
                    
                    # Periodic cleanup
                    if episode % 500 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    continue

            return {
                "rewards": episode_rewards,
                "lengths": episode_lengths,
                "final_epsilon": agent.epsilon,
                "algorithm": "Improved Vision DQN",
                "training_stats": training_stats
            }

        except Exception as e:
            print(f"Critical error in Improved DQN experiment: {e}")
            import traceback
            traceback.print_exc()
            return {
                "rewards": [],
                "lengths": [],
                "final_epsilon": 0.0,
                "algorithm": "Improved Vision DQN",
                "error": str(e)
            }
            
        finally:
            # Explicit cleanup
            if agent is not None:
                del agent.q_network
                del agent.target_network
                del agent.memory
                del agent
            
            if env is not None:
                del env
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # def run_honours_successor_experiment(self, episodes=5000, max_steps=200, seed=20):
    #     """Run the honours agent experiment, where the perfect reward map is given"""
    #     np.random.seed(seed)

    #     env = SimpleEnv(size=self.env_size)
    #     agent = SuccessorAgent(env)

    #     episode_rewards = []
    #     episode_lengths = []
    #     epsilon = 1
    #     epsilon_end = 0.05
    #     epsilon_decay = 0.9995

    #     for episode in tqdm(range(episodes), desc=f"Successor Agent (seed {seed})"):
    #         obs = env.reset()
    #         total_reward = 0
    #         steps = 0
    #         trajectory = []  # Track trajectory for failure analysis

    #         # Reset for new episode (from your code)
    #         agent.true_reward_map = np.zeros((env.size, env.size))
    #         agent.wvf = np.zeros(
    #             (agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32
    #         )
    #         agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

    #         current_state_idx = agent.get_state_index(obs)
    #         current_action = agent.sample_random_action(obs, epsilon=epsilon)
    #         current_exp = [current_state_idx, current_action, None, None, None]

    #         for step in range(max_steps):
    #             # Record position and action for trajectory
    #             agent_pos = tuple(env.agent_pos)
    #             trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
    #             obs, reward, done, _, _ = env.step(current_action)
    #             next_state_idx = agent.get_state_index(obs)

    #             # Complete experience
    #             current_exp[2] = next_state_idx
    #             current_exp[3] = reward
    #             current_exp[4] = done

    #             # Choose next action
    #             if step == 0 or episode < 1:  # Warmup period
    #                 next_action = agent.sample_random_action(obs, epsilon=epsilon)
    #             else:
    #                 next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

    #             next_exp = [next_state_idx, next_action, None, None, None]

    #             # Update agent
    #             agent.update(current_exp, None if done else next_exp)

    #             # Vision Model
    #             # Update the agent's true_reward_map based on current observation
    #             agent_position = tuple(env.agent_pos)

    #             # Get the current environment grids reward map
    #             grid = env.grid.encode()
    #             object_layer = grid[..., 0]
    #             predicted_reward_map_2d = grid[..., 0]
    #             predicted_reward_map_2d[object_layer == 2] = 0.0   
    #             predicted_reward_map_2d[object_layer == 1] = 0.0   
    #             predicted_reward_map_2d[object_layer == 8] = 1.0

    #             # Mark position as visited
    #             agent.visited_positions[agent_position[0], agent_position[1]] = True

    #             agent.reward_maps.fill(0)  # Reset all maps to zero
    #             for y in range(agent.grid_size):
    #                 for x in range(agent.grid_size):
    #                     curr_reward = predicted_reward_map_2d[y, x]
    #                     idx = y * agent.grid_size + x
    #                     reward_threshold = 0.5
    #                     if curr_reward > reward_threshold:
    #                         # changed from = reward to 1
    #                         agent.reward_maps[idx, y, x] = 1
    #                     else:
    #                         agent.reward_maps[idx, y, x] = 0

    #             M_flat = np.mean(agent.M, axis=0)
    #             R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
    #             V_all = M_flat @ R_flat_all.T
    #             agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

    #             total_reward += reward
    #             steps += 1
    #             current_exp = next_exp
    #             current_action = next_action

    #             if done:
    #                 break

    #         # Check for failure in last 100 episodes and save trajectory plot
    #         if episode >= episodes - 100 and not done:
    #             self.plot_and_save_trajectory("Honours Successor", episode, trajectory, env.size, seed)

    #         epsilon = max(epsilon_end, epsilon * epsilon_decay)
    #         episode_rewards.append(total_reward)
    #         episode_lengths.append(steps)

    #     return {
    #         "rewards": episode_rewards,
    #         "lengths": episode_lengths,
    #         "final_epsilon": epsilon,
    #         "algorithm": "Honours Successor",
    #     }
    
    # def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20):
    #     """Run Master agent experiment"""
    #     np.random.seed(seed)

    #     env = SimpleEnv(size=self.env_size)
    #     agent = SuccessorAgent(env)

    #     # Setup vision model
    #     input_shape = (env.size, env.size, 1)
    #     ae_model = build_autoencoder(input_shape)
    #     ae_model.compile(optimizer="adam", loss="mse")

    #     episode_rewards = []
    #     episode_lengths = []
    #     epsilon = 1
    #     epsilon_end = 0.05
    #     epsilon_decay = 0.9995

    #     for episode in tqdm(range(episodes), desc=f"Masters Successor (seed {seed})"):
    #         obs = env.reset()
    #         total_reward = 0
    #         steps = 0
    #         trajectory = []  # Track trajectory for failure analysis

    #         # Reset for new episode (from your code)
    #         agent.true_reward_map = np.zeros((env.size, env.size))
    #         agent.wvf = np.zeros(
    #             (agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32
    #         )
    #         agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

    #         current_state_idx = agent.get_state_index(obs)
    #         current_action = agent.sample_random_action(obs, epsilon=epsilon)
    #         current_exp = [current_state_idx, current_action, None, None, None]

    #         for step in range(max_steps):
    #             # Record position and action for trajectory
    #             agent_pos = tuple(env.agent_pos)
    #             trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
    #             obs, reward, done, _, _ = env.step(current_action)
    #             next_state_idx = agent.get_state_index(obs)

    #             # Complete experience
    #             current_exp[2] = next_state_idx
    #             current_exp[3] = reward
    #             current_exp[4] = done

    #             # Choose next action
    #             if step == 0 or episode < 1:  # Warmup period
    #                 next_action = agent.sample_random_action(obs, epsilon=epsilon)
    #             else:
    #                 next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

    #             next_exp = [next_state_idx, next_action, None, None, None]

    #             # Update agent
    #             agent.update(current_exp, None if done else next_exp)

    #             # Vision Model
    #             # Update the agent's true_reward_map based on current observation
    #             agent_position = tuple(env.agent_pos)

    #             # Get the current environment grid
    #             grid = env.grid.encode()
    #             normalized_grid = np.zeros_like(
    #                 grid[..., 0], dtype=np.float32
    #             )  # Shape: (H, W)

    #             # Setting up input for the AE to obtain it's prediction of the space
    #             object_layer = grid[..., 0]
    #             normalized_grid[object_layer == 2] = 0.0  # Wall
    #             normalized_grid[object_layer == 1] = 0.0  # Open space
    #             normalized_grid[object_layer == 8] = 1.0  # Reward (e.g. goal object)

    #             # Reshape for the autoencoder (add batch and channel dims)
    #             input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    #             # Get the predicted reward map from the AE
    #             predicted_reward_map = ae_model.predict(input_grid, verbose=0)
    #             predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]

    #             # Mark position as visited
    #             agent.visited_positions[agent_position[0], agent_position[1]] = True

    #             # Learning Signal
    #             if done and step < max_steps:
    #                 agent.true_reward_map[agent_position[0], agent_position[1]] = 1
    #             else:
    #                 agent.true_reward_map[agent_position[0], agent_position[1]] = 0

    #             # Update the rest of the true_reward_map with AE predictions
    #             for y in range(agent.true_reward_map.shape[0]):
    #                 for x in range(agent.true_reward_map.shape[1]):
    #                     if not agent.visited_positions[y, x]:
    #                         predicted_value = predicted_reward_map_2d[y, x]
    #                         if predicted_value > 0.001:
    #                             agent.true_reward_map[y, x] = predicted_value
    #                         else:
    #                             agent.true_reward_map[y, x] = 0

    #             # Train the vision model
    #             trigger_ae_training = False
    #             train_vision_threshold = 0.1
    #             if (abs(predicted_reward_map_2d[agent_position[0], agent_position[1]]- agent.true_reward_map[agent_position[0], agent_position[1]])> train_vision_threshold):
    #                 trigger_ae_training = True

    #             if trigger_ae_training:

    #                 target = agent.true_reward_map[np.newaxis, ..., np.newaxis]

    #                 # Train the model for a single step
    #                 history = ae_model.fit(
    #                     input_grid,  # Input: current environment grid
    #                     target,  # Target: agent's true_reward_map
    #                     epochs=1,  # Just one training step
    #                     batch_size=1,  # Single sample
    #                     verbose=0,  # Suppress output for cleaner logs
    #                 )
    #                 step_loss = history.history["loss"][0]

    #             agent.reward_maps.fill(0)  # Reset all maps to zero

    #             for y in range(agent.grid_size):
    #                 for x in range(agent.grid_size):
    #                     curr_reward = agent.true_reward_map[y, x]
    #                     idx = y * agent.grid_size + x
    #                     reward_threshold = 0.5
    #                     if curr_reward > reward_threshold:
    #                         agent.reward_maps[idx, y, x] = 1
    #                     else:
    #                         agent.reward_maps[idx, y, x] = 0

    #             # Update agent WVF
    #             M_flat = np.mean(agent.M, axis=0)
    #             R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
    #             V_all = M_flat @ R_flat_all.T
    #             agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

    #             total_reward += reward
    #             steps += 1
    #             current_exp = next_exp
    #             current_action = next_action

    #             if done:
    #                 break

    #         # Check for failure in last 100 episodes and save trajectory plot
    #         if episode >= episodes - 100 and not done:
    #             self.plot_and_save_trajectory("Masters Successor", episode, trajectory, env.size, seed)

    #         epsilon = max(epsilon_end, epsilon * epsilon_decay)
    #         episode_rewards.append(total_reward)
    #         episode_lengths.append(steps)

    #     return {
    #         "rewards": episode_rewards,
    #         "lengths": episode_lengths,
    #         "final_epsilon": epsilon,
    #         "algorithm": "Masters Successor",
    #     }
    
    # def run_sarsa_sr_experiment(self, episodes=5000, max_steps=200, seed=20):
    #     """Run SARSA SR baseline experiment"""
    #     np.random.seed(seed)
        
    #     # avoid circular imports
    #     from agents import SARSASRAgent
        
    #     env = SimpleEnv(size=self.env_size)
    #     agent = SARSASRAgent(env)
        
    #     episode_rewards = []
    #     episode_lengths = []
        
    #     for episode in tqdm(range(episodes), desc=f"SARSA SR (seed {seed})"):
    #         obs = env.reset()
    #         agent.reset_episode()
    #         total_reward = 0
    #         steps = 0
            
    #         state_idx = agent.get_state_index(obs)
    #         action = agent.choose_action(state_idx)
            
    #         for step in range(max_steps):
    #             obs, reward, done, _, _ = env.step(action)
    #             next_state_idx = agent.get_state_index(obs)
                
    #             if done:
    #                 # Terminal state update
    #                 agent.update(state_idx, action, reward, next_state_idx, 0, done)
    #                 total_reward += reward
    #                 steps += 1
    #                 break
    #             else:
    #                 # Choose next action for SARSA update
    #                 next_action = agent.choose_action(next_state_idx)
                    
    #                 # SARSA update with actual next action
    #                 agent.update(state_idx, action, reward, next_state_idx, next_action, done)
                    
    #                 # Move to next state-action pair
    #                 state_idx = next_state_idx
    #                 action = next_action
                
    #             total_reward += reward
    #             steps += 1
            
    #         agent.decay_epsilon()
    #         episode_rewards.append(total_reward)
    #         episode_lengths.append(steps)
        
    #     return {
    #         'rewards': episode_rewards,
    #         'lengths': episode_lengths,
    #         'final_epsilon': agent.epsilon,
    #         'algorithm': 'SARSA SR'
    #     }

    # def run_vision_only_experiment(self, episodes=5000, max_steps=200, seed=20):
    #     """Run Vision-Only baseline experiment"""
    #     np.random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)

    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     env = SimpleEnv(size=self.env_size)
    #     agent = ImprovedVisionOnlyAgent(env)
      
    #     # Setup autoencoder
    #     input_shape = (1, env.size, env.size) 
    #     ae_model = build_autoencoder(input_shape).to(device)
   
    #     optimizer = optim.Adam(ae_model.parameters(), lr=0.001),
    #     loss_fn = nn.MSELoss()

    #     ae_model.compile(
    #         optimizer = optimizer,
    #         loss_fn = loss_fn
    #     )
        
    #     episode_rewards = []
    #     episode_lengths = []
    #     epsilon = 1.0
    #     epsilon_end = 0.05
    #     epsilon_decay = 0.995
    #     train_every = 10
        
    #     for episode in tqdm(range(episodes), desc=f"Vision-Only Agent (seed {seed})"):
    #         obs = env.reset()
    #         total_reward = 0
    #         steps = 0
            
    #         prev_pos = tuple(env.agent_pos)
            
    #         for step in range(max_steps):
    #             # Choose action
    #             if episode < 50:  # Extended warm-up with random actions
    #                 action = env.action_space.sample()
    #             else:
    #                 action = agent.sample_action_from_values(obs, epsilon=epsilon)
                
    #             # Take action
    #             obs, reward, done, _, _ = env.step(action)
    #             next_pos = tuple(env.agent_pos)
                
    #             # Update value map
    #             agent.update_value_map(prev_pos, action, reward, next_pos, done)
                
    #             # Train autoencoder periodically
    #             if step % train_every == 0 or done:
    #                 input_data, target_data = agent.prepare_training_data()

    #                 if input_data is not None:
    #                     # Convert to torch tensors
    #                     input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    #                     target_tensor = torch.tensor(target_data, dtype=torch.float32).to(device)

    #                     ae_model.train()
    #                     optimizer.zero_grad()
    #                     output = ae_model(input_tensor)
    #                     loss = loss_fn(output, target_tensor)
    #                     loss.backward()
    #                     optimizer.step()

    #                     # Prediction for agent
    #                     ae_model.eval()
    #                     with torch.no_grad():
    #                         pred = ae_model(input_tensor)
    #                         pred = pred.detach().cpu().numpy()
    #                         agent.predicted_value_map = pred[0, 0]  # shape: (1, 1, H, W)
                
    #             total_reward += reward
    #             steps += 1
    #             prev_pos = next_pos
                
    #             if done:
    #                 break
            
    #         # Decay epsilon
    #         epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
    #         # Store statistics
    #         episode_rewards.append(total_reward)
    #         episode_lengths.append(steps)
        
    #     return {
    #         "rewards": episode_rewards,
    #         "lengths": episode_lengths,
    #         "final_epsilon": epsilon,
    #         "algorithm": "Vision-Only",
    #     }

    def run_comparison_experiment(self, episodes=5000):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")
            
            # Run Q-learning
            # qlearning_results = self.run_qlearning_experiment(episodes=episodes, seed=seed)
            
            # Run DQN
            dqn_results = self.run_vision_dqn_experiment(episodes=episodes, seed=seed)
            
            # Run SARSA SR
            # sarsa_sr_results = self.run_sarsa_sr_experiment(episodes=episodes, seed=seed)

            # Run Honours successor
            # honours_results = self.run_honours_successor_experiment(episodes=episodes, seed=seed)
            
            # Run Masters successor
            # successor_results = self.run_successor_experiment(episodes=episodes, seed=seed)
            
            # Run Vision-Only agent
            # vision_results = self.run_vision_only_experiment(episodes=episodes, seed=seed)
            
            # Store results
            # algorithms = ['Q-Learning', 'DQN', 'SARSA SR', 'Masters Successor', 'Honours Successor', 'Vision-Only']
            algorithms = ['DQN']  # For now, only DQN is run
            # results_list = [qlearning_results, dqn_results, sarsa_sr_results, successor_results, honours_results, vision_results]
            results_list = [dqn_results]  # For now, only DQN is run
            
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
            final_performance = np.mean(
                [np.mean(run["rewards"][-100:]) for run in runs]
            )
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


def main():
    """Run the experiment comparison"""
    print("Starting baseline comparison experiment...")

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=10001)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()