import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import (
    SuccessorAgentPartialSARSA, 
    SuccessorAgentPartialQLearning, 
    DQNAgentPartial, 
    LSTM_DQN_Agent,
    GoalConditionedLSTMDQN
)
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import (
    overlay_values_on_grid, visualize_sr, save_all_reward_maps, 
    save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path, getch
)
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper
from utils.sr_comparison import SRComparator

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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

    # ========================================================================
    # SUCCESSOR REPRESENTATION EXPERIMENTS (UNCHANGED)
    # ========================================================================
    
    def run_successor_experiment_sarsa(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run Master agent experiment with path integration - SARSA version"""
        
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

        path_integration_errors = []

        for episode in tqdm(range(episodes), desc=f"Masters Successor SARSA (seed {seed})"):
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

                # Take action in environment
                obs, reward, done, _, _ = env.step(current_action)
                
                # Update internal state based on action taken
                agent.update_internal_state(current_action)
                
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

                # ============================= SR UPDATE =============================
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
                    
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Masters Successor SARSA w/ Path Integration",
            "path_integration_errors": path_integration_errors,
        }

    def run_successor_experiment_q_learning(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run Master agent experiment with path integration - Q-Learning version"""
        
        np.random.seed(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)

        agent = SuccessorAgentPartialQLearning(env) 

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

        path_integration_errors = []

        for episode in tqdm(range(episodes), desc=f"Masters Successor Q-Learning (seed {seed})"):
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
                agent_pos = agent.internal_pos
                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                agent_view = obs['image'][0]  
                normalized_grid = np.zeros((7, 7), dtype=np.float32)
                normalized_grid[agent_view == 2] = 0.0
                normalized_grid[agent_view == 1] = 0.0  
                normalized_grid[agent_view == 8] = 1.0 

                if step == 0:
                    done = False
                if step > 0:
                    if done:
                        normalized_grid[6, 3] = 1.0

                step_info = {
                    'agent_view': obs['image'][0].copy(),
                    'agent_pos': tuple(agent.internal_pos),
                    'agent_dir': agent.internal_dir,
                    'normalized_grid': normalized_grid.copy()
                }
                trajectory_buffer.append(step_info)

                obs, reward, done, _, _ = env.step(current_action)
                agent.update_internal_state(current_action)

                next_state_idx = agent.get_state_index(obs)
                obs['image'] = obs['image'].T

                current_exp[2] = next_state_idx
                current_exp[3] = reward
                current_exp[4] = done

                next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                next_exp = [next_state_idx, next_action, None, None, None]
                agent.update(current_exp, next_exp)

                agent_position = agent.internal_pos
                agent_view = obs['image'][0]  
                normalized_grid = np.zeros((7, 7), dtype=np.float32)
                normalized_grid[agent_view == 2] = 0.0
                normalized_grid[agent_view == 1] = 0.0  
                normalized_grid[agent_view == 8] = 1.0 

                if done:
                    normalized_grid[6, 3] = 1.0

                input_grid = normalized_grid[np.newaxis, ..., np.newaxis] 

                with torch.no_grad():
                    ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                    predicted_reward_map_tensor = ae_model(ae_input_tensor)
                    predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()

                agent.visited_positions[agent_position[1], agent_position[0]] = True

                if done and step < max_steps:
                    agent.true_reward_map[agent_position[1], agent_position[0]] = 1

                    if len(trajectory_buffer) > 0:
                        batch_inputs = []
                        batch_targets = []
                        
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
                        
                        current_target_7x7 = self._create_target_view_with_reward(
                            tuple(agent.internal_pos),
                            agent.internal_dir,
                            agent_position,
                            agent.true_reward_map
                        )
                        batch_inputs.append(normalized_grid)
                        batch_targets.append(current_target_7x7)
                        
                        self._train_ae_on_batch(ae_model, optimizer, loss_fn, 
                                            batch_inputs, batch_targets, device)

                agent_x, agent_y = agent_position
                ego_center_x = 3
                ego_center_y = 6
                agent_dir = agent.internal_dir
                
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
                            if not agent.visited_positions[global_y, global_x]:
                                predicted_value = predicted_reward_map_2d[view_y, view_x]
                                agent.true_reward_map[global_y, global_x] = predicted_value

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

                agent.reward_maps.fill(0)

                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        curr_reward = agent.true_reward_map[y, x]
                        idx = y * agent.grid_size + x
                        if agent.true_reward_map[y, x] >= 0.5:
                            agent.reward_maps[idx, y, x] = curr_reward

                MOVE_FORWARD = 2
                M_forward = agent.M[MOVE_FORWARD, :, :]
                R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
                V_all = M_forward @ R_flat_all.T
                agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

                total_reward += reward
                steps += 1
                current_exp = next_exp
                current_action = next_action

                if done:
                    break

            ae_triggers_per_episode.append(ae_triggers_this_episode)
            path_integration_errors.append(episode_path_errors)
                    
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Masters Successor Q-Learning w/ Path Integration",
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

        agent = DQNAgentPartial(env, 
                               learning_rate=0.001,
                               gamma=0.99,
                               epsilon_start=1.0,
                               epsilon_end=0.05,
                               epsilon_decay=0.9995,
                               memory_size=10000,
                               batch_size=32,
                               target_update_freq=100)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_shape = (7, 7, 1)
        ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        ae_triggers_per_episode = [] 
        episode_rewards = []
        episode_lengths = []
        path_integration_errors = []
        dqn_losses = []

        for episode in tqdm(range(episodes), desc=f"DQN Partial Observable (seed {seed})"):
            obs = env.reset()
            
            agent.reset_path_integration()
            agent.initialize_path_integration(obs)
            
            total_reward = 0
            steps = 0
            trajectory = []
            ae_triggers_this_episode = 0
            episode_path_errors = 0
            episode_dqn_losses = []

            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

            current_obs = obs
            current_state = agent.get_dqn_state(current_obs)

            if 'image' in obs:
                obs['image'] = obs['image'].T
            
            for step in range(max_steps):
                agent_pos = agent.internal_pos
                
                if manual:
                    print(f"Episode {episode}, Step {step}")
                    key = getch().lower()
                    
                    if key == 'w':
                        current_action = 2
                    elif key == 'a':
                        current_action = 0
                    elif key == 'd':
                        current_action = 1
                    elif key == 's':
                        current_action = 5
                    elif key == 'q':
                        manual = False
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                    elif key == '\r' or key == '\n':
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                    else:
                        current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                else:
                    current_action = agent.select_action_dqn(current_obs, agent.epsilon)

                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                obs, reward, done, _, _ = env.step(current_action)

                if 'image' in obs:
                    obs['image'] = obs['image'].T
                
                agent.update_internal_state(current_action)
                
                if episode % 200 == 0:
                    is_accurate, error_msg = agent.verify_path_integration(obs)
                    if not is_accurate:
                        episode_path_errors += 1
                        if episode_path_errors == 1:
                            print(f"Episode {episode}, Step {step}: {error_msg}")

                next_obs = obs.copy()
                next_state = agent.get_dqn_state(next_obs)

                agent.remember(current_state, current_action, reward, next_state, done)
                
                if len(agent.memory) >= agent.batch_size:
                    dqn_loss = agent.train_dqn()
                    episode_dqn_losses.append(dqn_loss)

                total_reward += reward
                steps += 1
                current_obs = next_obs
                current_state = next_state

                if done:
                    break

            agent.decay_epsilon()
            
            ae_triggers_per_episode.append(ae_triggers_this_episode)
            path_integration_errors.append(episode_path_errors)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_dqn_losses:
                dqn_losses.append(np.mean(episode_dqn_losses))
            else:
                dqn_losses.append(0.0)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "DQN with Path Integration & Vision",
            "path_integration_errors": path_integration_errors,
            "dqn_losses": dqn_losses,
            "ae_triggers": ae_triggers_per_episode,
        }

    # ========================================================================
    # LSTM-DQN EXPERIMENT (UPDATED)
    # ========================================================================
    
    def run_lstm_dqn_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """
        Run LSTM-DQN agent experiment.
        
        CHANGES FROM ORIGINAL:
        - Updated to match document 2 implementation
        - Fixed frame stack update logic (only once per step)
        - Changed loss tracking from 'dqn_losses' to 'lstm_losses'
        - Updated visualization paths
        """
        
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
            learning_rate=0.00005,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=5000,
            batch_size=16,
            target_update_freq=500
        )

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        lstm_losses = []  # CHANGED: was 'dqn_losses' in original

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
                # FIXED: Update frame stack ONCE per step
                frame = agent._extract_frame(obs)
                agent.frame_stack.push(frame)
                
                # Get the stacked state 
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
            
            # Visualizations 
            if episode % 1000 == 0:
                # Loss plot
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
                plt.savefig(generate_save_path(f'lstm_dqn/lstm_dqn_loss/loss_up_to_ep_{episode}.png'))
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
                plt.savefig(generate_save_path(f'lstm_dqn/lstm_dqn_rewards/rewards_up_to_ep_{episode}.png'))
                plt.close()
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "LSTM-DQN with Frame Stacking",
            "lstm_losses": lstm_losses,  # CHANGED: was 'dqn_losses'
        }

    
    
    def run_lstm_wvf_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """
        Run LSTM_WVF_Agent experiment (UVF baseline, view-based).
        
        This agent:
        - Uses VIEW-BASED goal conditioning Q(s, a, g) where g = goal position in 7×7 view
        - LSTM for memory over partial observations
        - NO vision model (no reward predictor)
        - Frame stacking for visual context
        - Sequence-based training
        
        Handles random goal positions:
        - Goals spawn at different (x, y) each episode
        - Agent conditions on RELATIVE position in its 7×7 view
        - Learns: "when goal at (x,y) in view, take action a"
        - This generalizes across episodes naturally!
        
        Parallels the 3D WVF compositional agent but without task composition.
        """
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)
        
        # Import the agent
        from lstm_wvf_agent import LSTM_WVF_Agent
        
        # Initialize agent
        agent = LSTM_WVF_Agent(
            env,
            frame_stack_k=4,
            sequence_length=16,
            lstm_hidden_dim=128,
            learning_rate=0.00005,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=5000,
            batch_size=16,
            target_update_freq=200
        )
        
        # Tracking
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        
        for episode in tqdm(range(episodes), desc=f"LSTM_WVF_Agent (seed {seed})"):
            # Reset environment
            obs, _ = env.reset()
            if isinstance(obs, dict) and 'image' in obs:
                obs['image'] = obs['image'].T
            
            # Reset agent for new episode
            stacked_obs = agent.reset_episode(obs)
            
            total_reward = 0
            steps = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Update frame stack
                frame = agent._extract_frame(obs)
                agent.frame_stack.push(frame)
                stacked_obs = agent.get_stacked_state()
                
                # Select action
                if manual:
                    print(f"Episode {episode}, Step {step}")
                    key = getch().lower()
                    
                    if key == 'w':
                        action = 2  # forward
                    elif key == 'a':
                        action = 0  # turn left
                    elif key == 'd':
                        action = 1  # turn right
                    elif key == 'q':
                        manual = False
                        action = agent.select_action(obs)
                    elif key == '\r' or key == '\n':
                        action = agent.select_action(obs)
                    else:
                        action = agent.select_action(obs)
                else:
                    action = agent.select_action(obs)
                
                # Take action in environment
                next_obs, reward, done, truncated, info = env.step(action)
                
                if isinstance(next_obs, dict) and 'image' in next_obs:
                    next_obs['image'] = next_obs['image'].T
                
                # Update frame stack with next observation
                next_frame = agent._extract_frame(next_obs)
                agent.frame_stack.push(next_frame)
                next_stacked_obs = agent.get_stacked_state()
                
                # Store transition
                agent.store_transition(stacked_obs, action, reward, next_stacked_obs, done)
                
                # Update counters
                total_reward += reward
                steps += 1
                
                # Move to next observation
                obs = next_obs
                stacked_obs = next_stacked_obs
                
                if done or truncated:
                    break
            
            # Process episode for replay
            agent.process_episode()
            
            # Train after episode
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                episode_losses.append(loss)
                training_losses.append(loss)
            else:
                training_losses.append(0.0)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Periodic visualization
            if episode % 1000 == 0 and episode > 0:
                # Loss plot
                plt.figure(figsize=(10, 5))
                plt.plot(training_losses, alpha=0.7, label='Training Loss')
                if len(training_losses) >= 50:
                    smoothed = np.convolve(training_losses, np.ones(50)/50, mode='valid')
                    plt.plot(range(25, len(training_losses) - 24), smoothed,
                            color='red', linewidth=2, label='Smoothed')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.title(f'LSTM_WVF_Agent Loss (up to ep {episode})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(generate_save_path(f'lstm_wvf/loss/loss_ep_{episode}.png'))
                plt.close()
                
                # Reward plot
                plt.figure(figsize=(10, 5))
                plt.plot(episode_rewards, alpha=0.7)
                if len(episode_rewards) >= 50:
                    smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
                    plt.plot(range(25, len(episode_rewards) - 24), smoothed,
                            color='green', linewidth=2, label='Smoothed')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'LSTM_WVF_Agent Learning Curve (up to ep {episode})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(generate_save_path(f'lstm_wvf/rewards/rewards_ep_{episode}.png'))
                plt.close()
        
        print(f"\nLSTM_WVF_Agent Summary for seed {seed}:")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average reward (final 100): {np.mean(episode_rewards[-100:]):.3f}")
        print(f"Average length (final 100): {np.mean(episode_lengths[-100:]):.1f}")
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "training_losses": training_losses,
            "final_epsilon": agent.epsilon,
            "algorithm": "LSTM_WVF_Agent (UVF, view-based)"
        }
    # ========================================================================
    # COMPARISON FRAMEWORK (UPDATED)
    # ========================================================================
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual=False):
        """
        Run comparison between all agents across multiple seeds.
        
        UPDATED: Now includes 5 algorithms:
        1. Masters Successor SARSA
        2. Masters Successor Q-Learning  
        3. DQN (with path integration & vision)
        4. LSTM-DQN
        5. LSTM-WVF (NEW)
        """
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n{'='*60}")
            print(f"Running experiments with seed {seed}")
            print(f"{'='*60}")

            # NEW: LSTM-WVF
            lstm_wvf_results = self.run_lstm_wvf_experiment(
                episodes=episodes, max_steps=max_steps, seed=seed, manual=manual
            )

            # Run all algorithms
            successor_results_sarsa = self.run_successor_experiment_sarsa(
                episodes=episodes, max_steps=max_steps, seed=seed, manual=manual
            )

            successor_results_q_learning = self.run_successor_experiment_q_learning(
                episodes=episodes, max_steps=max_steps, seed=seed, manual=manual
            )

            dqn_results = self.run_dqn_experiment(
                episodes=episodes, max_steps=max_steps, seed=seed, manual=manual
            )

            lstm_dqn_results = self.run_lstm_dqn_experiment(
                episodes=episodes, max_steps=max_steps, seed=seed, manual=manual
            )

            

            # Store results
            algorithms = [
                'Masters Successor SARSA', 
                'Masters Successor Q-Learning', 
                'DQN', 
                'LSTM-DQN',
                'LSTM-WVF'  # NEW
            ]
            results_list = [
                successor_results_sarsa, 
                successor_results_q_learning, 
                dqn_results, 
                lstm_dqn_results,
                lstm_wvf_results  # NEW
            ]
            
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

    # ========================================================================
    # ANALYSIS AND VISUALIZATION (UPDATED)
    # ========================================================================
    
    def analyze_results(self, window=100):
        """
        Analyze and plot comparison results.
        
        UPDATED: Extended to 3x3 grid to accommodate new metrics from LSTM-WVF
        """
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # CHANGED: Extended to 3x3 grid for additional metrics
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))

        # Plot 1: Learning curves (rewards)
        ax1 = axes[0, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

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
            if "path_integration_errors" in runs[0]:
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

        # Plot 4: Final performance comparison
        ax4 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])
            final_rewards[alg_name] = final_100

        ax4.boxplot(final_rewards.values(), labels=final_rewards.keys())
        ax4.set_ylabel("Reward")
        ax4.set_title("Final Performance (Last 100 Episodes)")
        ax4.grid(True)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        # Plot 5: LSTM/DQN Loss Comparison
        ax5 = axes[1, 1]
        for alg_name, runs in self.results.items():
            # Check for different loss types
            if "lstm_losses" in runs[0]:
                all_losses = np.array([run["lstm_losses"] for run in runs])
                mean_losses = np.mean(all_losses, axis=0)
                mean_smooth = pd.Series(mean_losses).rolling(window).mean()
                x = range(len(mean_smooth))
                ax5.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)

        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Loss")
        ax5.set_title("LSTM-DQN Training Loss")
        ax5.legend()
        ax5.grid(True)

        # NEW Plot 6: WVF Loss (for LSTM-WVF)
        ax6 = axes[1, 2]
        for alg_name, runs in self.results.items():
            if "wvf_losses" in runs[0]:
                all_losses = np.array([run["wvf_losses"] for run in runs])
                mean_losses = np.mean(all_losses, axis=0)
                mean_smooth = pd.Series(mean_losses).rolling(window).mean()
                x = range(len(mean_smooth))
                ax6.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)

        ax6.set_xlabel("Episode")
        ax6.set_ylabel("WVF Loss")
        ax6.set_title("LSTM-WVF Q-Network Loss")
        ax6.legend()
        ax6.grid(True)

        # NEW Plot 7: Reward Predictor Loss
        ax7 = axes[2, 0]
        for alg_name, runs in self.results.items():
            if "rp_losses" in runs[0]:
                all_losses = np.array([run["rp_losses"] for run in runs])
                mean_losses = np.mean(all_losses, axis=0)
                mean_smooth = pd.Series(mean_losses).rolling(window).mean()
                x = range(len(mean_smooth))
                ax7.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)

        ax7.set_xlabel("Episode")
        ax7.set_ylabel("RP Loss")
        ax7.set_title("Reward Predictor Loss (LSTM-WVF)")
        ax7.legend()
        ax7.grid(True)

        # NEW Plot 8: RP Training Triggers
        ax8 = axes[2, 1]
        for alg_name, runs in self.results.items():
            if "rp_triggers" in runs[0]:
                all_triggers = np.array([run["rp_triggers"] for run in runs])
                mean_triggers = np.mean(all_triggers, axis=0)
                mean_smooth = pd.Series(mean_triggers).rolling(window).mean()
                x = range(len(mean_smooth))
                ax8.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)

        ax8.set_xlabel("Episode")
        ax8.set_ylabel("RP Triggers")
        ax8.set_title("Reward Predictor Training Frequency")
        ax8.legend()
        ax8.grid(True)

        # Plot 9: Summary statistics
        ax9 = axes[2, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            
            # Path integration statistics
            total_path_errors = 0
            if "path_integration_errors" in runs[0]:
                total_path_errors = np.sum([np.sum(run["path_integration_errors"]) for run in runs])

            summary_data.append({
                "Algorithm": alg_name[:20] + "..." if len(alg_name) > 20 else alg_name,
                "Final Perf": f"{final_performance:.3f}",
                "Convergence": convergence_episode,
                "Path Errors": total_path_errors,
            })

        summary_df = pd.DataFrame(summary_data)
        ax9.axis("tight")
        ax9.axis("off")
        table = ax9.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax9.set_title("Summary Statistics")

        plt.tight_layout()
        save_path = generate_save_path("experiment_comparison_all_algorithms.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

        # Save numerical results
        self.save_results()

        return summary_df

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _find_convergence_episode(self, all_rewards, window):
        """Find approximate convergence episode"""
        mean_rewards = np.mean(all_rewards, axis=0)
        smoothed = pd.Series(mean_rewards).rolling(window).mean()

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

        results_file = generate_save_path(f"experiment_results_all_{timestamp}.json")

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
                
                # Add optional fields if available
                optional_fields = [
                    "path_integration_errors", 
                    "lstm_losses",  # CHANGED: was dqn_losses
                    "wvf_losses",  # NEW
                    "rp_losses",  # NEW
                    "rp_triggers",  # NEW
                    "ae_triggers"
                ]
                
                for key in optional_fields:
                    if key in run:
                        if key in ["lstm_losses", "wvf_losses", "rp_losses"]:
                            json_run[key] = [float(x) for x in run[key]]
                        else:
                            json_run[key] = [int(x) for x in run[key]]
                
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")

    def _create_target_view_with_reward(self, past_agent_pos, past_agent_dir, reward_pos, reward_map):
        """Create 7x7 target view from past agent position showing reward location"""
        target_7x7 = np.zeros((7, 7), dtype=np.float32)
        
        ego_center_x, ego_center_y = 3, 6
        past_x, past_y = past_agent_pos
        reward_x, reward_y = reward_pos
        
        for view_y in range(7):
            for view_x in range(7):
                dx_ego = view_x - ego_center_x
                dy_ego = view_y - ego_center_y
                
                if past_agent_dir == 3:
                    dx_world, dy_world = dx_ego, dy_ego
                elif past_agent_dir == 0:
                    dx_world, dy_world = -dy_ego, dx_ego
                elif past_agent_dir == 1:
                    dx_world, dy_world = -dx_ego, -dy_ego
                elif past_agent_dir == 2:
                    dx_world, dy_world = dy_ego, -dx_ego
                
                global_x = past_x + dx_world
                global_y = past_y + dy_world
                
                if (global_x == reward_x and global_y == reward_y):
                    target_7x7[view_y, view_x] = 1.0
                else:
                    target_7x7[view_y, view_x] = 0.0
        
        return target_7x7

    def _train_ae_on_batch(self, model, optimizer, loss_fn, inputs, targets, device):
        """Train autoencoder on batch of trajectory data"""
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

    def _create_target_7x7(self, agent):
        """
        Create target 7x7 reward map from agent's current knowledge.
        Used for LSTM-WVF reward predictor training.
        """
        target = np.zeros((7, 7), dtype=np.float32)
        
        agent_x, agent_y = agent.internal_pos
        agent_dir = agent.internal_dir
        ego_center_x, ego_center_y = 3, 6
        
        for view_y in range(7):
            for view_x in range(7):
                dx_ego = view_x - ego_center_x
                dy_ego = view_y - ego_center_y
                
                if agent_dir == 3:
                    dx_world, dy_world = dx_ego, dy_ego
                elif agent_dir == 0:
                    dx_world, dy_world = -dy_ego, dx_ego
                elif agent_dir == 1:
                    dx_world, dy_world = -dx_ego, -dy_ego
                elif agent_dir == 2:
                    dx_world, dy_world = dy_ego, -dx_ego
                else:
                    dx_world, dy_world = dx_ego, dy_ego
                
                global_x = agent_x + dx_world
                global_y = agent_y + dy_world
                
                if 0 <= global_x < agent.grid_size and 0 <= global_y < agent.grid_size:
                    target[view_y, view_x] = agent.true_reward_map[global_y, global_x]
        
        return target

    def _get_manual_action(self, agent, obs, episode, step):
        """Get action from manual input for debugging"""
        print(f"Episode {episode}, Step {step} - W=fwd, A=left, D=right, Q=auto")
        key = getch().lower()
        
        if key == 'w':
            return 2  # forward
        elif key == 'a':
            return 0  # turn left
        elif key == 'd':
            return 1  # turn right
        elif key == 'q':
            return agent.select_action(obs)
        elif key == '\r' or key == '\n':
            return agent.select_action(obs)
        else:
            return agent.select_action(obs)

        
    def _save_lstm_wvf_visualizations(self, agent, episode, wvf_losses, rp_losses,
                                    episode_rewards, rp_triggers):
        """
        OPTIMIZED: Skip expensive Q-value heatmap computation.
        """
        
        # WVF Loss plot
        if len(wvf_losses) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(wvf_losses, alpha=0.7, label='WVF Loss')
            if len(wvf_losses) >= 50:
                smoothed = np.convolve(wvf_losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(wvf_losses) - 24), smoothed, 'r-', lw=2, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title(f'LSTM-WVF Q-Network Loss (ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'lstm_wvf/wvf_loss/loss_ep_{episode}.png'))
            plt.close()
        
        # RP Loss plot
        if len(rp_losses) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(rp_losses, alpha=0.7, label='RP Loss')
            if len(rp_losses) >= 50:
                smoothed = np.convolve(rp_losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(rp_losses) - 24), smoothed, 'b-', lw=2, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title(f'Reward Predictor Loss (ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'lstm_wvf/rp_loss/loss_ep_{episode}.png'))
            plt.close()
        
        # RP Triggers plot
        if len(rp_triggers) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(rp_triggers, alpha=0.7, label='RP Training Triggers')
            if len(rp_triggers) >= 50:
                smoothed = np.convolve(rp_triggers, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(rp_triggers) - 24), smoothed, 'g-', lw=2, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Triggers per Episode')
            plt.title(f'Reward Predictor Training Frequency (ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'lstm_wvf/rp_triggers/triggers_ep_{episode}.png'))
            plt.close()
        
        # Rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, alpha=0.7)
        if len(episode_rewards) >= 50:
            smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(25, len(episode_rewards) - 24), smoothed, 'g-', lw=2, label='Smoothed')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'LSTM-WVF Learning Curve (ep {episode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(generate_save_path(f'lstm_wvf/rewards/rewards_ep_{episode}.png'))
        plt.close()
        
        # === OPTIMIZATION: SKIP expensive Q-values heatmap ===
        # The original code did 100 forward passes here - removed for speed
        # Uncomment below if you need it for debugging (will be slow):
        """
        if hasattr(agent, 'get_all_q_values'):
            q_values = agent.get_all_q_values(skip_if_slow=False)  # Set False to compute
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            action_names = ['Turn Left', 'Turn Right', 'Move Forward']
            for a in range(3):
                im = axes[a].imshow(q_values[:, :, a], cmap='viridis', origin='lower')
                axes[a].set_title(f'{action_names[a]} Q-values')
                plt.colorbar(im, ax=axes[a])
            plt.tight_layout()
            plt.savefig(generate_save_path(f'lstm_wvf/qvalues/qvalues_ep_{episode}.png'))
            plt.close()
        """
        
        # Reward map (this is cheap - keep it)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axes[0].imshow(agent.true_reward_map, cmap='viridis', origin='lower')
        axes[0].set_title('Learned Reward Map')
        axes[0].plot(agent.internal_pos[0], agent.internal_pos[1], 'r*', markersize=15, label='Agent')
        plt.colorbar(im1, ax=axes[0])
        axes[0].legend()
        
        im2 = axes[1].imshow(agent.visited_positions.astype(float), cmap='Blues', origin='lower')
        axes[1].set_title('Visited Positions')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(generate_save_path(f'lstm_wvf/maps/maps_ep_{episode}.png'))
        plt.close()



def main():
    """Run the experiment comparison with all algorithms"""
    print("="*60)
    print("Starting comprehensive experiment comparison")
    print("="*60)
    print("\nAlgorithms to compare:")
    print("1. Masters Successor SARSA")
    print("2. Masters Successor Q-Learning")
    print("3. DQN (with path integration & vision)")
    print("4. LSTM-DQN (updated)")
    print("5. LSTM-WVF (goal-conditioned learning)")
    print("="*60)

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=3)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=7000, max_steps=200, manual=False)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\n" + "="*60)
    print("Experiment Summary:")
    print("="*60)
    print(summary)
    print("\n" + "="*60)
    print("Experiment completed! Check the results/ folder for plots and data.")
    print("="*60)


if __name__ == "__main__":
    main()