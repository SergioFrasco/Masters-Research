import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf
import math


from minigrid.core.world_object import Goal, Wall
from tqdm import tqdm
from env import SimpleEnv, data_collector
from models import build_autoencoder, focal_mse_loss, load_trained_autoencoder, weighted_focal_mse_loss
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf
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
    
def construct_sr():
     # Create environment and agent
    env = SimpleEnv(size=10)
    agent = SuccessorAgent(env, learning_rate=0.1, gamma=0.99)

    # Training loop
    n_episodes = 2000
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        current_exp = None
        
        while not done:
            state_idx = agent.get_state_index(obs)
            action = agent.sample_action(obs, epsilon=0.1)
            
            next_obs, reward, done, truncated, info = env.step(action)
            next_state_idx = agent.get_state_index(next_obs)
            
            # Store experience
            next_exp = (state_idx, action, next_state_idx, reward, done)
            
            # Update agent
            if current_exp is not None:
                agent.update(current_exp, next_exp)
            
            current_exp = next_exp
            obs = next_obs
            
            if done:
                # Final update
                agent.update(current_exp)

    
    averaged_M = np.mean(agent.M, axis=0)
    plt.imsave('results/averaged_M.png', averaged_M, cmap='hot')
    np.save('models/successor_representation.npy', averaged_M)

def test_world_value_function():
    """Test and visualize the World Value Function using saved models"""
    # Initialize environment
    env = SimpleEnv(size=10)
    obs, _ = env.reset()
    
    # Load saved models
    autoencoder = load_trained_autoencoder()
    successor_matrix = np.load('models/successor_representation.npy')
    
    # Create a binary grid representation where 1s represent goals
    grid_state = np.zeros((env.size, env.size))
    for i in range(env.size):
        for j in range(env.size):
            cell = env.grid.get(i, j)
            if isinstance(cell, Goal):
                grid_state[i, j] = 1
            elif isinstance(cell, Wall):
                grid_state[i, j] = 0.5
    
    # Reshape for the autoencoder (adding batch and channel dimensions)
    grid_state = grid_state.reshape(1, env.size, env.size, 1)
    
    # Get vision model prediction
    reconstructed = autoencoder.predict(grid_state)
    
    # Create reward location array
    reward_locations = np.zeros(env.size * env.size)
    reward_threshold = 0.75
    
    # Convert reconstructed image to flattened reward array
    for i in range(env.size):
        for j in range(env.size):
            if reconstructed[0, i, j, 0] > reward_threshold:
                state_idx = i * env.size + j
                reward_locations[state_idx] = 1
    
    # Compute World Value Function
    world_value_function = np.dot(successor_matrix, reward_locations)
    
    # Reshape WVF for visualization
    wvf_grid = world_value_function.reshape(env.size, env.size)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original environment
    ax1.imshow(grid_state[0, :, :, 0], cmap='gray')
    ax1.set_title("Original Environment")
    
    # Plot vision model output
    ax2.imshow(reconstructed[0, :, :, 0], cmap='gray')
    ax2.set_title("Vision Model Detection")
    
    # Plot World Value Function
    wvf_plot = ax3.imshow(wvf_grid, cmap='hot')
    ax3.set_title("World Value Function")
    plt.colorbar(wvf_plot, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('results/world_value_function.png')
    plt.close()
    
    return world_value_function, wvf_grid

def visualize_agent_trajectory(env, wvf_grid, n_steps=100):
    """Visualize an agent following the World Value Function"""
    obs, _ = env.reset()
    trajectory = [env.agent_pos]
    
    for _ in range(n_steps):
        # Get current position
        x, y = env.agent_pos
        
        # Get neighboring positions
        neighbors = [
            (x-1, y), (x+1, y),  # Left, Right
            (x, y-1), (x, y+1)   # Up, Down
        ]
        
        # Filter valid positions and get their values
        valid_neighbors = []
        neighbor_values = []
        
        for nx, ny in neighbors:
            if 0 <= nx < env.size and 0 <= ny < env.size:
                valid_neighbors.append((nx, ny))
                neighbor_values.append(wvf_grid[ny, nx])
        
        # Choose direction with highest value
        if valid_neighbors:
            best_idx = np.argmax(neighbor_values)
            next_pos = valid_neighbors[best_idx]
            
            # Move agent (simplified)
            env.agent_pos = next_pos
            trajectory.append(next_pos)
        
        # Check if goal reached
        if isinstance(env.grid.get(*env.agent_pos), Goal):
            break
    
    # Visualize trajectory
    plt.figure(figsize=(8, 8))
    plt.imshow(wvf_grid, cmap='hot')
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'w-', linewidth=2, label='Agent Path')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')
    
    plt.colorbar(label='Value')
    plt.legend()
    plt.title('Agent Trajectory on World Value Function')
    plt.savefig('results/agent_trajectory.png')
    plt.close()

def train_successor_agent(agent, env, episodes=500, ae_model=None, max_steps=200, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, train_vision_threshold=0.1):
    """
    Training loop for SuccessorAgent in MiniGrid environment with vision model integration
    """
    episode_rewards = []
    epsilon = epsilon_start
    
    # Initialize explored positions mask
    agent.true_reward_map_explored = np.zeros((env.size, env.size), dtype=bool)
    
    for episode in tqdm(range(episodes), "Training Successor Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0

        # For every new episode, reset the reward map, and WVF, the SR stays consistent with the environment i.e doesn't reset
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.true_reward_map_explored = np.zeros((env.size, env.size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]
        
        for step in range(max_steps):
            # Take action and observe result
            obs, reward, done, _, _ = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx  # next state
            current_exp[3] = reward          # reward
            current_exp[4] = done            # done flag
            

            # Here we need to sample from WVF.
            # 1. Build the WVF for this moment in time
            # 2. Maximize over the WVF to grab all Maps that contain goals
            # 3. Choose a Random one of these Maps
            # 4. Sample and Action from this map with decaying epsilon probability

            reward_threshold = 0.75
            # Check if each 10x10 map contains any value > threshold
            mask = (agent.wvf > reward_threshold).any(axis=(1, 2))  # shape: (100,)
            # Use mask to select maps
            max_wvfs = agent.wvf[mask]  # shape: (N, 10, 10) where N <= 100
            
            # Get next action, for the first step just use a random action as the WVF is only setup after the first step, thereafter use WVF
            # if step == 0 or len(max_wvfs) == 0:
                # print("First Normal Action Taken")
            next_action = agent.sample_action(obs, epsilon=epsilon)
                
            # else:
            #     print("WVF Action Taken")
            #     random_map_index = np.random.randint(0, len(max_wvfs))
            #     chosen_map = max_wvfs[random_map_index]
            #     next_action = agent.sample_wvf_action(obs, epsilon = epsilon, chosen_map = chosen_map)


                # # Compute a suitable grid size (square-ish)
                # cols = int(np.ceil(np.sqrt(len(max_wvfs))))
                        
                # rows = int(np.ceil(len(max_wvfs) / cols))

                # fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

                # # Flatten in case axs is a 2D array when rows, cols > 1
                # axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

                # for i in range(len(axs)):
                #     ax = axs[i]
                #     if i < len(max_wvfs):
                #         im = ax.imshow(max_wvfs[i], cmap='viridis', vmin=0, vmax=1)
                #         ax.set_title(f'Map {i}')
                #     ax.axis('off')

                # plt.tight_layout()
                # plt.savefig(f"results/max_wvfs_{episode}")

            
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
            normalized_grid[object_layer == 2] = 0.0   # Wall - should this be a 0?? or a number, this is for input
            normalized_grid[object_layer == 1] = 0.0   # Open space
            normalized_grid[object_layer == 8] = 1.0   # Reward (e.g. goal object)
            
            # Rotate the grid to match render_mode = human 
            normalized_grid = np.flipud(normalized_grid)
            normalized_grid = np.rot90(normalized_grid, k=-1)
            
            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
            
            # Get the predicted reward map from the AE
            predicted_reward_map = ae_model.predict(input_grid, verbose=0)
            predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]
            
            # Update the rest of the true_reward_map with AE predictions
            for y in range(agent.true_reward_map.shape[0]):
                for x in range(agent.true_reward_map.shape[1]):
                    if (x, y) != agent_position:  # Skip the reward position
                        # Get the predicted value for this position from the AE
                        predicted_value = predicted_reward_map_2d[y, x]
                        agent.true_reward_map[y, x] = predicted_value

            if done:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 1
                # print(agent.true_reward_map)
            else:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 0


            trigger_ae_training = False

            if abs(predicted_reward_map_2d[agent_position[1], agent_position[0]] - agent.true_reward_map[agent_position[1], agent_position[0]]) > train_vision_threshold:
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
                    # Only track if we are sure its a reward
                    if reward > 0.75:
                        idx = y * agent.grid_size + x
                        agent.reward_maps[idx, y, x] = reward
                    else:
                        idx = y * agent.grid_size + x
                        agent.reward_maps[idx, y, x] = 0

            
            # dot product the SR with these reward Maps
            # 1. SR: M_flat[s, s'] = expected future occupancy of s' from s
            M_flat = np.mean(agent.M, axis=0)  # shape: (100, 100), average accross actions

            # Attempt 3
            # Compute the value function for each reward map
            # Compute the value function for each reward map
            for i in range(agent.state_size):  # Loop through reward maps
                R = agent.reward_maps[i, :, :]  # (10, 10) reward map
                R_flat = R.flatten()  # (100,) vector

                # Dot product with SR to get value function over all states
                V = np.dot(M_flat, R_flat)  # V is shape (100,)
                
                # Reshape to (10, 10) and store
                agent.wvf[i] = V.reshape(agent.grid_size, agent.grid_size)




            # Reward found, next episode
            if done:
                break
                
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)

         # Generate visualizations occasionally
        if episode % 10 == 0:
            save_all_reward_maps(agent, save_path=f"results/reward_maps_episode_{episode}")
            save_all_wvf(agent, save_path=f"results/wvf_episode_{episode}")
            averaged_M = np.mean(agent.M, axis=0)
            plt.imsave(f'results/averaged_M_{episode}.png', averaged_M, cmap='hot')

            # print("Actual: \n", normalized_grid)
            # print("Agents Guess: \n", agent.true_reward_map)
            # print("Input Grid: \n", input_grid)
            # Create visualization of current state
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grid (environment)
            ax1.imshow(normalized_grid, cmap='gray')
            ax1.set_title("Environment Grid")
            
            # Agent's true reward map
            ax2.imshow(agent.true_reward_map, cmap='viridis')
            # Overlay dots on positions the agent has actually visited
            # visited_y, visited_x = np.where(agent.true_reward_map_explored)
            # ax2.scatter(visited_x, visited_y, color='red', s=5)
            ax2.set_title("Agent's Reward Map (red=visited)")
            
            # AE prediction
            ax3.imshow(predicted_reward_map_2d, cmap='viridis')
            ax3.set_title("AE Prediction")
            
            plt.tight_layout()
            plt.savefig(f'results/episode_{episode}.png')
            plt.close()


    
    ae_model.save('results/current/vision_model.h5')
    print("Training Complete, Vision Model Saved!")
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
    ae_model.compile(optimizer='adam', loss=focal_mse_loss)

    # Train the agent
    rewards = train_successor_agent(agent, env, ae_model = ae_model) 
    averaged_M = np.mean(agent.M, axis=0)
    plt.imsave('results/averaged_M.png', averaged_M, cmap='hot')
    # np.save('models/successor_representation.npy', averaged_M)


if __name__ == "__main__":
    main()

