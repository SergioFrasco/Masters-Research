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
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred
from models.construct_sr import constructSR
from agents import SuccessorAgent

reward_threshold = 0.75

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable GPU if not needed
tf.config.set_visible_devices([], "GPU")

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)
sys.path.append(".")
    

def train_successor_agent(agent, env, episodes=401, ae_model=None, max_steps=150, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, train_vision_threshold=0.1):
    """
    Training loop for SuccessorAgent in MiniGrid environment with vision model integration, SR tracking, and WVF formation
    """
    episode_rewards = []
    epsilon = epsilon_start

    print_flag = True
    
    for episode in tqdm(range(episodes), "Training Successor Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0

        # For every new episode, reset the reward map, and WVF, the SR stays consistent with the environment i.e doesn't reset
        agent.true_reward_map = np.zeros((env.size, env.size))
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
            # 4. Sample an Action from this map with decaying epsilon probability

            # Reset max_wvfs maps to zero
            max_wvfs = np.empty((0, agent.grid_size, agent.grid_size), dtype=np.float32)
            # Check if each 10x10 map contains any value > threshold
            mask = (agent.wvf > reward_threshold).any(axis=(1, 2))  # shape: (100,)
            # Use mask to select maps
            max_wvfs = agent.wvf[mask]  # shape: (N, 10, 10) where N <= 100
            
            # Get next action, for the first step just use a q-learned action as the WVF is only setup after the first step, thereafter use WVF
            # Also checks if we actually have a max map. ie if we're not cofident in our WVF we sample a q-learned action
            if step == 0 or len(max_wvfs) == 0:
                # print("Normal Action Taken")
                next_action = agent.sample_action(obs, epsilon=epsilon)
            
            # Sample an action from the WVF
            else:
                if print_flag:
                    # Typically this triggers around episode 20
                    print("First WVF Action Taken")
                    print_flag = False
                
                # Choose the map at random, should this be random? or just go toward the max one all the time?
                # We'll have to check average return to see it's performance
                random_map_index = np.random.randint(0, len(max_wvfs))
                chosen_map = max_wvfs[random_map_index]
                next_action = agent.sample_action_with_wvf(obs, epsilon = epsilon, chosen_reward_map = chosen_map)
                
                # save_max_wvf_maps(max_wvfs, episode = episode)
                # Printing the array of maps
                plt.close('all')  # to close all open figures and save memory

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
                    if reward > reward_threshold:
                        idx = y * agent.grid_size + x
                        agent.reward_maps[idx, y, x] = reward
                    else:
                        idx = y * agent.grid_size + x
                        agent.reward_maps[idx, y, x] = 0

            
            # dot product the SR with these reward Maps
            # 1. SR: M_flat[s, s'] = expected future occupancy of s' from s
            M_flat = np.mean(agent.M, axis=0)  # shape: (100, 100), average accross actions

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
        if episode % 100 == 0:
            save_all_reward_maps(agent, save_path=f"results/reward_maps_episode_{episode}")
            save_all_wvf(agent, save_path=f"results/wvf_episode_{episode}")
            averaged_M = np.mean(agent.M, axis=0)
            plt.imsave(f'results/averaged_M_{episode}.png', averaged_M, cmap='hot')
            save_env_map_pred(agent = agent, normalized_grid = normalized_grid, predicted_reward_map_2d = predicted_reward_map_2d, episode = episode)
        
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
    # averaged_M = np.mean(agent.M, axis=0)
    # plt.imsave('results/averaged_M.png', averaged_M, cmap='hot')
    # np.save('models/successor_representation.npy', averaged_M)

        # Plot rewards over episodes
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rewards_over_episodes.png")
    plt.show()



if __name__ == "__main__":
    main()

