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

Reward_Threshold = 0.75

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
    
    # Convert reconstructed image to flattened reward array
    for i in range(env.size):
        for j in range(env.size):
            if reconstructed[0, i, j, 0] > Reward_Threshold:
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