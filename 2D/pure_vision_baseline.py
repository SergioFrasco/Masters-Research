# -------------------------------
# Vision Model Value Prediction
# -------------------------------
#
# The vision model (autoencoder) is trained to predict the value function
# of the environment from a 2D visual grid representation.
#
# ✅ Training Setup:
# - Input: A 2D grid image of the environment (walls, agent, goal).
# - Target: The ground truth value map learned by the agent via TD learning.
# - Loss: Mean Squared Error (MSE) between predicted and true value maps.
#
# 🔁 Training Loop (called every few steps):
# 1. Collect (input_data, target_data) via `agent.prepare_training_data()`.
#    - input_data: Current grid state (encoded).
#    - target_data: Agent’s learned value map (Q-values per cell).
# 2. Train the autoencoder with: model.fit(input_data, target_data)
#
# 🧠 Temporal Difference (TD) Learning:
# - Q-learning is used internally by the agent to compute target values.
# - The update rule used is:
#     V(s) ← V(s) + α [r + γ max_a' V(s') - V(s)]
#
# 🎯 Purpose:
# - The model learns to map raw grid layouts to spatial value predictions.
# - It bootstraps off Q-learning without relying on SR or latent features.
# - This creates a pure vision-based value function baseline for comparison.


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf
import math
import pandas as pd
import glob

from collections import deque
from agents import ImprovedVisionOnlyAgent
from minigrid.core.world_object import Goal, Wall
from tqdm import tqdm
from env import SimpleEnv, data_collector
from models import build_autoencoder, focal_mse_loss, load_trained_autoencoder, weighted_focal_mse_loss
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
from utils import create_video_from_images, get_latest_run_dir
from models.construct_sr import constructSR

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable GPU if not needed
tf.config.set_visible_devices([], "GPU")

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)
sys.path.append(".")



def train_improved_vision_agent(agent, env, episodes=2000, ae_model=None, max_steps=200,
                               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                               train_every=10):
    """
    Improved training loop with better experience collection and model updates
    """
    episode_rewards = []
    step_counts = []
    training_losses = []
    epsilon = epsilon_start
    
    # Create directories for saving
    os.makedirs(generate_save_path("vision_maps"), exist_ok=True)
    
    # Create log file
    log_file_path = generate_save_path("improved_vision_training.log")
    
    for episode in tqdm(range(episodes), "Training Improved Vision Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        prev_pos = tuple(env.agent_pos)
        
        for step in range(max_steps):
            # Choose action
            if episode < 50:  # Extended warm-up with random actions
                action = env.action_space.sample()
            else:
                action = agent.sample_action_from_values(obs, epsilon=epsilon)
            
            # Take action
            obs, reward, done, _, _ = env.step(action)
            next_pos = tuple(env.agent_pos)
            
            # Update value map
            agent.update_value_map(prev_pos, action, reward, next_pos, done)
            
            # Train autoencoder periodically
            if step % train_every == 0 or done:
                input_data, target_data = agent.prepare_training_data()
                
                if input_data is not None and ae_model is not None:
                    # Train with multiple epochs for better convergence
                    history = ae_model.fit(
                        input_data, target_data,
                        epochs=3,
                        batch_size=1,
                        verbose=0
                    )
                    
                    # Update prediction
                    predicted_values = ae_model.predict(input_data, verbose=0)
                    agent.predicted_value_map = predicted_values[0, :, :, 0]
                    
                    training_losses.append(history.history['loss'][-1])
            
            total_reward += reward
            step_count += 1
            prev_pos = next_pos
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store statistics
        episode_rewards.append(total_reward)
        step_counts.append(step_count)
        
        # Save visualizations
        if episode % 200 == 0:
            save_improved_vision_maps(agent, episode, generate_save_path)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(step_counts[-100:])
            print(f"Episode {episode}: Avg Reward={avg_reward:.3f}, Avg Steps={avg_steps:.1f}, Epsilon={epsilon:.3f}")
    
    # Save final model
    if ae_model is not None:
        ae_model.save(generate_save_path('improved_vision_model.h5'))
    
    # Generate final plots
    plot_improved_results(episode_rewards, step_counts, training_losses, generate_save_path)
    
    return episode_rewards, step_counts, training_losses


def save_improved_vision_maps(agent, episode, generate_save_path):
    """Save improved visualizations"""
    # Get current environment state
    grid = agent.env.grid.encode()
    object_layer = grid[..., 0]
    
    normalized_grid = np.zeros_like(object_layer, dtype=np.float32)
    normalized_grid[object_layer == 2] = 0.0   # Wall
    normalized_grid[object_layer == 1] = 0.0   # Open space  
    normalized_grid[object_layer == 8] = 1.0   # Goal
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Environment
    axes[0,0].imshow(normalized_grid, cmap='gray')
    axes[0,0].set_title(f'Environment (Episode {episode})')
    
    # Predicted value map
    im1 = axes[0,1].imshow(agent.predicted_value_map, cmap='hot', interpolation='nearest')
    axes[0,1].set_title('Predicted Value Map')
    plt.colorbar(im1, ax=axes[0,1])
    
    # True value map
    im2 = axes[1,0].imshow(agent.true_value_map, cmap='hot', interpolation='nearest')
    axes[1,0].set_title('True Value Map')
    plt.colorbar(im2, ax=axes[1,0])
    
    # Visit counts
    im3 = axes[1,1].imshow(agent.visit_counts, cmap='Blues', interpolation='nearest')
    axes[1,1].set_title('Visit Counts')
    plt.colorbar(im3, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'vision_maps/episode_{episode}_improved.png'))
    plt.close()


def plot_improved_results(episode_rewards, step_counts, training_losses, generate_save_path):
    """Generate improved result plots"""
    window = 50
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    rewards_series = pd.Series(episode_rewards)
    smoothed_rewards = rewards_series.rolling(window).mean()
    
    axes[0,0].plot(episode_rewards, alpha=0.3, label='Raw')
    axes[0,0].plot(smoothed_rewards, linewidth=2, label='Smoothed')
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Steps
    steps_series = pd.Series(step_counts)
    smoothed_steps = steps_series.rolling(window).mean()
    
    axes[0,1].plot(step_counts, alpha=0.3, label='Raw')
    axes[0,1].plot(smoothed_steps, linewidth=2, label='Smoothed')
    axes[0,1].set_title('Steps per Episode')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Training loss
    if training_losses:
        axes[1,0].plot(training_losses)
        axes[1,0].set_title('Training Loss')
        axes[1,0].set_xlabel('Training Step')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].grid(True)
    
    # Step histogram
    axes[1,1].hist(step_counts, bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Step Count Distribution')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(np.mean(step_counts), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(step_counts):.1f}')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(generate_save_path('improved_vision_results.png'))
    plt.close()


def main():
    """Main function to run improved vision-only agent training"""
    # Setup environment
    env = SimpleEnv(size=10)
    
    # Setup improved vision-only agent
    agent = ImprovedVisionOnlyAgent(env)
    
    # Setup autoencoder
    input_shape = (env.size, env.size, 1)
    ae_model = build_autoencoder(input_shape)
    
    # Use a better loss function and optimizer
    ae_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train the improved agent
    print("Training Improved Vision-Only Agent")
    rewards, steps, losses = train_improved_vision_agent(
        agent, env, 
        ae_model=ae_model, 
        episodes=5001,
        train_every=5
    )
    
    print(f"\nTraining completed.")
    print(f"Final episode reward: {rewards[-1]}")
    print(f"Average reward over last 100 episodes: {np.mean(rewards[-100:]):.3f}")
    if steps:
        print(f"Average steps over last 100 episodes: {np.mean(steps[-100:]):.1f}")
        print(f"Best (minimum) steps achieved: {min(steps)}")
    
    return rewards, steps, losses


if __name__ == "__main__":
    main()