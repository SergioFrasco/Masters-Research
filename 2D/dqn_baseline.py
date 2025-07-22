import numpy as np
import tensorflow as tf
from collections import deque
import random

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from agents import DQNAgent
from env import SimpleEnv
from utils.plotting import generate_save_path
from utils import get_latest_run_dir, create_video_from_images

def evaluate_dqn_performance(agent, env, episode, log_file_path):
    """
    Evaluate DQN performance by checking Q-values at goal vs neighbors.
    Similar to your evaluate_goal_state_values function but for DQN.
    """
    current_pos = env.agent_pos
    x, y = current_pos
    
    # Get current state
    current_state = agent.get_state_vector()
    current_q_values = agent.get_q_values(current_state)
    
    # Get Q-value for staying (this is tricky since there's no "stay" action)
    # We'll use the maximum Q-value as a proxy for the state's desirability
    current_max_q = np.max(current_q_values)
    
    # Check neighboring positions by simulating what Q-values would be
    neighbor_max_qs = []
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    for nx, ny in neighbors:
        if (0 <= nx < agent.grid_size and 0 <= ny < agent.grid_size):
            # Check if it's a valid position
            cell = env.grid.get(nx, ny)
            if cell is None or not hasattr(cell, '__class__') or cell.__class__.__name__ != 'Wall':
                # Simulate state at neighbor position
                goal_x, goal_y = agent._find_goal_position()
                neighbor_state = np.array([
                    nx / (agent.grid_size - 1),
                    ny / (agent.grid_size - 1),
                    current_state[2],  # Same direction
                    goal_x / (agent.grid_size - 1),
                    goal_y / (agent.grid_size - 1)
                ], dtype=np.float32)
                
                neighbor_q_values = agent.get_q_values(neighbor_state)
                neighbor_max_qs.append(np.max(neighbor_q_values))
    
    # Determine if current state has highest Q-value
    if len(neighbor_max_qs) == 0:
        result = f"Episode {episode}: No valid neighbors to compare\n"
    else:
        max_neighbor_q = max(neighbor_max_qs)
        if current_max_q >= max_neighbor_q:
            result = f"Episode {episode}: Goal state has highest Q-value (goal: {current_max_q:.3f}, max neighbor: {max_neighbor_q:.3f})\n"
        else:
            result = f"Episode {episode}: Goal state does not have highest Q-value (goal: {current_max_q:.3f}, max neighbor: {max_neighbor_q:.3f})\n"
    
    # Write to log file
    with open(log_file_path, 'a') as f:
        f.write(result)
    
    return result.strip()

def train_dqn_agent(agent, env, episodes=20001, max_steps=200, save_interval=250):
    """
    Training loop for DQN agent in MiniGrid environment.
    """
    episode_rewards = []
    step_counts = []
    losses = []
    epsilon_values = []
    
    # Tracking where the agent goes
    state_occupancy = np.zeros((env.size, env.size), dtype=np.int32)
    
    # Create log file for performance evaluations
    log_file_path = generate_save_path("dqn_performance_evaluations.txt")
    with open(log_file_path, 'w') as f:
        f.write("DQN Performance Evaluations\n")
        f.write("="*50 + "\n")
    
    # Training statistics
    q_value_stats = defaultdict(list)
    
    for episode in tqdm(range(episodes), "Training DQN Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        episode_loss = []
        
        # Get initial state
        current_state = agent.get_state_vector(obs)
        
        for step in range(max_steps):
            # Choose action
            action = agent.get_action(current_state)
            
            # Take action
            obs, reward, done, _, _ = env.step(action)
            next_state = agent.get_state_vector(obs)
            
            # Track agent position
            agent_pos = tuple(env.agent_pos)
            state_occupancy[agent_pos[0], agent_pos[1]] += 1
            
            # Store experience
            agent.remember(current_state, action, reward, next_state, done)
            
            # Train the network
            if len(agent.memory) >= agent.batch_size:
                loss, _ = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
            
            total_reward += reward
            step_count += 1
            current_state = next_state
            
            # Track Q-values for analysis
            if step % 10 == 0:  # Sample every 10 steps to avoid too much data
                q_values = agent.get_q_values(current_state)
                q_value_stats['max_q'].append(np.max(q_values))
                q_value_stats['mean_q'].append(np.mean(q_values))
                q_value_stats['std_q'].append(np.std(q_values))
            
            if done:
                # Evaluate performance at goal
                if episode % 100 == 0:
                    evaluate_dqn_performance(agent, env, episode, log_file_path)
                break
        
        # FIXED: Always append step count, use actual steps taken
        step_counts.append(step_count)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        else:
            losses.append(0)
        
        # Save visualizations periodically
        if episode % save_interval == 0 and episode > 0:
            save_dqn_visualizations(agent, env, episode, q_value_stats)
    
    # Save final model
    agent.save_model(generate_save_path('dqn_model.keras'))  # Using .keras format
    
    # Generate final plots
    plot_training_results(episode_rewards, step_counts, losses, epsilon_values, q_value_stats)
    
    return episode_rewards, step_counts, losses

def save_dqn_visualizations(agent, env, episode, q_value_stats):
    """Save DQN-specific visualizations."""
    
    # Q-value heatmap for the current goal position
    goal_x, goal_y = agent._find_goal_position()
    q_value_grid = np.zeros((agent.grid_size, agent.grid_size))
    
    for y in range(agent.grid_size):
        for x in range(agent.grid_size):
            # Check if position is valid (not a wall)
            cell = env.grid.get(x, y)
            if cell is None or not hasattr(cell, '__class__') or cell.__class__.__name__ != 'Wall':
                # Create state for this position
                state = np.array([
                    x / (agent.grid_size - 1),
                    y / (agent.grid_size - 1),
                    0.0,  # Facing north
                    goal_x / (agent.grid_size - 1),
                    goal_y / (agent.grid_size - 1)
                ], dtype=np.float32)
                
                q_values = agent.get_q_values(state)
                q_value_grid[y, x] = np.max(q_values)
            else:
                q_value_grid[y, x] = np.nan  # Mark walls as NaN
    
    # Plot Q-value heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(q_value_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Max Q-value')
    plt.title(f'Q-value Heatmap (Episode {episode})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Mark goal position
    plt.scatter(goal_x, goal_y, c='red', s=100, marker='*', label='Goal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'q_values/q_heatmap_episode_{episode}.png'))
    plt.close()

def plot_training_results(episode_rewards, step_counts, losses, epsilon_values, q_value_stats):
    """Plot comprehensive training results."""
    
    window = 100
    
    # FIXED: Ensure all arrays have the same length
    min_length = min(len(episode_rewards), len(step_counts), len(losses), len(epsilon_values))
    episode_rewards = episode_rewards[:min_length]
    step_counts = step_counts[:min_length]
    losses = losses[:min_length]
    epsilon_values = epsilon_values[:min_length]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DQN Training Results', fontsize=16)
    
    # Episode rewards
    rewards_series = pd.Series(episode_rewards)
    rolling_rewards = rewards_series.rolling(window).mean()
    
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    axes[0, 0].plot(rolling_rewards, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Step counts
    steps_series = pd.Series(step_counts)
    rolling_steps = steps_series.rolling(window).mean()
    
    axes[0, 1].plot(step_counts, alpha=0.3, label='Steps per Episode')
    axes[0, 1].plot(rolling_steps, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    loss_series = pd.Series(losses)
    rolling_loss = loss_series.rolling(window).mean()
    
    axes[0, 2].plot(losses, alpha=0.3, label='Training Loss')
    axes[0, 2].plot(rolling_loss, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Training Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Epsilon decay
    axes[1, 0].plot(epsilon_values)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Epsilon Decay')
    axes[1, 0].grid(True)
    
    # Q-value statistics
    if q_value_stats['max_q']:
        max_q_series = pd.Series(q_value_stats['max_q'])
        mean_q_series = pd.Series(q_value_stats['mean_q'])
        
        axes[1, 1].plot(max_q_series.rolling(50).mean(), label='Max Q-value')
        axes[1, 1].plot(mean_q_series.rolling(50).mean(), label='Mean Q-value')
        axes[1, 1].set_xlabel('Training Steps (sampled)')
        axes[1, 1].set_ylabel('Q-value')
        axes[1, 1].set_title('Q-value Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Learning curve (reward vs steps) - FIXED: Now both arrays have same length
    axes[1, 2].scatter(step_counts, episode_rewards, alpha=0.5)
    axes[1, 2].set_xlabel('Steps to Goal')
    axes[1, 2].set_ylabel('Episode Reward')
    axes[1, 2].set_title('Learning Efficiency')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(generate_save_path("dqn_training_results.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup environment
    env = SimpleEnv(size=10)
    
    # Setup DQN agent
    agent = DQNAgent(env, 
                     learning_rate=0.001,
                     gamma=0.95,
                     epsilon_start=1.0,
                     epsilon_end=0.01,
                     epsilon_decay=0.9995,
                     memory_size=10000,
                     batch_size=32,
                     target_update_freq=100)
    
    # Train the agent
    rewards, steps, losses = train_dqn_agent(agent, env, episodes=10001)
    
    print(f"Training completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Average steps (last 100 episodes): {np.mean(steps[-100:]):.2f}")

if __name__ == "__main__":
    main()