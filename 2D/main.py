import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf
from minigrid.core.world_object import Goal, Wall

from tqdm import tqdm
from env import SimpleEnv, collect_data
from models import load_trained_autoencoder
from models import focal_mse_loss
from utils.plotting import overlay_values_on_grid, visualize_sr
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

def train_successor_agent(
    agent,
    env,
    episodes=500,
    max_steps=100,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
):
    """
    Training loop for SuccessorAgent in MiniGrid environment
    
    Args:
        agent: SuccessorAgent instance
        env: MiniGrid environment
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        epsilon_start: Starting exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Rate at which epsilon decays
    """
    episode_rewards = []
    epsilon = epsilon_start
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]
        
        for step in range(max_steps):
            # Take action and observe result
            obs, reward, done, _ = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx  # next state
            current_exp[3] = reward          # reward
            current_exp[4] = done            # done flag
            
            # Get next action
            next_action = agent.sample_action(obs, epsilon=epsilon)
            
            # Create next experience tuple
            next_exp = [next_state_idx, next_action, None, None, None]
            
            # Update agent
            error_w, error_sr = agent.update(current_exp, None if done else next_exp)
            
            total_reward += reward
            step_count += 1
            
            # Prepare for next step
            current_exp = next_exp
            current_action = next_action
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Steps: {step_count}")
            print("------------------------")
    
    return episode_rewards


def main():
    # Collecting sample images from the environment
    # collect_data()
    # print("Finished Collecting Sample Environments")
    
    # --------------------- Vision Based Reward Model ------------------------
  
    # dataset = np.load("datasets/grid_dataset.npy")
    # print("Loading Data Set")

    # autoencoder = load_trained_autoencoder()
    # print("Loading Vision Model")

    # autoencoder.compile(optimizer="adam", loss=focal_mse_loss)
    # reconstructed = autoencoder.predict(dataset)
    # print("Completed Running Vision Model On Dataset")

    # # Visualize the original and reconstructed grids (first sample)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Original grid
    # ax1.imshow(dataset[0, ..., 0], cmap='gray')
    # ax1.set_title("Original Grid")
    # overlay_values_on_grid(dataset[0, ..., 0], ax1)

    # # Reconstructed grid
    # ax2.imshow(reconstructed[0, ..., 0], cmap='gray')
    # ax2.set_title("Reconstructed Grid")
    # overlay_values_on_grid(reconstructed[0, ..., 0], ax2)
    # plt.tight_layout()
    # plt.savefig('results/comparison.png')

    # Retrieve the specific SR for the Given environment

    # Build the SR map for the given Environment 
    # construct_sr()

    # Now we have both the vision model and the transition dynamics. lets use them together to build WVF's

    # Test and visualize World Value Function
    # world_value_function, wvf_grid = test_world_value_function()
    
    # # Create environment for trajectory visualization
    # env = SimpleEnv(size=10)
    # visualize_agent_trajectory(env, wvf_grid)

    # --------- Where i got up to before the meeting ------------
    # Now we look to train the autoencoder as the agent moves through the environment
    env = SimpleEnv(size=10)
    agent = SuccessorAgent(env)
    rewards = train_successor_agent(agent, env)


   

if __name__ == "__main__":
    main()