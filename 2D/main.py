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
from models import build_autoencoder


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

class GroundTruthMap:
    """Manages the ground truth map of discovered rewards"""
    def __init__(self, size):
        self.size = size
        self.reset()
        
    def reset(self):
        """Initialize empty ground truth map"""
        self.map = np.zeros((self.size, self.size, 1))
        self.discovered_rewards = set()
        
    def update(self, pos):
        """Update map with newly discovered reward"""
        pos_tuple = tuple(pos)
        if pos_tuple not in self.discovered_rewards:
            self.map[pos[0], pos[1], 0] = 1
            self.discovered_rewards.add(pos_tuple)
            return True  # New discovery
        return False  # Already known

def train_agent_with_vision(
    agent,
    env,
    autoencoder,
    n_episodes=100,
    max_steps=1000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    ae_epochs_per_update=10,
    ae_batch_size=1
):
    """
    Combined training loop that relies on environment reset for new reward placement
    """
    epsilon = epsilon_start
    total_rewards_history = []
    ground_truth = GroundTruthMap(env.size)
    
    # Main training loop
    for episode in range(n_episodes):
        print(f"\nStarting Episode {episode + 1}/{n_episodes}")
        
        # Reset environment (this automatically places new rewards)
        obs = env.reset()
        # Reset ground truth map for new episode
        ground_truth.reset()
        
        total_reward = 0
        step_count = 0
        
        # Initialize first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]
        
        # Episode loop
        for step in range(max_steps):
            # Take action
            obs, reward, done, truncated, dict = env.step(current_action)
            next_state_idx = agent.get_state_index(obs)
            
            # Handle reward discovery
            if reward > 0:
                is_new_reward = ground_truth.update(env.agent_pos)
                
                if is_new_reward:
                    print(f"New reward found at {env.agent_pos}")
                    
                    # Train autoencoder on updated ground truth map
                    current_map = ground_truth.map[np.newaxis, ...]  # Add batch dimension
                    history = autoencoder.fit(
                        current_map, 
                        current_map,
                        epochs=ae_epochs_per_update,
                        batch_size=ae_batch_size,
                        verbose=0
                    )
                    
                    # Print training feedback
                    final_loss = history.history['loss'][-1]
                    print(f"Autoencoder loss after update: {final_loss:.4f}")
            
            # Complete current experience tuple
            current_exp[2] = next_state_idx
            current_exp[3] = reward
            current_exp[4] = done
            
            # Get next action
            next_action = agent.sample_action(obs, epsilon=epsilon)
            next_exp = [next_state_idx, next_action, None, None, None]
            
            # Update successor agent
            error_w, error_sr = agent.update(current_exp, None if done else next_exp)
            
            total_reward += reward
            step_count += 1
            
            # Setup for next step
            current_exp = next_exp
            current_action = next_action
            
            if done:
                print(f"Episode finished after {step_count} steps")
                print(f"Found {len(ground_truth.discovered_rewards)} rewards")
                break
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode results
        total_rewards_history.append(total_reward)
        
        # Print episode summary
        print(f"Episode {episode + 1} Summary:")
        print(f"Total Reward: {total_reward}")
        print(f"Epsilon: {epsilon:.3f}")
        if len(total_rewards_history) >= 10:
            print(f"Average Reward (last 10): {np.mean(total_rewards_history[-10:]):.2f}")
    
    return total_rewards_history, autoencoder

def test_autoencoder(model_path, test_image, save_path=None):
    """
    Test a trained autoencoder with a new input image and visualize results
    
    Args:
        model_path (str): Path to the saved autoencoder model
        test_image (np.array): Input image of shape (height, width, 1)
        save_path (str, optional): Path to save the comparison plot
    """
    # Load the trained model
    autoencoder = load_trained_autoencoder(model_path)
    
    # Ensure input image has batch dimension and correct shape
    if len(test_image.shape) == 2:
        test_image = test_image[..., np.newaxis]
    test_input = test_image[np.newaxis, ...]
    
    # Generate reconstruction
    reconstructed = autoencoder.predict(test_input)
    
    # Remove batch dimension for plotting
    reconstructed = reconstructed[0]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original
    im1 = ax1.imshow(test_image[..., 0], cmap='gray')
    ax1.set_title("Original Image")
    plt.colorbar(im1, ax=ax1)
    
    # Plot reconstruction
    im2 = ax2.imshow(reconstructed[..., 0], cmap='gray')
    ax2.set_title("Reconstructed Image")
    plt.colorbar(im2, ax=ax2)
    
    # Add reconstruction error as text
    mse = np.mean((test_image - reconstructed) ** 2)
    plt.figtext(0.5, 0.01, f'MSE: {mse:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    return reconstructed, mse

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
    # Initialize environment and agents
    
    # Initialize environment and agents
    # env = SimpleEnv(size=10)  # Environment size 10x10
    # successor_agent = SuccessorAgent(env)
    # autoencoder = build_autoencoder((env.size, env.size, 1))  # Matching environment size
    # autoencoder.compile(optimizer='adam', loss=focal_mse_loss)

    # # Run training
    # rewards, trained_autoencoder = train_agent_with_vision(
    #     successor_agent, 
    #     env, 
    #     autoencoder,
    #     n_episodes=1000,
    #     max_steps = 10000
    # )

    # # Save trained model
    # trained_autoencoder.save('trained_autoencoder.h5')

    # Create a test image (example: random rewards)
    size = 10  # Match your environment size
    test_image = np.zeros((size, size, 1))
    # Place some random rewards (1s) in the image
    n_rewards = 2
    random_positions = np.random.choice(size*size, n_rewards, replace=False)
    for pos in random_positions:
        x, y = pos // size, pos % size
        test_image[x, y, 0] = 1

    # Test the autoencoder
    reconstructed, mse = test_autoencoder(
        model_path='trained_autoencoder.h5',
        test_image=test_image,
        save_path='reconstruction_test.png'
    )
    print(f"Reconstruction MSE: {mse:.4f}")

    

if __name__ == "__main__":
    main()