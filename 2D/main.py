import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf

from minigrid.core.world_object import Goal, Wall
from tqdm import tqdm
from env import SimpleEnv, data_collector
from models import build_autoencoder, focal_mse_loss, load_trained_autoencoder, weighted_focal_mse_loss
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

def train_successor_agent(agent, env, episodes=500, ae_model=None, max_steps=100, epsilon_start=1.0, 
                         epsilon_end=0.01, epsilon_decay=0.995, train_vision_threshold=0.1):
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

        # For every new episode, reset the reward map
        # if episode == 0:  # Only initialize once at the beginning
        agent.true_reward_map = np.zeros((env.size, env.size))
        agent.true_reward_map_explored = np.zeros((env.size, env.size))
        
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
            
            # ------------------Train the vision model----------------
            # Update the agent's true_reward_map based on current observation
            agent_position = tuple(env.agent_pos)

            # Transform coordinates to match the normalized grid transformation
            # If normalized_grid uses a flipped and rotated coordinate system:
            transformed_y = env.size - 1 - agent_position[1]  # Flip Y
            transformed_x = agent_position[0]  # Keep X as is (adjust as needed)
            
            # Get the current environment grid
            grid = env.grid.encode()
            normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)
            
            # Object types are in grid[..., 0]
            object_layer = grid[..., 0]
            normalized_grid[object_layer == 2] = 0.0   # Wall
            normalized_grid[object_layer == 1] = 0.0   # Open space
            normalized_grid[object_layer == 8] = 1.0   # Reward (e.g. goal object)
            
            # Rotate the grid to match render_mode = human 
            # normalized_grid = np.flipud(normalized_grid)
            # normalized_grid = np.rot90(normalized_grid, k=-1)
            
            # Reshape for the autoencoder (add batch and channel dims)
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
            
            # Get the predicted reward map from the AE
            predicted_reward_map = ae_model.predict(input_grid, verbose=0)
            predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]
            
            # Calculate the prediction error for positions the agent has visited
            if np.any(agent.true_reward_map_explored):
                prediction_errors = []
                for y, x in zip(*np.where(agent.true_reward_map_explored)):
                    pred = predicted_reward_map_2d[y, x]
                    true = agent.true_reward_map[y, x]
                    prediction_errors.append(abs(pred - true))
                
                mean_error = np.mean(prediction_errors)
                max_error = np.max(prediction_errors)
                
                # Decide whether to train based on error statistics
                trigger_ae_training = max_error > train_vision_threshold
                
                if trigger_ae_training:
                    if episode % 50 == 0:  # Reduce verbosity
                        print(f"AE Training triggered - Mean error: {mean_error:.4f}, Max error: {max_error:.4f}")
                    
                    # Prepare target for training (agent's true reward map)
                    target = np.expand_dims(agent.true_reward_map, axis=0)  # Add batch dimension
                    target = np.expand_dims(target, axis=-1)  # Add channel dimension
                    
                    # Create a training mask based on explored positions
                    mask = np.expand_dims(agent.true_reward_map_explored, axis=0)
                    mask = np.expand_dims(mask, axis=-1)
                    
                    # Custom training step with weighted loss based on exploration mask
                    with tf.GradientTape() as tape:
                        predictions = ae_model(input_grid, training=True)
                        # Apply exploration mask to focus training on visited positions
                        masked_y_true = target * mask
                        masked_y_pred = predictions * mask
                        # Use weighted loss function to emphasize reward pixels
                        loss_value = weighted_focal_mse_loss(masked_y_true, masked_y_pred)
                    
                    # Get gradients and apply them
                    grads = tape.gradient(loss_value, ae_model.trainable_weights)
                    ae_model.optimizer.apply_gradients(zip(grads, ae_model.trainable_weights))
            
            # Reward found, next episode
            if done:
                break
                
        # Generate visualizations occasionally
        if episode % 1 == 0:

            print("Actual: \n", normalized_grid)
            print("Agents Guess: \n", agent.true_reward_map)
            # print("Input Grid: \n", input_grid)
            # Create visualization of current state
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grid (environment)
            ax1.imshow(normalized_grid, cmap='gray')
            ax1.set_title("Environment Grid")
            
            # Agent's true reward map
            ax2.imshow(agent.true_reward_map, cmap='viridis')
            # Overlay dots on positions the agent has actually visited
            visited_y, visited_x = np.where(agent.true_reward_map_explored)
            ax2.scatter(visited_x, visited_y, color='red', s=5)
            ax2.set_title("Agent's Reward Map (red=visited)")
            
            # AE prediction
            ax3.imshow(predicted_reward_map_2d, cmap='viridis')
            ax3.set_title("AE Prediction")
            
            plt.tight_layout()
            plt.savefig(f'results/episode_{episode}.png')
            plt.close()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        
        # Print progress occasionally
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Steps: {step_count}")
            print("------------------------")
    
    ae_model.save('results/current/vision_model.h5')
    print("Training Complete, Vision Model Saved!")
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

if __name__ == "__main__":
    main()