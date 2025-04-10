import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf
from minigrid.core.world_object import Goal, Wall

from tqdm import tqdm
from env import SimpleEnv, data_collector
from models import build_autoencoder, focal_mse_loss, load_trained_autoencoder
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

def train_successor_agent(agent, env, episodes=500, ae_model=None, max_steps=100, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, train_vision_threshold = 0.15):
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
    
    for episode in tqdm(range(episodes), "Training Successor Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0

        # For every new episode, the agent should have no understanding of what the reward space looks like
        # agent.true_reward_map = np.zeros((env.grid_size, env.grid_size))

        # For every new episode, the agent should understand as much of the reward space as it's vision can predict
        # I dont think changing this actually makes a difference as it's only updated once per episode and not used elsewhere
        agent.true_reward_map = np.zeros((env.size, env.size))
        
        # Store first experience
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]

        # Variable to track where the agent finds the reward, to assist updating agent.true_reward_map
        reward_position = None
        
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
            

            # ------------------Train the vision model for every step if the threshold is not met----------------
            # Update The agents true_reward_map

            agent_position = tuple(env.agent_pos) 
            
            # If the agent achieved a reward, drop a 1 on the current point of it's true reward map. Otherwise drop a 0.
            if done:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 1
            else:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 0

            # Convert the environment grid to the same format used during training
            grid = env.grid.encode()
            normalized_grid = np.zeros_like(grid, dtype=np.float32)
            normalized_grid[grid == 2] = 0.0   # Walls
            normalized_grid[grid == 1] = 0.0   # Open space
            normalized_grid[grid == 8] = 1.0   # Rewards

            # Extract the first channel and reshape for the autoencoder
            input_grid = normalized_grid[..., 0]
            input_grid = np.expand_dims(input_grid, axis=0)  # Add batch dimension (1, height, width)
            input_grid = np.expand_dims(input_grid, axis=-1)  # Add channel dimension (1, height, width, 1)

            # Get the predicted reward map from the AE
            predicted_reward_map = ae_model.predict(input_grid)
            
            trigger_ae_training = False
            # Update the rest of the true_reward_map with AE predictions
            for y in range(agent.true_reward_map.shape[0]):
                for x in range(agent.true_reward_map.shape[1]):
                    if (x, y) != agent_position:  # Skip the reward position
                        # Get the predicted value for this position from the AE
                        predicted_value = predicted_reward_map[0, y, x, 0]
                        agent.true_reward_map[y, x] = predicted_value
                    else:
                        # -------For the AE training threshold calculation------
                        # If the predicted value is threshold different from the true_reward_map: trigger vision model training
                        if abs(predicted_reward_map[0, y, x, 0] - agent.true_reward_map[agent_position[1], agent_position[0]]) > train_vision_threshold:
                            vision_prediction_error = abs(predicted_reward_map[0, y, x, 0] - agent.true_reward_map[agent_position[1], agent_position[0]])
                            trigger_ae_training = True

            # Checking true_reward_map formulation

            # Visualize the agents true reward map for this episode
            # print("True Reward Map:")
            # for row in agent.true_reward_map:
            #     print(" ".join(f"{val:.2f}" for val in row))
            
            # we then look to train the AE on this single step, where the input is the image from the environment and the loss propagation
            # is between this input image and the agents true_reward_map.
            if trigger_ae_training:
                # print("Vision Model Training Triggered with difference:", vision_prediction_error)
    
                # Prepare input for training (same format as prediction input)
                # input_grid is already prepared above

                # Silence AE training in terminal, so we can see RL agent training progress instead
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
                # Alternatively, for older TF versions:
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                # Make sure prediction is silent too
                old_stdout = sys.stdout  # Save current stdout
                sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null

                # Prepare target output (agent's true reward map)
                target = np.expand_dims(agent.true_reward_map, axis=0)  # Add batch dimension
                target = np.expand_dims(target, axis=-1)  # Add channel dimension
                
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
            

            # Reward found, next episode
            if done: 
                break


        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        
        # Print progress
        # if (episode + 1) % 100 == 0:
        #     avg_reward = np.mean(episode_rewards[-100:])
        #     print(f"Episode {episode + 1}/{episodes}")
        #     print(f"Average Reward: {avg_reward:.2f}")
        #     print(f"Epsilon: {epsilon:.3f}")
        #     print(f"Steps: {step_count}")
        #     print("------------------------")
    
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