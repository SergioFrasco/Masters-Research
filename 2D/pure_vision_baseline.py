import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from collections import deque
import random
from env import SimpleEnv
from utils.plotting import generate_save_path
import json
import time

class SpatialValueCNN(keras.Model):
    """CNN that predicts value function for every cell in the environment"""
    
    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        
        # Encoder: Extract features from environment image
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
        ])
        
        # Decoder: Produce dense value map
        self.decoder = keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(1, 3, padding='same', activation='linear'),  # Output value map
        ])
        
        # Optional: Add skip connections for better spatial resolution
        self.skip_conv = layers.Conv2D(1, 1, padding='same', activation='linear')
        
    def call(self, inputs, training=None):
        # Encode environment features
        encoded = self.encoder(inputs, training=training)
        
        # Skip connection from input
        skip = self.skip_conv(inputs, training=training)
        
        # Decode to value map
        decoded = self.decoder(encoded, training=training)
        
        # Combine with skip connection
        value_map = decoded + skip
        
        return tf.squeeze(value_map, axis=-1)  # Remove channel dimension -> [B, H, W]

class ExperienceReplayBuffer:
    """Experience replay buffer for spatial value learning"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        """Add experience tuple: (image, agent_pos, reward, next_image, next_agent_pos, done)"""
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and stack experiences
        images, agent_positions, rewards, next_images, next_agent_positions, dones = zip(*batch)
        
        return {
            'images': np.array(images),
            'agent_positions': np.array(agent_positions),
            'rewards': np.array(rewards),
            'next_images': np.array(next_images),
            'next_agent_positions': np.array(next_agent_positions),
            'dones': np.array(dones)
        }
    
    def __len__(self):
        return len(self.buffer)

class SpatialValueAgent:
    """Agent that learns spatial value functions over the entire environment"""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.grid_size = env.size
        self.action_size = 3  # left, right, forward
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Build value network and target network
        self.value_network = SpatialValueCNN(self.grid_size)
        self.target_network = SpatialValueCNN(self.grid_size)
        
        # Initialize networks with dummy input
        dummy_input = tf.zeros((1, self.grid_size, self.grid_size, 3))
        self.value_network(dummy_input)
        self.target_network(dummy_input)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 32
        self.train_freq = 4  # Train every 4 steps
        self.target_update_freq = 100  # Update target network every 100 training steps
        self.training_step = 0
        
        # Metrics
        self.episode_rewards = []
        self.training_losses = []
        self.value_predictions = []
        
    def render_env_as_image(self):
        """Convert environment to image representation"""
        grid = self.env.grid.encode()
        
        # Create multi-channel image
        # Channel 0: Objects (walls, goals, etc.)
        # Channel 1: Agent position
        # Channel 2: Agent direction
        
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Object layer - normalize different object types
        object_layer = grid[..., 0]
        image[..., 0] = object_layer / 10.0  # Normalize object IDs
        
        # Agent position layer
        agent_x, agent_y = self.env.agent_pos
        image[agent_x, agent_y, 1] = 1.0
        
        # Agent direction layer
        image[agent_x, agent_y, 2] = self.env.agent_dir / 4.0  # Normalize direction
        
        return image
    
    def get_action(self, obs, use_epsilon=True):
        """Choose action using epsilon-greedy policy based on value map"""
        if use_epsilon and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Get current environment image
        image = self.render_env_as_image()
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict value map
        value_map = self.value_network(image_batch, training=False)[0]  # [H, W]
        
        # Find best action by simulating each action
        current_pos = self.env.agent_pos
        current_dir = self.env.agent_dir
        
        best_action = 0
        best_value = -np.inf
        
        for action in range(self.action_size):
            next_pos, next_dir = self.simulate_action(current_pos, current_dir, action)
            
            if self.is_valid_position(next_pos):
                x, y = next_pos
                predicted_value = value_map[x, y].numpy()
                
                if predicted_value > best_value:
                    best_value = predicted_value
                    best_action = action
        
        return best_action
    
    def simulate_action(self, pos, direction, action):
        """Simulate what position/direction we'd be in after taking an action"""
        x, y = pos
        
        if action == 0:  # Turn left
            new_dir = (direction - 1) % 4
            return (x, y), new_dir
        elif action == 1:  # Turn right
            new_dir = (direction + 1) % 4
            return (x, y), new_dir
        else:  # Move forward
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            return (x + dx, y + dy), direction
    
    def is_valid_position(self, pos):
        """Check if position is valid"""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        cell = self.env.grid.get(x, y)
        from minigrid.core.world_object import Wall
        return cell is None or not isinstance(cell, Wall)
    
    def store_experience(self, image, agent_pos, reward, next_image, next_agent_pos, done):
        """Store experience in replay buffer"""
        experience = (image, agent_pos, reward, next_image, next_agent_pos, done)
        self.replay_buffer.push(experience)
    
    def train_step(self):
        """Perform one training step using TD(0) updates"""
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0
        
        images = tf.constant(batch['images'])
        agent_positions = batch['agent_positions']
        rewards = tf.constant(batch['rewards'], dtype=tf.float32)
        next_images = tf.constant(batch['next_images'])
        next_agent_positions = batch['next_agent_positions']
        dones = tf.constant(batch['dones'], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Predict current value maps
            current_value_maps = self.value_network(images, training=True)  # [B, H, W]
            
            # Predict next value maps using target network
            next_value_maps = self.target_network(next_images, training=False)  # [B, H, W]
            
            # Extract values at agent positions
            batch_indices = tf.range(self.batch_size)
            
            # Current state values V(s)
            current_positions = tf.stack([
                batch_indices,
                tf.constant([pos[0] for pos in agent_positions]),
                tf.constant([pos[1] for pos in agent_positions])
            ], axis=1)
            current_values = tf.gather_nd(current_value_maps, current_positions)
            
            # Next state values V(s')
            next_positions = tf.stack([
                batch_indices,
                tf.constant([pos[0] for pos in next_agent_positions]),
                tf.constant([pos[1] for pos in next_agent_positions])
            ], axis=1)
            next_values = tf.gather_nd(next_value_maps, next_positions)
            
            # Compute TD targets: r + γ * V(s') * (1 - done)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            
            # Compute TD error
            td_error = tf.reduce_mean(tf.square(current_values - td_targets))
            
            # Optional: Add regularization to encourage smooth value maps
            # Spatial smoothness penalty
            dx = current_value_maps[:, 1:, :] - current_value_maps[:, :-1, :]
            dy = current_value_maps[:, :, 1:] - current_value_maps[:, :, :-1]
            smoothness_loss = tf.reduce_mean(tf.square(dx)) + tf.reduce_mean(tf.square(dy))
            
            total_loss = td_error + 0.01 * smoothness_loss
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return total_loss.numpy()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.value_network.get_weights())
    
    def get_current_value_map(self):
        """Get current value map prediction for visualization"""
        image = self.render_env_as_image()
        image_batch = np.expand_dims(image, axis=0)
        value_map = self.value_network(image_batch, training=False)[0]
        return value_map.numpy()
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
        self.value_network.save_weights(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
        self.value_network.load_weights(filepath)
        self.update_target_network()

class SpatialValueMonitor:
    """Monitor training progress and visualize value maps"""
    
    def __init__(self, save_dir="spatial_value_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.value_map_snapshots = []
        
    def log_episode(self, episode, reward, length, loss, agent):
        """Log episode results"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss > 0:
            self.training_losses.append(loss)
        
        # Save value map snapshot periodically
        if episode % 100 == 0:
            value_map = agent.get_current_value_map()
            self.value_map_snapshots.append({
                'episode': episode,
                'value_map': value_map.copy(),
                'agent_pos': agent.env.agent_pos
            })
        
        # Print progress
        if episode % 50 == 0:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                  f"Length: {length}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    def plot_results(self):
        """Plot comprehensive training results"""
        fig = plt.figure(figsize=(20, 12))
        
        # Episode rewards
        ax1 = plt.subplot(2, 4, 1)
        plt.plot(self.episode_rewards, alpha=0.3, label='Episode Rewards')
        if len(self.episode_rewards) >= 50:
            smoothed = pd.Series(self.episode_rewards).rolling(50).mean()
            plt.plot(smoothed, linewidth=2, label='Smoothed')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training loss
        ax2 = plt.subplot(2, 4, 2)
        if self.training_losses:
            plt.plot(self.training_losses, alpha=0.7)
            plt.title('Training Loss (TD Error)')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        # Episode lengths
        ax3 = plt.subplot(2, 4, 3)
        plt.plot(self.episode_lengths, alpha=0.3, label='Episode Length')
        if len(self.episode_lengths) >= 50:
            smoothed = pd.Series(self.episode_lengths).rolling(50).mean()
            plt.plot(smoothed, linewidth=2, label='Smoothed')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Value map evolution
        if self.value_map_snapshots:
            n_snapshots = min(4, len(self.value_map_snapshots))
            
            for i, snapshot in enumerate(self.value_map_snapshots[-n_snapshots:]):
                ax = plt.subplot(2, 4, 4 + i + 1)
                im = plt.imshow(snapshot['value_map'], cmap='viridis')
                
                # Mark agent position
                agent_x, agent_y = snapshot['agent_pos']
                plt.scatter(agent_y, agent_x, c='red', s=100, marker='o', edgecolors='white', linewidth=2)
                
                plt.title(f'Value Map\nEpisode {snapshot["episode"]}')
                plt.colorbar(im, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
        plt.show()

def run_spatial_value_experiment(env_size=10, episodes=3000):
    """Run the spatial value function experiment"""
    
    # Initialize environment and agent
    env = SimpleEnv(size=env_size)
    agent = SpatialValueAgent(env, learning_rate=0.001, gamma=0.99)
    monitor = SpatialValueMonitor()
    
    print("Starting Spatial Value Function Learning...")
    print(f"Environment size: {env_size}x{env_size}")
    print(f"Training episodes: {episodes}")
    
    for episode in tqdm(range(episodes), desc="Training Spatial Value Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        episode_loss = 0
        loss_count = 0
        
        # Get initial state
        current_image = agent.render_env_as_image()
        current_pos = env.agent_pos
        
        for step in range(200):  # Max steps per episode
            # Choose action
            action = agent.get_action(obs)
            
            # Take action
            next_obs, reward, done, _, _ = env.step(action)
            
            # Get next state
            next_image = agent.render_env_as_image()
            next_pos = env.agent_pos
            
            # Store experience
            agent.store_experience(current_image, current_pos, reward, next_image, next_pos, done)
            
            # Train if we have enough experiences
            if len(agent.replay_buffer) >= agent.batch_size and step % agent.train_freq == 0:
                loss = agent.train_step()
                episode_loss += loss
                loss_count += 1
            
            # Update state
            current_image = next_image
            current_pos = next_pos
            obs = next_obs
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Update exploration
        agent.update_epsilon()
        
        # Log episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        monitor.log_episode(episode, total_reward, step_count, avg_loss, agent)
        
        # Save model periodically
        if episode % 500 == 0 and episode > 0:
            agent.save_model(os.path.join(monitor.save_dir, f'spatial_value_model_{episode}.h5'))
    
    # Final visualization
    monitor.plot_results()
    
    # Save final model
    agent.save_model(os.path.join(monitor.save_dir, 'spatial_value_model_final.h5'))
    
    return monitor, agent

# Run the experiment
if __name__ == "__main__":
    monitor, agent = run_spatial_value_experiment(env_size=10, episodes=4001)