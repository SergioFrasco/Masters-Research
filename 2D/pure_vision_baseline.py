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

class SpatialAttentionModule(layers.Layer):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv_query = layers.Conv2D(filters // 8, 1, activation='relu')
        self.conv_key = layers.Conv2D(filters // 8, 1, activation='relu')
        self.conv_value = layers.Conv2D(filters, 1, activation='relu')
        self.conv_output = layers.Conv2D(filters, 1, activation='relu')
        self.softmax = layers.Softmax(axis=-1)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Generate query, key, value
        query = self.conv_query(inputs, training=training)
        key = self.conv_key(inputs, training=training)
        value = self.conv_value(inputs, training=training)
        
        # Reshape for attention computation
        query = tf.reshape(query, [batch_size, height * width, -1])
        key = tf.reshape(key, [batch_size, height * width, -1])
        value = tf.reshape(value, [batch_size, height * width, -1])
        
        # Compute attention weights
        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = self.softmax(attention_weights)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, value)
        attended = tf.reshape(attended, [batch_size, height, width, self.filters])
        
        # Final output projection
        output = self.conv_output(attended, training=training)
        
        # Residual connection
        return inputs + output

class SpatialValueCNN(keras.Model):
    """Improved CNN with skip connections and attention mechanisms"""
    
    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        
        # Encoder with skip connections
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.skip1 = layers.Conv2D(32, 1, padding='same', activation='linear')
        
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.skip2 = layers.Conv2D(64, 1, padding='same', activation='linear')
        
        self.conv5 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv6 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.skip3 = layers.Conv2D(128, 1, padding='same', activation='linear')
        
        # Attention mechanism
        self.attention = SpatialAttentionModule(128)
        
        # Decoder with skip connections
        self.deconv1 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.deconv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.skip4 = layers.Conv2D(64, 1, padding='same', activation='linear')
        
        self.deconv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.deconv4 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.skip5 = layers.Conv2D(32, 1, padding='same', activation='linear')
        
        self.deconv5 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.deconv6 = layers.Conv2D(1, 3, padding='same', activation='linear')
        
        # Final skip connection from input
        self.final_skip = layers.Conv2D(1, 1, padding='same', activation='linear')
        
    def call(self, inputs, training=None):
        # Encoder path with skip connections
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(x1, training=training)
        skip1_out = self.skip1(x1, training=training)
        x2 = x2 + skip1_out  # Skip connection
        
        x3 = self.conv3(x2, training=training)
        x4 = self.conv4(x3, training=training)
        skip2_out = self.skip2(x3, training=training)
        x4 = x4 + skip2_out  # Skip connection
        
        x5 = self.conv5(x4, training=training)
        x6 = self.conv6(x5, training=training)
        skip3_out = self.skip3(x5, training=training)
        x6 = x6 + skip3_out  # Skip connection
        
        # Apply attention mechanism
        attended = self.attention(x6, training=training)
        
        # Decoder path with skip connections
        d1 = self.deconv1(attended, training=training)
        d2 = self.deconv2(d1, training=training)
        skip4_out = self.skip4(x4, training=training)  # Skip from encoder
        d2 = d2 + skip4_out  # Skip connection
        
        d3 = self.deconv3(d2, training=training)
        d4 = self.deconv4(d3, training=training)
        skip5_out = self.skip5(x2, training=training)  # Skip from encoder
        d4 = d4 + skip5_out  # Skip connection
        
        d5 = self.deconv5(d4, training=training)
        d6 = self.deconv6(d5, training=training)
        
        # Final skip connection from input
        final_skip = self.final_skip(inputs, training=training)
        value_map = d6 + final_skip
        
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
    """Agent that learns spatial value functions with improved architecture and training"""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.grid_size = env.size
        self.action_size = 3  # left, right, forward
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Training parameters with adaptive learning rate
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.fine_tune_lr = learning_rate * 0.1  # Lower LR for fine-tuning
        self.convergence_threshold = 0.01  # Loss threshold for switching to fine-tuning
        self.fine_tune_mode = False
        
        # Adaptive smoothness penalty
        self.smoothness_weight = 0.01
        self.smoothness_weights = [0.001, 0.005, 0.01, 0.05, 0.1]  # Different values to try
        self.current_smoothness_idx = 2  # Start with 0.01
        self.smoothness_update_freq = 500  # Update every 500 episodes
        
        # Build value network and target network
        self.value_network = SpatialValueCNN(self.grid_size)
        self.target_network = SpatialValueCNN(self.grid_size)
        
        # Initialize networks with dummy input
        dummy_input = tf.zeros((1, self.grid_size, self.grid_size, 3))
        self.value_network(dummy_input)
        self.target_network(dummy_input)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer with learning rate scheduling
        self.optimizer = keras.optimizers.Adam(learning_rate=self.current_learning_rate)
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 32
        self.train_freq = 4  # Train every 4 steps
        self.target_update_freq = 100  # Update target network every 100 training steps
        self.training_step = 0
        
        # Metrics for adaptive training
        self.episode_rewards = []
        self.training_losses = []
        self.recent_losses = deque(maxlen=100)  # Track recent losses for convergence detection
        self.value_predictions = []
    
    def update_learning_rate(self):
        """Update learning rate based on training progress"""
        if len(self.recent_losses) >= 50:
            avg_recent_loss = np.mean(list(self.recent_losses)[-50:])
            
            # Switch to fine-tuning mode if loss is converging
            if avg_recent_loss < self.convergence_threshold and not self.fine_tune_mode:
                self.fine_tune_mode = True
                self.current_learning_rate = self.fine_tune_lr
                self.optimizer.learning_rate.assign(self.current_learning_rate)
                print(f"Switching to fine-tuning mode with LR: {self.current_learning_rate}")
    
    def update_smoothness_weight(self, episode):
        """Experiment with different smoothness penalty weights"""
        if episode % self.smoothness_update_freq == 0 and episode > 0:
            # Cycle through different smoothness weights
            self.current_smoothness_idx = (self.current_smoothness_idx + 1) % len(self.smoothness_weights)
            self.smoothness_weight = self.smoothness_weights[self.current_smoothness_idx]
            print(f"Updated smoothness weight to: {self.smoothness_weight}")
    
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
    
    def get_reward_map(self):
        """Extract reward map from environment"""
        reward_map = np.zeros((self.grid_size, self.grid_size))
        
        # Iterate through all cells in the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.env.grid.get(x, y)
                if cell is not None:
                    # Check if it's a goal or reward-giving object
                    if hasattr(cell, 'type') and cell.type == 'goal':
                        reward_map[x, y] = 1.0  # Goal gives positive reward
                    elif hasattr(cell, 'type') and cell.type == 'lava':
                        reward_map[x, y] = -1.0  # Lava gives negative reward
                    elif hasattr(cell, 'reward'):
                        reward_map[x, y] = cell.reward
        
        return reward_map
    
    def get_environment_layout(self):
        """Get the static environment layout (walls, goals, etc.)"""
        layout = np.zeros((self.grid_size, self.grid_size))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.env.grid.get(x, y)
                if cell is not None:
                    if hasattr(cell, 'type'):
                        if cell.type == 'wall':
                            layout[x, y] = -1  # Walls
                        elif cell.type == 'goal':
                            layout[x, y] = 1   # Goals
                        elif cell.type == 'lava':
                            layout[x, y] = -0.5  # Lava/obstacles
        
        return layout
    
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
        """Perform one training step with adaptive regularization"""
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
            
            # Enhanced spatial regularization
            # Spatial smoothness penalty (gradient-based)
            dx = current_value_maps[:, 1:, :] - current_value_maps[:, :-1, :]
            dy = current_value_maps[:, :, 1:] - current_value_maps[:, :, :-1]
            smoothness_loss = tf.reduce_mean(tf.square(dx)) + tf.reduce_mean(tf.square(dy))
            
            # Second-order smoothness (Laplacian)
            laplacian_x = dx[:, 1:, :] - dx[:, :-1, :]
            laplacian_y = dy[:, :, 1:] - dy[:, :, :-1]
            second_order_smoothness = tf.reduce_mean(tf.square(laplacian_x)) + tf.reduce_mean(tf.square(laplacian_y))
            
            # Total loss with adaptive regularization
            total_loss = td_error + self.smoothness_weight * (smoothness_loss + 0.1 * second_order_smoothness)
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        
        self.training_step += 1
        
        # Track loss for adaptive learning rate
        self.recent_losses.append(total_loss.numpy())
        
        # Update learning rate based on convergence
        self.update_learning_rate()
        
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
        self.learning_rates = []
        self.smoothness_weights = []
        self.value_map_snapshots = []
        
    def log_episode(self, episode, reward, length, loss, agent):
        """Log episode results with additional metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.learning_rates.append(agent.current_learning_rate)
        self.smoothness_weights.append(agent.smoothness_weight)
        
        if loss > 0:
            self.training_losses.append(loss)
        
        # Update smoothness weight periodically
        agent.update_smoothness_weight(episode)
        
        # Save value map snapshot periodically
        if episode % 100 == 0:
            value_map = agent.get_current_value_map()
            reward_map = agent.get_reward_map()
            env_layout = agent.get_environment_layout()
            
            self.value_map_snapshots.append({
                'episode': episode,
                'value_map': value_map.copy(),
                'reward_map': reward_map.copy(),
                'env_layout': env_layout.copy(),
                'agent_pos': agent.env.agent_pos,
                'learning_rate': agent.current_learning_rate,
                'smoothness_weight': agent.smoothness_weight,
                'fine_tune_mode': agent.fine_tune_mode
            })
        
        # Print progress with additional info
        if episode % 50 == 0:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            mode = "Fine-tune" if agent.fine_tune_mode else "Initial"
            print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                  f"Length: {length}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.3f}, "
                  f"LR: {agent.current_learning_rate:.6f}, Smoothness: {agent.smoothness_weight:.3f}, "
                  f"Mode: {mode}")
    
    def plot_results(self):
        """Plot enhanced training metrics"""
        fig = plt.figure(figsize=(20, 10))
        
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
        
        # Learning rate evolution
        ax4 = plt.subplot(2, 4, 4)
        plt.plot(self.learning_rates, linewidth=2, color='orange')
        plt.title('Learning Rate Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Smoothness weight evolution
        ax5 = plt.subplot(2, 4, 5)
        plt.plot(self.smoothness_weights, linewidth=2, color='green')
        plt.title('Smoothness Weight Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Smoothness Weight')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Value map quality over time (if available)
        if self.value_map_snapshots:
            ax6 = plt.subplot(2, 4, 6)
            episodes = [snap['episode'] for snap in self.value_map_snapshots]
            value_ranges = [np.max(snap['value_map']) - np.min(snap['value_map']) 
                           for snap in self.value_map_snapshots]
            plt.plot(episodes, value_ranges, 'o-', linewidth=2, color='purple')
            plt.title('Value Map Dynamic Range')
            plt.xlabel('Episode')
            plt.ylabel('Value Range')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'enhanced_training_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed comparison plots
        if self.value_map_snapshots:
            self.plot_detailed_comparison()
    
    def plot_detailed_comparison(self):
        """Create detailed comparison plots showing multiple episodes"""
        if not self.value_map_snapshots:
            return
        
        # Get up to 4 snapshots to display
        n_snapshots = min(4, len(self.value_map_snapshots))
        if n_snapshots == 0:
            return
            
        selected_snapshots = self.value_map_snapshots[-n_snapshots:]
        
        # Create figure with 3 rows and n_snapshots columns
        fig = plt.figure(figsize=(6 * n_snapshots, 18))
        
        for i, snapshot in enumerate(selected_snapshots):
            episode_num = snapshot['episode']
            agent_x, agent_y = snapshot['agent_pos']
            env_layout = snapshot['env_layout']
            reward_map = snapshot['reward_map']
            value_map = snapshot['value_map']
            lr = snapshot['learning_rate']
            smooth_weight = snapshot['smoothness_weight']
            fine_tune = snapshot['fine_tune_mode']
            
            # Masks for different environment elements
            walls_mask = env_layout == -1
            goals_mask = env_layout == 1
            obstacles_mask = env_layout == -0.5
            
            # Row 1: Environment Layout
            ax1 = plt.subplot(3, n_snapshots, i + 1)
            im1 = ax1.imshow(env_layout, cmap='RdYlBu', alpha=0.8)
            
            # Mark agent position
            ax1.scatter(agent_y, agent_x, c='orange', s=150, marker='o', 
                       edgecolors='white', linewidth=3, label='Agent')
            
            mode_text = "Fine-tune" if fine_tune else "Initial"
            ax1.set_title(f'Environment Layout\n(Episode {episode_num}, {mode_text})', 
                         fontsize=12, fontweight='bold')
            ax1.legend()
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Row 2: Reward Space
            ax2 = plt.subplot(3, n_snapshots, n_snapshots + i + 1)
            im2 = ax2.imshow(reward_map, cmap='RdYlGn', alpha=0.8, vmin=-1, vmax=1)
            
            # Overlay walls
            if np.any(walls_mask):
                ax2.imshow(np.where(walls_mask, 0.5, np.nan), cmap='gray', alpha=0.9, vmin=0, vmax=1)
            
            # Mark agent position
            ax2.scatter(agent_y, agent_x, c='orange', s=150, marker='o', 
                       edgecolors='white', linewidth=3, label='Agent')
            
            ax2.set_title(f'Reward Space\n(LR: {lr:.6f}, Smooth: {smooth_weight:.3f})', 
                         fontsize=12, fontweight='bold')
            ax2.legend()
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # Row 3: Learned Value Map
            ax3 = plt.subplot(3, n_snapshots, 2 * n_snapshots + i + 1)
            im3 = ax3.imshow(value_map, cmap='viridis', alpha=0.9)
            
            # Overlay environment structure
            if np.any(walls_mask):
                ax3.imshow(np.where(walls_mask, 1, np.nan), cmap='gray', alpha=0.7, vmin=0, vmax=1)
            
            # Mark goals with red squares
            if np.any(goals_mask):
                goal_positions = np.where(goals_mask)
                for gx, gy in zip(goal_positions[0], goal_positions[1]):
                    ax3.scatter(gy, gx, c='red', s=200, marker='s', alpha=0.8, 
                              edgecolors='white', linewidth=2)
            
            # Mark agent position
            ax3.scatter(agent_y, agent_x, c='orange', s=150, marker='o', 
                       edgecolors='white', linewidth=3, label='Agent')
            
            value_range = np.max(value_map) - np.min(value_map)
            ax3.set_title(f'Learned Value Map\n(Range: {value_range:.3f})', 
                         fontsize=12, fontweight='bold')
            ax3.legend()
            plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'enhanced_detailed_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def run_spatial_value_experiment(env_size=10, episodes=3000):
    """Run the enhanced spatial value function experiment"""
    
    # Initialize environment and agent
    env = SimpleEnv(size=env_size)
    agent = SpatialValueAgent(env, learning_rate=0.001, gamma=0.99)
    monitor = SpatialValueMonitor()
    
    print("Starting Enhanced Spatial Value Function Learning...")
    print(f"Environment size: {env_size}x{env_size}")
    print(f"Training episodes: {episodes}")
    print(f"Architecture: Skip connections + Spatial attention")
    print(f"Training: Adaptive learning rate + Variable smoothness penalty")
    
    for episode in tqdm(range(episodes), desc="Training Enhanced Spatial Value Agent"):
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
            agent.save_model(os.path.join(monitor.save_dir, f'enhanced_spatial_value_model_{episode}.h5'))
    
    # Final visualization
    monitor.plot_results()
    
    # Save final model
    agent.save_model(os.path.join(monitor.save_dir, 'enhanced_spatial_value_model_final.h5'))
    
    print(f"\nTraining completed!")
    print(f"Final learning rate: {agent.current_learning_rate:.6f}")
    print(f"Final smoothness weight: {agent.smoothness_weight:.3f}")
    print(f"Fine-tuning mode: {'Yes' if agent.fine_tune_mode else 'No'}")
    
    return monitor, agent

# Run the enhanced experiment
if __name__ == "__main__":
    monitor, agent = run_spatial_value_experiment(env_size=10, episodes=6001)