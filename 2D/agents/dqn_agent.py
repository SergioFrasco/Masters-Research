import numpy as np
import tensorflow as tf
from collections import deque
import random
import numpy as np

class VisionDQNAgent:
    """
    Vision-based Deep Q-Network agent for fair comparison with Successor agent.
    Processes raw grid observations instead of getting privileged goal position info.
    """
    
    def __init__(self, env, action_size=3, learning_rate=0.001, 
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=100):
        
        self.env = env
        self.grid_size = env.size
        self.action_size = action_size  # [turn_left, turn_right, move_forward]
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_vision_network()
        self.target_network = self._build_vision_network()
        self.update_target_network()
        
        # Training tracking
        self.training_step = 0
    
    def _build_vision_network(self):
        """
        Build a convolutional neural network for processing grid observations.
        
        Architecture:
        Grid Input (10x10x1) → Conv layers → Flatten → Combine with agent_dir → Dense → Q-values
        """
        # Grid input branch
        grid_input = tf.keras.layers.Input(shape=(self.grid_size, self.grid_size, 2), name='grid_input')
        
        # Convolutional layers for spatial feature extraction
        # Use small kernels and multiple layers to build up receptive field
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        
        # Optional: Add another conv layer for more complex patterns
        conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        
        # Flatten spatial features
        flattened = tf.keras.layers.Flatten()(conv3)
        
        # Agent direction input (non-spatial information)
        dir_input = tf.keras.layers.Input(shape=(1,), name='direction_input')
        
        # Combine spatial and directional information
        combined = tf.keras.layers.concatenate([flattened, dir_input])
        
        # Fully connected layers for decision making
        dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        
        # Output layer: Q-values for each action
        q_values = tf.keras.layers.Dense(self.action_size, activation='linear', name='q_values')(dense2)
        
        # Create the model
        model = tf.keras.Model(inputs=[grid_input, dir_input], outputs=q_values)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                    loss='mse')
        
        return model

    # Old network building method, worked when given explicit goal position
    # def _build_network(self):
    #     """Build the neural network for Q-value approximation."""
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dense(self.action_size, activation='linear')
    #     ])
        
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
    #                  loss='mse')
    #     return model
    
    def get_state_vector(self, obs=None):
        """
        Convert environment state to state vector.
        Returns: [agent_x, agent_y, agent_direction, goal_x, goal_y]
        """
        # Get agent position and direction
        agent_x, agent_y = self.env.agent_pos
        agent_dir = self.env.agent_dir
        
        # Find goal position
        goal_x, goal_y = self._find_goal_position()
        
        # Normalize positions to [0, 1] range
        state = np.array([
            agent_x / (self.grid_size - 1),
            agent_y / (self.grid_size - 1),
            agent_dir / 3.0,  # Direction is 0-3, normalize to [0, 1]
            goal_x / (self.grid_size - 1),
            goal_y / (self.grid_size - 1)
        ], dtype=np.float32)
        
        return state
    
    def get_vision_state(self, obs=None):
        """
        Get vision-based state representation for fair comparison with Successor agent.
        
        Returns:
            dict with:
                'grid': normalized grid observation (height, width, 2) - includes agent position
                'agent_dir': normalized agent direction (scalar)
        """
        # Get the raw grid encoding from environment
        grid = self.env.grid.encode()
        
        # Extract object layer (first channel contains object types)
        object_layer = grid[..., 0].astype(np.float32)
        
        # Create two-channel representation
        # Channel 0: Environment objects (walls, goals)
        # Channel 1: Agent position
        
        env_channel = np.zeros_like(object_layer, dtype=np.float32)
        agent_channel = np.zeros_like(object_layer, dtype=np.float32)
        
        # Environment channel - same normalization as Successor agent
        env_channel[object_layer == 2] = 0.0  # Wall → 0
        env_channel[object_layer == 1] = 0.0  # Open space → 0  
        env_channel[object_layer == 8] = 1.0  # Goal → 1
        
        # Agent channel - mark agent position
        agent_x, agent_y = self.env.agent_pos
        agent_channel[agent_y, agent_x] = 1.0  # Agent → 1
        
        # Stack channels: (height, width, 2)
        grid_input = np.stack([env_channel, agent_channel], axis=-1)
        
        # Get agent direction (still useful non-visual info)
        agent_dir = self.env.agent_dir / 3.0  # Normalize to [0, 1]
        
        return {
            'grid': grid_input,
            'agent_dir': agent_dir
        }

    def _find_goal_position(self):
        """
        Find goal position - keep this method but only use it for debugging/analysis.
        The network should NOT have access to this information.
        """
        grid = self.env.grid.encode()
        object_layer = grid[..., 0]
        
        goal_positions = np.where(object_layer == 8)
        if len(goal_positions[0]) > 0:
            return goal_positions[1][0], goal_positions[0][0]  # x, y
        else:
            return self.grid_size // 2, self.grid_size // 2
        
    def get_action(self, state_dict, epsilon=None):
        """
        Choose action using epsilon-greedy policy with vision-based state.
        
        Args:
            state_dict: Dictionary with 'grid' and 'agent_dir' keys
            epsilon: Exploration rate (uses self.epsilon if None)
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        else:
            # Prepare inputs for the network
            grid_input = state_dict['grid'][np.newaxis, ...]  # Add batch dimension
            dir_input = np.array([[state_dict['agent_dir']]])  # Shape: (1, 1)
            
            # Get Q-values from network
            q_values = self.q_network.predict([grid_input, dir_input], verbose=0)
            return np.argmax(q_values[0])
    
    def remember(self, state_dict, action, reward, next_state_dict, done):
        """
        Store experience in replay buffer.
        Now stores state dictionaries instead of simple vectors.
        """
        self.memory.append((state_dict, action, reward, next_state_dict, done))
    
    def replay(self):
        """
        Train the network on a batch of experiences.
        Updated to handle vision-based states.
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate batch into components
        state_dicts = [e[0] for e in batch]
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_state_dicts = [e[3] for e in batch]
        dones = np.array([e[4] for e in batch])
        
        # Prepare batch inputs for current states
        grid_batch = np.array([s['grid'] for s in state_dicts])
        dir_batch = np.array([[s['agent_dir']] for s in state_dicts])
        
        # Prepare batch inputs for next states  
        next_grid_batch = np.array([s['grid'] for s in next_state_dicts])
        next_dir_batch = np.array([[s['agent_dir']] for s in next_state_dicts])
        
        # Current Q-values
        current_q_values = self.q_network.predict([grid_batch, dir_batch], verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict([next_grid_batch, next_dir_batch], verbose=0)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        history = self.q_network.fit([grid_batch, dir_batch], targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss, None
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the trained model."""
        self.q_network.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.q_network = tf.keras.models.load_model(filepath)
        self.update_target_network()
    
    def get_q_values(self, state_dict):
        """Get Q-values for all actions in given state."""
        grid_input = state_dict['grid'][np.newaxis, ...]
        dir_input = np.array([[state_dict['agent_dir']]])
        return self.q_network.predict([grid_input, dir_input], verbose=0)[0]

