import numpy as np
import tensorflow as tf
from collections import deque
import random
import numpy as np

class DQNAgent:
    """
    Deep Q-Network agent for MiniGrid environment.
    
    State representation: [agent_x, agent_y, agent_direction, goal_x, goal_y]
    Action space: [turn_left, turn_right, move_forward] = [0, 1, 2]
    """
    
    def __init__(self, env, state_size=5, action_size=3, learning_rate=0.001, 
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=100):
        
        self.env = env
        self.grid_size = env.size
        self.state_size = state_size  # [agent_x, agent_y, agent_dir, goal_x, goal_y]
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
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Training tracking
        self.training_step = 0
        
    def _build_network(self):
        """Build the neural network for Q-value approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
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
    
    def _find_goal_position(self):
        """Find the goal position in the current environment."""
        grid = self.env.grid.encode()
        object_layer = grid[..., 0]
        
        # Find where goal (object type 8) is located
        goal_positions = np.where(object_layer == 8)
        if len(goal_positions[0]) > 0:
            return goal_positions[1][0], goal_positions[0][0]  # x, y
        else:
            # If no goal found, return center as fallback
            return self.grid_size // 2, self.grid_size // 2
    
    def get_action(self, state, epsilon=None):
        """
        Choose action using epsilon-greedy policy.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        else:
            state = state.reshape(1, -1)
            q_values = self.q_network.predict(state, verbose=0)
            return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate batch into components
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss, None  # Return None for second value to match your interface
    
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
    
    def get_q_values(self, state):
        """Get Q-values for all actions in given state."""
        state = state.reshape(1, -1)
        return self.q_network.predict(state, verbose=0)[0]

