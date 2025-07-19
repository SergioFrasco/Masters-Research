import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Configure TensorFlow for maximum memory efficiency
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# GPU memory configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Limit GPU memory usage
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

class OptimizedDQNAgent:
    """
    Heavily optimized DQN Agent for memory-constrained environments
    """
    
    def __init__(self, env, state_shape=(10, 10, 1), action_size=3):
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.grid_size = env.size
        
        # Ultra-conservative hyperparameters
        self.memory = deque(maxlen=500)  # Very small replay buffer
        self.epsilon = 0.9  # Start lower
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.01  # Higher learning rate for faster convergence
        self.gamma = 0.9  # Slightly lower discount
        self.batch_size = 8  # Very small batch size
        self.target_update_freq = 100
        self.train_freq = 8  # Train less frequently
        
        # Build simpler model
        self.q_network = self._build_simple_model()
        self.target_network = self._build_simple_model()
        self.update_target_network()
        
        self.training_step = 0
        self.episode_count = 0
        
    def _build_simple_model(self):
        """Build ultra-lightweight model"""
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=self.state_shape),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        # Use SGD instead of Adam for memory efficiency
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def get_simple_state(self):
        """Create minimal state representation"""
        # Get basic environment info
        agent_x, agent_y = self.env.agent_pos
        
        # Create simple state vector instead of full grid
        state = np.zeros(self.state_shape, dtype=np.float32)
        
        # Simplified representation - just mark agent position
        state[agent_y, agent_x, 0] = 1.0
        
        # Add goal information if available
        try:
            goal_pos = self.env.goal_pos
            if goal_pos:
                goal_x, goal_y = goal_pos
                state[goal_y, goal_x, 0] = 0.5
        except:
            pass
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with memory management"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_batch = np.expand_dims(state, 0)
        q_values = self.q_network.predict(state_batch, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Simplified training with aggressive memory management"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample and convert to numpy arrays immediately
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get predictions
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train with single epoch
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        # Clear variables and force garbage collection
        del states, actions, rewards, next_states, dones
        del current_q_values, next_q_values
        
        if self.training_step % 50 == 0:
            gc.collect()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath):
        """Save model"""
        self.q_network.save(filepath)


def train_optimized_agent(agent, env, episodes=500, max_steps=100):
    """
    Ultra-efficient training loop
    """
    scores = []
    successful_episodes = 0
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        # Reset environment
        obs = env.reset()
        state = agent.get_simple_state()
        
        total_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # Choose and execute action
            action = agent.act(state)
            next_obs, reward, done, _, _ = env.step(action)
            next_state = agent.get_simple_state()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Train less frequently
            if step % agent.train_freq == 0 and len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            if done:
                if reward > 0:  # Successful episode
                    successful_episodes += 1
                break
        
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        agent.episode_count += 1
        
        # Print progress less frequently
        if episode % 50 == 0:
            recent_avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            success_rate = successful_episodes / (episode + 1)
            print(f"Episode {episode}: Avg Score={recent_avg:.3f}, "
                  f"Success Rate={success_rate:.2%}, Epsilon={agent.epsilon:.3f}")
            
            # Force cleanup
            gc.collect()
    
    return scores, successful_episodes


def test_agent(agent, env, episodes=20, max_steps=100):
    """
    Quick testing function
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during testing
    
    test_scores = []
    successful_tests = 0
    
    for episode in range(episodes):
        obs = env.reset()
        state = agent.get_simple_state()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            obs, reward, done, _, _ = env.step(action)
            state = agent.get_simple_state()
            total_reward += reward
            
            if done:
                if reward > 0:
                    successful_tests += 1
                break
        
        test_scores.append(total_reward)
    
    agent.epsilon = original_epsilon
    return test_scores, successful_tests


def plot_simple_results(scores, save_path=None):
    """
    Simple plotting function
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Moving average
    window = min(50, len(scores) // 5)
    if window > 1:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(scores)), moving_avg)
        plt.title(f'Moving Average (window={window})')
    else:
        plt.plot(scores)
        plt.title('Scores')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """
    Main training function
    """
    # Import environment
    try:
        from env import SimpleEnv
    except ImportError:
        print("Error: Could not import SimpleEnv. Make sure env.py is in the same directory.")
        return

    print("Optimized DQN Agent - Memory Efficient Version")
    print("=" * 50)
    
    # Create environment and agent
    env = SimpleEnv(size=10)
    agent = OptimizedDQNAgent(env, state_shape=(10, 10, 1), action_size=3)
    
    # Train with fewer episodes
    print("Training agent...")
    scores, successful_episodes = train_optimized_agent(agent, env, episodes=1000, max_steps=100)
    
    # Save model
    try:
        agent.save_model('optimized_dqn_model.h5')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    # Test agent
    print("\nTesting agent...")
    test_scores, successful_tests = test_agent(agent, env, episodes=200, max_steps=100)
    
    # Results
    print(f"\nTraining Results:")
    print(f"- Total Episodes: {len(scores)}")
    print(f"- Successful Episodes: {successful_episodes}")
    print(f"- Success Rate: {successful_episodes/len(scores):.2%}")
    print(f"- Average Score (last 50): {np.mean(scores[-50:]):.3f}")
    
    print(f"\nTesting Results:")
    print(f"- Test Episodes: {len(test_scores)}")
    print(f"- Successful Tests: {successful_tests}")
    print(f"- Test Success Rate: {successful_tests/len(test_scores):.2%}")
    print(f"- Average Test Score: {np.mean(test_scores):.3f}")
    
    # Plot results
    plot_simple_results(scores, 'optimized_dqn_results.png')
    
    print("\nOptimized DQN training complete!")


if __name__ == "__main__":
    main()