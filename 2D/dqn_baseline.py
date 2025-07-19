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
    Optimized DQN Agent for directional navigation in grid environments
    """
    
    def __init__(self, env, state_shape=(10, 10, 4), action_size=3):
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.grid_size = env.size
        
        # Fixed hyperparameters for directional navigation
        self.memory = deque(maxlen=3000)  # Even larger buffer
        self.epsilon = 0.9  # Start high but not too high
        self.epsilon_min = 0.1  # Keep minimum exploration
        self.epsilon_decay = 0.9995  # Much slower decay - key fix!
        self.learning_rate = 0.0005  # Even lower learning rate
        self.gamma = 0.99  # Higher discount for long-term planning
        self.batch_size = 64  # Larger batch for more stable learning
        self.target_update_freq = 100  # Less frequent target updates
        self.train_freq = 1  # Train every step for more learning
        
        # Build improved model
        self.q_network = self._build_directional_model()
        self.target_network = self._build_directional_model()
        self.update_target_network()
        
        self.training_step = 0
        self.episode_count = 0
        
        # Debug tracking
        self.action_names = ['Turn Left', 'Turn Right', 'Move Forward']
        self.recent_actions = deque(maxlen=20)
        
    def _build_directional_model(self):
        """Build model suitable for directional navigation"""
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.state_shape, padding='same'),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _get_front_position(self, x, y, direction):
        """Get position directly in front of agent based on direction"""
        if direction == 0:  # North
            return x, y - 1
        elif direction == 1:  # East  
            return x + 1, y
        elif direction == 2:  # South
            return x, y + 1
        elif direction == 3:  # West
            return x - 1, y
        return x, y
    
    def get_directional_state(self):
        """Create proper state representation for directional navigation"""
        agent_x, agent_y = self.env.agent_pos
        
        # Get agent direction (assuming env has agent_dir attribute)
        agent_dir = getattr(self.env, 'agent_dir', 0)
        
        # Create multi-channel state representation
        state = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        
        # Channel 0: Agent position
        state[agent_y, agent_x, 0] = 1.0
        
        # Channel 1: Goal position  
        if hasattr(self.env, 'goal_pos') and self.env.goal_pos is not None:
            goal_x, goal_y = self.env.goal_pos
            if 0 <= goal_y < self.grid_size and 0 <= goal_x < self.grid_size:
                state[goal_y, goal_x, 1] = 1.0
        
        # Channel 2: Agent orientation - use one-hot encoding
        # Create direction channels: N, E, S, W
        dir_channel = np.zeros((self.grid_size, self.grid_size))
        if agent_dir == 0:  # North
            dir_channel[max(0, agent_y-1):agent_y+1, agent_x] = 1.0
        elif agent_dir == 1:  # East  
            dir_channel[agent_y, agent_x:min(self.grid_size, agent_x+2)] = 1.0
        elif agent_dir == 2:  # South
            dir_channel[agent_y:min(self.grid_size, agent_y+2), agent_x] = 1.0
        elif agent_dir == 3:  # West
            dir_channel[agent_y, max(0, agent_x-1):agent_x+1] = 1.0
        state[:, :, 2] = dir_channel
        
        # Channel 3: Distance and direction to goal
        if hasattr(self.env, 'goal_pos') and self.env.goal_pos is not None:
            goal_x, goal_y = self.env.goal_pos
            
            # Manhattan distance field to goal
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
                    # Normalize distance (max possible distance in 10x10 grid is 18)
                    normalized_dist = 1.0 - (manhattan_dist / 18.0)
                    state[y, x, 3] = max(0, normalized_dist)
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with memory management"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection with debugging"""
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            state_batch = np.expand_dims(state, 0)
            q_values = self.q_network.predict(state_batch, verbose=0)
            action = np.argmax(q_values[0])
        
        # Track recent actions for debugging
        self.recent_actions.append(action)
        
        return action
    
    def replay(self):
        """Improved training with better memory management"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get predictions
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train with single epoch
        history = self.q_network.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        # Clear variables and force garbage collection
        del states, actions, rewards, next_states, dones
        del current_q_values, next_q_values, targets
        
        if self.training_step % 100 == 0:
            gc.collect()
        
        return loss
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def get_debug_info(self):
        """Get debugging information"""
        if len(self.recent_actions) == 0:
            return "No recent actions"
        
        action_counts = {i: 0 for i in range(self.action_size)}
        for action in self.recent_actions:
            action_counts[action] += 1
        
        debug_info = []
        for action_id, count in action_counts.items():
            percentage = (count / len(self.recent_actions)) * 100
            debug_info.append(f"{self.action_names[action_id]}: {count} ({percentage:.1f}%)")
        
        return " | ".join(debug_info)
    
    def save_model(self, filepath):
        """Save model"""
        self.q_network.save(filepath)


def train_optimized_agent(agent, env, episodes=500, max_steps=200):
    """
    Training loop with comprehensive debugging
    """
    scores = []
    successful_episodes = 0
    losses = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Environment: {env.size}x{env.size} grid")
    print(f"Action space: {agent.action_names}")
    print("=" * 70)
    
    for episode in range(episodes):
        # Reset environment
        obs = env.reset()
        state = agent.get_directional_state()
        
        total_reward = 0
        step_count = 0
        episode_loss = 0
        
        # Store initial positions for debugging
        initial_agent_pos = getattr(env, 'agent_pos', (0, 0))
        initial_goal_pos = getattr(env, 'goal_pos', (0, 0))
        initial_agent_dir = getattr(env, 'agent_dir', 0)
        
        for step in range(max_steps):
            # Choose and execute action
            action = agent.act(state)
            next_obs, reward, done, _, _ = env.step(action)
            next_state = agent.get_directional_state()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train every step for maximum learning
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss > 0:
                    episode_loss += loss
            
            # Update
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                if reward > 0:  # Successful episode
                    successful_episodes += 1
                break
        
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        losses.append(episode_loss / max(1, step_count // agent.train_freq))
        agent.episode_count += 1
        
        # Detailed debug output
        if episode % 10 == 0 or (episode < 50 and episode % 5 == 0):
            recent_avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            success_rate = successful_episodes / (episode + 1)
            recent_success = np.sum([1 for s in scores[-20:] if s > 0]) / min(20, len(scores))
            avg_loss = np.mean(losses[-20:]) if len(losses) >= 20 else (np.mean(losses) if losses else 0)
            
            print(f"\nEpisode {episode:3d}: Score={total_reward:5.1f}, Steps={step_count:3d}")
            print(f"    Agent: {initial_agent_pos} -> {getattr(env, 'agent_pos', 'unknown')}")
            print(f"    Goal:  {initial_goal_pos}, Agent Dir: {initial_agent_dir} -> {getattr(env, 'agent_dir', 'unknown')}")
            print(f"    Avg Score: {recent_avg:5.2f} | Success Rate: {success_rate:5.1%} (Recent: {recent_success:5.1%})")
            print(f"    Epsilon: {agent.epsilon:.4f} | Avg Loss: {avg_loss:.4f}")
            print(f"    Actions: {agent.get_debug_info()}")
            
            # Memory usage info
            print(f"    Memory: {len(agent.memory)}/{agent.memory.maxlen}")
            
            # Check for problematic behavior
            if len(agent.recent_actions) >= 10:
                last_10_actions = list(agent.recent_actions)[-10:]
                if all(a == last_10_actions[0] for a in last_10_actions):
                    print(f"    ⚠️  WARNING: Agent stuck repeating action {agent.action_names[last_10_actions[0]]}")
            
            if episode % 50 == 0:
                gc.collect()
        
        # Early success detection
        if episode >= 100 and episode % 100 == 0:
            recent_100_success = np.sum([1 for s in scores[-100:] if s > 0]) / 100
            if recent_100_success > 0.8:
                print(f"\n🎉 High success rate achieved! {recent_100_success:.1%} over last 100 episodes")
    
    return scores, successful_episodes, losses


def test_agent(agent, env, episodes=20, max_steps=200):
    """
    Testing function with detailed output
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during testing
    
    test_scores = []
    successful_tests = 0
    test_details = []
    
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    for episode in range(episodes):
        obs = env.reset()
        state = agent.get_directional_state()
        total_reward = 0
        steps = 0
        
        initial_pos = getattr(env, 'agent_pos', (0, 0))
        goal_pos = getattr(env, 'goal_pos', (0, 0))
        path = [initial_pos]
        
        for step in range(max_steps):
            action = agent.act(state)
            obs, reward, done, _, _ = env.step(action)
            state = agent.get_directional_state()
            total_reward += reward
            steps += 1
            
            current_pos = getattr(env, 'agent_pos', (0, 0))
            path.append(current_pos)
            
            if done:
                if reward > 0:
                    successful_tests += 1
                break
        
        test_scores.append(total_reward)
        test_details.append({
            'episode': episode,
            'score': total_reward,
            'steps': steps,
            'success': reward > 0,
            'path': path,
            'goal': goal_pos
        })
        
        if episode < 5:  # Show first few test runs in detail
            print(f"Test {episode+1:2d}: {'SUCCESS' if reward > 0 else 'FAILED'} | "
                  f"Score: {total_reward:4.1f} | Steps: {steps:3d}")
            print(f"    Start: {initial_pos} -> Goal: {goal_pos}")
            print(f"    Path length: {len(path)-1}")
    
    agent.epsilon = original_epsilon
    
    # Summary
    print(f"\nTest Summary:")
    print(f"  Success Rate: {successful_tests}/{episodes} ({successful_tests/episodes:.1%})")
    print(f"  Average Score: {np.mean(test_scores):.2f}")
    print(f"  Average Steps: {np.mean([d['steps'] for d in test_details]):.1f}")
    
    return test_scores, successful_tests, test_details


def plot_comprehensive_results(scores, losses, save_path=None):
    """
    Comprehensive plotting function with loss tracking
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training scores
    axes[0,0].plot(scores)
    axes[0,0].set_title('Training Scores Over Time')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].grid(True)
    axes[0,0].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Success (Score=1)')
    axes[0,0].legend()
    
    # Moving average
    window = min(50, len(scores) // 5)
    if window > 1:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[0,1].plot(range(window-1, len(scores)), moving_avg)
        axes[0,1].set_title(f'Moving Average Score (window={window})')
    else:
        axes[0,1].plot(scores)
        axes[0,1].set_title('Scores')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Average Score')
    axes[0,1].grid(True)
    axes[0,1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # Success rate over time
    success_window = 50
    success_rates = []
    for i in range(success_window, len(scores) + 1):
        recent_scores = scores[i-success_window:i]
        success_rate = sum(1 for s in recent_scores if s > 0) / success_window
        success_rates.append(success_rate * 100)
    
    if success_rates:
        axes[1,0].plot(range(success_window, len(scores) + 1), success_rates)
        axes[1,0].set_title(f'Success Rate Over Time (window={success_window})')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Success Rate (%)')
        axes[1,0].grid(True)
        axes[1,0].set_ylim(0, 100)
    
    # Training loss
    if losses and any(l > 0 for l in losses):
        valid_losses = [l for l in losses if l > 0]
        axes[1,1].plot(valid_losses)
        axes[1,1].set_title('Training Loss Over Time')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Average Loss')
        axes[1,1].grid(True)
    else:
        axes[1,1].text(0.5, 0.5, 'No Loss Data Available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Training Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """
    Main training function with comprehensive debugging
    """
    # Import environment
    try:
        from env import SimpleEnv
    except ImportError:
        print("Error: Could not import SimpleEnv. Make sure env.py is in the same directory.")
        return

    print("🤖 Optimized DQN Agent - Directional Navigation Version")
    print("=" * 70)
    
    # Create environment and agent
    env = SimpleEnv(size=10)
    
    # Print environment info
    print(f"Environment created: {env.size}x{env.size} grid")
    if hasattr(env, 'agent_pos'):
        print(f"Agent starting position: {env.agent_pos}")
    if hasattr(env, 'agent_dir'):
        print(f"Agent starting direction: {env.agent_dir}")
    if hasattr(env, 'goal_pos'):
        print(f"Goal position: {env.goal_pos}")
    
    agent = OptimizedDQNAgent(env, state_shape=(10, 10, 4), action_size=3)
    
    print(f"\nAgent Configuration:")
    print(f"  State shape: {agent.state_shape}")
    print(f"  Action space: {agent.action_size} ({', '.join(agent.action_names)})")
    print(f"  Memory buffer: {agent.memory.maxlen}")
    print(f"  Initial epsilon: {agent.epsilon}")
    print(f"  Batch size: {agent.batch_size}")
    
    # Train with better parameters
    print(f"\n🚀 Starting training...")
    scores, successful_episodes, losses = train_optimized_agent(agent, env, episodes=500, max_steps=200)
    
    # Save model
    try:
        agent.save_model('directional_dqn_model.h5')
        print("\n💾 Model saved successfully!")
    except Exception as e:
        print(f"\n❌ Could not save model: {e}")
    
    # Test agent
    print(f"\n🧪 Testing trained agent...")
    test_scores, successful_tests, test_details = test_agent(agent, env, episodes=20, max_steps=200)
    
    # Final Results
    print(f"\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Training Results:")
    print(f"  • Total Episodes: {len(scores)}")
    print(f"  • Successful Episodes: {successful_episodes} ({successful_episodes/len(scores):.1%})")
    print(f"  • Average Score (last 50): {np.mean(scores[-50:]):.3f}")
    print(f"  • Final Epsilon: {agent.epsilon:.3f}")
    
    print(f"\nTesting Results:")
    print(f"  • Test Episodes: {len(test_scores)}")
    print(f"  • Successful Tests: {successful_tests} ({successful_tests/len(test_scores):.1%})")
    print(f"  • Average Test Score: {np.mean(test_scores):.3f}")
    
    # Learning progress analysis
    if len(scores) >= 100:
        early_success = np.sum([1 for s in scores[:100] if s > 0]) / 100
        late_success = np.sum([1 for s in scores[-100:] if s > 0]) / 100
        improvement = late_success - early_success
        print(f"\nLearning Progress:")
        print(f"  • Early success rate (first 100): {early_success:.1%}")
        print(f"  • Late success rate (last 100): {late_success:.1%}")
        print(f"  • Improvement: {improvement:+.1%}")
        
        if improvement > 0.2:
            print("  ✅ Good learning progress detected!")
        elif improvement > 0:
            print("  📈 Some learning progress detected")
        else:
            print("  ⚠️  Limited learning progress - may need tuning")
    
    # Plot results
    plot_comprehensive_results(scores, losses, 'directional_dqn_results.png')
    
    print(f"\n🎉 Training complete! Check the plots for detailed analysis.")


if __name__ == "__main__":
    main()