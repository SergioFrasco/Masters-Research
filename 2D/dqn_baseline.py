import numpy as np
import tensorflow as tf
from collections import deque
import random

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
    

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Assuming your imports work the same way
from env import SimpleEnv
from utils.plotting import generate_save_path
from utils import get_latest_run_dir, create_video_from_images

def evaluate_dqn_performance(agent, env, episode, log_file_path):
    """
    Evaluate DQN performance by checking Q-values at goal vs neighbors.
    Similar to your evaluate_goal_state_values function but for DQN.
    """
    current_pos = env.agent_pos
    x, y = current_pos
    
    # Get current state
    current_state = agent.get_state_vector()
    current_q_values = agent.get_q_values(current_state)
    
    # Get Q-value for staying (this is tricky since there's no "stay" action)
    # We'll use the maximum Q-value as a proxy for the state's desirability
    current_max_q = np.max(current_q_values)
    
    # Check neighboring positions by simulating what Q-values would be
    neighbor_max_qs = []
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    for nx, ny in neighbors:
        if (0 <= nx < agent.grid_size and 0 <= ny < agent.grid_size):
            # Check if it's a valid position
            cell = env.grid.get(nx, ny)
            if cell is None or not hasattr(cell, '__class__') or cell.__class__.__name__ != 'Wall':
                # Simulate state at neighbor position
                goal_x, goal_y = agent._find_goal_position()
                neighbor_state = np.array([
                    nx / (agent.grid_size - 1),
                    ny / (agent.grid_size - 1),
                    current_state[2],  # Same direction
                    goal_x / (agent.grid_size - 1),
                    goal_y / (agent.grid_size - 1)
                ], dtype=np.float32)
                
                neighbor_q_values = agent.get_q_values(neighbor_state)
                neighbor_max_qs.append(np.max(neighbor_q_values))
    
    # Determine if current state has highest Q-value
    if len(neighbor_max_qs) == 0:
        result = f"Episode {episode}: No valid neighbors to compare\n"
    else:
        max_neighbor_q = max(neighbor_max_qs)
        if current_max_q >= max_neighbor_q:
            result = f"Episode {episode}: Goal state has highest Q-value (goal: {current_max_q:.3f}, max neighbor: {max_neighbor_q:.3f})\n"
        else:
            result = f"Episode {episode}: Goal state does not have highest Q-value (goal: {current_max_q:.3f}, max neighbor: {max_neighbor_q:.3f})\n"
    
    # Write to log file
    with open(log_file_path, 'a') as f:
        f.write(result)
    
    return result.strip()

def train_dqn_agent(agent, env, episodes=20001, max_steps=200, save_interval=250):
    """
    Training loop for DQN agent in MiniGrid environment.
    """
    episode_rewards = []
    step_counts = []
    losses = []
    epsilon_values = []
    
    # Tracking where the agent goes
    state_occupancy = np.zeros((env.size, env.size), dtype=np.int32)
    
    # Create log file for performance evaluations
    log_file_path = generate_save_path("dqn_performance_evaluations.txt")
    with open(log_file_path, 'w') as f:
        f.write("DQN Performance Evaluations\n")
        f.write("="*50 + "\n")
    
    # Training statistics
    q_value_stats = defaultdict(list)
    
    for episode in tqdm(range(episodes), "Training DQN Agent"):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        episode_loss = []
        
        # Get initial state
        current_state = agent.get_state_vector(obs)
        
        for step in range(max_steps):
            # Choose action
            action = agent.get_action(current_state)
            
            # Take action
            obs, reward, done, _, _ = env.step(action)
            next_state = agent.get_state_vector(obs)
            
            # Track agent position
            agent_pos = tuple(env.agent_pos)
            state_occupancy[agent_pos[0], agent_pos[1]] += 1
            
            # Store experience
            agent.remember(current_state, action, reward, next_state, done)
            
            # Train the network
            if len(agent.memory) >= agent.batch_size:
                loss, _ = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
            
            total_reward += reward
            step_count += 1
            current_state = next_state
            
            # Track Q-values for analysis
            if step % 10 == 0:  # Sample every 10 steps to avoid too much data
                q_values = agent.get_q_values(current_state)
                q_value_stats['max_q'].append(np.max(q_values))
                q_value_stats['mean_q'].append(np.mean(q_values))
                q_value_stats['std_q'].append(np.std(q_values))
            
            if done:
                # Evaluate performance at goal
                if episode % 100 == 0:
                    evaluate_dqn_performance(agent, env, episode, log_file_path)
                
                step_counts.append(step + 1)
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        else:
            losses.append(0)
        
        # Save visualizations periodically
        if episode % save_interval == 0 and episode > 0:
            save_dqn_visualizations(agent, env, episode, q_value_stats)
    
    # Save final model
    agent.save_model(generate_save_path('dqn_model.h5'))
    
    # Generate final plots
    plot_training_results(episode_rewards, step_counts, losses, epsilon_values, q_value_stats)
    
    return episode_rewards, step_counts, losses

def save_dqn_visualizations(agent, env, episode, q_value_stats):
    """Save DQN-specific visualizations."""
    
    # Q-value heatmap for the current goal position
    goal_x, goal_y = agent._find_goal_position()
    q_value_grid = np.zeros((agent.grid_size, agent.grid_size))
    
    for y in range(agent.grid_size):
        for x in range(agent.grid_size):
            # Check if position is valid (not a wall)
            cell = env.grid.get(x, y)
            if cell is None or not hasattr(cell, '__class__') or cell.__class__.__name__ != 'Wall':
                # Create state for this position
                state = np.array([
                    x / (agent.grid_size - 1),
                    y / (agent.grid_size - 1),
                    0.0,  # Facing north
                    goal_x / (agent.grid_size - 1),
                    goal_y / (agent.grid_size - 1)
                ], dtype=np.float32)
                
                q_values = agent.get_q_values(state)
                q_value_grid[y, x] = np.max(q_values)
            else:
                q_value_grid[y, x] = np.nan  # Mark walls as NaN
    
    # Plot Q-value heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(q_value_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Max Q-value')
    plt.title(f'Q-value Heatmap (Episode {episode})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Mark goal position
    plt.scatter(goal_x, goal_y, c='red', s=100, marker='*', label='Goal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(generate_save_path(f'q_values/q_heatmap_episode_{episode}.png'))
    plt.close()

def plot_training_results(episode_rewards, step_counts, losses, epsilon_values, q_value_stats):
    """Plot comprehensive training results."""
    
    window = 100
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DQN Training Results', fontsize=16)
    
    # Episode rewards
    rewards_series = pd.Series(episode_rewards)
    rolling_rewards = rewards_series.rolling(window).mean()
    
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    axes[0, 0].plot(rolling_rewards, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Step counts
    steps_series = pd.Series(step_counts)
    rolling_steps = steps_series.rolling(window).mean()
    
    axes[0, 1].plot(step_counts, alpha=0.3, label='Steps per Episode')
    axes[0, 1].plot(rolling_steps, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    loss_series = pd.Series(losses)
    rolling_loss = loss_series.rolling(window).mean()
    
    axes[0, 2].plot(losses, alpha=0.3, label='Training Loss')
    axes[0, 2].plot(rolling_loss, label=f'Rolling Avg (window={window})', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Training Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Epsilon decay
    axes[1, 0].plot(epsilon_values)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Epsilon Decay')
    axes[1, 0].grid(True)
    
    # Q-value statistics
    if q_value_stats['max_q']:
        max_q_series = pd.Series(q_value_stats['max_q'])
        mean_q_series = pd.Series(q_value_stats['mean_q'])
        
        axes[1, 1].plot(max_q_series.rolling(50).mean(), label='Max Q-value')
        axes[1, 1].plot(mean_q_series.rolling(50).mean(), label='Mean Q-value')
        axes[1, 1].set_xlabel('Training Steps (sampled)')
        axes[1, 1].set_ylabel('Q-value')
        axes[1, 1].set_title('Q-value Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Learning curve (reward vs steps)
    axes[1, 2].scatter(step_counts, episode_rewards, alpha=0.5)
    axes[1, 2].set_xlabel('Steps to Goal')
    axes[1, 2].set_ylabel('Episode Reward')
    axes[1, 2].set_title('Learning Efficiency')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(generate_save_path("dqn_training_results.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup environment
    env = SimpleEnv(size=10)
    
    # Setup DQN agent
    agent = DQNAgent(env, 
                     learning_rate=0.001,
                     gamma=0.95,
                     epsilon_start=1.0,
                     epsilon_end=0.01,
                     epsilon_decay=0.995,
                     memory_size=10000,
                     batch_size=32,
                     target_update_freq=100)
    
    # Train the agent
    rewards, steps, losses = train_dqn_agent(agent, env, episodes=1001)
    
    print(f"Training completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Average steps (last 100 episodes): {np.mean(steps[-100:]):.2f}")

if __name__ == "__main__":
    main()


def evaluate_trained_dqn(agent, env, num_episodes=100, max_steps=200, render=False):
    """
    Comprehensive evaluation of trained DQN agent.
    """
    print("Starting comprehensive DQN evaluation...")
    
    results = {
        'episode_rewards': [],
        'episode_steps': [],
        'success_rate': 0,
        'path_efficiency': [],
        'q_value_analysis': defaultdict(list),
        'policy_analysis': defaultdict(list),
        'goal_positions': [],
        'starting_positions': [],
        'paths_taken': []
    }
    
    successful_episodes = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        path = []
        
        # Record starting position and goal
        start_pos = tuple(env.agent_pos)
        goal_pos = agent._find_goal_position()
        results['starting_positions'].append(start_pos)
        results['goal_positions'].append(goal_pos)
        
        # Calculate optimal path length (Manhattan distance)
        optimal_steps = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
        
        current_state = agent.get_state_vector(obs)
        
        for step in range(max_steps):
            # Record position for path analysis
            path.append(tuple(env.agent_pos))
            
            # Get action (no exploration during evaluation)
            action = agent.get_action(current_state, epsilon=0.0)
            q_values = agent.get_q_values(current_state)
            
            # Record Q-value and policy statistics
            results['q_value_analysis']['max_q'].append(np.max(q_values))
            results['q_value_analysis']['mean_q'].append(np.mean(q_values))
            results['q_value_analysis']['action_taken'].append(action)
            results['q_value_analysis']['q_action'].append(q_values[action])
            
            # Take action
            obs, reward, done, _, _ = env.step(action)
            current_state = agent.get_state_vector(obs)
            
            total_reward += reward
            steps += 1
            
            if done:
                successful_episodes += 1
                path.append(tuple(env.agent_pos))  # Final position
                break
        
        results['episode_rewards'].append(total_reward)
        results['episode_steps'].append(steps)
        results['paths_taken'].append(path)
        
        # Calculate path efficiency
        if optimal_steps > 0:
            efficiency = optimal_steps / steps if steps > 0 else 0
        else:
            efficiency = 1.0 if steps == 1 else 0
        results['path_efficiency'].append(efficiency)
    
    results['success_rate'] = successful_episodes / num_episodes
    
    return results

def analyze_policy_quality(agent, env):
    """
    Analyze the quality of the learned policy by examining Q-values across the grid.
    """
    print("Analyzing policy quality...")
    
    analysis = {
        'q_value_grids': {},
        'action_grids': {},
        'policy_consistency': []
    }
    
    # Test multiple goal positions
    test_goals = [(1, 1), (8, 8), (1, 8), (8, 1), (4, 4)]
    
    for goal_idx, (goal_x, goal_y) in enumerate(test_goals):
        q_grid = np.zeros((agent.grid_size, agent.grid_size))
        action_grid = np.zeros((agent.grid_size, agent.grid_size))
        
        for y in range(agent.grid_size):
            for x in range(agent.grid_size):
                # Skip if position is a wall (assuming walls exist)
                try:
                    cell = env.grid.get(x, y)
                    if cell is not None and hasattr(cell, '__class__') and cell.__class__.__name__ == 'Wall':
                        q_grid[y, x] = np.nan
                        action_grid[y, x] = -1
                        continue
                except:
                    pass
                
                # Create state for this position and goal
                state = np.array([
                    x / (agent.grid_size - 1),
                    y / (agent.grid_size - 1),
                    0.0,  # Facing north
                    goal_x / (agent.grid_size - 1),
                    goal_y / (agent.grid_size - 1)
                ], dtype=np.float32)
                
                q_values = agent.get_q_values(state)
                q_grid[y, x] = np.max(q_values)
                action_grid[y, x] = np.argmax(q_values)
        
        analysis['q_value_grids'][f'goal_{goal_x}_{goal_y}'] = q_grid
        analysis['action_grids'][f'goal_{goal_x}_{goal_y}'] = action_grid
        
        # Check if Q-values are highest at goal position
        goal_q = q_grid[goal_y, goal_x]
        if not np.isnan(goal_q):
            surrounding_q = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = goal_y + dy, goal_x + dx
                    if 0 <= ny < agent.grid_size and 0 <= nx < agent.grid_size:
                        if not np.isnan(q_grid[ny, nx]):
                            surrounding_q.append(q_grid[ny, nx])
            
            if surrounding_q:
                is_consistent = goal_q >= max(surrounding_q)
                analysis['policy_consistency'].append(is_consistent)
    
    return analysis

def create_comprehensive_plots(results, policy_analysis, agent):
    """
    Create comprehensive visualization plots for DQN evaluation.
    """
    print("Creating comprehensive evaluation plots...")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # 1. Episode rewards distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results['episode_rewards'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Episode Rewards Distribution')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Steps to goal distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results['episode_steps'], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Steps to Goal Distribution')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Path efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(results['path_efficiency'], bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title('Path Efficiency Distribution')
    ax3.set_xlabel('Efficiency (optimal_steps/actual_steps)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Success rate pie chart
    ax4 = fig.add_subplot(gs[0, 3])
    success_rate = results['success_rate']
    ax4.pie([success_rate, 1-success_rate], 
            labels=[f'Success ({success_rate:.1%})', f'Failure ({1-success_rate:.1%})'],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Success Rate')
    
    # 5. Q-value evolution during episodes
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.plot(results['q_value_analysis']['max_q'], label='Max Q-value', alpha=0.7)
    ax5.plot(results['q_value_analysis']['mean_q'], label='Mean Q-value', alpha=0.7)
    ax5.set_title('Q-value Evolution')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Q-value')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6-10. Q-value heatmaps for different goal positions
    goal_positions = [(1, 1), (8, 8), (1, 8), (8, 1), (4, 4)]
    for i, (gx, gy) in enumerate(goal_positions):
        ax = fig.add_subplot(gs[1, i])
        key = f'goal_{gx}_{gy}'
        if key in policy_analysis['q_value_grids']:
            q_grid = policy_analysis['q_value_grids'][key]
            im = ax.imshow(q_grid, cmap='viridis', interpolation='nearest')
            ax.scatter(gx, gy, c='red', s=100, marker='*')
            ax.set_title(f'Q-values (Goal: {gx}, {gy})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 11-15. Action preference heatmaps
    action_names = ['Turn Left', 'Turn Right', 'Move Forward']
    for i, (gx, gy) in enumerate(goal_positions):
        ax = fig.add_subplot(gs[2, i])
        key = f'goal_{gx}_{gy}'
        if key in policy_analysis['action_grids']:
            action_grid = policy_analysis['action_grids'][key]
            # Create custom colormap for actions
            cmap = plt.cm.get_cmap('Set3')
            im = ax.imshow(action_grid, cmap=cmap, interpolation='nearest', vmin=-0.5, vmax=2.5)
            ax.scatter(gx, gy, c='red', s=100, marker='*')
            ax.set_title(f'Best Actions (Goal: {gx}, {gy})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar with action labels
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['L', 'R', 'F'])
    
    # 16. Starting positions heatmap
    ax16 = fig.add_subplot(gs[3, 0])
    start_heatmap = np.zeros((agent.grid_size, agent.grid_size))
    for x, y in results['starting_positions']:
        start_heatmap[y, x] += 1
    im = ax16.imshow(start_heatmap, cmap='Blues', interpolation='nearest')
    ax16.set_title('Starting Positions Heatmap')
    ax16.set_xlabel('X')
    ax16.set_ylabel('Y')
    plt.colorbar(im, ax=ax16, fraction=0.046, pad=0.04)
    
    # 17. Goal positions heatmap
    ax17 = fig.add_subplot(gs[3, 1])
    goal_heatmap = np.zeros((agent.grid_size, agent.grid_size))
    for x, y in results['goal_positions']:
        goal_heatmap[y, x] += 1
    im = ax17.imshow(goal_heatmap, cmap='Reds', interpolation='nearest')
    ax17.set_title('Goal Positions Heatmap')
    ax17.set_xlabel('X')
    ax17.set_ylabel('Y')
    plt.colorbar(im, ax=ax17, fraction=0.046, pad=0.04)
    
    # 18. Action distribution
    ax18 = fig.add_subplot(gs[3, 2])
    action_counts = [0, 0, 0]
    for action in results['q_value_analysis']['action_taken']:
        action_counts[action] += 1
    ax18.bar(range(3), action_counts, color=['blue', 'orange', 'green'])
    ax18.set_title('Action Distribution')
    ax18.set_xlabel('Action')
    ax18.set_ylabel('Count')
    ax18.set_xticks(range(3))
    ax18.set_xticklabels(['Turn Left', 'Turn Right', 'Move Forward'])
    ax18.grid(True, alpha=0.3)
    
    # 19. Reward vs Steps scatter
    ax19 = fig.add_subplot(gs[3, 3])
    ax19.scatter(results['episode_steps'], results['episode_rewards'], alpha=0.6)
    ax19.set_title('Reward vs Steps')
    ax19.set_xlabel('Steps to Goal')
    ax19.set_ylabel('Total Reward')
    ax19.grid(True, alpha=0.3)
    
    # 20. Policy consistency
    ax20 = fig.add_subplot(gs[3, 4])
    consistency_rate = np.mean(policy_analysis['policy_consistency']) if policy_analysis['policy_consistency'] else 0
    ax20.pie([consistency_rate, 1-consistency_rate], 
            labels=[f'Consistent ({consistency_rate:.1%})', f'Inconsistent ({1-consistency_rate:.1%})'],
            autopct='%1.1f%%', startangle=90)
    ax20.set_title('Policy Consistency\n(Goal has highest Q-value)')
    
    plt.suptitle('DQN Agent Comprehensive Evaluation', fontsize=16, y=0.98)
    plt.savefig(generate_save_path("dqn_comprehensive_evaluation.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_learning_behavior(agent, env, num_test_episodes=50):
    """
    Analyze specific learning behaviors and patterns.
    """
    print("Analyzing learning behavior patterns...")
    
    behaviors = {
        'wall_collision_attempts': 0,
        'optimal_moves': 0,
        'suboptimal_moves': 0,
        'exploration_vs_exploitation': {'exploration': 0, 'exploitation': 0},
        'directional_bias': {'turn_left': 0, 'turn_right': 0, 'move_forward': 0},
        'goal_seeking_efficiency': []
    }
    
    for episode in range(num_test_episodes):
        obs = env.reset()
        current_state = agent.get_state_vector(obs)
        start_pos = tuple(env.agent_pos)
        goal_pos = agent._find_goal_position()
        
        # Calculate initial distance to goal
        initial_distance = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
        
        for step in range(200):  # Max steps
            # Get Q-values and action
            q_values = agent.get_q_values(current_state)
            action = np.argmax(q_values)  # Best action according to policy
            
            # Analyze action choice
            behaviors['directional_bias'][['turn_left', 'turn_right', 'move_forward'][action]] += 1
            
            # Take action and observe result
            old_pos = tuple(env.agent_pos)
            obs, reward, done, _, _ = env.step(action)
            new_pos = tuple(env.agent_pos)
            
            # Check if agent tried to move but position didn't change (wall collision)
            if action == 2 and old_pos == new_pos:  # Move forward but didn't move
                behaviors['wall_collision_attempts'] += 1
            
            # Check if move brought agent closer to goal
            if new_pos != old_pos:  # Agent actually moved
                old_distance = abs(old_pos[0] - goal_pos[0]) + abs(old_pos[1] - goal_pos[1])
                new_distance = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
                
                if new_distance < old_distance:
                    behaviors['optimal_moves'] += 1
                elif new_distance > old_distance:
                    behaviors['suboptimal_moves'] += 1
            
            current_state = agent.get_state_vector(obs)
            
            if done:
                # Calculate efficiency for this episode
                final_distance = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
                efficiency = (initial_distance - final_distance) / (step + 1)
                behaviors['goal_seeking_efficiency'].append(efficiency)
                break
    
    return behaviors

def generate_performance_report(results, policy_analysis, behavior_analysis, agent):
    """
    Generate a comprehensive text report of the agent's performance.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DQN AGENT PERFORMANCE REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Basic Performance Metrics
    report_lines.append("BASIC PERFORMANCE METRICS:")
    report_lines.append("-" * 30)
    report_lines.append(f"Success Rate: {results['success_rate']:.1%}")
    report_lines.append(f"Average Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    report_lines.append(f"Average Steps to Goal: {np.mean(results['episode_steps']):.1f} ± {np.std(results['episode_steps']):.1f}")
    report_lines.append(f"Average Path Efficiency: {np.mean(results['path_efficiency']):.2f} ± {np.std(results['path_efficiency']):.2f}")
    report_lines.append("")
    
    # Q-value Analysis
    report_lines.append("Q-VALUE ANALYSIS:")
    report_lines.append("-" * 20)
    report_lines.append(f"Mean Q-value Range: {np.min(results['q_value_analysis']['mean_q']):.3f} to {np.max(results['q_value_analysis']['mean_q']):.3f}")
    report_lines.append(f"Max Q-value Range: {np.min(results['q_value_analysis']['max_q']):.3f} to {np.max(results['q_value_analysis']['max_q']):.3f}")
    report_lines.append("")
    
    # Policy Analysis
    report_lines.append("POLICY ANALYSIS:")
    report_lines.append("-" * 17)
    if policy_analysis['policy_consistency']:
        consistency_rate = np.mean(policy_analysis['policy_consistency'])
        report_lines.append(f"Policy Consistency: {consistency_rate:.1%}")
        report_lines.append("(Percentage of goal positions where goal state has highest Q-value)")
    report_lines.append("")
    
    # Behavioral Analysis
    report_lines.append("BEHAVIORAL ANALYSIS:")
    report_lines.append("-" * 21)
    total_moves = behavior_analysis['optimal_moves'] + behavior_analysis['suboptimal_moves']
    if total_moves > 0:
        optimal_ratio = behavior_analysis['optimal_moves'] / total_moves
        report_lines.append(f"Optimal Move Ratio: {optimal_ratio:.1%}")
    
    report_lines.append(f"Wall Collision Attempts: {behavior_analysis['wall_collision_attempts']}")
    
    # Action distribution
    total_actions = sum(behavior_analysis['directional_bias'].values())
    if total_actions > 0:
        report_lines.append("Action Distribution:")
        for action, count in behavior_analysis['directional_bias'].items():
            percentage = count / total_actions * 100
            report_lines.append(f"  {action}: {percentage:.1f}%")
    
    if behavior_analysis['goal_seeking_efficiency']:
        avg_efficiency = np.mean(behavior_analysis['goal_seeking_efficiency'])
        report_lines.append(f"Goal Seeking Efficiency: {avg_efficiency:.3f}")
    
    report_lines.append("")
    
    # Network Architecture Summary
    report_lines.append("NETWORK ARCHITECTURE:")
    report_lines.append("-" * 22)
    report_lines.append(f"State Size: {agent.state_size}")
    report_lines.append(f"Action Size: {agent.action_size}")
    report_lines.append(f"Memory Size: {len(agent.memory)}")
    report_lines.append(f"Current Epsilon: {agent.epsilon:.4f}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 16)
    
    if results['success_rate'] < 0.8:
        report_lines.append("• Low success rate suggests need for longer training or hyperparameter tuning")
    
    if np.mean(results['path_efficiency']) < 0.7:
        report_lines.append("• Low path efficiency indicates suboptimal policy - consider reward shaping")
    
    if behavior_analysis['wall_collision_attempts'] > 50:
        report_lines.append("• High wall collision attempts suggest poor spatial awareness")
    
    if policy_analysis['policy_consistency'] and np.mean(policy_analysis['policy_consistency']) < 0.8:
        report_lines.append("• Low policy consistency indicates unstable value function")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    # Save report to file
    report_text = "\n".join(report_lines)
    with open(generate_save_path("dqn_performance_report.txt"), 'w') as f:
        f.write(report_text)
    
    return report_text

def main_evaluation():
    """
    Main evaluation function that runs all analyses.
    """
    print("Starting comprehensive DQN agent evaluation...")
    
    # Load environment and agent
    env = SimpleEnv(size=10)
    
    # Create and load trained agent (you'll need to load your trained model)
    agent = DQNAgent(env)
    # agent.load_model(generate_save_path('dqn_model.h5'))  # Uncomment when you have a trained model
    
    # Run comprehensive evaluation
    results = evaluate_trained_dqn(agent, env, num_episodes=200)
    
    # Analyze policy quality
    policy_analysis = analyze_policy_quality(agent, env)
    
    # Analyze learning behaviors
    behavior_analysis = analyze_learning_behavior(agent, env)
    
    # Create visualizations
    create_comprehensive_plots(results, policy_analysis, agent)
    
    # Generate performance report
    report = generate_performance_report(results, policy_analysis, behavior_analysis, agent)
    print("\nPERFORMANCE REPORT:")
    print(report)
    
    return results, policy_analysis, behavior_analysis

