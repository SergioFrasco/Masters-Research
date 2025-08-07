import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Autoencoder for vision-based reward prediction
class Autoencoder(nn.Module):
    def __init__(self, input_channels=1):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # Output between 0 and 1 for reward prediction
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Simple DQN network for engineered features
class FeatureDQN(nn.Module):
    def __init__(self, input_size=5, action_size=4, hidden_size=128):
        super(FeatureDQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class VisionDQNAgent:
    """
    Agent that uses vision model to extract features for DQN
    """
    
    def __init__(self, env, action_size=4, learning_rate=0.001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=1000):
        
        self.env = env
        self.grid_size = env.size
        self.action_size = action_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN hyperparameters
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
        
        # Vision Model (Autoencoder)
        self.vision_model = Autoencoder(input_channels=1).to(self.device)
        self.vision_optimizer = optim.Adam(self.vision_model.parameters(), lr=0.001)
        self.vision_loss_fn = nn.MSELoss()
        
        # DQN Networks (for engineered features)
        self.q_network = FeatureDQN(input_size=5, action_size=self.action_size).to(self.device)
        self.target_network = FeatureDQN(input_size=5, action_size=self.action_size).to(self.device)
        self.update_target_network()
        
        self.dqn_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.dqn_loss_fn = nn.MSELoss()
        
        # Training tracking
        self.training_step = 0
        self.episode_step = 0
        
        # Vision model training
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.train_vision_threshold = 0.1
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
        self.recent_vision_losses = deque(maxlen=100)
    
    def get_visual_input(self, obs):
        """
        Convert environment observation to visual input for autoencoder
        """
        # Get the current environment grid
        grid = self.env.grid.encode()
        normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)
        
        # Map objects to visual representation
        object_layer = grid[..., 0]
        normalized_grid[object_layer == 2] = 0.0   # Wall
        normalized_grid[object_layer == 1] = 0.0   # Open space  
        normalized_grid[object_layer == 8] = 1.0   # Goal/Reward
        
        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        visual_input = normalized_grid[np.newaxis, np.newaxis, ...]
        return torch.tensor(visual_input, dtype=torch.float32).to(self.device)
    
    def predict_reward_locations(self, obs):
        """
        Use vision model to predict reward locations
        Returns: reward_map as numpy array (H, W)
        """
        visual_input = self.get_visual_input(obs)
        
        with torch.no_grad():
            self.vision_model.eval()
            predicted_reward_map = self.vision_model(visual_input)
            return predicted_reward_map.squeeze().cpu().numpy()
    
    def extract_engineered_features(self, obs, predicted_reward_map):
        """
        Extract [agent_x, agent_y, goal_x, goal_y, distance_to_goal] from observation and predictions
        """
        # Get agent position from environment
        agent_x, agent_y = self.env.agent_pos
        
        # Find goal position from predicted reward map
        # Find the location with highest predicted reward
        goal_positions = np.where(predicted_reward_map > 0.5)
        
        if len(goal_positions[0]) > 0:
            # If multiple goals, take the one with highest prediction
            max_idx = np.argmax(predicted_reward_map[goal_positions])
            goal_x = goal_positions[1][max_idx]  # x is second dimension
            goal_y = goal_positions[0][max_idx]  # y is first dimension
        else:
            # If no clear goal detected, use center as fallback
            goal_x = self.grid_size // 2
            goal_y = self.grid_size // 2
        
        # Calculate distance to goal
        distance_to_goal = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
        
        # Normalize features to reasonable ranges
        features = np.array([
            agent_x / self.grid_size,           # Normalize to [0, 1]
            agent_y / self.grid_size,           # Normalize to [0, 1]  
            goal_x / self.grid_size,            # Normalize to [0, 1]
            goal_y / self.grid_size,            # Normalize to [0, 1]
            distance_to_goal / (self.grid_size * np.sqrt(2))  # Normalize by max possible distance
        ], dtype=np.float32)
        
        return features
    
    def get_action(self, features, epsilon=None):
        """
        Choose action based on engineered features using DQN
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        # Convert features to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(feature_tensor)
            
        # Track Q-values for analysis
        self.recent_q_values.append(torch.max(q_values).item())
        
        return torch.argmax(q_values).item()
    
    def remember(self, features, action, reward, next_features, done):
        """Store experience in replay buffer"""
        self.memory.append((features, action, reward, next_features, done))
    
    def train_vision_model(self, obs, agent_position, done, step, max_steps):
        """
        Train the vision model using the same approach as in your successor agent
        """
        # Get visual input
        visual_input = self.get_visual_input(obs)
        
        # Get current prediction
        with torch.no_grad():
            predicted_reward_map = self.vision_model(visual_input).squeeze().cpu().numpy()
        
        # Mark position as visited
        self.visited_positions[agent_position[0], agent_position[1]] = True
        
        # Learning Signal: Update true reward map based on actual experience
        if done and step < max_steps:
            # Successful completion - mark current position as rewarding
            self.true_reward_map[agent_position[0], agent_position[1]] = 1.0
        else:
            # No reward at this position
            self.true_reward_map[agent_position[0], agent_position[1]] = 0.0
        
        # Update unvisited positions with model predictions
        for y in range(self.true_reward_map.shape[0]):
            for x in range(self.true_reward_map.shape[1]):
                if not self.visited_positions[y, x]:
                    predicted_value = predicted_reward_map[y, x]
                    if predicted_value > 0.001:
                        self.true_reward_map[y, x] = predicted_value
                    else:
                        self.true_reward_map[y, x] = 0.0
        
        # Check if we should trigger training
        prediction_error = abs(predicted_reward_map[agent_position[0], agent_position[1]] - 
                             self.true_reward_map[agent_position[0], agent_position[1]])
        
        if prediction_error > self.train_vision_threshold:
            # Train the vision model
            target_tensor = torch.tensor(
                self.true_reward_map[np.newaxis, np.newaxis, ...], 
                dtype=torch.float32
            ).to(self.device)
            
            self.vision_model.train()
            self.vision_optimizer.zero_grad()
            output = self.vision_model(visual_input)
            loss = self.vision_loss_fn(output, target_tensor)
            loss.backward()
            self.vision_optimizer.step()
            
            self.recent_vision_losses.append(loss.item())
            return loss.item()
        
        return None
    
    def train_dqn(self):
        """
        Train the DQN using engineered features
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        features = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_features = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(features)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            target_q_values = self.target_network(next_features)
            max_next_q = torch.max(target_q_values, dim=1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Compute loss
        loss = self.dqn_loss_fn(current_q, target_q)
        
        # Optimize
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.dqn_optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        self.recent_losses.append(loss.item())
        return loss.item(), torch.mean(current_q_values).item()
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_step = 0
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size))
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
    
    def step(self):
        """Call this at each environment step"""
        self.episode_step += 1
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_dqn_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_vision_loss': np.mean(self.recent_vision_losses) if self.recent_vision_losses else 0,
            'avg_q_value': np.mean(self.recent_q_values) if self.recent_q_values else 0,
            'memory_size': len(self.memory)
        }


def run_hybrid_vision_dqn_experiment(env_class, env_size=10, episodes=5000, max_steps=200, seed=20):
    """
    Run experiment with Hybrid Vision-DQN agent
    """
    # Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    env = None
    agent = None
    
    try:
        # Create environment
        env = env_class(size=env_size)
        
        # Create hybrid agent
        agent = HybridVisionDQNAgent(
            env,
            action_size=4,  # Adjust based on your environment
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32,
            target_update_freq=1000
        )
        
        episode_rewards = []
        episode_lengths = []
        training_stats = []
        
        print(f"Starting Hybrid Vision-DQN with seed {seed}")
        print("Vision model learns reward locations, DQN uses extracted features")
        
        for episode in range(episodes):
            try:
                obs = env.reset()
                agent.reset_episode()
                total_reward = 0
                steps = 0
                
                # Get initial features
                predicted_reward_map = agent.predict_reward_locations(obs)
                current_features = agent.extract_engineered_features(obs, predicted_reward_map)
                
                for step in range(max_steps):
                    # Choose action based on engineered features
                    action = agent.get_action(current_features)
                    
                    # Take action
                    obs, reward, done, _, _ = env.step(action)
                    agent_position = tuple(env.agent_pos)
                    
                    # Train vision model
                    vision_loss = agent.train_vision_model(obs, agent_position, done, step, max_steps)
                    
                    # Get next features
                    predicted_reward_map = agent.predict_reward_locations(obs)
                    next_features = agent.extract_engineered_features(obs, predicted_reward_map)
                    
                    # Store experience
                    agent.remember(current_features, action, reward, next_features, done)
                    
                    # Train DQN
                    dqn_loss, avg_q = agent.train_dqn()
                    
                    # Update
                    agent.step()
                    total_reward += reward
                    steps += 1
                    current_features = next_features
                    
                    if done:
                        break
                
                # Decay epsilon
                agent.decay_epsilon()
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                # Collect training statistics
                if episode % 500 == 0:
                    stats = agent.get_stats()
                    training_stats.append({
                        'episode': episode,
                        **stats
                    })
                    
                    recent_success_rate = np.mean([r > 0 for r in episode_rewards[-100:]])
                    recent_avg_reward = np.mean(episode_rewards[-100:])
                    
                    print(f"Episode {episode}: "
                          f"Success Rate={recent_success_rate:.2f}, "
                          f"Avg Reward={recent_avg_reward:.2f}, "
                          f"Epsilon={stats['epsilon']:.3f}, "
                          f"DQN Loss={stats['avg_dqn_loss']:.4f}, "
                          f"Vision Loss={stats['avg_vision_loss']:.4f}")
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "Hybrid Vision-DQN",
            "training_stats": training_stats
        }
    
    except Exception as e:
        print(f"Critical error in Hybrid Vision-DQN experiment: {e}")
        import traceback
        traceback.print_exc()
        return {
            "rewards": [],
            "lengths": [],
            "final_epsilon": 0.0,
            "algorithm": "Hybrid Vision-DQN",
            "error": str(e)
        }
    
    finally:
        # Cleanup
        if agent is not None:
            del agent.vision_model
            del agent.q_network
            del agent.target_network
            del agent.memory
            del agent
        
        if env is not None:
            del env
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()