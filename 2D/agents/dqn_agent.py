import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedVisionQNetwork(nn.Module):
    def __init__(self, grid_size, action_size):
        super(ImprovedVisionQNetwork, self).__init__()
        
        # Simplified CNN architecture - no BatchNorm for more stable training
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        
        # Maintain spatial resolution - no aggressive pooling
        self.flattened_size = 16 * grid_size * grid_size
        
        # Simpler fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)
        
        # Light dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, grid):
        # Process grid through CNN
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten but maintain spatial information
        x = x.view(x.size(0), -1)
        
        # Forward through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return self.out(x)

class VisionDQNAgent:
    """
    Simplified Vision-based Deep Q-Network agent focused on learning from raw grid observation.
    """
    
    def __init__(self, env, action_size=3, learning_rate=0.001, 
                 gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=500,
                 learning_starts=200, train_freq=1):
        
        self.env = env
        self.grid_size = env.size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # More aggressive hyperparameters for faster learning
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        
        # Smaller replay buffer for more recent experiences
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = ImprovedVisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.target_network = ImprovedVisionQNetwork(self.grid_size, self.action_size).to(self.device)
        self.update_target_network()
        
        # Standard Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=self.learning_rate
        )
        
        # MSE loss for cleaner gradients
        self.loss_fn = nn.MSELoss()
        
        # Training tracking
        self.training_step = 0
        self.episode_step = 0
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
    
    def get_vision_state(self, obs=None):
        """
        Pure vision-based state representation - only using the raw grid observation.
        No privileged information about agent position or direction.
        """
        # Get the raw grid encoding from environment
        grid = self.env.grid.encode()
        
        # Extract only the object layer - this is what the agent "sees"
        object_layer = grid[..., 0].astype(np.float32)
        
        # Simple normalization: 
        # 0 = empty space, 1 = wall (obstacle), 2 = goal, 3 = agent
        vision_grid = np.zeros_like(object_layer, dtype=np.float32)
        
        # Map object types to simple values
        vision_grid[object_layer == 1] = 0.0   # Empty space
        vision_grid[object_layer == 2] = 1.0   # Wall (obstacle)
        vision_grid[object_layer == 8] = 2.0   # Goal (what we want to learn to recognize)
        vision_grid[object_layer == 10] = 3.0  # Agent (self-position)
        
        # Normalize to [0, 1] range
        vision_grid = vision_grid / 3.0
        
        # Add batch and channel dimensions: (1, H, W) -> (1, 1, H, W) when needed
        return vision_grid
    
    def get_action(self, vision_grid, epsilon=None):
        """
        Action selection with faster epsilon decay for more exploration early on.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # More exploration during early training
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        grid_tensor = torch.tensor(vision_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(grid_tensor)
            
        # Track Q-values for analysis
        self.recent_q_values.append(torch.max(q_values).item())
        
        return torch.argmax(q_values).item()
    
    def remember(self, vision_grid, action, reward, next_vision_grid, done):
        """Store experience with reward shaping for better learning signal."""
        # Simple reward shaping: small negative for each step to encourage efficiency
        shaped_reward = reward
        if not done and reward == 0:
            shaped_reward = -0.01  # Small penalty for each step
        
        self.memory.append((vision_grid, action, shaped_reward, next_vision_grid, done))
    
    def replay(self):
        """
        Simplified training with standard DQN updates.
        """
        if len(self.memory) < self.learning_starts:
            return None, None
        
        if self.episode_step % self.train_freq != 0:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        vision_grids = [b[0] for b in batch]
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_vision_grids = [b[3] for b in batch]
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        # Convert vision grids to tensors
        grid_batch = torch.stack([
            torch.tensor(grid, dtype=torch.float32).unsqueeze(0) for grid in vision_grids
        ]).to(self.device)

        next_grid_batch = torch.stack([
            torch.tensor(grid, dtype=torch.float32).unsqueeze(0) for grid in next_vision_grids
        ]).to(self.device)

        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(grid_batch)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            target_q_values = self.target_network(next_grid_batch)
            max_next_q = torch.max(target_q_values, dim=1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        # Track loss
        self.recent_losses.append(loss.item())
        
        return loss.item(), torch.mean(current_q_values).item()
    
    def update_target_network(self):
        """Hard update of target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Faster epsilon decay for more exploration early, then exploitation."""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def step(self):
        """Call this at each environment step."""
        self.episode_step += 1
    
    def reset_episode(self):
        """Call this at the start of each episode."""
        self.episode_step = 0
    
    def get_stats(self):
        """Get training statistics for monitoring."""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_q_value': np.mean(self.recent_q_values) if self.recent_q_values else 0,
            'memory_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
    
    def get_q_values(self, vision_grid):
        """Get Q-values for all actions in given state."""
        grid_tensor = torch.tensor(vision_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(grid_tensor)
        return q_values[0].cpu().numpy()

def run_vision_dqn_experiment(env_class, env_size=10, episodes=5000, max_steps=200, seed=20):
    """
    Run experiment with improved Vision-based DQN agent
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

    # Force CPU usage to avoid GPU memory issues
    device = torch.device("cpu")
    
    env = None
    agent = None
    
    try:
        # Create environment
        env = env_class(size=env_size)
        
        # Create improved agent with more aggressive parameters
        agent = VisionDQNAgent(
            env, 
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=0.9,      # Start with high exploration
            epsilon_end=0.05,       # Keep some exploration
            epsilon_decay=0.995,    # Faster decay
            memory_size=10000,      # Smaller buffer for recent experiences
            batch_size=32,
            target_update_freq=250, # More frequent target updates
            learning_starts=200,    # Start learning sooner
            train_freq=1            # Train every step
        )
        
        # Force agent to use CPU
        agent.device = device
        agent.q_network = agent.q_network.to(device)
        agent.target_network = agent.target_network.to(device)

        episode_rewards = []
        episode_lengths = []
        training_stats = []

        print(f"Starting Improved Vision DQN with seed {seed}")
        print(f"Agent sees grid as: empty=0.0, wall=0.33, goal=0.67, agent=1.0")

        for episode in range(episodes):
            try:
                obs = env.reset()
                agent.reset_episode()
                total_reward = 0
                steps = 0

                # Get initial state using pure vision
                current_vision = agent.get_vision_state(obs)

                for step in range(max_steps):
                    # Choose action based on vision only
                    action = agent.get_action(current_vision)
                    
                    # Take action
                    obs, reward, done, _, _ = env.step(action)
                    next_vision = agent.get_vision_state(obs)
                    
                    # Store experience
                    agent.remember(current_vision, action, reward, next_vision, done)
                    
                    # Train the network
                    loss, avg_q = agent.replay()
                    
                    # Step the agent
                    agent.step()
                    
                    total_reward += reward
                    steps += 1
                    current_vision = next_vision
                    
                    if done:
                        break

                # Decay epsilon after each episode
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
                          f"Avg Loss={stats['avg_loss']:.4f}")

            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "Improved Vision DQN",
            "training_stats": training_stats
        }

    except Exception as e:
        print(f"Critical error in Improved DQN experiment: {e}")
        import traceback
        traceback.print_exc()
        return {
            "rewards": [],
            "lengths": [],
            "final_epsilon": 0.0,
            "algorithm": "Improved Vision DQN",
            "error": str(e)
        }
        
    finally:
        # Explicit cleanup
        if agent is not None:
            del agent.q_network
            del agent.target_network
            del agent.memory
            del agent
        
        if env is not None:
            del env
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()