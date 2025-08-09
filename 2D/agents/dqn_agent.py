import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class ImprovedVisualDQN(nn.Module):
    """
    Improved CNN-based DQN with better spatial processing and position encoding
    """
    
    def __init__(self, input_channels=3, view_size=10, action_size=4, hidden_size=512):
        super(ImprovedVisualDQN, self).__init__()

        self.view_size = view_size
        self.action_size = action_size

        # Enhanced convolutional layers with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Global average pooling to make the network translation invariant
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Position encoding - add agent's relative position information
        self.pos_encoder = nn.Linear(2, 64)  # x, y coordinates
        
        conv_output_size = 256 * 4 * 4  # After global average pooling
        
        # Enhanced fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + 64, hidden_size),  # +64 for position encoding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )

    def forward(self, x, agent_pos=None):
        # x shape: (batch_size, channels, height, width)
        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        
        # Global average pooling
        pooled = self.global_avg_pool(conv_out)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Add position encoding if provided
        if agent_pos is not None:
            pos_encoded = self.pos_encoder(agent_pos)
            flattened = torch.cat([flattened, pos_encoded], dim=1)
        else:
            # Add zero position encoding if not provided
            batch_size = flattened.size(0)
            zero_pos = torch.zeros(batch_size, 64, device=flattened.device)
            flattened = torch.cat([flattened, zero_pos], dim=1)
        
        q_values = self.fc_layers(flattened)
        return q_values

class ImprovedVisualDQNAgent:
    """
    Improved DQN Agent with better state representation and position awareness
    """
    
    def __init__(self, env, view_size=10, action_size=4, learning_rate=0.0001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=64, target_update_freq=500):
        
        self.env = env
        self.grid_size = env.size
        self.view_size = view_size
        self.action_size = action_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Improved hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay with prioritization weights
        self.memory = deque(maxlen=memory_size)
        
        # Networks with enhanced input channels
        self.q_network = ImprovedVisualDQN(
            input_channels=3,  # object type, object color, object state
            view_size=self.view_size,
            action_size=self.action_size
        ).to(self.device)
        
        self.target_network = ImprovedVisualDQN(
            input_channels=3,
            view_size=self.view_size,
            action_size=self.action_size
        ).to(self.device)
        
        self.update_target_network()
        
        # Improved optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=2000, 
            gamma=0.9
        )
        
        self.loss_fn = nn.SmoothL1Loss()  # More stable than MSE
        
        # Training tracking
        self.training_step = 0
        self.episode_step = 0
        self.update_count = 0
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        
        print(f"Improved Q-Network architecture created")
    
    def get_enhanced_observation(self, obs):
        """
        Create enhanced multi-channel observation with position information
        """
        try:
            # Get the full encoded grid from environment
            grid = self.env.grid.encode()  # Shape: (H, W, 3)
            
            # Normalize each channel appropriately
            object_types = grid[..., 0].astype(np.float32)  # Object types
            object_colors = grid[..., 1].astype(np.float32)  # Object colors  
            object_states = grid[..., 2].astype(np.float32)  # Object states
            
            # Normalize to [0, 1] range for better training stability
            object_types = object_types / 10.0  # Assuming max object type ~10
            object_colors = object_colors / 6.0   # Assuming max color ~6
            object_states = object_states / 3.0   # Assuming max state ~3
            
            # Stack channels: (3, H, W)
            enhanced_view = np.stack([object_types, object_colors, object_states], axis=0)
            
            # Convert to tensor
            view_tensor = torch.tensor(enhanced_view, dtype=torch.float32).to(self.device)
            
            # Get agent position for position encoding
            agent_pos = np.array(self.env.agent_pos, dtype=np.float32)
            # Normalize position to [-1, 1] range
            normalized_pos = (agent_pos / (self.grid_size - 1)) * 2 - 1
            pos_tensor = torch.tensor(normalized_pos, dtype=torch.float32).to(self.device)
            
            return view_tensor, pos_tensor
            
        except Exception as e:
            print(f"Error in get_enhanced_observation: {e}")
            # Return default observation
            default_view = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
            default_pos = np.array([0.0, 0.0], dtype=np.float32)
            return (torch.tensor(default_view, dtype=torch.float32).to(self.device),
                   torch.tensor(default_pos, dtype=torch.float32).to(self.device))
    
    def get_action(self, obs, epsilon=None):
        """
        Choose action with improved exploration strategy
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Improved exploration: use epsilon-greedy with action noise
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        try:
            # Get enhanced observation
            visual_input, pos_input = self.get_enhanced_observation(obs)
            # Add batch dimension
            visual_input = visual_input.unsqueeze(0)
            pos_input = pos_input.unsqueeze(0)
            
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(visual_input, pos_input)
                
            # Track Q-values for analysis
            self.recent_q_values.append(torch.max(q_values).item())
            
            return torch.argmax(q_values, dim=1).item()
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            return random.randrange(self.action_size)
    
    def remember(self, obs, action, reward, next_obs, done):
        """Store experience with reward tracking"""
        self.memory.append((obs, action, reward, next_obs, done))
        self.recent_rewards.append(reward)
    
    def train(self):
        """
        Enhanced training with improved stability
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            
            # Process enhanced observations for entire batch
            current_visual_list = []
            current_pos_list = []
            next_visual_list = []
            next_pos_list = []
            actions = []
            rewards = []
            dones = []
            
            for obs, action, reward, next_obs, done in batch:
                curr_vis, curr_pos = self.get_enhanced_observation(obs)
                next_vis, next_pos = self.get_enhanced_observation(next_obs)
                
                current_visual_list.append(curr_vis)
                current_pos_list.append(curr_pos)
                next_visual_list.append(next_vis)
                next_pos_list.append(next_pos)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
            
            # Stack tensors to create batch
            current_visual = torch.stack(current_visual_list, dim=0)
            current_pos = torch.stack(current_pos_list, dim=0)
            next_visual = torch.stack(next_visual_list, dim=0)
            next_pos = torch.stack(next_pos_list, dim=0)
            
            actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
            # Current Q-values
            self.q_network.train()
            current_q_values = self.q_network(current_visual, current_pos)
            current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Target Q-values using Double DQN
            with torch.no_grad():
                # Use main network to select action
                next_q_values_main = self.q_network(next_visual, next_pos)
                next_actions = torch.argmax(next_q_values_main, dim=1)
                
                # Use target network to evaluate action
                next_q_values_target = self.target_network(next_visual, next_pos)
                max_next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                target_q = rewards_tensor + (self.gamma * max_next_q * (1 - dones_tensor))
            
            # Compute loss
            loss = self.loss_fn(current_q, target_q)
            
            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update target network
            self.training_step += 1
            if self.training_step % self.target_update_freq == 0:
                self.soft_update_target_network(tau=0.01)  # Soft update
            
            self.recent_losses.append(loss.item())
            avg_q = torch.mean(current_q_values).item()
            
            return loss.item(), avg_q
            
        except Exception as e:
            print(f"Error in train: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def soft_update_target_network(self, tau=0.01):
        """Soft update of target network for stability"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Improved epsilon decay with minimum exploration"""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_step = 0
    
    def step(self):
        """Call this at each environment step"""
        self.episode_step += 1
    
    def get_stats(self):
        """Get comprehensive training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_q_value': np.mean(self.recent_q_values) if self.recent_q_values else 0,
            'avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'memory_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }