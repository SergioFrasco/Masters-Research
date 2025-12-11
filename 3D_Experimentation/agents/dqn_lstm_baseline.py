"""
LSTM-DQN Agent for MiniWorld 3D Environment - HYBRID APPROACH

Combines:
1. Frame stacking (k=4) for short-term spatial memory
2. Small LSTM (128 units) for medium-term temporal reasoning
3. All the stability features from the base DQN

This helps the agent remember objects it saw before turning around.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class FrameStack:
    """
    Stack the last k frames to give the agent short-term memory.
    
    For a 3-channel RGB image, stacking k=4 frames gives (12, H, W).
    This allows the agent to perceive motion and recent history.
    """
    
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame):
        """Reset with a new frame, filling the stack with copies"""
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stacked()
    
    def step(self, frame):
        """Add a new frame and return stacked observation"""
        self.frames.append(frame.copy())
        return self._get_stacked()
    
    def _get_stacked(self):
        """Stack frames along channel dimension: (k*C, H, W)"""
        return np.concatenate(list(self.frames), axis=0)
    
    def __len__(self):
        return len(self.frames)


class HybridLSTM_DQN3D(nn.Module):
    """
    Hybrid architecture: Frame Stacking + CNN + Small LSTM + FC
    
    Flow:
    1. Stacked frames (12, 60, 80) → CNN → features (flattened)
    2. Features → LSTM (128 hidden) → temporal embedding
    3. Temporal embedding → FC layers → Q-values
    
    The LSTM is kept SMALL (128 units) to avoid instability.
    """
    
    def __init__(self, input_shape=(12, 60, 80), action_size=3, 
                 hidden_size=256, lstm_size=128):
        super(HybridLSTM_DQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN processes stacked frames
        # Input: (12, 60, 80) if k=4 stacking
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # Small LSTM for temporal memory
        # Single layer, small hidden size for stability
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
            return int(np.prod(output.shape[1:]))
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # Orthogonal initialization for LSTM weights (more stable)
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x, hidden=None):
        """
        Forward pass with optional hidden state.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: Q-values for each action
            hidden: Updated LSTM hidden state
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = self.conv(x)
        features = features.view(batch_size, -1)  # Flatten
        
        # Reshape for LSTM: (batch, seq_len=1, features)
        features = features.unsqueeze(1)
        
        # LSTM processing
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            lstm_out, hidden = self.lstm(features)
        
        # Take the last output from the sequence
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Fully connected layers
        q_values = self.fc(lstm_out)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class DuelingHybridLSTM_DQN3D(nn.Module):
    """
    Dueling version of the Hybrid LSTM-DQN.
    Separates value and advantage streams after the LSTM.
    """
    
    def __init__(self, input_shape=(12, 60, 80), action_size=3, 
                 hidden_size=256, lstm_size=128):
        super(DuelingHybridLSTM_DQN3D, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # Shared CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
            return int(np.prod(output.shape[1:]))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # CNN features
        features = self.conv(x)
        features = features.view(batch_size, -1)
        features = features.unsqueeze(1)
        
        # LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            lstm_out, hidden = self.lstm(features)
        
        lstm_out = lstm_out[:, -1, :]
        
        # Dueling architecture
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        
        # Combine with mean subtraction
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for LSTM training.
    
    Stores complete episodes rather than individual transitions.
    This allows us to sample sequences with proper temporal ordering.
    """
    
    def __init__(self, capacity=5000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push_transition(self, state, action, reward, next_state, done):
        """Add a transition to the current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        """Finalize the current episode and store it"""
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=8):
        """
        Sample a batch of sequences from stored episodes.
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
        
        Returns:
            List of sequences, each a list of (s, a, r, s', done) tuples
        """
        batch = []
        
        for _ in range(batch_size):
            # Sample a random episode
            episode = random.choice(self.episodes)
            
            if len(episode) > seq_len:
                # Sample a random starting point
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                # Use entire episode if shorter than seq_len
                sequence = episode
            
            batch.append(sequence)
        
        return batch
    
    def __len__(self):
        return len(self.episodes)


class LSTMDQNAgent3D:
    """
    Hybrid LSTM-DQN Agent with frame stacking.
    
    Key features:
    1. Frame stacking (k=4) for short-term memory
    2. Small LSTM (128 units) for temporal reasoning
    3. Episode-based replay buffer
    4. Hidden state management during episodes
    5. All stability features (soft updates, double DQN, grad clipping)
    """
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
                 memory_size=5000, batch_size=32, seq_len=8,
                 hidden_size=256, lstm_size=128, use_dueling=True,
                 tau=0.005, use_double_dqn=True, grad_clip=10.0):
        
        self.env = env
        self.action_dim = 3  # turn_left, turn_right, move_forward
        self.k_frames = k_frames
        self.seq_len = seq_len
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Stability parameters
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.grad_clip = grad_clip
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LSTM-DQN Agent using device: {self.device}")
        print(f"Frame stacking: k={k_frames}, LSTM size: {lstm_size}")
        print(f"Stability settings: tau={tau}, double_dqn={use_double_dqn}, grad_clip={grad_clip}")
        
        # Get observation shape
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        # Determine single frame shape (C, H, W)
        if sample_img.shape[0] in [3, 4]:
            self.single_frame_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.single_frame_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        # Stacked observation shape
        self.obs_shape = (k_frames * 3, self.single_frame_shape[1], self.single_frame_shape[2])
        
        print(f"Single frame shape: {self.single_frame_shape}")
        print(f"Stacked observation shape: {self.obs_shape}")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Initialize networks
        NetworkClass = DuelingHybridLSTM_DQN3D if use_dueling else HybridLSTM_DQN3D
        self.q_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            lstm_size=lstm_size
        ).to(self.device)
        
        self.target_network = NetworkClass(
            input_shape=self.obs_shape,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            lstm_size=lstm_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Episode-based replay buffer
        self.memory = EpisodeReplayBuffer(capacity=memory_size)
        
        # Hidden state tracking during episodes
        self.current_hidden = None
        
        # Training tracking
        self.update_counter = 0
        self.training_steps = 0
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total network parameters: {total_params:,}")
    
    def soft_update_target_network(self):
        """Soft update target network using Polyak averaging"""
        for target_param, online_param in zip(self.target_network.parameters(), 
                                               self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame tensor (C, H, W)"""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if isinstance(img, np.ndarray):
            if img.shape[0] in [3, 4]:
                pass
            else:
                img = np.transpose(img, (2, 0, 1))
            
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            
            if img.shape[0] == 4:
                img = img[:3]
        
        return img
    
    def reset_episode(self, obs):
        """
        Reset for a new episode.
        
        Returns:
            stacked_obs: Frame-stacked observation
        """
        # Preprocess and stack frames
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        # Reset LSTM hidden state
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        return stacked
    
    def step_episode(self, obs):
        """
        Process a new observation during an episode.
        
        Returns:
            stacked_obs: Frame-stacked observation
        """
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.step(frame)
        return stacked
    
    def select_action(self, stacked_obs, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        Maintains hidden state across the episode.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Convert stacked observation to tensor
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_network(state, self.current_hidden)
            # Detach hidden state to prevent backprop through time during action selection
            self.current_hidden = (self.current_hidden[0].detach(), 
                                  self.current_hidden[1].detach())
            return q_values.argmax().item()
    
    def remember(self, stacked_obs, action, reward, next_stacked_obs, done):
        """Store transition in episode buffer"""
        self.memory.push_transition(stacked_obs, action, reward, next_stacked_obs, done)
    
    def train_step(self):
        """
        Perform one training step with sequence sampling.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample sequences from episodes
        sequences = self.memory.sample(self.batch_size, self.seq_len)
        
        total_loss = 0.0
        
        for sequence in sequences:
            # Initialize hidden state for this sequence
            hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
            target_hidden = self.target_network.init_hidden(batch_size=1, device=self.device)
            
            for (state, action, reward, next_state, done) in sequence:
                # Convert to tensors
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
                reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
                done_tensor = torch.tensor([done], dtype=torch.bool).to(self.device)
                
                # Current Q value
                q_values, hidden = self.q_network(state_tensor, hidden)
                current_q = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze()
                
                # Target Q value
                with torch.no_grad():
                    if self.use_double_dqn:
                        # Double DQN
                        next_q_values, _ = self.q_network(next_state_tensor, 
                                                         (hidden[0].detach(), hidden[1].detach()))
                        next_action = next_q_values.argmax(1, keepdim=True)
                        
                        target_next_q_values, target_hidden = self.target_network(
                            next_state_tensor, target_hidden
                        )
                        next_q = target_next_q_values.gather(1, next_action).squeeze()
                    else:
                        # Standard DQN
                        target_next_q_values, target_hidden = self.target_network(
                            next_state_tensor, target_hidden
                        )
                        next_q = target_next_q_values.max(1)[0]
                    
                    target_q = reward_tensor + (self.gamma * next_q * ~done_tensor)
                
                # Compute loss
                loss = F.smooth_l1_loss(current_q, target_q)
                total_loss += loss.item()
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                
                # Detach hidden state for next step
                hidden = (hidden[0].detach(), hidden[1].detach())
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
        
        # Soft update target network
        self.soft_update_target_network()
        
        self.update_counter += 1
        self.training_steps += 1
        
        return total_loss / (len(sequences) * len(sequences[0]))
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to starting value"""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {filepath}")
    
    def get_q_values(self, stacked_obs):
        """Get Q-values for debugging/visualization"""
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.q_network(state, self.current_hidden)
            return q_values.squeeze().cpu().numpy()