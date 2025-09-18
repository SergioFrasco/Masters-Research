import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class LSTM_DQN(nn.Module):
    """
    LSTM-based Deep Q-Network for partial observability
    
    Architecture flow:
    1. Frame Stack (4 frames) → CNN → Features
    2. Features → LSTM → Temporal representation
    3. Temporal representation → MLP → Q-values
    """
    
    def __init__(self, frame_stack_size=4, cnn_output_dim=256, lstm_hidden_dim=128, 
                 output_size=6, dropout_rate=0.1):
        """
        Args:
            frame_stack_size: Number of frames to stack (usually 4)
            cnn_output_dim: Dimension of CNN output features
            lstm_hidden_dim: Hidden dimension of LSTM
            output_size: Number of actions
            dropout_rate: Dropout probability for regularization
        """
        super(LSTM_DQN, self).__init__()
        
        # Store dimensions for later use
        self.lstm_hidden_dim = lstm_hidden_dim
        self.frame_stack_size = frame_stack_size
        
        # CNN for processing stacked frames
        # Input: (batch, frame_stack * channels, height, width)
        # For MiniGrid: (batch, 4, 7, 7) assuming grayscale
        self.cnn = nn.Sequential(
            # First conv layer: detect basic features
            nn.Conv2d(frame_stack_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv layer: combine features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
            
            # Third conv layer: high-level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Flatten for fully connected layers
            nn.Flatten(),
            
            # Project to fixed dimension
            # For 7x7 input after pooling: 3x3x64 = 576
            nn.Linear(3 * 3 * 64, cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM for temporal processing
        # Input: (batch, sequence_length, cnn_output_dim)
        # Output: (batch, sequence_length, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,  # Single LSTM layer (can increase for more capacity)
            batch_first=True,  # Input format: (batch, seq, features)
            dropout=0  # No dropout with single layer
        )
        
        # MLP head for Q-value prediction
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)  # Output Q-values for each action
        )
        
    def forward(self, x, hidden_state=None, return_hidden=False):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, sequence_length, frame_stack, height, width)
               OR (batch, frame_stack, height, width) for single step
            hidden_state: Tuple of (h, c) for LSTM hidden and cell states
                         If None, uses zeros
            return_hidden: Whether to return the updated hidden state
        
        Returns:
            q_values: Q-values for each action
            hidden_state (optional): Updated LSTM hidden state
        """
        # Check if input is a sequence or single step
        if len(x.shape) == 4:  # Single step: (batch, frame_stack, height, width)
            batch_size = x.shape[0]
            sequence_length = 1
            x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, frame_stack, H, W)
        else:  # Sequence: (batch, seq_len, frame_stack, height, width)
            batch_size = x.shape[0]
            sequence_length = x.shape[1]
        
        # Process each timestep through CNN
        # Reshape to process all timesteps at once
        x_reshaped = x.view(batch_size * sequence_length, self.frame_stack_size, 7, 7)
        
        # Extract CNN features for all timesteps
        cnn_features = self.cnn(x_reshaped)  # (batch*seq_len, cnn_output_dim)
        
        # Reshape back to sequence format
        cnn_features = cnn_features.view(batch_size, sequence_length, -1)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(x.device)
            c_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(x.device)
            hidden_state = (h_0, c_0)
        
        # Process through LSTM
        lstm_out, new_hidden = self.lstm(cnn_features, hidden_state)
        # lstm_out shape: (batch, sequence_length, lstm_hidden_dim)
        
        # Get Q-values from MLP
        # Process all timesteps
        lstm_out_reshaped = lstm_out.contiguous().view(batch_size * sequence_length, -1)
        q_values = self.mlp(lstm_out_reshaped)
        q_values = q_values.view(batch_size, sequence_length, -1)
        
        # If single step, remove sequence dimension
        if sequence_length == 1:
            q_values = q_values.squeeze(1)
        
        if return_hidden:
            return q_values, new_hidden
        return q_values


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences instead of individual transitions
    This is crucial for LSTM training as it needs temporal context
    """
    
    def __init__(self, capacity=10000, sequence_length=32):
        """
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of each sequence
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.sequences = deque(maxlen=capacity)
        
    def push_sequence(self, sequence):
        """
        Store a sequence of transitions
        
        Args:
            sequence: List of (state, action, reward, next_state, done) tuples
        """
        if len(sequence) >= self.sequence_length:
            # Store only sequences of the required length
            self.sequences.append(sequence[:self.sequence_length])
    
    def sample(self, batch_size):
        """
        Sample a batch of sequences
        
        Returns:
            Batch of sequences, each of length sequence_length
        """
        return random.sample(self.sequences, min(batch_size, len(self.sequences)))
    
    def __len__(self):
        return len(self.sequences)


class FrameStack:
    """
    Maintains a stack of the most recent frames
    This provides motion information and short-term context
    """
    
    def __init__(self, k=4):
        """
        Args:
            k: Number of frames to stack
        """
        self.k = k
        self.frames = deque(maxlen=k)
        
    def reset(self, initial_frame):
        """
        Reset the stack with an initial frame
        
        Args:
            initial_frame: The first observation
        """
        # Clear the deque and fill with copies of initial frame
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(initial_frame)
    
    def push(self, frame):
        """
        Add a new frame to the stack
        
        Args:
            frame: New observation to add
        """
        self.frames.append(frame)
    
    def get_stack(self):
        """
        Get the current frame stack as a numpy array
        
        Returns:
            Stacked frames of shape (k, H, W)
        """
        return np.array(list(self.frames))


class LSTM_DQN_Agent:
    """
    Complete LSTM-DQN agent with frame stacking and sequence-based training
    """
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.9995, memory_size=5000,
                 batch_size=8, sequence_length=32, frame_stack_k=4,
                 target_update_freq=100, lstm_hidden_dim=128):
        """
        Initialize the LSTM-DQN agent
        
        Key differences from vanilla DQN:
        - Uses frame stacking for immediate temporal context
        - Maintains LSTM hidden states across steps
        - Stores and samples sequences instead of transitions
        """
        self.env = env
        self.action_dim = 3  # MiniGrid action space
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_update_freq = target_update_freq
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Frame stacking
        self.frame_stack_k = frame_stack_k
        self.frame_stack = FrameStack(k=frame_stack_k)
        
        # Neural networks
        self.q_network = LSTM_DQN(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            output_size=self.action_dim
        ).to(self.device)
        
        self.target_network = LSTM_DQN(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            output_size=self.action_dim
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Sequence replay buffer
        self.memory = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        # Current episode buffer (accumulates transitions until episode ends)
        self.current_episode = []
        
        # LSTM hidden state tracking
        self.hidden_state = None
        
        # Training counter
        self.update_counter = 0
        
        print(f"LSTM-DQN Agent initialized")
        print(f"Frame stack size: {frame_stack_k}")
        print(f"Sequence length: {sequence_length}")
        print(f"LSTM hidden dim: {lstm_hidden_dim}")
        print(f"Using device: {self.device}")
    

    def reset_episode(self, initial_obs):
        """
        Reset for a new episode
        
        Args:
            initial_obs: First observation of the episode
        """
        # Extract the visual part properly
        frame = self._extract_frame(initial_obs)
        
        # Reset frame stack with initial observation
        self.frame_stack.reset(frame)
        
        # Reset LSTM hidden state
        self.hidden_state = None
        
        # Clear current episode buffer
        self.current_episode = []

    def get_stacked_state(self, obs):
        """
        Process observation and get stacked frame representation
        
        Args:
            obs: Current observation
            
        Returns:
            Tensor of stacked frames ready for network input
        """
        # Extract frame from observation
        frame = self._extract_frame(obs)
        
        # DON'T push to frame stack here - it's already been pushed!
        # This was causing duplicate frames
        
        # Get stacked frames
        stacked = self.frame_stack.get_stack()
        
        # Ensure we have numeric data
        stacked = np.array(stacked, dtype=np.float32)
        
        # Convert to tensor and normalize
        stacked_tensor = torch.FloatTensor(stacked).to(self.device)
        stacked_tensor = stacked_tensor / 10.0  # Simple normalization
        
        return stacked_tensor

    def _extract_frame(self, obs):
        """
        Helper method to consistently extract frame from observation
        
        Args:
            obs: Observation which could be dict or array
            
        Returns:
            Numpy array of the frame
        """
        if isinstance(obs, dict):
            if 'image' in obs:
                frame = obs['image']
                # Handle different shapes
                if len(frame.shape) == 3:
                    # Take first channel if multi-channel
                    frame = frame[0] if frame.shape[0] < frame.shape[1] else frame[:,:,0]
                # Ensure 2D array
                if len(frame.shape) != 2:
                    frame = frame.squeeze()
            else:
                # Fallback - create empty frame
                print("Warning: No 'image' key in observation dict")
                frame = np.zeros((7, 7), dtype=np.float32)
        else:
            # Assume it's already an array
            frame = obs
            if len(frame.shape) == 3:
                frame = frame[0] if frame.shape[0] < frame.shape[1] else frame[:,:,0]
        
        # Ensure it's a float array
        frame = np.array(frame, dtype=np.float32)
        
        # Ensure it's 2D
        if len(frame.shape) != 2:
            print(f"Warning: Unexpected frame shape {frame.shape}, reshaping to 2D")
            frame = frame.reshape(7, 7)
        
        return frame

    def select_action(self, obs, epsilon=None):
        """
        Select action using epsilon-greedy policy with LSTM
        
        IMPORTANT: This method should NOT update the frame stack!
        The frame stack should only be updated once per step.
        
        Args:
            obs: Current observation
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Get current stack without pushing new frame
        stacked = self.frame_stack.get_stack()
        stacked = np.array(stacked, dtype=np.float32)
        state = torch.FloatTensor(stacked).to(self.device) / 10.0
        
        with torch.no_grad():
            # Add batch dimension
            state_batch = state.unsqueeze(0)
            
            # Forward pass with hidden state
            q_values, self.hidden_state = self.q_network(
                state_batch, 
                self.hidden_state,
                return_hidden=True
            )
            
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in current episode buffer
        
        Args:
            state: Stacked frames at time t
            action: Action taken
            reward: Reward received
            next_state: Stacked frames at time t+1
            done: Episode termination flag
        """
        self.current_episode.append((state, action, reward, next_state, done))
        
        # If episode is done, process it into sequences
        if done:
            self.process_episode()
    
    def process_episode(self):
        """
        Process completed episode into sequences for storage
        This is where we create overlapping sequences from the episode
        """
        episode_length = len(self.current_episode)
        
        # Create sequences with sliding window
        for start_idx in range(episode_length - self.sequence_length + 1):
            sequence = self.current_episode[start_idx:start_idx + self.sequence_length]
            self.memory.push_sequence(sequence)
    
    def train(self):
        """
        Train the LSTM-DQN using sequence-based learning
        
        This is the most crucial difference from vanilla DQN:
        We process entire sequences in order, maintaining hidden states
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch of sequences
        batch_sequences = self.memory.sample(self.batch_size)
        
        total_loss = 0.0
        
        for sequence in batch_sequences:
            # Extract states, actions, rewards, next_states, dones from sequence
            states = torch.stack([s[0] for s in sequence]).to(self.device)
            actions = torch.tensor([s[1] for s in sequence], dtype=torch.long).to(self.device)
            rewards = torch.tensor([s[2] for s in sequence], dtype=torch.float32).to(self.device)
            next_states = torch.stack([s[3] for s in sequence]).to(self.device)
            dones = torch.tensor([s[4] for s in sequence], dtype=torch.bool).to(self.device)
            
            # Add batch dimension (batch_size=1 for each sequence)
            states = states.unsqueeze(0)  # (1, seq_len, frame_stack, H, W)
            next_states = next_states.unsqueeze(0)
            
            # Forward pass through Q-network
            # Reset hidden state for each sequence (important!)
            current_q_values = self.q_network(states, hidden_state=None)  # (1, seq_len, action_dim)
            current_q_values = current_q_values.squeeze(0)  # (seq_len, action_dim)
            
            # Select Q-values for taken actions
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values using target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states, hidden_state=None)
                next_q_values = next_q_values.squeeze(0).max(1)[0]  # (seq_len,)
                
                # Compute targets with Bellman equation
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss for this sequence
            loss = F.mse_loss(current_q_values, target_q_values)
            total_loss += loss.item()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return total_loss / len(batch_sequences)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# Example usage showing the complete flow
def demonstrate_lstm_dqn_flow():
    """
    Demonstrates the complete data flow through LSTM-DQN
    """
    print("=== LSTM-DQN Data Flow Demonstration ===\n")
    
    # Mock environment setup
    class MockEnv:
        def __init__(self):
            self.observation_space = {'image': np.zeros((7, 7))}
            
        def reset(self):
            return {'image': np.random.rand(7, 7)}
        
        def step(self, action):
            obs = {'image': np.random.rand(7, 7)}
            reward = np.random.random()
            done = np.random.random() > 0.95
            return obs, reward, done, {}, {}
    
    env = MockEnv()
    agent = LSTM_DQN_Agent(env, sequence_length=10)
    
    print("1. Episode starts - Frame stack initialized with first observation")
    initial_obs = env.reset()
    agent.reset_episode(initial_obs)
    print(f"   Frame stack shape: {agent.frame_stack.get_stack().shape}")
    
    print("\n2. Taking actions - Frames accumulate in stack")
    for step in range(5):
        # Get current stacked state
        current_state = agent.get_stacked_state(initial_obs)
        print(f"   Step {step}: Stacked state shape: {current_state.shape}")
        
        # Select action (LSTM maintains hidden state internally)
        action = agent.select_action(initial_obs)
        
        # Environment step
        next_obs, reward, done, _, _ = env.step(action)
        
        # Get next stacked state
        next_state = agent.get_stacked_state(next_obs)
        
        # Store in episode buffer (not yet in replay buffer!)
        agent.store_transition(current_state, action, reward, next_state, done)
        
        initial_obs = next_obs
        
        if done:
            print(f"   Episode ended at step {step}")
            break
    
    print("\n3. Episode complete - Creating sequences for replay buffer")
    print(f"   Episode length: {len(agent.current_episode)}")
    print(f"   Number of sequences created: {max(0, len(agent.current_episode) - agent.sequence_length + 1)}")
    
    print("\n4. Training on sequences (if enough data)")
    if len(agent.memory) >= agent.batch_size:
        loss = agent.train()
        print(f"   Training loss: {loss:.4f}")
    else:
        print(f"   Not enough sequences yet ({len(agent.memory)}/{agent.batch_size})")
    
    print("\n=== Key Differences from Vanilla DQN ===")
    print("• Frame stacking provides immediate temporal context (motion)")
    print("• LSTM hidden state carries information across entire episode")
    print("• Sequences are processed in order during training")
    print("• Gradients flow through time (backprop through time)")
    print("• Each sequence starts with fresh hidden state during training")


if __name__ == "__main__":
    demonstrate_lstm_dqn_flow()