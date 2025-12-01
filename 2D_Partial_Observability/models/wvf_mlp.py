"""
LSTM-WVF (World Value Function) Model

Combines:
1. Frame stacking for immediate temporal context (motion detection)
2. LSTM for long-term memory across episode (partial observability)
3. Goal conditioning for multi-goal navigation (WVF)

Architecture:
    Frame Stack (4 frames) → CNN → Features
                                      ↓
                              LSTM → Temporal Features
                                      ↓
                        [Temporal Features + Goal(x,y)] → MLP → Q(s,g,a)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class LSTM_WVF(nn.Module):
    """
    LSTM-based World Value Function for partial observability with goal conditioning.
    
    Key differences from standard LSTM-DQN:
    - Goal position (x, y) is concatenated before the MLP head
    - Outputs Q(s, g, a) - value of action a in state s for reaching goal g
    - Can evaluate any goal position, enabling multi-goal navigation
    """
    
    def __init__(self, frame_stack_size=4, cnn_output_dim=256, lstm_hidden_dim=128,
                 goal_dim=2, num_actions=3, dropout_rate=0.1, view_size=7):
        """
        Args:
            frame_stack_size: Number of frames to stack (usually 4)
            cnn_output_dim: Dimension of CNN output features
            lstm_hidden_dim: Hidden dimension of LSTM
            goal_dim: Dimension of goal input (2 for x,y position)
            num_actions: Number of actions (3 for MiniGrid: left, right, forward)
            dropout_rate: Dropout probability for regularization
            view_size: Size of the observation view (7 for MiniGrid default)
        """
        super(LSTM_WVF, self).__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.frame_stack_size = frame_stack_size
        self.view_size = view_size
        self.goal_dim = goal_dim
        
        # CNN for processing stacked frames
        # Input: (batch, frame_stack, height, width) = (batch, 4, 7, 7)
        self.cnn = nn.Sequential(
            # First conv layer: detect basic features (walls, objects)
            nn.Conv2d(frame_stack_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv layer: combine features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 7x7 -> 3x3
            
            # Third conv layer: high-level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Flatten(),
            
            # Project to fixed dimension
            # For 7x7 input after pooling: 3x3x64 = 576
            nn.Linear(3 * 3 * 64, cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM for temporal processing (handles partial observability)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # MLP head for Q-value prediction
        # Input: LSTM output (lstm_hidden_dim) + goal position (goal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x, goal, hidden_state=None, return_hidden=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, frame_stack, H, W)
               OR (batch, frame_stack, H, W) for single step
            goal: Goal position tensor of shape (batch, 2) - normalized (x, y)
            hidden_state: Tuple of (h, c) for LSTM hidden and cell states
            return_hidden: Whether to return the updated hidden state
        
        Returns:
            q_values: Q(s, g, a) for each action
            hidden_state (optional): Updated LSTM hidden state
        """
        # Handle single step vs sequence input
        if len(x.shape) == 4:  # Single step: (batch, frame_stack, H, W)
            batch_size = x.shape[0]
            sequence_length = 1
            x = x.unsqueeze(1)  # Add sequence dim: (batch, 1, frame_stack, H, W)
        else:  # Sequence: (batch, seq_len, frame_stack, H, W)
            batch_size = x.shape[0]
            sequence_length = x.shape[1]
        
        # Process each timestep through CNN
        x_reshaped = x.view(batch_size * sequence_length, self.frame_stack_size, 
                           self.view_size, self.view_size)
        cnn_features = self.cnn(x_reshaped)  # (batch*seq_len, cnn_output_dim)
        cnn_features = cnn_features.view(batch_size, sequence_length, -1)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(x.device)
            c_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(x.device)
            hidden_state = (h_0, c_0)
        
        # Process through LSTM
        lstm_out, new_hidden = self.lstm(cnn_features, hidden_state)
        # lstm_out: (batch, seq_len, lstm_hidden_dim)
        
        # Expand goal to match sequence length if needed
        if len(goal.shape) == 2:  # (batch, goal_dim)
            goal = goal.unsqueeze(1).expand(-1, sequence_length, -1)
        # goal: (batch, seq_len, goal_dim)
        
        # Concatenate LSTM output with goal
        lstm_with_goal = torch.cat([lstm_out, goal], dim=-1)
        # lstm_with_goal: (batch, seq_len, lstm_hidden_dim + goal_dim)
        
        # Get Q-values from MLP
        lstm_with_goal_flat = lstm_with_goal.view(batch_size * sequence_length, -1)
        q_values = self.mlp(lstm_with_goal_flat)
        q_values = q_values.view(batch_size, sequence_length, -1)
        
        # Remove sequence dimension if single step
        if sequence_length == 1:
            q_values = q_values.squeeze(1)
        
        if return_hidden:
            return q_values, new_hidden
        return q_values


class FrameStack:
    """
    Maintains a stack of the most recent frames.
    Provides motion information and short-term temporal context.
    """
    
    def __init__(self, k=4):
        """
        Args:
            k: Number of frames to stack
        """
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, initial_frame):
        """Reset the stack with copies of the initial frame."""
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(initial_frame.copy())
    
    def push(self, frame):
        """Add a new frame to the stack."""
        self.frames.append(frame.copy())
    
    def get_stack(self):
        """Get the current frame stack as a numpy array of shape (k, H, W)."""
        return np.array(list(self.frames))


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences for LSTM training.
    
    Key difference from standard replay buffer:
    - Stores entire sequences instead of individual transitions
    - Each sequence includes goal information for WVF training
    """
    
    def __init__(self, capacity=5000, sequence_length=16):
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
        Store a sequence of transitions.
        
        Args:
            sequence: List of (state, action, reward, next_state, done, goal) tuples
                     where state/next_state are stacked frames
        """
        if len(sequence) >= self.sequence_length:
            self.sequences.append(sequence[:self.sequence_length])
    
    def sample(self, batch_size):
        """Sample a batch of sequences."""
        return random.sample(self.sequences, min(batch_size, len(self.sequences)))
    
    def __len__(self):
        return len(self.sequences)


class RewardPredictor(nn.Module):
    """
    Autoencoder that learns to predict reward locations from partial observations.
    
    This replaces explicit goal detection (like the CubeDetector in 3D):
    - Input: 7x7 egocentric view (what the agent sees)
    - Output: 7x7 predicted reward map (where rewards might be)
    
    Training signal comes from:
    1. Ground truth when agent finds a goal (done signal)
    2. Retrospective training on past trajectory when goal is found
    """
    
    def __init__(self, input_channels=1, hidden_dim=64):
        super(RewardPredictor, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7 -> 3x3
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 3x3 -> 6x6
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(7, 7), mode='nearest'),  # 6x6 -> 7x7
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 1, 7, 7)
        
        Returns:
            Predicted reward map of shape (batch, 1, 7, 7)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# For backwards compatibility with imports
Autoencoder = RewardPredictor


if __name__ == "__main__":
    # Test the components
    print("Testing LSTM-WVF components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test LSTM-WVF network
    print("\n1. Testing LSTM_WVF network...")
    model = LSTM_WVF(
        frame_stack_size=4,
        lstm_hidden_dim=128,
        goal_dim=2,
        num_actions=3
    ).to(device)
    
    # Single step input
    batch_size = 2
    x = torch.randn(batch_size, 4, 7, 7).to(device)  # (batch, frames, H, W)
    goal = torch.rand(batch_size, 2).to(device)  # Normalized goal position
    
    q_values, hidden = model(x, goal, return_hidden=True)
    print(f"   Single step - Input: {x.shape}, Goal: {goal.shape}")
    print(f"   Output Q-values: {q_values.shape}")  # Should be (batch, 3)
    print(f"   Hidden state h: {hidden[0].shape}, c: {hidden[1].shape}")
    
    # Sequence input
    seq_len = 8
    x_seq = torch.randn(batch_size, seq_len, 4, 7, 7).to(device)
    q_values_seq = model(x_seq, goal)
    print(f"   Sequence - Input: {x_seq.shape}")
    print(f"   Output Q-values: {q_values_seq.shape}")  # Should be (batch, seq_len, 3)
    
    # Test Frame Stack
    print("\n2. Testing FrameStack...")
    frame_stack = FrameStack(k=4)
    initial_frame = np.random.rand(7, 7).astype(np.float32)
    frame_stack.reset(initial_frame)
    
    for i in range(3):
        new_frame = np.random.rand(7, 7).astype(np.float32)
        frame_stack.push(new_frame)
    
    stacked = frame_stack.get_stack()
    print(f"   Stacked frames shape: {stacked.shape}")  # Should be (4, 7, 7)
    
    # Test Reward Predictor
    print("\n3. Testing RewardPredictor...")
    reward_predictor = RewardPredictor(input_channels=1).to(device)
    
    obs = torch.randn(batch_size, 1, 7, 7).to(device)
    pred_rewards = reward_predictor(obs)
    print(f"   Input observation: {obs.shape}")
    print(f"   Predicted reward map: {pred_rewards.shape}")  # Should be (batch, 1, 7, 7)
    print(f"   Output range: [{pred_rewards.min():.3f}, {pred_rewards.max():.3f}]")
    
    # Test Sequence Replay Buffer
    print("\n4. Testing SequenceReplayBuffer...")
    buffer = SequenceReplayBuffer(capacity=100, sequence_length=8)
    
    # Create a dummy sequence
    dummy_sequence = []
    for i in range(10):
        state = torch.randn(4, 7, 7)
        action = np.random.randint(0, 3)
        reward = 0.0 if i < 9 else 1.0
        next_state = torch.randn(4, 7, 7)
        done = (i == 9)
        goal = (5, 5)
        dummy_sequence.append((state, action, reward, next_state, done, goal))
    
    buffer.push_sequence(dummy_sequence)
    print(f"   Buffer size after adding 1 sequence: {len(buffer)}")
    
    sampled = buffer.sample(1)
    print(f"   Sampled sequence length: {len(sampled[0])}")
    
    print("\n✓ All components working correctly!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLSTM-WVF Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")