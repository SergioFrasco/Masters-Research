"""
DRQN Agent with LSTM Memory for Partially Observable Environments
Deep Recurrent Q-Network implementation for MiniWorld navigation tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class DRQN(nn.Module):
    """Deep Recurrent Q-Network with LSTM for temporal memory"""
    
    def __init__(self, input_size, hidden_size=128, lstm_hidden=128, 
                 num_lstm_layers=1, output_size=3):
        super(DRQN, self).__init__()
        
        # Feature extraction layers (processes each timestep)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Output layer for Q-values
        self.fc_out = nn.Linear(lstm_hidden, output_size)
        
        self.hidden_size = lstm_hidden
        self.num_layers = num_lstm_layers
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            hidden: Tuple of (h_n, c_n) LSTM hidden states, or None to initialize
            
        Returns:
            q_values: Shape (batch, seq_len, output_size) or (batch, output_size)
            hidden: Updated hidden state tuple (h_n, c_n)
        """
        # Handle single timestep vs sequence input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, input_size)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Feature extraction for each timestep
        x = x.reshape(batch_size * seq_len, -1)  # (batch*seq, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq, hidden)
        
        # LSTM processes the sequence
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq, lstm_hidden)
        
        # Compute Q-values for each timestep
        q_values = self.fc_out(lstm_out)  # (batch, seq, output_size)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)  # (batch, output_size)
            
        return q_values, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state to zeros"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class EpisodeBuffer:
    """Buffer to store transitions from a single episode"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.states)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def get_sequence(self, start_idx, length):
        """Extract a sequence of transitions starting from start_idx"""
        end_idx = min(start_idx + length, len(self))
        return {
            'states': self.states[start_idx:end_idx],
            'actions': self.actions[start_idx:end_idx],
            'rewards': self.rewards[start_idx:end_idx],
            'next_states': self.next_states[start_idx:end_idx],
            'dones': self.dones[start_idx:end_idx]
        }


class SequenceReplayBuffer:
    """Replay buffer that stores and samples sequences for DRQN training"""
    
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
    
    def add_episode(self, episode_buffer):
        """
        Add sequences from a completed episode to the replay buffer.
        Uses overlapping sequences with 50% overlap for better coverage.
        """
        episode_len = len(episode_buffer)
        
        if episode_len == 0:
            return
        
        if episode_len <= self.sequence_length:
            # Store entire episode if shorter than sequence_length
            sequence = {
                'states': list(episode_buffer.states),
                'actions': list(episode_buffer.actions),
                'rewards': list(episode_buffer.rewards),
                'next_states': list(episode_buffer.next_states),
                'dones': list(episode_buffer.dones),
                'length': episode_len
            }
            self.buffer.append(sequence)
        else:
            # Store overlapping sequences (50% overlap)
            stride = max(1, self.sequence_length // 2)
            for start in range(0, episode_len - self.sequence_length + 1, stride):
                seq_data = episode_buffer.get_sequence(start, self.sequence_length)
                sequence = {
                    'states': seq_data['states'],
                    'actions': seq_data['actions'],
                    'rewards': seq_data['rewards'],
                    'next_states': seq_data['next_states'],
                    'dones': seq_data['dones'],
                    'length': len(seq_data['states'])
                }
                self.buffer.append(sequence)
            
            # Also add the final sequence if not covered
            if (episode_len - self.sequence_length) % stride != 0:
                start = episode_len - self.sequence_length
                seq_data = episode_buffer.get_sequence(start, self.sequence_length)
                sequence = {
                    'states': seq_data['states'],
                    'actions': seq_data['actions'],
                    'rewards': seq_data['rewards'],
                    'next_states': seq_data['next_states'],
                    'dones': seq_data['dones'],
                    'length': len(seq_data['states'])
                }
                self.buffer.append(sequence)
    
    def sample(self, batch_size):
        """Sample a batch of sequences"""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DRQNAgentPartial:
    """
    DRQN Agent with LSTM memory for partially observable environments.
    Designed for MiniWorld navigation with egocentric observations.
    """
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.9995, memory_size=5000,
                 batch_size=32, target_update_freq=100, hidden_dim=128,
                 lstm_hidden=128, num_lstm_layers=1, sequence_length=8,
                 burn_in_length=4):
        """
        Initialize DRQN Agent.
        
        Args:
            env: Environment instance
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            memory_size: Maximum number of sequences in replay buffer
            batch_size: Number of sequences per training batch
            target_update_freq: Steps between target network updates
            hidden_dim: Hidden layer size for feature extraction
            lstm_hidden: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            sequence_length: Length of sequences for training
            burn_in_length: Steps to run LSTM before computing loss (for hidden state warmup)
        """
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3  # left, right, forward
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sequence parameters
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        # State representation: 13x13 egocentric view + position (2) + direction (4)
        self.view_size = 13 * 13
        self.state_dim = self.view_size + 2 + 4  # 175 features
        
        # Initialize networks
        self.q_network = DRQN(
            input_size=self.state_dim,
            hidden_size=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            output_size=self.action_dim
        ).to(self.device)
        
        self.target_network = DRQN(
            input_size=self.state_dim,
            hidden_size=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            output_size=self.action_dim
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer for sequences
        self.replay_buffer = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        # Episode buffer for current episode
        self.episode_buffer = EpisodeBuffer()
        
        # Current hidden state (persists across steps within an episode)
        self.hidden_state = None
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Path integration (kept for compatibility)
        self.internal_pos = None
        self.internal_dir = None
        
        print(f"DRQN Agent initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - LSTM hidden: {lstm_hidden}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Burn-in length: {burn_in_length}")
        print(f"  - Device: {self.device}")

    def reset_for_new_episode(self):
        """Reset agent state at the start of a new episode"""
        # Reset LSTM hidden state
        self.hidden_state = None
        
        # Clear episode buffer
        self.episode_buffer.clear()
        
        # Reset path integration
        self.internal_pos = None
        self.internal_dir = None

    def _get_agent_pos_from_env(self):
        """Get agent position directly from environment"""
        x = int(round(self.env.agent.pos[0] / self.env.grid_size))
        z = int(round(self.env.agent.pos[2] / self.env.grid_size))
        return (x, z)
    
    def _get_agent_dir_from_env(self):
        """Get agent direction directly from environment"""
        angle = self.env.agent.dir
        degrees = (np.degrees(angle) % 360)
        if degrees < 45 or degrees >= 315:
            return 0  # East (+X)
        elif 45 <= degrees < 135:
            return 3  # North (-Z)
        elif 135 <= degrees < 225:
            return 2  # West (-X)
        else:
            return 1  # South (+Z)

    def create_egocentric_observation(self, goal_pos_red=None, goal_pos_blue=None, matrix_size=13):
        """
        Create an egocentric observation matrix.
        Agent is at bottom-middle, facing upward.
        Goals are marked relative to agent's position and orientation.
        """
        ego_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        
        agent_row = matrix_size - 1
        agent_col = matrix_size // 2

        def place_goal(pos, value):
            if pos is None:
                return
            gx, gz = pos
            ego_row = agent_row - gz
            ego_col = agent_col - gx
            if 0 <= ego_row < matrix_size and 0 <= ego_col < matrix_size:
                ego_matrix[int(ego_row), int(ego_col)] = value

        place_goal(goal_pos_red, 1.0)
        place_goal(goal_pos_blue, 1.0)

        return ego_matrix

    def get_state_tensor(self, obs):
        """
        Convert observation to state tensor for network input.
        Does not handle temporal stacking - LSTM handles that.
        """
        # Flatten egocentric view
        view_flat = obs.flatten().astype(np.float32)
        
        # Normalized position
        pos_x = self._get_agent_pos_from_env()[0] / (self.grid_size - 1)
        pos_z = self._get_agent_pos_from_env()[1] / (self.grid_size - 1)
        position = np.array([pos_x, pos_z], dtype=np.float32)
        
        # One-hot direction
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate all features
        state = np.concatenate([view_flat, position, direction_onehot])
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, obs, epsilon=None):
        """
        Select action using DRQN with epsilon-greedy exploration.
        Maintains hidden state across timesteps within an episode.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self.get_state_tensor(obs)
        
        with torch.no_grad():
            # Pass through network with current hidden state
            q_values, self.hidden_state = self.q_network(
                state.unsqueeze(0),  # (1, input_size)
                self.hidden_state
            )
            return q_values.argmax().item()

    # Alias for compatibility with existing code
    def select_action_dqn(self, obs, epsilon):
        return self.select_action(obs, epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the episode buffer"""
        self.episode_buffer.add(state, action, reward, next_state, done)
        
        # At episode end, add sequences to replay buffer
        if done:
            self.replay_buffer.add_episode(self.episode_buffer)
            self.episode_buffer.clear()

    # Alias for compatibility
    def remember(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)

    def train(self):
        """
        Train DRQN on sampled sequences from replay buffer.
        Uses burn-in to establish hidden state before computing loss.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch of sequences
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Find max sequence length in batch for padding
        max_len = max(seq['length'] for seq in batch)
        
        # Prepare batch tensors
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        masks_list = []
        
        for seq in batch:
            seq_len = seq['length']
            
            # Stack sequence tensors
            states = torch.stack(seq['states'])
            actions = torch.tensor(seq['actions'], dtype=torch.long, device=self.device)
            rewards = torch.tensor(seq['rewards'], dtype=torch.float32, device=self.device)
            next_states = torch.stack(seq['next_states'])
            dones = torch.tensor(seq['dones'], dtype=torch.bool, device=self.device)
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
            
            # Pad sequences to max_len
            if seq_len < max_len:
                pad_len = max_len - seq_len
                
                # Pad with zeros/False
                states = F.pad(states, (0, 0, 0, pad_len))
                actions = F.pad(actions, (0, pad_len))
                rewards = F.pad(rewards, (0, pad_len))
                next_states = F.pad(next_states, (0, 0, 0, pad_len))
                dones = F.pad(dones, (0, pad_len), value=True)
                mask = F.pad(mask, (0, pad_len), value=False)
            
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            next_states_list.append(next_states)
            dones_list.append(dones)
            masks_list.append(mask)
        
        # Stack into batch tensors: (batch, seq_len, ...)
        states = torch.stack(states_list)  # (batch, seq, state_dim)
        actions = torch.stack(actions_list)  # (batch, seq)
        rewards = torch.stack(rewards_list)  # (batch, seq)
        next_states = torch.stack(next_states_list)  # (batch, seq, state_dim)
        dones = torch.stack(dones_list)  # (batch, seq)
        masks = torch.stack(masks_list)  # (batch, seq)
        
        # Apply burn-in: don't compute loss for first burn_in_length steps
        if self.burn_in_length > 0 and max_len > self.burn_in_length:
            # Zero out mask for burn-in period
            masks[:, :self.burn_in_length] = False
        
        # Forward pass through Q-network
        q_values, _ = self.q_network(states, None)  # (batch, seq, actions)
        current_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (batch, seq)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states, None)
            max_next_q = next_q_values.max(dim=2)[0]  # (batch, seq)
            target_q = rewards + (self.gamma * max_next_q * ~dones)
        
        # Compute masked loss (ignore padded and burn-in timesteps)
        td_error = (current_q - target_q) ** 2
        
        # Only compute loss where mask is True
        if masks.sum() > 0:
            masked_loss = (td_error * masks).sum() / masks.sum()
        else:
            masked_loss = td_error.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return masked_loss.item()

    # Alias for compatibility
    def train_dqn(self):
        return self.train()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_dqn_state(self, obs):
        """Compatibility wrapper for get_state_tensor"""
        return self.get_state_tensor(obs)

    @property
    def memory(self):
        """Compatibility property for checking replay buffer size"""
        return self.replay_buffer

    def save_model(self, filepath):
        """Save trained model and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"Model loaded from {filepath}")


# Convenience function for creating agent
def create_drqn_agent(env, **kwargs):
    """Factory function to create a DRQN agent with default parameters"""
    default_params = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.9995,
        'memory_size': 5000,
        'batch_size': 32,
        'target_update_freq': 100,
        'hidden_dim': 128,
        'lstm_hidden': 128,
        'num_lstm_layers': 1,
        'sequence_length': 8,
        'burn_in_length': 4
    }
    default_params.update(kwargs)
    return DRQNAgentPartial(env, **default_params)