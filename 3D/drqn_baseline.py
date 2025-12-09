"""
DRQN Agent
Changes:
1. REMOVED intrinsic rewards (exploration bonus, distance shaping, speed bonus)
2. REMOVED extra goal features from state (now matches DQN's 175 dims)
3. Kept Double DQN and soft updates (these are legitimate algorithmic improvements)
4. Added option to use hard target updates like DQN for even fairer comparison
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class DRQN(nn.Module):
    """Deep Recurrent Q-Network with LSTM"""
    
    def __init__(self, input_size, hidden_size=128, lstm_hidden=128, 
                 num_lstm_layers=1, output_size=3):
        super(DRQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(lstm_hidden, output_size)
        
        self.hidden_size = lstm_hidden
        self.num_layers = num_lstm_layers
        
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        x = x.reshape(batch_size * seq_len, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(batch_size, seq_len, -1)
        
        lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.fc_out(lstm_out)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
            
        return q_values, hidden
    
    def init_hidden(self, batch_size, device):
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
        """Add sequences from a completed episode"""
        episode_len = len(episode_buffer)
        
        if episode_len == 0:
            return
        
        if episode_len <= self.sequence_length:
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
    
    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DRQNAgentFair:
    """
    DRQN Agent v3
    
    Key changes for fair comparison:
    1. State dimension matches DQN exactly (175 features)
       - ego_obs (13*13=169) + position (2) + direction_onehot (4)
       - NO extra goal features
    2. NO intrinsic rewards - only uses environment reward
    3. Same epsilon schedule as DQN
    4. Option for hard target updates (like DQN) or soft updates
    
    The ONLY advantage DRQN should have is the LSTM memory.
    """
    
    def __init__(self, env, learning_rate=0.0005, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.9995, memory_size=10000,
                 batch_size=32, target_update_freq=100, hidden_dim=128,
                 lstm_hidden=64, num_lstm_layers=1, sequence_length=8,
                 burn_in_length=4, use_double_dqn=True, 
                 use_soft_update=False, tau=0.005):
        """
        Args:
            use_soft_update: If False, uses hard target updates like DQN (fairer)
                           If True, uses soft updates (slightly better but still fair)
            All other hyperparameters match DQN where applicable
        """
        
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.use_double_dqn = use_double_dqn
        self.use_soft_update = use_soft_update
        self.tau = tau
        
        # State representation - MATCHES DQN EXACTLY
        # ego_obs (13*13=169) + position (2) + direction_onehot (4) = 175
        self.view_size = 13 * 13
        self.state_dim = self.view_size + 2 + 4  # 175 features (same as DQN!)
        
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
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        self.episode_buffer = EpisodeBuffer()
        self.hidden_state = None
        
        # Hyperparameters - MATCH DQN
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Same as DQN: 0.9995
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq  # Same as DQN: 100
        self.update_counter = 0
        
        # For visualization (store current goals but DON'T use in state)
        self.current_goal_red = None
        self.current_goal_blue = None
        
        print(f"DRQN Agent v3 (Fair Comparison) initialized:")
        print(f"  - State dim: {self.state_dim} (matches DQN)")
        print(f"  - NO intrinsic rewards")
        print(f"  - NO extra goal features")
        print(f"  - LSTM hidden: {lstm_hidden}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Double DQN: {use_double_dqn}")
        print(f"  - Soft updates: {use_soft_update} (tau={tau if use_soft_update else 'N/A'})")
        print(f"  - Target update freq: {target_update_freq}")
        print(f"  - Epsilon decay: {epsilon_decay}")
        print(f"  - Device: {self.device}")

    def reset_for_new_episode(self):
        """Reset agent state at the start of a new episode"""
        self.hidden_state = None
        
        # Save episode buffer
        if len(self.episode_buffer) > 0:
            self.replay_buffer.add_episode(self.episode_buffer)
        self.episode_buffer.clear()
        
        self.current_goal_red = None
        self.current_goal_blue = None

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
        """Create an egocentric observation matrix"""
        ego_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        
        agent_row = matrix_size - 1
        agent_col = matrix_size // 2
        
        # Mark agent position
        ego_matrix[agent_row, agent_col] = 0.5

        def place_goal(pos, value):
            if pos is None:
                return None
            gx, gz = pos
            ego_row_goal = agent_row - gz
            ego_col_goal = agent_col - gx
            if 0 <= ego_row_goal < matrix_size and 0 <= ego_col_goal < matrix_size:
                ego_matrix[int(ego_row_goal), int(ego_col_goal)] = value
                return (ego_row_goal, ego_col_goal)
            return None

        place_goal(goal_pos_red, 1.0)
        place_goal(goal_pos_blue, 0.8)
        
        # Store for visualization only (NOT used in state)
        self.current_goal_red = goal_pos_red
        self.current_goal_blue = goal_pos_blue

        return ego_matrix

    def get_state_tensor(self, obs, goal_pos_red=None, goal_pos_blue=None):
        """
        Convert observation to state tensor.
        
        MATCHES DQN EXACTLY - 175 features:
        - ego_obs flattened (169)
        - normalized position (2)
        - direction one-hot (4)
        
        NOTE: goal_pos_red and goal_pos_blue are accepted for API compatibility
        but are NOT included in the state vector for fair comparison.
        """
        # 1. Flatten egocentric observation (13x13 = 169)
        view_flat = obs.flatten().astype(np.float32)
        
        # 2. Normalized position (2)
        pos_x = self._get_agent_pos_from_env()[0] / (self.grid_size - 1)
        pos_z = self._get_agent_pos_from_env()[1] / (self.grid_size - 1)
        position = np.array([pos_x, pos_z], dtype=np.float32)
        
        # 3. Direction one-hot (4)
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate: 169 + 2 + 4 = 175 (same as DQN!)
        state = np.concatenate([view_flat, position, direction_onehot])
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, obs, epsilon=None):
        """Select action using DRQN with epsilon-greedy exploration"""
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self.get_state_tensor(obs, self.current_goal_red, self.current_goal_blue)
        
        with torch.no_grad():
            q_values, self.hidden_state = self.q_network(
                state.unsqueeze(0),
                self.hidden_state
            )
            return q_values.argmax().item()

    def select_action_dqn(self, obs, epsilon):
        """Alias for compatibility"""
        return self.select_action(obs, epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the episode buffer"""
        self.episode_buffer.add(state, action, reward, next_state, done)
        
        if done:
            self.replay_buffer.add_episode(self.episode_buffer)
            self.episode_buffer.clear()

    def remember(self, state, action, reward, next_state, done):
        """Alias for compatibility"""
        self.store_transition(state, action, reward, next_state, done)

    def _hard_update_target(self):
        """Hard update target network (like DQN)"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _soft_update_target(self):
        """Soft update target network: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                              self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def train(self):
        """Train DRQN with Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        batch = self.replay_buffer.sample(self.batch_size)
        max_len = max(seq['length'] for seq in batch)
        
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        masks_list = []
        
        for seq in batch:
            seq_len = seq['length']
            
            states = torch.stack(seq['states'])
            actions = torch.tensor(seq['actions'], dtype=torch.long, device=self.device)
            rewards = torch.tensor(seq['rewards'], dtype=torch.float32, device=self.device)
            next_states = torch.stack(seq['next_states'])
            dones = torch.tensor(seq['dones'], dtype=torch.bool, device=self.device)
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
            
            if seq_len < max_len:
                pad_len = max_len - seq_len
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
        
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)
        rewards = torch.stack(rewards_list)
        next_states = torch.stack(next_states_list)
        dones = torch.stack(dones_list)
        masks = torch.stack(masks_list)
        
        # Apply burn-in
        if self.burn_in_length > 0 and max_len > self.burn_in_length:
            masks[:, :self.burn_in_length] = False
        
        # Current Q values
        q_values, _ = self.q_network(states, None)
        current_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select action
                next_q_online, _ = self.q_network(next_states, None)
                best_actions = next_q_online.argmax(dim=2, keepdim=True)
                
                # Use target network to evaluate
                next_q_target, _ = self.target_network(next_states, None)
                max_next_q = next_q_target.gather(2, best_actions).squeeze(-1)
            else:
                # Standard DQN
                next_q_values, _ = self.target_network(next_states, None)
                max_next_q = next_q_values.max(dim=2)[0]
            
            target_q = rewards + (self.gamma * max_next_q * ~dones)
        
        # Compute loss
        td_error = (current_q - target_q) ** 2
        
        if masks.sum() > 0:
            masked_loss = (td_error * masks).sum() / masks.sum()
        else:
            masked_loss = td_error.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        
        # Update target network
        if self.use_soft_update:
            self._soft_update_target()
        else:
            # Hard update like DQN
            if self.update_counter % self.target_update_freq == 0:
                self._hard_update_target()
        
        return masked_loss.item()

    def train_dqn(self):
        """Alias for compatibility"""
        return self.train()

    def decay_epsilon(self):
        """Decay epsilon - same schedule as DQN"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_dqn_state(self, obs):
        """Alias for compatibility"""
        return self.get_state_tensor(obs, self.current_goal_red, self.current_goal_blue)

    @property
    def memory(self):
        return self.replay_buffer

    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"Model loaded from {filepath}")