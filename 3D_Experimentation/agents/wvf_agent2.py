"""
World Value Functions (WVF) Agent for Compositional RL

Based on Nangue Tasse et al.'s approach:
- Train separate Q-networks for each primitive feature (red, blue, box, sphere)
- Compose Q-values at evaluation time using min operation
- Normalization to ensure Q-values are on similar scales

Architecture per primitive: Frame Stacking (k=4) + CNN + Small LSTM + FC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy


class FrameStack:
    """Stack the last k frames for short-term memory."""
    
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame):
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stacked()
    
    def step(self, frame):
        self.frames.append(frame.copy())
        return self._get_stacked()
    
    def _get_stacked(self):
        return np.concatenate(list(self.frames), axis=0)


class PrimitiveQNetwork(nn.Module):
    """
    Q-Network for a single primitive feature.
    Dueling architecture with LSTM for temporal reasoning.
    """
    
    def __init__(self, input_shape=(12, 60, 80), action_size=3, 
                 hidden_size=128, lstm_size=64):
        super(PrimitiveQNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv(dummy)
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
        
        # CNN
        features = self.conv(x)
        features = features.view(batch_size, -1).unsqueeze(1)
        
        # LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            lstm_out, hidden = self.lstm(features)
        
        lstm_out = lstm_out[:, -1, :]
        
        # Dueling
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """Episode-based replay buffer for LSTM training."""
    
    def __init__(self, capacity=2000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            self.end_episode()
    
    def end_episode(self):
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=4):
        batch = []
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            if len(episode) > seq_len:
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                sequence = episode
            batch.append(sequence)
        return batch
    
    def __len__(self):
        return len(self.episodes)
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = []


class WorldValueFunctionAgent:
    """
    World Value Functions Agent
    
    Maintains 4 separate Q-networks for primitive features:
    - Q_red: value of reaching any red object
    - Q_blue: value of reaching any blue object  
    - Q_box: value of reaching any box
    - Q_sphere: value of reaching any sphere
    
    Composition at evaluation: Q_composed = min(Q_feature1, Q_feature2)
    """
    
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=128, lstm_size=64, tau=0.005, grad_clip=10.0):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.seq_len = seq_len
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.grad_clip = grad_clip
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WVF Agent using device: {self.device}")
        
        # Get observation shape
        sample_obs = env.reset()[0]
        if isinstance(sample_obs, dict) and 'image' in sample_obs:
            sample_img = sample_obs['image']
        else:
            sample_img = sample_obs
        
        if sample_img.shape[0] in [3, 4]:
            self.single_frame_shape = (3, sample_img.shape[1], sample_img.shape[2])
        else:
            self.single_frame_shape = (3, sample_img.shape[0], sample_img.shape[1])
        
        self.obs_shape = (k_frames * 3, self.single_frame_shape[1], self.single_frame_shape[2])
        print(f"Observation shape: {self.obs_shape}")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Initialize Q-networks for each primitive
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.memories = {}
        
        for primitive in self.PRIMITIVES:
            # Online network
            self.q_networks[primitive] = PrimitiveQNetwork(
                input_shape=self.obs_shape,
                action_size=self.action_dim,
                hidden_size=hidden_size,
                lstm_size=lstm_size
            ).to(self.device)
            
            # Target network
            self.target_networks[primitive] = PrimitiveQNetwork(
                input_shape=self.obs_shape,
                action_size=self.action_dim,
                hidden_size=hidden_size,
                lstm_size=lstm_size
            ).to(self.device)
            self.target_networks[primitive].load_state_dict(
                self.q_networks[primitive].state_dict()
            )
            self.target_networks[primitive].eval()
            
            # Optimizer
            self.optimizers[primitive] = optim.Adam(
                self.q_networks[primitive].parameters(), 
                lr=learning_rate
            )
            
            # Replay buffer
            self.memories[primitive] = EpisodeReplayBuffer(capacity=memory_size)
        
        # Current training state
        self.current_primitive = None
        self.epsilon = epsilon_start
        self.current_hidden = None
        
        # Track Q-value statistics for normalization
        self.q_stats = {p: {'min': 0, 'max': 1} for p in self.PRIMITIVES}
        
        total_params = sum(p.numel() for p in self.q_networks['red'].parameters())
        print(f"Parameters per primitive network: {total_params:,}")
        print(f"Total parameters (4 networks): {total_params * 4:,}")
    
    def set_training_primitive(self, primitive):
        """Set which primitive we're currently training."""
        assert primitive in self.PRIMITIVES, f"Unknown primitive: {primitive}"
        self.current_primitive = primitive
        self.epsilon = self.epsilon_start
        print(f"Now training primitive: {primitive}")
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame tensor."""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
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
        """Reset for new episode."""
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        if self.current_primitive:
            self.current_hidden = self.q_networks[self.current_primitive].init_hidden(
                batch_size=1, device=self.device
            )
        
        return stacked
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, epsilon=None, primitive=None):
        """Select action using epsilon-greedy for training."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if primitive is None:
            primitive = self.current_primitive
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.q_networks[primitive](
                state, self.current_hidden
            )
            self.current_hidden = (
                self.current_hidden[0].detach(),
                self.current_hidden[1].detach()
            )
            return q_values.argmax().item()
    
    def select_action_composed(self, stacked_obs, features, normalize=True):
        """
        Select action using composed Q-values for evaluation.
        
        Args:
            stacked_obs: Frame-stacked observation
            features: List of primitive features to compose, e.g. ['red', 'box']
            normalize: Whether to normalize Q-values before composition
        
        Returns:
            action: Best action according to composed Q-values
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        q_values_list = []
        
        with torch.no_grad():
            for feature in features:
                hidden = self.q_networks[feature].init_hidden(
                    batch_size=1, device=self.device
                )
                q_vals, _ = self.q_networks[feature](state, hidden)
                
                if normalize:
                    # Normalize to roughly [0, 1] range using tracked statistics
                    q_min = self.q_stats[feature]['min']
                    q_max = self.q_stats[feature]['max']
                    if q_max - q_min > 1e-6:
                        q_vals = (q_vals - q_min) / (q_max - q_min)
                    else:
                        q_vals = q_vals * 0  # All same, doesn't matter
                
                q_values_list.append(q_vals)
            
            # Stack and take element-wise minimum
            q_stacked = torch.stack(q_values_list, dim=0)  # (n_features, 1, n_actions)
            q_composed = q_stacked.min(dim=0)[0]  # (1, n_actions)
            
            return q_composed.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in current primitive's buffer."""
        if self.current_primitive:
            self.memories[self.current_primitive].push(
                state, action, reward, next_state, done
            )
    
    def soft_update_target(self, primitive):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_networks[primitive].parameters(),
            self.q_networks[primitive].parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Training step for current primitive."""
        primitive = self.current_primitive
        memory = self.memories[primitive]
        
        if len(memory) < self.batch_size:
            return 0.0
        
        sequences = memory.sample(self.batch_size, self.seq_len)
        
        # Pad and convert to tensors
        max_len = max(len(seq) for seq in sequences)
        
        states_batch, actions_batch, rewards_batch = [], [], []
        next_states_batch, dones_batch, lengths = [], [], []
        
        for seq in sequences:
            seq_len = len(seq)
            lengths.append(seq_len)
            
            states = [s[0] for s in seq]
            actions = [s[1] for s in seq]
            rewards = [s[2] for s in seq]
            next_states = [s[3] for s in seq]
            dones = [s[4] for s in seq]
            
            # Pad if needed
            if seq_len < max_len:
                pad_len = max_len - seq_len
                states.extend([states[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([0.0] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([True] * pad_len)
            
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(dones_batch).to(self.device)
        
        batch_size, seq_len = states_t.shape[:2]
        
        # Forward pass
        hidden = self.q_networks[primitive].init_hidden(batch_size, self.device)
        target_hidden = self.target_networks[primitive].init_hidden(batch_size, self.device)
        
        q_values_list = []
        for t in range(seq_len):
            q_vals, hidden = self.q_networks[primitive](states_t[:, t], hidden)
            q_values_list.append(q_vals)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        q_values = torch.stack(q_values_list, dim=1)
        current_q = q_values.gather(2, actions_t.unsqueeze(2)).squeeze(2)
        
        # Update Q-value statistics for normalization
        with torch.no_grad():
            q_flat = q_values.view(-1)
            self.q_stats[primitive]['min'] = min(
                self.q_stats[primitive]['min'],
                q_flat.min().item()
            )
            self.q_stats[primitive]['max'] = max(
                self.q_stats[primitive]['max'],
                q_flat.max().item()
            )
        
        # Double DQN target
        with torch.no_grad():
            # Online network selects actions
            next_q_list = []
            hidden_copy = self.q_networks[primitive].init_hidden(batch_size, self.device)
            for t in range(seq_len):
                nq, hidden_copy = self.q_networks[primitive](next_states_t[:, t], hidden_copy)
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_values = torch.stack(next_q_list, dim=1)
            next_actions = next_q_values.argmax(2, keepdim=True)
            
            # Target network evaluates
            target_q_list = []
            for t in range(seq_len):
                tq, target_hidden = self.target_networks[primitive](
                    next_states_t[:, t], target_hidden
                )
                target_q_list.append(tq)
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
            
            target_q_values = torch.stack(target_q_list, dim=1)
            next_q = target_q_values.gather(2, next_actions).squeeze(2)
            
            target_q = rewards_t + (self.gamma * next_q * ~dones_t)
        
        # Masked loss
        loss_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length] = 1.0
        
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        # Backward
        self.optimizers[primitive].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_networks[primitive].parameters(), 
            max_norm=self.grad_clip
        )
        self.optimizers[primitive].step()
        
        # Soft update
        self.soft_update_target(primitive)
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save all primitive networks."""
        checkpoint = {
            'q_stats': self.q_stats,
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
        }
        
        for primitive in self.PRIMITIVES:
            checkpoint[f'q_network_{primitive}'] = self.q_networks[primitive].state_dict()
            checkpoint[f'target_network_{primitive}'] = self.target_networks[primitive].state_dict()
            checkpoint[f'optimizer_{primitive}'] = self.optimizers[primitive].state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"WVF model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load all primitive networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_stats = checkpoint['q_stats']
        
        for primitive in self.PRIMITIVES:
            self.q_networks[primitive].load_state_dict(
                checkpoint[f'q_network_{primitive}']
            )
            self.target_networks[primitive].load_state_dict(
                checkpoint[f'target_network_{primitive}']
            )
            self.optimizers[primitive].load_state_dict(
                checkpoint[f'optimizer_{primitive}']
            )
        
        print(f"WVF model loaded from {filepath}")
    
    def get_q_values_all_primitives(self, stacked_obs):
        """Get Q-values from all primitives (for debugging)."""
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        result = {}
        with torch.no_grad():
            for primitive in self.PRIMITIVES:
                hidden = self.q_networks[primitive].init_hidden(1, self.device)
                q_vals, _ = self.q_networks[primitive](state, hidden)
                result[primitive] = q_vals.squeeze().cpu().numpy()
        
        return result