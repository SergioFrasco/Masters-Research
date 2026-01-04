"""
========================================================================
GOAL-CONDITIONED LSTM-DQN AGENT - VIEW-BASED (NO VISION MODEL)
========================================================================

This is the CORRECTED implementation for your 2D Minigrid baseline.

KEY FEATURE: View-Based Goal Conditioning
- Agent conditions on goal position WITHIN its 7×7 partial observation
- When goal is visible: uses its position in the view
- When goal is NOT visible: uses center position (exploration mode)
- This handles changing goal positions across episodes naturally!

Why this works:
- Goals spawn randomly each episode at different absolute positions
- But relative position in agent's view is consistent and learnable
- Network learns: "when goal is at (x,y) in my view, take action a"
- No vision model needed - just uses what's already in the observation

NO VISION MODEL:
- Just reads goal position from observation (like reading agent position)
- No learned reward predictor
- No neural network for vision
- Just np.argwhere(view == 8) to find goal

Differences from 3D WVF:
- Conditions on GOAL POSITION (x, y) instead of task features
- No composition (2D env has identical goals)  
- Still uses LSTM for memory and goal conditioning for learning efficiency

Based on:
- Universal Value Functions (Schaul et al. 2015)
- World Value Functions (Nangue Tasse et al. 2020) - but simplified to UVF

Architecture:
    [Image frames (stacked) | Goal position (2D)] -> CNN -> LSTM -> Dueling Q

========================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class FrameStack:
    """Stack the last k frames for short-term visual memory."""
    
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame):
        """Reset with initial frame, repeated k times."""
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stacked()
    
    def push(self, frame):
        """Add new frame and return stacked."""
        self.frames.append(frame.copy())
        return self._get_stacked()
    
    def get_stack(self):
        """Get current stack without modifying."""
        return self._get_stacked()
    
    def _get_stacked(self):
        """Stack frames along channel dimension."""
        return np.concatenate(list(self.frames), axis=0)


class LSTM_WVF_Network(nn.Module):
    """
    Goal-Conditioned Q-Network with LSTM (UVF approach).
    
    Learns Q(s, a, g) where:
    - s = stacked partial observations (k frames)
    - a = action
    - g = goal position (x, y) normalized to [0, 1]
    
    Architecture:
        [Image (k channels) | Goal Tiled (2 channels)] -> CNN -> LSTM -> Dueling Q
    
    Input: (batch, k + 2, H, W) where k = frame stack size
    """
    
    def __init__(self, frame_stack_size=4, lstm_hidden_dim=128, 
                 num_actions=3, input_height=7, input_width=7):
        super(LSTM_WVF_Network, self).__init__()
        
        self.frame_stack_size = frame_stack_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_actions = num_actions
        
        # CNN input = stacked frames + goal position tiled
        cnn_input_channels = frame_stack_size + 2  # k frames + (x, y)
        
        # CNN backbone for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        self.conv_output_size = 64 * input_height * input_width
        
        # LSTM for temporal reasoning over partial observations
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Dueling architecture: V(s, g) and A(s, a, g)
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
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
    
    def tile_goal(self, state, goal):
        """
        Tile goal position across spatial dimensions and concatenate with state.
        
        Args:
            state: (batch, k, H, W) stacked frames
            goal: (batch, 2) normalized goal position [x, y] in [0, 1]
        
        Returns:
            combined: (batch, k + 2, H, W)
        """
        batch_size = state.size(0)
        H, W = state.size(2), state.size(3)
        
        # Tile goal: (batch, 2) -> (batch, 2, H, W)
        goal_expanded = goal.view(batch_size, 2, 1, 1)
        goal_tiled = goal_expanded.expand(batch_size, 2, H, W)
        
        # Concatenate: state + goal
        combined = torch.cat([state, goal_tiled], dim=1)
        
        return combined
    
    def forward(self, state, goal, hidden=None, return_hidden=False):
        """
        Forward pass with goal conditioning.
        
        Args:
            state: (batch, k, H, W) stacked frames
            goal: (batch, 2) normalized goal position
            hidden: Optional LSTM hidden state (h, c)
            return_hidden: Whether to return updated hidden state
        
        Returns:
            q_values: (batch, num_actions)
            hidden: (optional) Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # Tile goal and concatenate with state
        combined = self.tile_goal(state, goal)  # (batch, k+2, H, W)
        
        # CNN features
        conv_features = self.conv(combined)
        conv_features = conv_features.view(batch_size, -1)  # (batch, conv_output_size)
        
        # LSTM expects (batch, seq_len, features)
        conv_features = conv_features.unsqueeze(1)  # (batch, 1, conv_output_size)
        
        if hidden is not None:
            lstm_out, new_hidden = self.lstm(conv_features, hidden)
        else:
            lstm_out, new_hidden = self.lstm(conv_features)
        
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)
        
        # Dueling Q-values: Q(s,a,g) = V(s,g) + [A(s,a,g) - mean(A(s,a,g))]
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if return_hidden:
            return q_values, new_hidden
        return q_values
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return (h, c)


class SequenceReplayBuffer:
    """
    Episode-based replay buffer for LSTM training.
    Stores complete episodes and samples sequences.
    """
    
    def __init__(self, capacity=5000, sequence_length=16):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
        self.sequence_length = sequence_length
    
    def push(self, state, goal, action, reward, next_state, done):
        """Add transition to current episode."""
        self.current_episode.append((state, goal, action, reward, next_state, done))
        
        if done:
            self._finalize_episode()
    
    def _finalize_episode(self):
        """Store completed episode in buffer."""
        if len(self.current_episode) > 0:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample(self, batch_size):
        """Sample random sequences from stored episodes."""
        if len(self.episodes) < batch_size:
            return None
        
        sequences = []
        
        for _ in range(batch_size):
            # Sample random episode
            episode = random.choice(self.episodes)
            
            if len(episode) <= self.sequence_length:
                # Use entire episode if shorter than sequence length
                sequences.append(episode)
            else:
                # Sample random subsequence
                start_idx = random.randint(0, len(episode) - self.sequence_length)
                sequences.append(episode[start_idx:start_idx + self.sequence_length])
        
        return sequences
    
    def __len__(self):
        return len(self.episodes)


class LSTM_WVF_Agent:
    """
    LSTM_WVF_Agent - View-based goal conditioning (UVF baseline for 2D Minigrid).
    
    Key features:
    - **View-based goal conditioning**: Q(s, a, g) where g = goal position in 7×7 view
    - LSTM for memory over partial observations
    - No vision model (no reward predictor)
    - Frame stacking for short-term visual context
    - Sequence-based replay for LSTM training
    
    How it handles changing goal positions:
    - Goals spawn at random positions each episode
    - Agent conditions on goal position WITHIN its current 7×7 view
    - When goal visible: learns "goal at (x,y) in view -> take action a"
    - When goal NOT visible: uses default position (exploration mode)
    - This relative encoding generalizes across episodes!
    
    This parallels the 3D WVF agent but without composition.
    """
    
    def __init__(self, env, 
                 frame_stack_k=4,
                 sequence_length=16,
                 lstm_hidden_dim=128,
                 learning_rate=0.00005,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.9995,
                 memory_size=5000,
                 batch_size=16,
                 target_update_freq=200):
        
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3  # turn_left, turn_right, forward
        
        # Hyperparameters
        self.frame_stack_k = frame_stack_k
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LSTM_WVF_Agent (view-based) using device: {self.device}")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=frame_stack_k)
        
        # Networks
        self.q_network = LSTM_WVF_Network(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            num_actions=self.action_dim,
            input_height=7,
            input_width=7
        ).to(self.device)
        
        self.target_network = LSTM_WVF_Network(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            num_actions=self.action_dim,
            input_height=7,
            input_width=7
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        # Episode state
        self.current_episode = []
        self.hidden_state = None
        self.current_goal = None
        
        # Update counter
        self.update_counter = 0
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"  Network parameters: {total_params:,}")
        print(f"  Frame stack: {frame_stack_k} frames")
        print(f"  Sequence length: {sequence_length}")
        print(f"  LSTM hidden dim: {lstm_hidden_dim}")
    
    def _extract_frame(self, obs):
        """Extract single frame from observation."""
        if isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']
            if len(frame.shape) == 3:
                frame = frame[0] if frame.shape[0] < frame.shape[2] else frame[:, :, 0]
        else:
            frame = obs
            if len(frame.shape) == 3:
                frame = frame[0] if frame.shape[0] < frame.shape[2] else frame[:, :, 0]
        
        return np.array(frame, dtype=np.float32)
    
    def _get_goal_in_view(self, obs):
        """
        Extract goal position from agent's 7×7 partial observation.
        
        This is view-based goal conditioning - agent only conditions on goals it can see.
        When no goal is visible, returns center position (exploration mode).
        
        Args:
            obs: Current observation (dict or array)
        
        Returns:
            np.array([gx_norm, gy_norm]) normalized to [0, 1] within 7×7 view
        """
        # Extract 7×7 view
        if isinstance(obs, dict) and 'image' in obs:
            view = obs['image'][0]  # Shape: (7, 7)
        else:
            view = obs
            if len(view.shape) == 3:
                view = view[0] if view.shape[0] < view.shape[2] else view[:, :, 0]
        
        # Find goal in view (goal type == 8 in Minigrid)
        goal_positions = np.argwhere(view == 8)
        
        if len(goal_positions) == 0:
            # No goal visible - use neutral position (center of view)
            # This allows network to learn exploration behavior
            return np.array([0.5, 0.5], dtype=np.float32)
        
        # Take first (or closest) goal in view
        gy, gx = goal_positions[0]
        
        # Normalize to [0, 1] within the 7×7 view
        # View coordinates: (0,0) is top-left, (6,6) is bottom-right
        gx_norm = gx / 6.0
        gy_norm = gy / 6.0
        
        return np.array([gx_norm, gy_norm], dtype=np.float32)
    
    def reset_episode(self, obs):
        """Reset for new episode."""
        # Extract frame and initialize stack
        frame = self._extract_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        # Initialize LSTM hidden state
        self.hidden_state = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        # Get goal from current view (not from environment state)
        # This ensures we only use information the agent can actually see
        self.current_goal = self._get_goal_in_view(obs)
        
        # Store for episode replay
        self.current_episode = []
        
        return stacked
    
    def get_stacked_state(self):
        """Get current stacked state."""
        return self.frame_stack.get_stack()
    
    def select_action(self, obs, epsilon=None):
        """
        Select action using epsilon-greedy policy with goal conditioning.
        
        Goal is extracted from the current 7×7 view - agent only conditions on
        goals it can actually see. This ensures true partial observability.
        
        Args:
            obs: Current observation (can be dict or array)
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            action: Selected action (0=left, 1=right, 2=forward)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Update goal from current view (what agent can see NOW)
        self.current_goal = self._get_goal_in_view(obs)
        
        # Get current stacked state
        state = self.get_stacked_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Goal is already normalized from view (no need for additional normalization)
        goal_tensor = torch.FloatTensor(self.current_goal).unsqueeze(0).to(self.device)
        
        # Forward pass through Q-network
        self.q_network.eval()
        with torch.no_grad():
            q_values, new_hidden = self.q_network(
                state_tensor, goal_tensor, self.hidden_state, return_hidden=True
            )
            # Update hidden state for next step
            self.hidden_state = (new_hidden[0].detach(), new_hidden[1].detach())
        
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in current episode buffer.
        
        Note: self.current_goal is already normalized from _get_goal_in_view()
        """
        # Current goal is already normalized, use it directly
        self.current_episode.append((state, self.current_goal, action, reward, next_state, done))
    
    def process_episode(self):
        """Process completed episode into replay buffer."""
        for transition in self.current_episode:
            state, goal, action, reward, next_state, done = transition
            self.memory.push(state, goal, action, reward, next_state, done)
        
        self.current_episode = []
    
    def train(self):
        """Train Q-network on sampled sequences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample sequences
        sequences = self.memory.sample(self.batch_size)
        if sequences is None:
            return 0.0
        
        total_loss = 0.0
        
        self.q_network.train()
        
        for sequence in sequences:
            # Unpack sequence
            states = torch.stack([
                torch.FloatTensor(s[0]) for s in sequence
            ]).to(self.device)
            goals = torch.stack([
                torch.FloatTensor(s[1]) for s in sequence
            ]).to(self.device)
            actions = torch.tensor([s[2] for s in sequence], dtype=torch.long).to(self.device)
            rewards = torch.tensor([s[3] for s in sequence], dtype=torch.float32).to(self.device)
            next_states = torch.stack([
                torch.FloatTensor(s[4]) for s in sequence
            ]).to(self.device)
            dones = torch.tensor([s[5] for s in sequence], dtype=torch.bool).to(self.device)
            
            # Add batch dimension
            states = states.unsqueeze(0)  # (1, seq_len, k, H, W)
            goals = goals.unsqueeze(0)  # (1, seq_len, 2)
            next_states = next_states.unsqueeze(0)
            
            seq_len = states.size(1)
            
            # Initialize hidden states
            h_0 = torch.zeros(1, 1, self.lstm_hidden_dim).to(self.device)
            c_0 = torch.zeros(1, 1, self.lstm_hidden_dim).to(self.device)
            init_hidden = (h_0, c_0)
            
            # Forward pass through Q-network
            q_values_list = []
            hidden = init_hidden
            
            for t in range(seq_len):
                q_vals, hidden = self.q_network(
                    states[:, t], goals[:, t], hidden, return_hidden=True
                )
                q_values_list.append(q_vals)
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            q_values = torch.stack(q_values_list, dim=1).squeeze(0)  # (seq_len, num_actions)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-values (Double DQN)
            with torch.no_grad():
                # Get next actions from online network
                next_q_list = []
                hidden = init_hidden
                
                for t in range(seq_len):
                    nq, hidden = self.q_network(
                        next_states[:, t], goals[:, t], hidden, return_hidden=True
                    )
                    next_q_list.append(nq)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                
                next_q_values = torch.stack(next_q_list, dim=1).squeeze(0)
                next_actions = next_q_values.argmax(1, keepdim=True)
                
                # Evaluate with target network
                target_q_list = []
                hidden = init_hidden
                
                for t in range(seq_len):
                    tq, hidden = self.target_network(
                        next_states[:, t], goals[:, t], hidden, return_hidden=True
                    )
                    target_q_list.append(tq)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                
                target_q_values = torch.stack(target_q_list, dim=1).squeeze(0)
                next_q = target_q_values.gather(1, next_actions).squeeze(1)
                
                target_q = rewards + (self.gamma * next_q * ~dones)
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q, target_q)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return total_loss / len(sequences)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    print("LSTM_WVF_Agent (view-based, no vision model) loaded successfully.")
    print("\nKey features:")
    print("  ✓ Goal conditioning Q(s, a, g) - Universal Value Function approach")
    print("  ✓ LSTM for memory over partial observations")
    print("  ✓ Frame stacking for short-term visual context")
    print("  ✓ No vision model (no reward predictor)")
    print("  ✓ Sequence-based replay for temporal learning")
    print("\nThis is the UVF baseline that parallels 3D WVF compositional learning.")