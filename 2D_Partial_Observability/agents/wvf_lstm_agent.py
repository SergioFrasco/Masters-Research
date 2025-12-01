"""
LSTM-WVF Agent

Combines the best of three approaches:
1. LSTM-DQN: Frame stacking + LSTM memory for partial observability
2. WVF: Goal-conditioned Q-learning for multi-goal navigation  
3. Successor Agent: Learned reward prediction + path integration

Key Features:
- Does NOT require explicit goal detection (learns from rewards)
- Maintains LSTM hidden state across episode for memory
- Uses path integration for ego→allo coordinate transformation
- Retrospectively trains reward predictor when goals are found
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from models import LSTM_WVF, FrameStack, SequenceReplayBuffer, RewardPredictor


class LSTM_WVF_Agent:
    """
    LSTM-based World Value Function Agent for partial observability.
    
    Learns to navigate to goals without explicit goal detection by:
    1. Learning reward locations from experience (RewardPredictor)
    2. Using LSTM memory to handle partial observability
    3. Conditioning on goals for flexible multi-goal behavior
    """
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=5000, batch_size=8, sequence_length=16,
                 frame_stack_k=4, target_update_freq=100, lstm_hidden_dim=128,
                 trajectory_buffer_size=10, reward_threshold=0.5):
        """
        Initialize the LSTM-WVF agent.
        
        Args:
            env: MiniGrid environment
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            epsilon_start/end/decay: Exploration parameters
            memory_size: Capacity of sequence replay buffer
            batch_size: Number of sequences to sample for training
            sequence_length: Length of sequences for LSTM training
            frame_stack_k: Number of frames to stack
            target_update_freq: How often to update target network
            lstm_hidden_dim: Hidden dimension of LSTM
            trajectory_buffer_size: Steps to remember for retrospective training
            reward_threshold: Threshold for considering a position as a goal
        """
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3  # left, right, forward
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_update_freq = target_update_freq
        self.reward_threshold = reward_threshold
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ==================== Memory Components ====================
        # Frame stacking (immediate temporal context)
        self.frame_stack_k = frame_stack_k
        self.frame_stack = FrameStack(k=frame_stack_k)
        
        # LSTM hidden state (long-term memory)
        self.hidden_state = None
        
        # Sequence replay buffer
        self.memory = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        # Current episode buffer
        self.current_episode = []
        
        # Trajectory buffer for retrospective reward predictor training
        self.trajectory_buffer = deque(maxlen=trajectory_buffer_size)
        
        # ==================== Neural Networks ====================
        # Main Q-network (goal-conditioned)
        self.q_network = LSTM_WVF(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            goal_dim=2,
            num_actions=self.action_dim
        ).to(self.device)
        
        # Target Q-network
        self.target_network = LSTM_WVF(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            goal_dim=2,
            num_actions=self.action_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Reward predictor (learns to predict where rewards are)
        self.reward_predictor = RewardPredictor(input_channels=1).to(self.device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.rp_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=learning_rate * 10)
        
        # Loss functions
        self.q_loss_fn = nn.MSELoss()
        self.rp_loss_fn = nn.MSELoss()
        
        # ==================== Path Integration ====================
        self.internal_pos = None
        self.internal_dir = None
        self.path_initialized = False
        
        # ==================== Reward Map ====================
        # Allocentric reward map (accumulated from predictions and ground truth)
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # ==================== Training Counters ====================
        self.update_counter = 0
        self.rp_training_count = 0
        
        print(f"LSTM-WVF Agent initialized on {self.device}")
        print(f"  Frame stack: {frame_stack_k}, Sequence length: {sequence_length}")
        print(f"  LSTM hidden dim: {lstm_hidden_dim}")
        print(f"  Reward threshold: {reward_threshold}")
    
    # ==================== Episode Management ====================
    
    def reset_episode(self, initial_obs):
        """
        Reset for a new episode.
        
        Args:
            initial_obs: First observation of the episode
        """
        # Extract and initialize frame stack
        frame = self._extract_frame(initial_obs)
        self.frame_stack.reset(frame)
        
        # Reset LSTM hidden state
        self.hidden_state = None
        
        # Clear episode buffer
        self.current_episode = []
        
        # Clear trajectory buffer
        self.trajectory_buffer.clear()
        
        # Reset reward map and visited positions
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Initialize path integration
        self._initialize_path_integration()
    
    def _initialize_path_integration(self):
        """Initialize internal position and direction from environment."""
        self.internal_pos = list(self.env.agent_pos)
        self.internal_dir = self.env.agent_dir
        self.path_initialized = True
    
    def reset_path_integration(self):
        """Reset path integration state."""
        self.internal_pos = None
        self.internal_dir = None
        self.path_initialized = False
    
    # ==================== Observation Processing ====================
    
    def _extract_frame(self, obs):
        """
        Extract a single frame from observation.
        
        Args:
            obs: Observation dict with 'image' key
            
        Returns:
            2D numpy array of shape (7, 7)
        """
        if isinstance(obs, dict):
            if 'image' in obs:
                frame = obs['image']
                # Handle transposed observation
                if len(frame.shape) == 3:
                    # Take first channel (object type)
                    if frame.shape[0] < frame.shape[2]:
                        frame = frame[0]  # (C, H, W) -> (H, W)
                    else:
                        frame = frame[:, :, 0]  # (H, W, C) -> (H, W)
            else:
                frame = np.zeros((7, 7), dtype=np.float32)
        else:
            frame = obs
            if len(frame.shape) == 3:
                frame = frame[0] if frame.shape[0] < frame.shape[2] else frame[:, :, 0]
        
        return np.array(frame, dtype=np.float32)
    
    def _create_normalized_view(self, obs):
        """
        Create normalized 7x7 view for reward predictor.
        
        Converts object types to reward-relevant representation:
        - Walls: 0.0
        - Empty: 0.0  
        - Goal: 1.0
        
        Args:
            obs: Observation dict
            
        Returns:
            Normalized 7x7 array
        """
        frame = self._extract_frame(obs)
        normalized = np.zeros((7, 7), dtype=np.float32)
        
        # MiniGrid object types
        GOAL = 8
        
        normalized[frame == GOAL] = 1.0
        
        return normalized
    
    def get_stacked_state(self):
        """
        Get current stacked frame state as tensor.
        
        Returns:
            Tensor of shape (frame_stack, 7, 7)
        """
        stacked = self.frame_stack.get_stack()
        stacked = np.array(stacked, dtype=np.float32)
        return torch.FloatTensor(stacked).to(self.device) / 10.0
    
    # ==================== Path Integration ====================
    
    def update_internal_state(self, action):
        """
        Update internal position and direction based on action.
        
        Args:
            action: Action taken (0=left, 1=right, 2=forward)
        """
        if not self.path_initialized:
            return
        
        TURN_LEFT = 0
        TURN_RIGHT = 1
        MOVE_FORWARD = 2
        
        x, y = self.internal_pos
        direction = self.internal_dir
        
        if action == TURN_LEFT:
            self.internal_dir = (direction - 1) % 4
        elif action == TURN_RIGHT:
            self.internal_dir = (direction + 1) % 4
        elif action == MOVE_FORWARD:
            # Direction: 0=right, 1=down, 2=left, 3=up
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            new_x, new_y = x + dx, y + dy
            
            # Check if valid move
            if self._is_valid_position(new_x, new_y):
                self.internal_pos = [new_x, new_y]
    
    def _is_valid_position(self, x, y):
        """Check if position is valid (within bounds and not a wall)."""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        cell = self.env.grid.get(x, y)
        if cell is not None:
            from minigrid.core.world_object import Wall
            if isinstance(cell, Wall):
                return False
        return True
    
    # ==================== Reward Map Management ====================
    
    def get_goals_from_reward_map(self):
        """
        Extract goal positions from the reward map.
        
        Returns:
            List of (x, y) tuples where rewards are predicted/known
        """
        goals = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.true_reward_map[y, x] >= self.reward_threshold:
                    goals.append((x, y))
        return goals
    
    def _normalize_goal(self, goal_xy):
        """Normalize goal position to [0, 1] range."""
        x, y = goal_xy
        return np.array([x / (self.grid_size - 1), y / (self.grid_size - 1)], dtype=np.float32)
    
    def update_reward_map_from_prediction(self, obs):
        """
        Update allocentric reward map using reward predictor output.
        
        Maps 7x7 egocentric prediction to global coordinates using path integration.
        
        Args:
            obs: Current observation
        """
        # Create input for reward predictor
        normalized_view = self._create_normalized_view(obs)
        input_tensor = torch.tensor(
            normalized_view[np.newaxis, np.newaxis, ...], 
            dtype=torch.float32
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            self.reward_predictor.eval()
            predicted = self.reward_predictor(input_tensor)
            predicted_map = predicted.squeeze().cpu().numpy()
        
        # Map to global coordinates
        agent_x, agent_y = self.internal_pos
        agent_dir = self.internal_dir
        
        ego_center_x, ego_center_y = 3, 6  # Agent position in 7x7 view
        
        for view_y in range(7):
            for view_x in range(7):
                # Calculate offset from agent position in ego view
                dx_ego = view_x - ego_center_x
                dy_ego = view_y - ego_center_y
                
                # Rotate based on agent direction to get world offsets
                if agent_dir == 3:  # Up (north)
                    dx_world, dy_world = dx_ego, dy_ego
                elif agent_dir == 0:  # Right (east)
                    dx_world, dy_world = -dy_ego, dx_ego
                elif agent_dir == 1:  # Down (south)
                    dx_world, dy_world = -dx_ego, -dy_ego
                elif agent_dir == 2:  # Left (west)
                    dx_world, dy_world = dy_ego, -dx_ego
                else:
                    dx_world, dy_world = dx_ego, dy_ego
                
                global_x = agent_x + dx_world
                global_y = agent_y + dy_world
                
                # Update reward map if within bounds and not visited
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    if not self.visited_positions[global_y, global_x]:
                        self.true_reward_map[global_y, global_x] = predicted_map[view_y, view_x]
        
        # Mark current position as visited
        self.visited_positions[agent_y, agent_x] = True
    
    def mark_goal_found(self, reward_pos):
        """
        Mark a position as definitely containing a reward.
        
        Args:
            reward_pos: (x, y) position where reward was found
        """
        x, y = reward_pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.true_reward_map[y, x] = 1.0
    
    # ==================== Action Selection ====================
    
    def select_action(self, obs, epsilon=None):
        """
        Select action using goal-conditioned Q-values with epsilon-greedy exploration.
        
        Strategy:
        1. Get goals from reward map
        2. For each goal, compute Q(s, g, a) using LSTM
        3. Select action with highest Q-value across all goals
        
        Args:
            obs: Current observation
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Selected action (0, 1, or 2)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Get goals from reward map
        goals = self.get_goals_from_reward_map()
        
        # If no goals known, explore randomly
        if len(goals) == 0:
            return random.randint(0, self.action_dim - 1)
        
        # Get current state from frame stack
        state = self.get_stacked_state().unsqueeze(0)  # (1, frame_stack, 7, 7)
        
        best_q = float('-inf')
        best_action = None
        
        self.q_network.eval()
        with torch.no_grad():
            for goal in goals:
                # Normalize goal
                norm_goal = self._normalize_goal(goal)
                goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get Q-values for this goal
                q_values, new_hidden = self.q_network(
                    state, goal_tensor, self.hidden_state, return_hidden=True
                )
                
                # Find best action for this goal
                max_q = q_values.max().item()
                if max_q > best_q:
                    best_q = max_q
                    best_action = q_values.argmax().item()
                    # Store hidden state from best goal evaluation
                    self.hidden_state = new_hidden
        
        if best_action is None:
            return random.randint(0, self.action_dim - 1)
        
        return best_action
    
    # ==================== Experience Storage ====================
    
    def store_step_info(self, obs):
        """
        Store step information for retrospective reward predictor training.
        
        Args:
            obs: Current observation
        """
        step_info = {
            'normalized_view': self._create_normalized_view(obs),
            'agent_pos': tuple(self.internal_pos),
            'agent_dir': self.internal_dir
        }
        self.trajectory_buffer.append(step_info)
    
    def store_transition(self, state, action, reward, next_state, done, goals):
        """
        Store transition in current episode buffer.
        
        Args:
            state: Stacked frames tensor
            action: Action taken
            reward: Reward received
            next_state: Next stacked frames tensor
            done: Terminal flag
            goals: List of current goal positions
        """
        # Store with all current goals (for multi-goal training)
        for goal in goals:
            self.current_episode.append((state, action, reward, next_state, done, goal))
    
    def process_episode(self):
        """
        Process completed episode into sequences for replay buffer.
        Creates overlapping sequences for LSTM training.
        """
        episode_length = len(self.current_episode)
        
        if episode_length < self.sequence_length:
            return
        
        # Create sequences with sliding window (50% overlap)
        stride = max(1, self.sequence_length // 2)
        for start_idx in range(0, episode_length - self.sequence_length + 1, stride):
            sequence = self.current_episode[start_idx:start_idx + self.sequence_length]
            self.memory.push_sequence(sequence)
    
    # ==================== Training ====================
    
    def train_q_network(self):
        """
        Train the goal-conditioned Q-network using sequence-based learning.
        
        Returns:
            Average loss across batch
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch_sequences = self.memory.sample(self.batch_size)
        total_loss = 0.0
        
        self.q_network.train()
        
        for sequence in batch_sequences:
            # Extract sequence data
            states = torch.stack([s[0] for s in sequence]).to(self.device)
            actions = torch.tensor([s[1] for s in sequence], dtype=torch.long).to(self.device)
            rewards = torch.tensor([s[2] for s in sequence], dtype=torch.float32).to(self.device)
            next_states = torch.stack([s[3] for s in sequence]).to(self.device)
            dones = torch.tensor([s[4] for s in sequence], dtype=torch.bool).to(self.device)
            goals = torch.tensor(
                [self._normalize_goal(s[5]) for s in sequence], 
                dtype=torch.float32
            ).to(self.device)
            
            # Add batch dimension
            states = states.unsqueeze(0)  # (1, seq_len, frame_stack, H, W)
            next_states = next_states.unsqueeze(0)
            goals = goals.unsqueeze(0)  # (1, seq_len, 2)
            
            # Forward pass
            current_q_values = self.q_network(states, goals[:, 0, :], hidden_state=None)
            current_q_values = current_q_values.squeeze(0)  # (seq_len, num_actions)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets using target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states, goals[:, 0, :], hidden_state=None)
                next_q_values = next_q_values.squeeze(0)
                max_next_q = next_q_values.max(1)[0]
                targets = rewards + (self.gamma * max_next_q * ~dones)
            
            # Compute loss
            loss = self.q_loss_fn(current_q, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backprop
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.q_optimizer.step()
            
            total_loss += loss.item()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return total_loss / len(batch_sequences) if batch_sequences else 0.0
    
    def train_reward_predictor_retrospective(self, reward_pos):
        """
        Retrospectively train the reward predictor when a goal is found.
        
        For each past step in trajectory buffer, create a target showing
        where the reward was relative to that position.
        
        Args:
            reward_pos: (x, y) position where reward was found
        """
        if len(self.trajectory_buffer) == 0:
            return 0.0
        
        batch_inputs = []
        batch_targets = []
        
        for step_info in self.trajectory_buffer:
            target_view = self._create_target_view_with_reward(
                step_info['agent_pos'],
                step_info['agent_dir'],
                reward_pos
            )
            batch_inputs.append(step_info['normalized_view'])
            batch_targets.append(target_view)
        
        # Also include current position (agent is on goal)
        current_target = np.zeros((7, 7), dtype=np.float32)
        current_target[6, 3] = 1.0  # Goal at agent's position in ego view
        batch_inputs.append(self._create_normalized_view_at_goal())
        batch_targets.append(current_target)
        
        # Train
        return self._train_reward_predictor_batch(batch_inputs, batch_targets)
    
    def _create_normalized_view_at_goal(self):
        """Create normalized view when agent is standing on goal."""
        view = np.zeros((7, 7), dtype=np.float32)
        view[6, 3] = 1.0  # Agent position in egocentric view
        return view
    
    def _create_target_view_with_reward(self, past_pos, past_dir, reward_pos):
        """
        Create target 7x7 view showing where reward is from a past position.
        
        Args:
            past_pos: (x, y) agent position at past step
            past_dir: Agent direction at past step
            reward_pos: (x, y) global position of reward
            
        Returns:
            7x7 target array with 1.0 at reward location
        """
        target = np.zeros((7, 7), dtype=np.float32)
        
        past_x, past_y = past_pos
        reward_x, reward_y = reward_pos
        
        ego_center_x, ego_center_y = 3, 6
        
        # Calculate global offset from past position to reward
        dx_world = reward_x - past_x
        dy_world = reward_y - past_y
        
        # Rotate world offset to egocentric based on past direction
        if past_dir == 3:  # Up (north)
            dx_ego, dy_ego = dx_world, dy_world
        elif past_dir == 0:  # Right (east)
            dx_ego, dy_ego = dy_world, -dx_world
        elif past_dir == 1:  # Down (south)
            dx_ego, dy_ego = -dx_world, -dy_world
        elif past_dir == 2:  # Left (west)
            dx_ego, dy_ego = -dy_world, dx_world
        else:
            dx_ego, dy_ego = dx_world, dy_world
        
        # Calculate ego view coordinates
        view_x = ego_center_x + dx_ego
        view_y = ego_center_y + dy_ego
        
        # Check if within view bounds
        if 0 <= view_x < 7 and 0 <= view_y < 7:
            target[int(view_y), int(view_x)] = 1.0
        
        return target
    
    def _train_reward_predictor_batch(self, inputs, targets):
        """
        Train reward predictor on a batch.
        
        Args:
            inputs: List of normalized 7x7 views
            targets: List of target 7x7 reward maps
            
        Returns:
            Loss value
        """
        input_batch = np.stack([inp[np.newaxis, ...] for inp in inputs])
        target_batch = np.stack([tgt[np.newaxis, ...] for tgt in targets])
        
        input_tensor = torch.tensor(input_batch, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_batch, dtype=torch.float32).to(self.device)
        
        self.reward_predictor.train()
        self.rp_optimizer.zero_grad()
        
        output = self.reward_predictor(input_tensor)
        loss = self.rp_loss_fn(output, target_tensor)
        
        loss.backward()
        self.rp_optimizer.step()
        
        self.rp_training_count += 1
        
        return loss.item()
    
    def train_reward_predictor_online(self, obs, target_7x7):
        """
        Train reward predictor on current observation if prediction differs from known state.
        
        Args:
            obs: Current observation
            target_7x7: Known reward map in egocentric view
            
        Returns:
            Whether training was triggered, and loss if so
        """
        # Get prediction
        normalized_view = self._create_normalized_view(obs)
        input_tensor = torch.tensor(
            normalized_view[np.newaxis, np.newaxis, ...],
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            self.reward_predictor.eval()
            predicted = self.reward_predictor(input_tensor)
            predicted_map = predicted.squeeze().cpu().numpy()
        
        # Check error
        error = np.abs(predicted_map - target_7x7)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        if max_error > 0.05 or mean_error > 0.01:
            # Train on this sample
            return True, self._train_reward_predictor_batch([normalized_view], [target_7x7])
        
        return False, 0.0
    
    # ==================== Utility Methods ====================
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_all_q_values(self):
        """
        Get Q-values for all possible goals (for visualization).
        
        Returns:
            Array of shape (grid_size, grid_size, num_actions)
        """
        q_values = np.zeros((self.grid_size, self.grid_size, self.action_dim))
        state = self.get_stacked_state().unsqueeze(0)
        
        self.q_network.eval()
        with torch.no_grad():
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    norm_goal = self._normalize_goal((x, y))
                    goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q = self.q_network(state, goal_tensor)
                    q_values[y, x, :] = q.squeeze().cpu().numpy()
        
        return q_values


if __name__ == "__main__":
    print("LSTM-WVF Agent module loaded successfully.")
    print("\nUsage:")
    print("  from lstm_wvf_agent import LSTM_WVF_Agent")
    print("  agent = LSTM_WVF_Agent(env)")
    print("  agent.reset_episode(initial_obs)")
    print("  action = agent.select_action(obs)")