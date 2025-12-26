"""
Unified World Value Functions (WVF) Agent for Compositional RL

CORRECTED IMPLEMENTATION based on Nangue Tasse et al.'s actual approach.

Key insight from the Boolean Task Algebra paper:
- Learn Extended Value Functions (EVF): Q̄(s, g, a) for EACH goal g
- Extended reward: R̄_MIN penalty when reaching wrong goal
- This teaches agent how to reach each specific goal separately
- Composition: Q̄*_{B AND S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}

The agent learns Q̄ for ALL 4 goals during training on ANY task.
At test time, composition picks the goal in the intersection.

Based on:
- "A Boolean Task Algebra for Reinforcement Learning" (NeurIPS 2020)
- "World Value Functions" (RLDM 2022)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


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


class GoalConditionedEVFNetwork(nn.Module):
    """
    Extended Value Function Network - learns Q̄(s, g, a) for ALL goals.
    
    This is the correct WVF architecture:
    - Input: stacked frames (image observation)
    - Output: Q-values for EACH goal and EACH action
    - Shape: (num_goals, num_actions) = (4, 3)
    
    During training, we update Q̄ for ALL goals every step using the
    extended reward function (R̄_MIN for wrong goals).
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_goals=4, action_size=3,
                 hidden_size=256, lstm_size=128):
        super(GoalConditionedEVFNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_goals = num_goals
        self.action_size = action_size
        self.lstm_size = lstm_size
        
        # CNN backbone (shared across all goals)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        self._conv_output_size = self._get_conv_output_size(input_shape[0], 
                                                             input_shape[1], 
                                                             input_shape[2])
        
        # LSTM for temporal reasoning (shared)
        self.lstm = nn.LSTM(
            input_size=self._conv_output_size,
            hidden_size=lstm_size,
            num_layers=1,
            batch_first=True
        )
        
        # Separate heads for each goal (this is key!)
        # Each head outputs Q-values for all actions for that specific goal
        self.goal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
            for _ in range(num_goals)
        ])
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels, height, width):
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
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
    
    def forward(self, state, hidden=None):
        """
        Forward pass - outputs Q̄(s, g, a) for ALL goals.
        
        Args:
            state: (batch, C, H, W) image tensor
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: (batch, num_goals, action_size) - Q̄ for each goal
            hidden: Updated LSTM hidden state
        """
        batch_size = state.size(0)
        
        # CNN features (shared)
        conv_features = self.conv(state)
        conv_features = conv_features.view(batch_size, -1)  # (batch, conv_output)
        
        # LSTM expects (batch, seq_len, features)
        conv_features = conv_features.unsqueeze(1)  # (batch, 1, conv_output)
        
        if hidden is not None:
            lstm_out, hidden = self.lstm(conv_features, hidden)
        else:
            lstm_out, hidden = self.lstm(conv_features)
        
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_size)
        
        # Compute Q-values for each goal using separate heads
        q_per_goal = []
        for head in self.goal_heads:
            q_goal = head(lstm_out)  # (batch, action_size)
            q_per_goal.append(q_goal)
        
        # Stack: (batch, num_goals, action_size)
        q_values = torch.stack(q_per_goal, dim=1)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return (h, c)


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for LSTM training.
    Stores complete episodes and samples sequences for training.
    
    Now stores goal-specific rewards for ALL goals per transition.
    """
    
    def __init__(self, capacity=2000):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def push(self, state, action, rewards_per_goal, next_state, dones_per_goal):
        """
        Add a transition to the current episode.
        
        Args:
            state: Current observation
            action: Action taken
            rewards_per_goal: List of rewards for each goal (length = num_goals)
            next_state: Next observation
            dones_per_goal: List of done flags for each goal (length = num_goals)
        """
        self.current_episode.append((state, action, rewards_per_goal, next_state, dones_per_goal))
        
        # End episode if ANY goal is done (terminal state reached)
        if any(dones_per_goal):
            self.end_episode()
    
    def end_episode(self):
        """Finalize and store the current episode."""
        if len(self.current_episode) > 0:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample(self, batch_size, seq_len=4):
        """Sample sequences from stored episodes."""
        batch = []
        
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            
            if len(episode) > seq_len:
                start = random.randint(0, len(episode) - seq_len)
                sequence = episode[start:start + seq_len]
            else:
                sequence = list(episode)
            
            batch.append(sequence)
        
        return batch
    
    def __len__(self):
        return len(self.episodes)
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = []


class UnifiedWorldValueFunctionAgent:
    """
    CORRECTED World Value Function (WVF) Agent
    
    Based on Geraud Nangue Tasse's actual approach:
    
    1. Learn Q̄(s, g, a) - Extended Value Function for EACH goal g
    2. Use Extended Reward: R̄_MIN penalty when reaching wrong goal
    3. Update Q̄ for ALL goals every step (not just current task)
    4. Composition: Q̄*_{B AND S}(s, g, a) = min{Q̄*_B(s, g, a), Q̄*_S(s, g, a)}
    
    Goals (4 objects): red_box, blue_box, red_sphere, blue_sphere
    
    Tasks define which goals are "good":
        red -> {red_box, red_sphere}
        blue -> {blue_box, blue_sphere}
        box -> {red_box, blue_box}
        sphere -> {red_sphere, blue_sphere}
    
    Zero-shot composition:
        blue AND sphere -> min(Q̄_blue, Q̄_sphere) -> selects blue_sphere
    """
    
    # Goals = all 4 objects (terminal states)
    GOALS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}
    IDX_TO_GOAL = {i: g for i, g in enumerate(GOALS)}
    
    # Primitive tasks and their goal sets
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    TASK_TO_IDX = {t: i for i, t in enumerate(PRIMITIVES)}
    
    # Which goals are "good" (give +reward) for each task
    TASK_GOALS = {
        'red': ['red_box', 'red_sphere'],
        'blue': ['blue_box', 'blue_sphere'],
        'box': ['red_box', 'blue_box'],
        'sphere': ['red_sphere', 'blue_sphere'],
    }
    
    def __init__(self, env, k_frames=4, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=2000, batch_size=16, seq_len=4,
                 hidden_size=256, lstm_size=128,
                 tau=0.005, grad_clip=10.0,
                 r_correct=1.0, r_wrong=-0.1, step_penalty=-0.005,
                 r_bar_min=-10.0):
        """
        Args:
            r_bar_min: The R̄_MIN penalty for reaching wrong goal.
                       This is critical for learning goal-specific values!
                       Should be large negative (paper uses derived bound).
        """
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.num_goals = len(self.GOALS)
        self.seq_len = seq_len
        
        # Reward parameters
        self.r_correct = r_correct       # Reward for reaching correct goal for task
        self.r_wrong = r_wrong           # Reward for reaching wrong goal for task (but still a goal)
        self.step_penalty = step_penalty # Small step penalty
        self.r_bar_min = r_bar_min       # Extended reward penalty (R̄_MIN)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.grad_clip = grad_clip
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"UnifiedWorldValueFunctionAgent (CORRECTED EVF) using device: {self.device}")
        
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
        print(f"Number of goals: {self.num_goals}")
        print(f"Goals: {self.GOALS}")
        print(f"R̄_MIN penalty: {self.r_bar_min}")
        print(f"Learning Q̄(s, g, a) for each goal - CORRECT WVF approach")
        
        # Frame stacker
        self.frame_stack = FrameStack(k=k_frames)
        
        # Create networks
        self._create_networks()
        
        # Current episode state
        self.current_hidden = None
        self.current_task = None
    
    def _create_networks(self):
        """Create goal-conditioned EVF networks."""
        self.q_network = GoalConditionedEVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        
        self.target_network = GoalConditionedEVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            lstm_size=self.lstm_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = EpisodeReplayBuffer(capacity=self.memory_size)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"  Goal-conditioned EVF network parameters: {total_params:,}")
    
    def sample_task(self):
        """Randomly sample a primitive task for this episode."""
        return random.choice(self.PRIMITIVES)
    
    def compute_extended_rewards(self, info, current_task):
        """
        Compute extended rewards for ALL goals.
        
        This is the key insight from WVF:
        - For each goal g, compute R̄(s, g, a)
        - If we reached goal g: normal reward based on task
        - If we reached a DIFFERENT goal: R̄_MIN penalty
        - If no goal reached: step penalty
        
        Args:
            info: Environment info dict with 'contacted_object'
            current_task: Current primitive task (e.g., 'blue')
            
        Returns:
            rewards: List of rewards for each goal (length = num_goals)
            dones: List of done flags for each goal (length = num_goals)
            task_success: Whether current task was satisfied
        """
        contacted = info.get('contacted_object', None)
        task_goal_set = self.TASK_GOALS[current_task]
        
        rewards = []
        dones = []
        task_success = False
        
        if contacted is None:
            # No terminal state reached - step penalty for all goals
            for goal in self.GOALS:
                rewards.append(self.step_penalty)
                dones.append(False)
        else:
            # Terminal state reached!
            for goal in self.GOALS:
                if contacted == goal:
                    # We reached THIS specific goal
                    if goal in task_goal_set:
                        rewards.append(self.r_correct)  # Good for task!
                        task_success = True
                    else:
                        rewards.append(self.r_wrong)    # Wrong for task, but reached goal
                    dones.append(True)
                else:
                    # We reached a DIFFERENT goal - R̄_MIN penalty!
                    rewards.append(self.r_bar_min)
                    dones.append(True)  # Episode ends for all goals
        
        return rewards, dones, task_success
    
    def compute_reward(self, info, current_task):
        """
        Backward-compatible method for experiment_utils.
        Returns task reward and whether goal was reached.
        """
        rewards, dones, task_success = self.compute_extended_rewards(info, current_task)
        
        contacted = info.get('contacted_object', None)
        if contacted is None:
            return self.step_penalty, False
        
        task_goal_set = self.TASK_GOALS[current_task]
        if contacted in task_goal_set:
            return self.r_correct, True
        else:
            return self.r_wrong, True
    
    def preprocess_frame(self, obs):
        """Convert observation to single frame."""
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        if img.shape[0] not in [3, 4]:
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
    
    def reset_episode(self, obs, task_name=None):
        """
        Reset for new episode.
        
        Args:
            obs: Initial observation
            task_name: Optional task name for training
        """
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        # Initialize fresh hidden state for new episode
        self.current_hidden = self.q_network.init_hidden(batch_size=1, device=self.device)
        
        # Set current task if provided
        if task_name is not None:
            self.current_task = task_name
        
        return stacked
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def select_action(self, stacked_obs, task_idx=None, epsilon=None):
        """
        Select action for TRAINING using epsilon-greedy.
        
        During training, we select action based on the TASK's Q-values:
        Q_task(s, a) = max_{g in task_goals} Q̄(s, g, a)
        
        Args:
            stacked_obs: Stacked observation frames
            task_idx: Task index (not used in new implementation, kept for compatibility)
            epsilon: Exploration rate
            
        Returns:
            action: Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Q̄(s, g, a) for all goals: (1, num_goals, num_actions)
            q_all_goals, self.current_hidden = self.q_network(state, self.current_hidden)
            self.current_hidden = (self.current_hidden[0].detach(),
                                   self.current_hidden[1].detach())
            
            # For training, use current task's goal set
            if self.current_task is not None:
                task_goal_indices = [self.GOAL_TO_IDX[g] for g in self.TASK_GOALS[self.current_task]]
                # Q_task(s, a) = max over goals in task's goal set
                q_task = q_all_goals[0, task_goal_indices, :].max(dim=0)[0]  # (num_actions,)
                return q_task.argmax().item()
            else:
                # Fallback: max over all goals
                q_max = q_all_goals[0].max(dim=0)[0]  # (num_actions,)
                return q_max.argmax().item()
    
    def select_action_primitive(self, stacked_obs, task_name, use_target=True):
        """
        Select action for a PRIMITIVE task during evaluation.
        
        Uses the target network for more stable Q-estimates during eval.
        
        Args:
            stacked_obs: Current stacked observation
            task_name: Primitive task name (e.g., 'blue', 'red', 'box', 'sphere')
            use_target: Whether to use target network (more stable for eval)
            
        Returns:
            action: The greedy action for this primitive task
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use target network for eval (more stable)
            network = self.target_network if use_target else self.q_network
            q_all_goals, new_hidden = network(state, self.current_hidden)
            q_all_goals = q_all_goals[0]  # (num_goals, num_actions)
            
            # Update hidden state
            self.current_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            # Get goal indices for this primitive task
            task_goal_indices = [self.GOAL_TO_IDX[g] for g in self.TASK_GOALS[task_name]]
            
            # Q_task(s, a) = max over goals in task's goal set
            q_task = q_all_goals[task_goal_indices, :].max(dim=0)[0]  # (num_actions,)
            
            return q_task.argmax().item()
    
    def select_action_composed(self, stacked_obs, features, use_target=True):
        """
        Select action using Boolean composition (CORRECT WVF approach).
        
        For compositional tasks like "blue AND sphere":
        1. Find goals that satisfy ALL features (intersection)
        2. For valid goals, use their Q-values directly
        3. Take max over valid goals, then argmax over actions
        
        Uses target network for more stable Q-estimates during evaluation.
        
        Args:
            stacked_obs: Current stacked observation
            features: List of primitive tasks to compose (e.g., ['blue', 'sphere'])
            use_target: Whether to use target network (more stable for eval)
            
        Returns:
            action: The action that maximizes the composed Q-value
        """
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use target network for eval (more stable)
            network = self.target_network if use_target else self.q_network
            q_all_goals, new_hidden = network(state, self.current_hidden)
            q_all_goals = q_all_goals[0]  # (num_goals, num_actions)
            
            # Update hidden state
            self.current_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            # Find goals that satisfy ALL features (intersection)
            # For "blue AND sphere", only "blue_sphere" satisfies both
            valid_goal_indices = []
            for goal_idx, goal in enumerate(self.GOALS):
                satisfies_all = all(goal in self.TASK_GOALS[task] for task in features)
                if satisfies_all:
                    valid_goal_indices.append(goal_idx)
            
            if len(valid_goal_indices) == 0:
                # Fallback: no goal satisfies all features (shouldn't happen with valid tasks)
                print(f"WARNING: No goal satisfies all features {features}")
                q_final = q_all_goals.max(dim=0)[0]
            else:
                # Get Q-values only for valid goals (those in the intersection)
                valid_q_values = q_all_goals[valid_goal_indices, :]  # (num_valid_goals, num_actions)
                
                # Max over valid goals to get Q(s, a)
                q_final = valid_q_values.max(dim=0)[0]  # (num_actions,)
            
            return q_final.argmax().item()
    
    def remember(self, state, task_idx, action, reward, next_state, done):
        """
        Store transition - CORRECTED to store rewards for ALL goals.
        
        This method is called from experiment_utils with task_idx,
        but we need to compute extended rewards for all goals.
        
        Note: We need to call compute_extended_rewards separately since
        we don't have `info` here. This is handled in the training loop.
        """
        # This is a simplified version - the actual extended rewards
        # should be computed in the training loop where we have `info`
        pass  # Handled by remember_extended
    
    def remember_extended(self, state, action, rewards_per_goal, next_state, dones_per_goal):
        """
        Store transition with extended rewards for all goals.
        
        Args:
            state: Current observation
            action: Action taken
            rewards_per_goal: List of rewards for each goal
            next_state: Next observation
            dones_per_goal: List of done flags for each goal
        """
        self.memory.push(state, action, rewards_per_goal, next_state, dones_per_goal)
    
    def soft_update_target(self):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """
        Perform one training step - CORRECTED to update Q̄ for ALL goals.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        sequences = self.memory.sample(self.batch_size, self.seq_len)
        max_len = max(len(seq) for seq in sequences)
        
        # Prepare batched data
        states_batch = []
        actions_batch = []
        rewards_batch = []  # Now: (batch, seq, num_goals)
        next_states_batch = []
        dones_batch = []    # Now: (batch, seq, num_goals)
        lengths = []
        
        for seq in sequences:
            seq_len_actual = len(seq)
            lengths.append(seq_len_actual)
            
            states = [s[0] for s in seq]
            actions = [s[1] for s in seq]
            rewards = [s[2] for s in seq]  # List of lists
            next_states = [s[3] for s in seq]
            dones = [s[4] for s in seq]    # List of lists
            
            # Pad sequences to max length
            if seq_len_actual < max_len:
                pad_len = max_len - seq_len_actual
                states.extend([states[-1]] * pad_len)
                actions.extend([0] * pad_len)
                rewards.extend([rewards[-1]] * pad_len)
                next_states.extend([next_states[-1]] * pad_len)
                dones.extend([[True] * self.num_goals] * pad_len)
            
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards_batch)).to(self.device)  # (batch, seq, num_goals)
        next_states_t = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_t = torch.BoolTensor(np.array(dones_batch)).to(self.device)  # (batch, seq, num_goals)
        
        batch_size, seq_len_max = states_t.shape[:2]
        
        # Initialize hidden states
        hidden = self.q_network.init_hidden(batch_size, self.device)
        target_hidden = self.target_network.init_hidden(batch_size, self.device)
        
        # Forward pass through online network
        q_values_list = []
        for t in range(seq_len_max):
            q_vals, hidden = self.q_network(states_t[:, t], hidden)
            q_values_list.append(q_vals)  # (batch, num_goals, num_actions)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        # Stack: (batch, seq, num_goals, num_actions)
        q_values = torch.stack(q_values_list, dim=1)
        
        # Get Q-values for taken actions: (batch, seq, num_goals)
        actions_expanded = actions_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_goals, 1)
        current_q = q_values.gather(3, actions_expanded).squeeze(3)  # (batch, seq, num_goals)
        
        # Target network (Double DQN style)
        with torch.no_grad():
            # Get next actions from online network
            next_q_list = []
            hidden_copy = self.q_network.init_hidden(batch_size, self.device)
            for t in range(seq_len_max):
                nq, hidden_copy = self.q_network(next_states_t[:, t], hidden_copy)
                next_q_list.append(nq)
                hidden_copy = (hidden_copy[0].detach(), hidden_copy[1].detach())
            
            next_q_online = torch.stack(next_q_list, dim=1)  # (batch, seq, num_goals, num_actions)
            next_actions = next_q_online.argmax(dim=3, keepdim=True)  # (batch, seq, num_goals, 1)
            
            # Evaluate actions with target network
            target_q_list = []
            for t in range(seq_len_max):
                tq, target_hidden = self.target_network(next_states_t[:, t], target_hidden)
                target_q_list.append(tq)
                target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())
            
            target_q_values = torch.stack(target_q_list, dim=1)  # (batch, seq, num_goals, num_actions)
            next_q = target_q_values.gather(3, next_actions).squeeze(3)  # (batch, seq, num_goals)
            
            # TD target for each goal
            target_q = rewards_t + (self.gamma * next_q * ~dones_t)
        
        # Masking for variable-length sequences
        loss_mask = torch.zeros(batch_size, seq_len_max, self.num_goals, device=self.device)
        for i, length in enumerate(lengths):
            loss_mask[i, :length, :] = 1.0
        
        # Compute loss for ALL goals
        loss = F.smooth_l1_loss(current_q * loss_mask, target_q * loss_mask, reduction='sum')
        loss = loss / loss_mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        self.soft_update_target()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset exploration rate to initial value."""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_goals': self.num_goals,
            'hidden_size': self.hidden_size,
            'lstm_size': self.lstm_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"WVF model (CORRECTED EVF) saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hidden_size = checkpoint.get('hidden_size', 256)
        self.lstm_size = checkpoint.get('lstm_size', 128)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"WVF model (CORRECTED EVF) loaded from {filepath}")