"""
Task-Conditioned World Value Functions (WVF) Agent for Compositional RL

WORKAROUND IMPLEMENTATION for non-terminating wrong contacts.

Theoretical deviation:
- Original theory: Episode terminates on ANY goal contact, R̄_MIN for wrong goals
- This environment: Episode only terminates on CORRECT goal contact
- Workaround: Give strong negative reward on wrong contact without termination

Key modifications:
1. Wrong contact gives large negative reward (simulating R̄_MIN)
2. Track contacted objects to avoid repeated penalties
3. Modified replay buffer handles non-terminal "wrong goal" transitions
4. Composition formula remains: Q̄_{A∧B} = min{Q̄_A, Q̄_B}

WARNING: This is a practical workaround. Zero-shot performance may be degraded
compared to the theoretically-correct terminating version.
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


class TaskConditionedWVFNetwork(nn.Module):
    """
    Task-Conditioned World Value Function Network.
    
    Learns Q̄_τ(s, g, a) - the extended value function for task τ.
    """
    
    def __init__(self, input_shape=(12, 60, 80), num_goals=4, num_tasks=4,
                 action_size=3, hidden_size=256, task_embed_size=32):
        super(TaskConditionedWVFNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_goals = num_goals
        self.num_tasks = num_tasks
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.task_embed_size = task_embed_size
        
        # Task embedding - learned representation for each primitive task
        self.task_embedding = nn.Embedding(num_tasks, task_embed_size)
        
        # CNN backbone (shared across all tasks and goals)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self._conv_output_size = self._get_conv_output_size(input_shape)
        
        # Combine visual features with task embedding
        self.feature_combine = nn.Sequential(
            nn.Linear(self._conv_output_size + task_embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Dueling architecture: separate value and advantage streams per goal
        self.goal_value_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in range(num_goals)
        ])
        
        self.goal_advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )
            for _ in range(num_goals)
        ])
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, state, task_idx):
        """
        Forward pass - outputs Q̄_τ(s, g, a) for all goals g.
        
        Args:
            state: (batch, channels, height, width)
            task_idx: (batch,) - index of the primitive task τ
            
        Returns:
            q_values: (batch, num_goals, action_size)
        """
        batch_size = state.size(0)
        
        # Visual features
        conv_features = self.conv(state)
        conv_features = conv_features.view(batch_size, -1)
        
        # Task embedding
        task_emb = self.task_embedding(task_idx)
        
        # Combine visual features with task conditioning
        combined = torch.cat([conv_features, task_emb], dim=1)
        features = self.feature_combine(combined)
        
        # Compute Q-values for each goal using dueling architecture
        q_per_goal = []
        for g in range(self.num_goals):
            value = self.goal_value_streams[g](features)
            advantage = self.goal_advantage_streams[g](features)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            q_per_goal.append(q)
        
        q_values = torch.stack(q_per_goal, dim=1)
        return q_values


class NonTerminalWVFReplayBuffer:
    """
    Replay buffer for non-terminating wrong contacts.
    
    Key difference from standard WVF buffer:
    - Wrong contact is stored as a HIGH PENALTY transition but NOT terminal
    - This teaches the agent that wrong goals are bad without ending episode
    
    For each transition, we compute rewards for ALL goals:
    - Correct goal contact: +1, terminal=True
    - Wrong goal contact: R̄_MIN (e.g., -5), terminal=False (workaround!)
    - No contact: step_cost, terminal=False
    """
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, task_idx, action, next_state, contacted_goal_idx,
             is_correct_for_task, episode_terminated):
        """
        Store a transition.
        
        Args:
            state: Current observation
            task_idx: Index of the primitive task being trained
            action: Action taken
            next_state: Next observation  
            contacted_goal_idx: Index of goal contacted (-1 if none)
            is_correct_for_task: Whether contacted goal satisfies current task
            episode_terminated: Whether episode actually ended (only True for correct contact)
        """
        self.buffer.append({
            'state': state,
            'task_idx': task_idx,
            'action': action,
            'next_state': next_state,
            'contacted_goal_idx': contacted_goal_idx,
            'is_correct_for_task': is_correct_for_task,
            'episode_terminated': episode_terminated,
        })
    
    def sample(self, batch_size, num_goals, r_bar_min=-5.0, step_cost=-0.01):
        """
        Sample batch and compute extended rewards R̄ for all goals.
        
        WORKAROUND VERSION:
        - Wrong goal contact: R̄_MIN but NOT terminal (agent continues)
        - This is a deviation from theory but necessary for this env
        
        Args:
            batch_size: Number of transitions to sample
            num_goals: Number of goals
            r_bar_min: Penalty for wrong goal (less harsh than theory since not terminal)
            step_cost: Cost per step
            
        Returns:
            Tuple of arrays for training
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = []
        task_indices = []
        actions = []
        next_states = []
        rewards_per_goal = []
        dones_per_goal = []
        
        for transition in batch:
            states.append(transition['state'])
            task_indices.append(transition['task_idx'])
            actions.append(transition['action'])
            next_states.append(transition['next_state'])
            
            contacted = transition['contacted_goal_idx']
            is_correct = transition['is_correct_for_task']
            terminated = transition['episode_terminated']
            
            rewards = []
            dones = []
            
            for g in range(num_goals):
                if contacted == -1:
                    # No contact - normal step
                    rewards.append(step_cost)
                    dones.append(False)
                elif contacted == g:
                    # We contacted THIS goal g
                    if is_correct and terminated:
                        # Correct goal for task AND episode ended
                        rewards.append(1.0)
                        dones.append(True)
                    else:
                        # We contacted goal g but it wasn't correct for task
                        # OR episode didn't terminate (workaround case)
                        # Give positive reward for reaching this goal (mastery)
                        # but the TASK-SPECIFIC value will be low due to other goals
                        rewards.append(1.0)  # Reached the goal we're evaluating
                        dones.append(terminated)
                else:
                    # We contacted a DIFFERENT goal (contacted != g)
                    if contacted >= 0:
                        # WORKAROUND: R̄_MIN but NOT terminal
                        # This teaches: "if you were trying to reach g, 
                        # but you hit something else, that's bad"
                        rewards.append(r_bar_min)
                        dones.append(terminated)  # Only True if correct for task
                    else:
                        rewards.append(step_cost)
                        dones.append(False)
            
            rewards_per_goal.append(rewards)
            dones_per_goal.append(dones)
        
        return (
            np.array(states),
            np.array(task_indices),
            np.array(actions),
            np.array(next_states),
            np.array(rewards_per_goal, dtype=np.float32),
            np.array(dones_per_goal)
        )
    
    def __len__(self):
        return len(self.buffer)


class UnifiedWorldValueFunctionAgent:
    """
    Task-Conditioned World Value Function Agent - WORKAROUND VERSION
    
    For environments where wrong contacts don't terminate the episode.
    
    Key differences from theory-correct version:
    1. Wrong contacts give R̄_MIN but don't terminate
    2. Agent tracks contacted objects within episode to learn from them
    3. Composition still uses min formula
    
    The zero-shot transfer may be degraded but should still work because:
    - Agent learns which goals are "bad" for each task (via R̄_MIN)
    - Agent learns how to reach all goals (mastery)
    - min composition still selects goals good for ALL tasks
    """
    
    # Goal space G - all terminal states
    GOALS = ['red_box', 'blue_box', 'red_sphere', 'blue_sphere']
    GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}
    IDX_TO_GOAL = {i: g for i, g in enumerate(GOALS)}
    
    # Primitive tasks τ
    PRIMITIVES = ['red', 'blue', 'box', 'sphere']
    TASK_TO_IDX = {t: i for i, t in enumerate(PRIMITIVES)}
    IDX_TO_TASK = {i: t for i, t in enumerate(PRIMITIVES)}
    
    # Which goals satisfy each primitive task
    TASK_GOALS = {
        'red': {'red_box', 'red_sphere'},
        'blue': {'blue_box', 'blue_sphere'},
        'box': {'red_box', 'blue_box'},
        'sphere': {'red_sphere', 'blue_sphere'},
    }
    
    def __init__(self, env, k_frames=4, learning_rate=0.0003, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=64,
                 hidden_size=256, lstm_size=128,
                 tau=0.005, grad_clip=10.0,
                 r_bar_min=-5.0,  # Less harsh since not terminal
                 # Kept for API compatibility
                 seq_len=4, r_correct=1.0, r_wrong=-0.1, step_penalty=-0.01):
        
        self.env = env
        self.action_dim = 3
        self.k_frames = k_frames
        self.num_goals = len(self.GOALS)
        self.num_tasks = len(self.PRIMITIVES)
        
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
        self.memory_size = memory_size
        
        # Extended reward parameters
        self.r_bar_min = r_bar_min
        self.step_cost = step_penalty
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Task-Conditioned WVF Agent (WORKAROUND) using device: {self.device}")
        
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
        print(f"Number of goals: {self.num_goals} - {self.GOALS}")
        print(f"Number of tasks: {self.num_tasks} - {self.PRIMITIVES}")
        print(f"R̄_MIN (workaround): {self.r_bar_min}")
        print(f"NOTE: Using non-terminating workaround for wrong contacts")
        
        # Frame stacking
        self.frame_stack = FrameStack(k=k_frames)
        
        # Current task
        self.current_task = None
        self.current_task_idx = None
        
        # Track contacts within episode (for workaround)
        self.episode_contacts = set()
        
        # Create networks
        self._create_networks()
        
        # Training tracking
        self.training_steps = 0
    
    def _create_networks(self):
        """Create online and target networks."""
        
        self.q_network = TaskConditionedWVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            num_tasks=self.num_tasks,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            task_embed_size=32
        ).to(self.device)
        
        self.target_network = TaskConditionedWVFNetwork(
            input_shape=self.obs_shape,
            num_goals=self.num_goals,
            num_tasks=self.num_tasks,
            action_size=self.action_dim,
            hidden_size=self.hidden_size,
            task_embed_size=32
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = NonTerminalWVFReplayBuffer(capacity=self.memory_size)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Network parameters: {total_params:,}")
    
    def sample_task(self):
        """Sample a random primitive task for training."""
        return random.choice(self.PRIMITIVES)
    
    def set_task(self, task_name):
        """Set the current task."""
        self.current_task = task_name
        self.current_task_idx = self.TASK_TO_IDX[task_name]
    
    def preprocess_frame(self, obs):
        """Convert observation to proper format."""
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
        """Reset for new episode."""
        frame = self.preprocess_frame(obs)
        stacked = self.frame_stack.reset(frame)
        
        if task_name is not None:
            self.set_task(task_name)
        
        # Reset episode contact tracking
        self.episode_contacts = set()
        
        return stacked
    
    def step_episode(self, obs):
        """Process new observation."""
        frame = self.preprocess_frame(obs)
        return self.frame_stack.step(frame)
    
    def get_contacted_goal_idx(self, info):
        """
        Get index of contacted goal from environment info.
        Returns -1 if no contact.
        """
        contacted = info.get('contacted_object', None)
        if contacted is None or contacted not in self.GOAL_TO_IDX:
            return -1
        return self.GOAL_TO_IDX[contacted]
    
    def check_contact_correct_for_task(self, info, task_name):
        """Check if the contacted object is correct for the given task."""
        contacted = info.get('contacted_object', None)
        if contacted is None:
            return False
        return contacted in self.TASK_GOALS[task_name]
    
    def select_action(self, stacked_obs, epsilon=None):
        """
        Select action during TRAINING using current task's Q̄_τ.
        
        Policy: π(s) = argmax_a [max_g Q̄_τ(s, g, a)] for g in task's goal set
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        task_idx = torch.LongTensor([self.current_task_idx]).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state, task_idx)
            q_values = q_values[0]  # (num_goals, num_actions)
            
            # Get valid goals for current task
            valid_goals = self.TASK_GOALS[self.current_task]
            valid_goal_indices = [self.GOAL_TO_IDX[g] for g in valid_goals]
            
            # Get Q-values for valid goals only
            q_valid = q_values[valid_goal_indices]
            
            # max over goals, argmax over actions
            q_max_per_action = q_valid.max(dim=0)[0]
            return q_max_per_action.argmax().item()
    
    def select_action_composed(self, stacked_obs, features, epsilon=0.0):
        """
        ZERO-SHOT COMPOSITION: Select action for compositional task.
        
        Q̄_{A∧B}(s, g, a) = min{Q̄_A(s, g, a), Q̄_B(s, g, a)}
        π(s) = argmax_a [max_{g ∈ valid} Q̄_{A∧B}(s, g, a)]
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Q̄_τ for each task in composition
            q_per_task = []
            for feature in features:
                task_idx = torch.LongTensor([self.TASK_TO_IDX[feature]]).to(self.device)
                q_task = self.q_network(state, task_idx)
                q_per_task.append(q_task[0])
            
            # Stack and take min (composition)
            q_stacked = torch.stack(q_per_task, dim=0)
            q_composed = q_stacked.min(dim=0)[0]  # (num_goals, num_actions)
            
            # Find valid goals (intersection)
            valid_goals = set(self.TASK_GOALS[features[0]])
            for feature in features[1:]:
                valid_goals = valid_goals.intersection(self.TASK_GOALS[feature])
            
            if len(valid_goals) == 0:
                print(f"WARNING: No valid goals for composition {features}")
                q_max_per_action = q_composed.max(dim=0)[0]
                return q_max_per_action.argmax().item()
            
            valid_goal_indices = [self.GOAL_TO_IDX[g] for g in valid_goals]
            q_valid = q_composed[valid_goal_indices]
            
            q_max_per_action = q_valid.max(dim=0)[0]
            return q_max_per_action.argmax().item()
    
    def remember(self, state, task_idx, action, reward, next_state, done):
        """Backward-compatible remember."""
        pass
    
    def remember_with_goal(self, state, action, next_state, contacted_goal_idx,
                           is_correct_for_task, episode_terminated):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current stacked observation
            action: Action taken
            next_state: Next stacked observation
            contacted_goal_idx: Index of contacted goal (-1 if none)
            is_correct_for_task: Whether contact was correct for current task
            episode_terminated: Whether episode actually ended
        """
        # Track contacts for this episode
        if contacted_goal_idx >= 0:
            self.episode_contacts.add(contacted_goal_idx)
        
        self.memory.push(
            state=state,
            task_idx=self.current_task_idx,
            action=action,
            next_state=next_state,
            contacted_goal_idx=contacted_goal_idx,
            is_correct_for_task=is_correct_for_task,
            episode_terminated=episode_terminated
        )
    
    def train_step(self):
        """Train on a batch of transitions."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, task_indices, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size, self.num_goals,
            r_bar_min=self.r_bar_min,
            step_cost=self.step_cost
        )
        
        states_t = torch.FloatTensor(states).to(self.device)
        task_indices_t = torch.LongTensor(task_indices).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        
        batch_size = states_t.shape[0]
        
        # Current Q-values
        q_values = self.q_network(states_t, task_indices_t)
        actions_expanded = actions_t.view(batch_size, 1, 1).expand(-1, self.num_goals, 1)
        current_q = q_values.gather(2, actions_expanded).squeeze(2)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_q_online = self.q_network(next_states_t, task_indices_t)
            next_actions = next_q_online.argmax(dim=2, keepdim=True)
            
            next_q_target = self.target_network(next_states_t, task_indices_t)
            next_q = next_q_target.gather(2, next_actions).squeeze(2)
            
            target_q = rewards_t + self.gamma * next_q * (~dones_t).float()
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        self.training_steps += 1
        self._soft_update_target()
        
        return loss.item()
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset exploration rate."""
        self.epsilon = self.epsilon_start
    
    def update_task_success(self, task_name, success):
        """Track task success (for compatibility)."""
        pass
    
    def save_model(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'obs_shape': self.obs_shape,
            'k_frames': self.k_frames,
            'num_goals': self.num_goals,
            'num_tasks': self.num_tasks,
            'hidden_size': self.hidden_size,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, filepath)
        print(f"Task-Conditioned WVF model (WORKAROUND) saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        
        print(f"Task-Conditioned WVF model (WORKAROUND) loaded from {filepath}")