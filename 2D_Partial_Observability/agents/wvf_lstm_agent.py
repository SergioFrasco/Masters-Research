"""
LSTM-WVF Agent - OPTIMIZED VERSION

Performance fixes:
1. Single goal selection instead of iterating all goals
2. Store single transition per step (not duplicated per goal)
3. Higher thresholds for reward predictor training
4. Cached goal selection with periodic updates
5. Optional expensive computations
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
    OPTIMIZED VERSION - significantly faster training.
    """
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=5000, batch_size=8, sequence_length=16,
                 frame_stack_k=4, target_update_freq=100, lstm_hidden_dim=128,
                 trajectory_buffer_size=10, reward_threshold=0.7,
                 rp_confidence_threshold=0.8):
        
        self.env = env
        self.grid_size = env.size
        self.action_dim = 3
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_update_freq = target_update_freq
        self.reward_threshold = reward_threshold
        self.rp_confidence_threshold = rp_confidence_threshold
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.frame_stack_k = frame_stack_k
        self.frame_stack = FrameStack(k=frame_stack_k)
        
        self.hidden_state = None
        
        self.memory = SequenceReplayBuffer(
            capacity=memory_size,
            sequence_length=sequence_length
        )
        
        self.current_episode = []
        self.trajectory_buffer = deque(maxlen=trajectory_buffer_size)
        
        self.q_network = LSTM_WVF(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            goal_dim=2,
            num_actions=self.action_dim
        ).to(self.device)
        
        self.target_network = LSTM_WVF(
            frame_stack_size=frame_stack_k,
            lstm_hidden_dim=lstm_hidden_dim,
            goal_dim=2,
            num_actions=self.action_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.reward_predictor = RewardPredictor(input_channels=1).to(self.device)
        
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.rp_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=learning_rate * 10)
        
        self.q_loss_fn = nn.MSELoss()
        self.rp_loss_fn = nn.MSELoss()
        
        self.internal_pos = None
        self.internal_dir = None
        self.path_initialized = False
        
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.rp_confidence_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        self.discovered_goals = set()
        
        # === OPTIMIZATION: Cache current goal ===
        self.current_goal = None
        self.goal_update_frequency = 10  # Update goal selection every N steps
        self.steps_since_goal_update = 0
        
        self.update_counter = 0
        self.rp_training_count = 0
        
        print(f"LSTM-WVF Agent (OPTIMIZED) initialized on {self.device}")
    
    def reset_episode(self, initial_obs):
        """Reset for a new episode."""
        frame = self._extract_frame(initial_obs)
        self.frame_stack.reset(frame)
        
        self.hidden_state = None
        self.current_episode = []
        self.trajectory_buffer.clear()
        
        self.true_reward_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.rp_confidence_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # === OPTIMIZATION: Reset goal cache ===
        self.current_goal = None
        self.steps_since_goal_update = 0
        
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
    
    def _extract_frame(self, obs):
        """Extract a single frame from observation."""
        if isinstance(obs, dict):
            if 'image' in obs:
                frame = obs['image']
                if len(frame.shape) == 3:
                    if frame.shape[0] < frame.shape[2]:
                        frame = frame[0]
                    else:
                        frame = frame[:, :, 0]
            else:
                frame = np.zeros((7, 7), dtype=np.float32)
        else:
            frame = obs
            if len(frame.shape) == 3:
                frame = frame[0] if frame.shape[0] < frame.shape[2] else frame[:, :, 0]
        
        return np.array(frame, dtype=np.float32)
    
    def _create_normalized_view(self, obs):
        """Create normalized 7x7 view for reward predictor."""
        frame = self._extract_frame(obs)
        normalized = np.zeros((7, 7), dtype=np.float32)
        
        GOAL = 8
        normalized[frame == GOAL] = 1.0
        
        return normalized
    
    def get_stacked_state(self):
        """Get current stacked frame state as tensor."""
        stacked = self.frame_stack.get_stack()
        stacked = np.array(stacked, dtype=np.float32)
        return torch.FloatTensor(stacked).to(self.device) / 10.0
    
    def update_internal_state(self, action):
        """Update internal position and direction based on action."""
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
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            new_x, new_y = x + dx, y + dy
            
            if self._is_valid_position(new_x, new_y):
                self.internal_pos = [new_x, new_y]
    
    def _is_valid_position(self, x, y):
        """Check if position is valid."""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        cell = self.env.grid.get(x, y)
        if cell is not None:
            from minigrid.core.world_object import Wall
            if isinstance(cell, Wall):
                return False
        return True
    
    def get_goals_from_reward_map(self):
        """Extract goal positions from the reward map with confidence threshold."""
        goals = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (self.true_reward_map[y, x] >= self.reward_threshold and 
                    self.rp_confidence_map[y, x] >= self.rp_confidence_threshold):
                    goals.append((x, y))
        
        for goal in self.discovered_goals:
            if goal not in goals:
                goals.append(goal)
        
        return goals
    
    def _normalize_goal(self, goal_xy):
        """Normalize goal position to [0, 1] range."""
        x, y = goal_xy
        return np.array([x / (self.grid_size - 1), y / (self.grid_size - 1)], dtype=np.float32)
    
    def _select_best_goal(self, goals):
        """
        OPTIMIZATION: Select a single goal to pursue.
        Uses closest goal by Manhattan distance, with some randomness.
        """
        if len(goals) == 0:
            return (self.grid_size // 2, self.grid_size // 2)
        
        if len(goals) == 1:
            return goals[0]
        
        # Prioritize discovered (confirmed) goals
        confirmed = [g for g in goals if g in self.discovered_goals]
        if confirmed:
            goals = confirmed
        
        # Select closest goal with 80% probability, random otherwise
        if random.random() < 0.8 and self.internal_pos is not None:
            agent_x, agent_y = self.internal_pos
            distances = [abs(g[0] - agent_x) + abs(g[1] - agent_y) for g in goals]
            min_idx = np.argmin(distances)
            return goals[min_idx]
        else:
            return random.choice(goals)
    
    def update_reward_map_from_prediction(self, obs):
        """Update allocentric reward map using reward predictor output."""
        normalized_view = self._create_normalized_view(obs)
        input_tensor = torch.tensor(
            normalized_view[np.newaxis, np.newaxis, ...], 
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            self.reward_predictor.eval()
            predicted = self.reward_predictor(input_tensor)
            predicted_map = predicted.squeeze().cpu().numpy()
        
        agent_x, agent_y = self.internal_pos
        agent_dir = self.internal_dir
        
        ego_center_x, ego_center_y = 3, 6
        
        for view_y in range(7):
            for view_x in range(7):
                dx_ego = view_x - ego_center_x
                dy_ego = view_y - ego_center_y
                
                if agent_dir == 3:
                    dx_world, dy_world = dx_ego, dy_ego
                elif agent_dir == 0:
                    dx_world, dy_world = -dy_ego, dx_ego
                elif agent_dir == 1:
                    dx_world, dy_world = -dx_ego, -dy_ego
                elif agent_dir == 2:
                    dx_world, dy_world = dy_ego, -dx_ego
                else:
                    dx_world, dy_world = dx_ego, dy_ego
                
                global_x = agent_x + dx_world
                global_y = agent_y + dy_world
                
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    if not self.visited_positions[global_y, global_x]:
                        pred_val = predicted_map[view_y, view_x]
                        
                        if self.rp_training_count > 100:
                            old_val = self.true_reward_map[global_y, global_x]
                            self.true_reward_map[global_y, global_x] = 0.9 * old_val + 0.1 * pred_val
                            self.rp_confidence_map[global_y, global_x] = min(1.0, 
                                self.rp_confidence_map[global_y, global_x] + 0.01)
                        else:
                            self.true_reward_map[global_y, global_x] = pred_val
                            self.rp_confidence_map[global_y, global_x] = 0.5
        
        self.visited_positions[agent_y, agent_x] = True
    
    def mark_goal_found(self, reward_pos):
        """Mark a position as definitely containing a reward."""
        x, y = reward_pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.true_reward_map[y, x] = 1.0
            self.rp_confidence_map[y, x] = 1.0
            self.discovered_goals.add((x, y))
            # === OPTIMIZATION: Update current goal when we find one ===
            self.current_goal = (x, y)
    
    def select_action(self, obs, epsilon=None):
        """
        OPTIMIZED: Select action using a SINGLE goal instead of iterating all.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # === OPTIMIZATION: Use cached goal, update periodically ===
        self.steps_since_goal_update += 1
        if self.current_goal is None or self.steps_since_goal_update >= self.goal_update_frequency:
            goals = self.get_goals_from_reward_map()
            self.current_goal = self._select_best_goal(goals)
            self.steps_since_goal_update = 0
        
        state = self.get_stacked_state().unsqueeze(0)
        norm_goal = self._normalize_goal(self.current_goal)
        goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values, new_hidden = self.q_network(
                state, goal_tensor, self.hidden_state, return_hidden=True
            )
            self.hidden_state = new_hidden
        
        return q_values.argmax().item()
    
    def get_current_goal(self):
        """Return the currently selected goal for storage."""
        if self.current_goal is None:
            return (self.grid_size // 2, self.grid_size // 2)
        return self.current_goal
    
    def store_step_info(self, obs):
        """Store step information for retrospective reward predictor training."""
        step_info = {
            'normalized_view': self._create_normalized_view(obs),
            'agent_pos': tuple(self.internal_pos),
            'agent_dir': self.internal_dir
        }
        self.trajectory_buffer.append(step_info)
    
    def store_transition(self, state, action, reward, next_state, done, goal):
        """
        OPTIMIZED: Store a SINGLE transition with ONE goal.
        (Previously stored duplicates for each goal)
        """
        # Just store with the single provided goal
        self.current_episode.append((state, action, reward, next_state, done, goal))
    
    def process_episode(self):
        """Process completed episode into sequences for replay buffer."""
        episode_length = len(self.current_episode)
        
        if episode_length < self.sequence_length:
            return
        
        stride = max(1, self.sequence_length // 2)
        for start_idx in range(0, episode_length - self.sequence_length + 1, stride):
            sequence = self.current_episode[start_idx:start_idx + self.sequence_length]
            self.memory.push_sequence(sequence)
    
    def train_q_network(self):
        """Train the goal-conditioned Q-network using sequence-based learning."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch_sequences = self.memory.sample(self.batch_size)
        total_loss = 0.0
        
        self.q_network.train()
        
        for sequence in batch_sequences:
            states = torch.stack([s[0] for s in sequence]).to(self.device)
            actions = torch.tensor([s[1] for s in sequence], dtype=torch.long).to(self.device)
            rewards = torch.tensor([s[2] for s in sequence], dtype=torch.float32).to(self.device)
            next_states = torch.stack([s[3] for s in sequence]).to(self.device)
            dones = torch.tensor([s[4] for s in sequence], dtype=torch.bool).to(self.device)
            
            goals_list = [s[5] for s in sequence]
            norm_goals = torch.tensor([self._normalize_goal(g) for g in goals_list], 
                                     dtype=torch.float32).to(self.device)
            
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            norm_goals = norm_goals.unsqueeze(0)
            
            h_0 = torch.zeros(1, 1, self.q_network.lstm_hidden_dim).to(self.device)
            c_0 = torch.zeros(1, 1, self.q_network.lstm_hidden_dim).to(self.device)
            init_hidden = (h_0, c_0)
            
            current_q_values = self.q_network(states, norm_goals, hidden_state=init_hidden)
            current_q_values = current_q_values.squeeze(0)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q_values = self.target_network(next_states, norm_goals, hidden_state=init_hidden)
                next_q_values = next_q_values.squeeze(0)
                max_next_q = next_q_values.max(1)[0]
                targets = rewards + (self.gamma * max_next_q * ~dones)
            
            loss = self.q_loss_fn(current_q, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.q_optimizer.step()
            
            total_loss += loss.item()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return total_loss / len(batch_sequences) if batch_sequences else 0.0
    
    def train_reward_predictor_retrospective(self, reward_pos):
        """Retrospectively train the reward predictor when a goal is found."""
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
        
        current_target = np.zeros((7, 7), dtype=np.float32)
        current_target[6, 3] = 1.0
        batch_inputs.append(self._create_normalized_view_at_goal())
        batch_targets.append(current_target)
        
        return self._train_reward_predictor_batch(batch_inputs, batch_targets)
    
    def _create_normalized_view_at_goal(self):
        """Create normalized view when agent is standing on goal."""
        view = np.zeros((7, 7), dtype=np.float32)
        view[6, 3] = 1.0
        return view
    
    def _create_target_view_with_reward(self, past_pos, past_dir, reward_pos):
        """Create target 7x7 view showing where reward is from a past position."""
        target = np.zeros((7, 7), dtype=np.float32)
        
        past_x, past_y = past_pos
        reward_x, reward_y = reward_pos
        
        ego_center_x, ego_center_y = 3, 6
        
        dx_world = reward_x - past_x
        dy_world = reward_y - past_y
        
        if past_dir == 3:
            dx_ego, dy_ego = dx_world, dy_world
        elif past_dir == 0:
            dx_ego, dy_ego = dy_world, -dx_world
        elif past_dir == 1:
            dx_ego, dy_ego = -dx_world, -dy_world
        elif past_dir == 2:
            dx_ego, dy_ego = -dy_world, dx_world
        else:
            dx_ego, dy_ego = dx_world, dy_world
        
        view_x = ego_center_x + dx_ego
        view_y = ego_center_y + dy_ego
        
        if 0 <= view_x < 7 and 0 <= view_y < 7:
            target[int(view_y), int(view_x)] = 1.0
        
        return target
    
    def _train_reward_predictor_batch(self, inputs, targets):
        """Train reward predictor on a batch."""
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
        OPTIMIZED: Higher thresholds to reduce unnecessary training.
        """
        normalized_view = self._create_normalized_view(obs)
        input_tensor = torch.tensor(
            normalized_view[np.newaxis, np.newaxis, ...],
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            self.reward_predictor.eval()
            predicted = self.reward_predictor(input_tensor)
            predicted_map = predicted.squeeze().cpu().numpy()
        
        error = np.abs(predicted_map - target_7x7)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        # === OPTIMIZATION: Higher thresholds (was 0.05 and 0.01) ===
        if max_error > 0.15 or mean_error > 0.05:
            return True, self._train_reward_predictor_batch([normalized_view], [target_7x7])
        
        return False, 0.0
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_all_q_values(self, skip_if_slow=True):
        """
        Get Q-values for all possible goals.
        OPTIMIZED: Can be skipped for performance.
        """
        if skip_if_slow:
            # Return empty array to skip expensive computation
            return np.zeros((self.grid_size, self.grid_size, self.action_dim))
        
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
    print("LSTM-WVF Agent (OPTIMIZED) module loaded successfully.")