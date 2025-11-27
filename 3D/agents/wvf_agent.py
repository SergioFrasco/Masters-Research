import numpy as np
import torch


class WVFAgent:
    """
    WVF Agent using goal-conditioned MLP.
    
    MATCHES DQN BASELINE STATE REPRESENTATION:
        state = ego_obs (13*13=169) + position (2) + direction_onehot (4) = 175 dims
    
    WVF adds goal conditioning:
        input = state (175) + goal_xy (2) = 177 dims
        output = Q(s, g, a) for 3 actions
    
    Key design:
    - Goal (x, y) is an INPUT to the network
    - Network outputs Q(s, g, a) for a SPECIFIC goal
    - To evaluate all goals, we call the network once per goal
    
    This is what Geraud suggested:
    "Say you have the state (s) be the same one you are using for the DQN 
    (be it the reward map or feature vector from resnet, etc). Then concatenate 
    that with the goal vector (x,y position) to get the input vector (s,g) for your mlp."
    """
    
    def __init__(self, env, wvf_model, target_model, optimizer, gamma=0.99, device='cpu', 
                 grid_size=10, target_update_freq=100):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.grid_size = grid_size
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # WVF model and optimizer (Q-network)
        self.wvf_model = wvf_model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        
        # Target network (like DQN baseline)
        self.target_model = target_model
        self.target_model.load_state_dict(self.wvf_model.state_dict())
        self.target_model.eval()  # Target network always in eval mode
        self.target_update_freq = target_update_freq
        
        # Track the reward map (from vision model) - used to find goals
        self.true_reward_map = np.zeros((grid_size, grid_size))
        
        # Track visited positions
        self.visited_positions = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Current egocentric observation (13x13)
        self.current_ego_obs = None
        
        # Cache Q-values for visualization
        self.current_q_values = None
        
        # Stats
        self.update_count = 0
        self.loss_history = []
    
    def set_ego_observation(self, ego_obs):
        """Set current egocentric observation from vision system"""
        self.current_ego_obs = ego_obs
    
    def _get_state_features(self):
        """
        Get state features for the network.
        
        MATCHES DQN BASELINE EXACTLY:
            state = ego_obs (169) + position (2) + direction_onehot (4) = 175 dims
        
        This is what the DQN baseline uses in get_dqn_state():
            view_flat = obs.flatten()  # 13x13 = 169
            position = [pos_x / (grid_size-1), pos_z / (grid_size-1)]  # 2
            direction_onehot = one_hot(direction)  # 4
            state = concat([view_flat, position, direction_onehot])  # 175
        """
        # 1. Flatten egocentric observation (13x13 = 169)
        if self.current_ego_obs is None:
            view_flat = np.zeros(13 * 13, dtype=np.float32)
        else:
            view_flat = self.current_ego_obs.flatten().astype(np.float32)
        
        # 2. Normalized position (2)
        pos_x, pos_z = self._get_agent_pos_from_env()
        position = np.array([
            pos_x / (self.grid_size - 1),
            pos_z / (self.grid_size - 1)
        ], dtype=np.float32)
        
        # 3. One-hot direction (4)
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate: 169 + 2 + 4 = 175
        state_features = np.concatenate([view_flat, position, direction_onehot])
        
        return state_features
    
    def _get_goals_from_reward_map(self):
        """
        Extract goal positions from the reward map.
        Returns list of (x, y) tuples where rewards are present.
        """
        goals = []
        threshold = 0.5  # Consider anything > 0.5 as a goal
        
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                if self.true_reward_map[z, x] > threshold:
                    goals.append((x, z))
        
        return goals
    
    def _normalize_goal(self, goal_xy):
        """Normalize goal position to [0, 1] range"""
        x, y = goal_xy
        return (x / (self.grid_size - 1), y / (self.grid_size - 1))
    
    def get_q_values_for_goal(self, goal_xy):
        """
        Get Q-values for reaching a specific goal from current state.
        
        Args:
            goal_xy: (x, y) tuple of goal position
            
        Returns:
            q_values: numpy array of shape (3,) for 3 actions
        """
        state_features = self._get_state_features()
        norm_goal = self._normalize_goal(goal_xy)
        
        # Convert to tensors
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.wvf_model.eval()
            q_values = self.wvf_model(state_tensor, goal_tensor)
            return q_values.squeeze(0).cpu().numpy()
    
    def get_all_q_values(self):
        """
        Get Q-values for ALL possible goals (for visualization).
        
        Returns:
            q_values: numpy array of shape (grid_size, grid_size, num_actions)
        """
        q_values = np.zeros((self.grid_size, self.grid_size, self.action_size))
        state_features = self._get_state_features()
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.wvf_model.eval()
        with torch.no_grad():
            for z in range(self.grid_size):
                for x in range(self.grid_size):
                    norm_goal = self._normalize_goal((x, z))
                    goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q = self.wvf_model(state_tensor, goal_tensor)
                    q_values[z, x, :] = q.squeeze(0).cpu().numpy()
        
        return q_values
    
    def update_q_values(self):
        """Update cached Q-values for visualization"""
        self.current_q_values = self.get_all_q_values()
    
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """
        Sample action using WVF with epsilon-greedy exploration.
        
        Strategy:
        1. Find goals in the reward map
        2. For each goal, get Q-values
        3. Pick the best (goal, action) pair
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Get goals from reward map
        goals = self._get_goals_from_reward_map()
        
        # If no goals detected, explore randomly
        if len(goals) == 0:
            return np.random.randint(self.action_size)
        
        # Find best action across all goals
        best_q = float('-inf')
        best_action = None
        
        for goal in goals:
            q_values = self.get_q_values_for_goal(goal)
            max_q = np.max(q_values)
            
            if max_q > best_q:
                best_q = max_q
                best_action = np.argmax(q_values)
        
        if best_action is None:
            return np.random.randint(self.action_size)
        
        return best_action
    
    def update(self, experience, goal_xy):
        """
        Update WVF using TD learning for a specific goal.
        
        Uses TARGET NETWORK for computing TD target (like DQN baseline).
        
        Args:
            experience: [state_features, action, next_state_features, reward, done]
            goal_xy: The goal we're learning to reach
            
        Returns:
            loss value
        """
        state_features = experience[0]      # current state features (flattened ego obs)
        action = experience[1]              # action taken
        next_state_features = experience[2] # next state features
        reward = experience[3]              # reward received
        done = experience[4]                # terminal flag
        
        # Normalize goal
        norm_goal = self._normalize_goal(goal_xy)
        
        # Convert to tensors
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get current Q-value for (state, goal, action)
        self.wvf_model.train()
        current_q_values = self.wvf_model(state_tensor, goal_tensor)
        current_q = current_q_values[0, action]
        
        # Compute TD target using TARGET NETWORK (stabilizes learning)
        with torch.no_grad():
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32).to(self.device)
            else:
                # Use target network for next Q-values (key difference!)
                next_q_values = self.target_model(next_state_tensor, goal_tensor)
                max_next_q = torch.max(next_q_values)
                td_target = reward + self.gamma * max_next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, td_target)
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wvf_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        # Update target network periodically
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.wvf_model.state_dict())
        
        return loss_val
    
    def update_for_all_goals(self, experience):
        """
        Update WVF for ALL detected goals (like Algorithm 1 in the paper).
        
        This trains the network to reach any goal from the current state.
        """
        goals = self._get_goals_from_reward_map()
        
        if len(goals) == 0:
            return 0.0
        
        total_loss = 0.0
        for goal in goals:
            loss = self.update(experience, goal)
            total_loss += loss
        
        return total_loss / len(goals)
    
    # ============ Environment Interface Methods ============
    
    def _get_agent_pos_from_env(self):
        """Get agent position from environment"""
        x = int(round(self.env.agent.pos[0] / self.env.grid_size))
        z = int(round(self.env.agent.pos[2] / self.env.grid_size))
        return (x, z)
    
    def _get_agent_dir_from_env(self):
        """Get agent direction from environment"""
        angle = self.env.agent.dir
        degrees = (np.degrees(angle) % 360)
        if degrees < 45 or degrees >= 315:
            return 0  # East
        elif 45 <= degrees < 135:
            return 3  # North
        elif 135 <= degrees < 225:
            return 2  # West
        else:
            return 1  # South
    
    def get_state_index(self):
        """Convert current grid position to flat state index"""
        x, z = self._get_agent_pos_from_env()
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def reset(self):
        """Reset for new episode"""
        self.current_q_values = None
        self.current_ego_obs = None
    
    def create_egocentric_observation(self, goal_pos_red=None, goal_pos_blue=None, matrix_size=13):
        """
        Create an egocentric observation matrix.
        Agent is at bottom-middle (6, 12), facing upward.
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