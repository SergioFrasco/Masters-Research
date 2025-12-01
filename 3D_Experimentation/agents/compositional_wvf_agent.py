"""
Compositional WVF Agent with 4 feature-specific value functions.

Features: 'red', 'blue', 'box', 'sphere'

Each feature has its own:
    - WVF network: Q_feature(s, g, a)
    - Target network
    - Optimizer
    - Reward map
    - Egocentric observation

Composition: For a task like ['red', 'box'], we compose:
    Q_task(s, g, a) = min(Q_red(s, g, a), Q_box(s, g, a))

where g must be a goal position that satisfies ALL task features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CompositionalWVFAgent:
    """
    Compositional WVF Agent using 4 feature-specific goal-conditioned networks.
    
    Architecture:
        - 4 separate WVF networks (red, blue, box, sphere)
        - Each network: Q_feature(s, g, a) where s includes feature-specific ego obs
        - Composition via min() for multi-feature tasks
    
    Training:
        - Each network trained independently on its feature-specific reward
        - V_red gets reward when ANY red object is touched
        - V_box gets reward when ANY box is touched
        - etc.
    
    Action Selection:
        - Find goals satisfying ALL task features (intersection)
        - Compute composed Q = min(Q_f1, Q_f2, ...) for each valid goal
        - Select action with highest composed Q
    """
    
    def __init__(self, env, wvf_model_class, model_kwargs, lr=0.0005, gamma=0.99, 
                 device='cpu', grid_size=10, target_update_freq=100, 
                 confidence_threshold=0.5):
        """
        Args:
            env: The environment
            wvf_model_class: The WVF_MLP class (not instance)
            model_kwargs: Dict of kwargs for WVF_MLP (state_dim, num_actions, hidden_dim)
            lr: Learning rate for all optimizers
            gamma: Discount factor
            device: torch device
            grid_size: Size of the grid world
            target_update_freq: How often to update target networks
            confidence_threshold: Minimum confidence for vision detections
        """
        self.env = env
        self.gamma = gamma
        self.device = device
        self.grid_size = grid_size
        self.target_update_freq = target_update_freq
        self.confidence_threshold = confidence_threshold
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # Feature names
        self.feature_names = ['red', 'blue', 'box', 'sphere']
        
        # Mapping from objects to their features
        self.object_to_features = {
            'red_box': ['red', 'box'],
            'blue_box': ['blue', 'box'],
            'red_sphere': ['red', 'sphere'],
            'blue_sphere': ['blue', 'sphere'],
        }
        
        # Reverse mapping: feature to objects that have it
        self.feature_to_objects = {
            'red': ['red_box', 'red_sphere'],
            'blue': ['blue_box', 'blue_sphere'],
            'box': ['red_box', 'blue_box'],
            'sphere': ['red_sphere', 'blue_sphere'],
        }
        
        # Create 4 WVF networks, target networks, and optimizers
        self.wvf_models = {}
        self.target_models = {}
        self.optimizers = {}
        
        for feature in self.feature_names:
            # Main network
            self.wvf_models[feature] = wvf_model_class(**model_kwargs).to(device)
            
            # Target network
            self.target_models[feature] = wvf_model_class(**model_kwargs).to(device)
            self.target_models[feature].load_state_dict(self.wvf_models[feature].state_dict())
            self.target_models[feature].eval()
            
            # Optimizer
            self.optimizers[feature] = optim.Adam(
                self.wvf_models[feature].parameters(), lr=lr
            )
        
        self.loss_fn = nn.MSELoss()
        
        # 4 feature-specific reward maps (allocentric/global coordinates)
        self.feature_reward_maps = {
            feature: np.zeros((grid_size, grid_size), dtype=np.float32)
            for feature in self.feature_names
        }
        
        # 4 feature-specific egocentric observations (13x13)
        self.feature_ego_obs = {
            feature: np.zeros((13, 13), dtype=np.float32)
            for feature in self.feature_names
        }
        
        # Track visited positions (shared across features)
        self.visited_positions = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Stats tracking
        self.update_counts = {feature: 0 for feature in self.feature_names}
        self.loss_history = {feature: [] for feature in self.feature_names}
        
        # Cache for visualization
        self.current_q_values = None
        
        # Store last detection result for debugging
        self.last_detection = None
    
    # ==================== State Features ====================
    
    def _get_state_features_for_feature(self, feature_name):
        """
        Get state features for a specific feature's network.
        
        State = feature_ego_obs (13*13=169) + position (2) + direction_onehot (4) = 175 dims
        
        Each feature has its own egocentric observation showing only objects
        relevant to that feature.
        """
        # 1. Flatten feature-specific egocentric observation (13x13 = 169)
        ego_obs = self.feature_ego_obs[feature_name]
        view_flat = ego_obs.flatten().astype(np.float32)
        
        # 2. Normalized position (2)
        pos_x, pos_z = self._get_agent_pos_from_env()
        position = np.array([
            pos_x / max(self.grid_size - 1, 1),
            pos_z / max(self.grid_size - 1, 1)
        ], dtype=np.float32)
        
        # 3. One-hot direction (4)
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate: 169 + 2 + 4 = 175
        state_features = np.concatenate([view_flat, position, direction_onehot])
        
        return state_features
    
    def get_all_state_features(self):
        """Get state features for all 4 feature networks."""
        return {
            feature: self._get_state_features_for_feature(feature)
            for feature in self.feature_names
        }
    
    # ==================== Vision Integration ====================
    
    def update_from_detection(self, detection_result):
        """
        Update all feature reward maps and ego observations from vision detection.
        
        Args:
            detection_result: Dict from detect_cube() containing:
                - detected_objects: list of detected object names
                - positions: dict mapping object names to (dx, dz) egocentric positions
                - probabilities: dict mapping object names to confidence scores
        """
        self.last_detection = detection_result
        positions = detection_result.get('positions', {})
        probabilities = detection_result.get('probabilities', {})
        
        # Reset egocentric observations each step
        for feature in self.feature_names:
            self.feature_ego_obs[feature] = np.zeros((13, 13), dtype=np.float32)
        
        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()
        
        for obj_name, pos in positions.items():
            if pos is None:
                continue
            
            # Check confidence threshold
            confidence = probabilities.get(obj_name, 1.0)
            if confidence < self.confidence_threshold:
                continue
            
            dx, dz = pos
            dx = int(round(dx))
            dz = int(round(dz))
            
            # Update egocentric observation for each relevant feature
            for feature in self.object_to_features.get(obj_name, []):
                self._place_in_ego_obs(feature, dx, dz)
            
            # Convert to global coordinates and update reward maps
            global_x, global_z = self._ego_to_global(dx, dz, agent_x, agent_z, agent_dir)
            
            if 0 <= global_x < self.grid_size and 0 <= global_z < self.grid_size:
                for feature in self.object_to_features.get(obj_name, []):
                    self.feature_reward_maps[feature][global_z, global_x] = 1.0
    
    def _place_in_ego_obs(self, feature_name, dx, dz):
        """
        Place a detection in the feature's egocentric observation.
        
        Agent is at (6, 12) in the 13x13 grid, facing "up" (negative z in ego frame).
        dx, dz are relative to agent in egocentric frame:
            - dx: positive = right of agent
            - dz: negative = in front of agent
        """
        ego_center_x = 6
        ego_center_z = 12
        
        # Convert detection offset to ego observation coordinates
        ego_x = ego_center_x + dx
        ego_z = ego_center_z + dz  # dz is typically negative (in front)
        
        if 0 <= ego_x < 13 and 0 <= ego_z < 13:
            self.feature_ego_obs[feature_name][ego_z, ego_x] = 1.0
    
    def _ego_to_global(self, dx, dz, agent_x, agent_z, agent_dir):
        """
        Convert egocentric (dx, dz) to global (x, z) coordinates.
        
        Args:
            dx, dz: Egocentric offset (dx=right, dz=forward with negative being front)
            agent_x, agent_z: Agent's global position
            agent_dir: Agent's direction (0=East, 1=South, 2=West, 3=North)
            
        Returns:
            (global_x, global_z): Global coordinates
        """
        if agent_dir == 3:  # North (facing -z in world)
            global_x = agent_x + dx
            global_z = agent_z + dz
        elif agent_dir == 0:  # East (facing +x in world)
            global_x = agent_x - dz
            global_z = agent_z + dx
        elif agent_dir == 1:  # South (facing +z in world)
            global_x = agent_x - dx
            global_z = agent_z - dz
        elif agent_dir == 2:  # West (facing -x in world)
            global_x = agent_x + dz
            global_z = agent_z - dx
        else:
            # Default fallback
            global_x = agent_x + dx
            global_z = agent_z + dz
        
        return int(round(global_x)), int(round(global_z))
    
    # ==================== Goal Extraction ====================
    
    def _get_goals_for_feature(self, feature_name):
        """Extract goal positions from a specific feature's reward map."""
        goals = []
        threshold = 0.5
        reward_map = self.feature_reward_maps[feature_name]
        
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                if reward_map[z, x] > threshold:
                    goals.append((x, z))
        
        return goals
    
    def _get_goals_for_task(self, task):
        """
        Get goals that satisfy ALL features in the task (intersection).
        
        For task ['red', 'box'], returns positions where BOTH a red object
        AND a box object are present (i.e., red_box positions).
        
        Args:
            task: Dict with 'features' list
            
        Returns:
            List of (x, z) goal positions
        """
        features = task.get('features', [])
        
        if len(features) == 0:
            return []
        
        # Get goal sets for each feature
        goal_sets = [set(self._get_goals_for_feature(f)) for f in features]
        
        # Handle empty sets
        if any(len(gs) == 0 for gs in goal_sets):
            return []
        
        # Intersection of all goal sets
        valid_goals = goal_sets[0]
        for gs in goal_sets[1:]:
            valid_goals = valid_goals.intersection(gs)
        
        return list(valid_goals)
    
    def _normalize_goal(self, goal_xy):
        """Normalize goal position to [0, 1] range."""
        x, z = goal_xy
        return (
            x / max(self.grid_size - 1, 1), 
            z / max(self.grid_size - 1, 1)
        )
    
    # ==================== Q-Value Computation ====================
    
    def get_q_values_for_goal(self, feature_name, goal_xy, state_features=None):
        """
        Get Q-values for reaching a specific goal using a specific feature network.
        
        Args:
            feature_name: Which feature network to use
            goal_xy: (x, z) goal position
            state_features: Optional pre-computed state features
            
        Returns:
            q_values: numpy array of shape (3,) for 3 actions
        """
        if state_features is None:
            state_features = self._get_state_features_for_feature(feature_name)
        
        norm_goal = self._normalize_goal(goal_xy)
        
        state_tensor = torch.tensor(
            state_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(
            norm_goal, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        model = self.wvf_models[feature_name]
        model.eval()
        
        with torch.no_grad():
            q_values = model(state_tensor, goal_tensor)
            return q_values.squeeze(0).cpu().numpy()
    
    def get_composed_q_values_for_goal(self, task, goal_xy):
        """
        Get composed Q-values for a goal by taking min across task features.
        
        Args:
            task: Task dict with 'features' list
            goal_xy: (x, z) goal position
            
        Returns:
            composed_q: numpy array of shape (3,) - min across features
        """
        features = task.get('features', [])
        
        if len(features) == 0:
            return np.zeros(self.action_size)
        
        q_values_list = []
        for feature in features:
            state_features = self._get_state_features_for_feature(feature)
            q = self.get_q_values_for_goal(feature, goal_xy, state_features)
            q_values_list.append(q)
        
        # Compose via minimum (AND logic)
        composed_q = np.min(np.stack(q_values_list), axis=0)
        return composed_q
    
    def get_all_q_values_for_feature(self, feature_name):
        """
        Get Q-values for ALL possible goals using a specific feature network.
        Used for visualization.
        
        Returns:
            q_values: numpy array of shape (grid_size, grid_size, num_actions)
        """
        q_values = np.zeros((self.grid_size, self.grid_size, self.action_size))
        state_features = self._get_state_features_for_feature(feature_name)
        state_tensor = torch.tensor(
            state_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        model = self.wvf_models[feature_name]
        model.eval()
        
        with torch.no_grad():
            for z in range(self.grid_size):
                for x in range(self.grid_size):
                    norm_goal = self._normalize_goal((x, z))
                    goal_tensor = torch.tensor(
                        norm_goal, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                    q = model(state_tensor, goal_tensor)
                    q_values[z, x, :] = q.squeeze(0).cpu().numpy()
        
        return q_values
    
    # ==================== Action Selection ====================
    
    def sample_action_with_wvf(self, obs, task, epsilon=0.0):
        """
        Sample action using composed WVF with epsilon-greedy exploration.
        
        Strategy:
        1. Find goals that satisfy ALL task features
        2. For each valid goal, compute composed Q-values
        3. Pick action with highest composed Q-value
        
        Args:
            obs: Current observation (not directly used, included for API compatibility)
            task: Task dict with 'features' list
            epsilon: Exploration probability
            
        Returns:
            action: int (0, 1, or 2)
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Get goals that satisfy the task
        valid_goals = self._get_goals_for_task(task)
        
        if len(valid_goals) == 0:
            # No valid goals detected, explore randomly
            return np.random.randint(self.action_size)
        
        # Find best action across all valid goals
        best_q = float('-inf')
        best_action = None
        
        for goal in valid_goals:
            composed_q = self.get_composed_q_values_for_goal(task, goal)
            max_q = np.max(composed_q)
            
            if max_q > best_q:
                best_q = max_q
                best_action = np.argmax(composed_q)
        
        if best_action is None:
            return np.random.randint(self.action_size)
        
        return int(best_action)
    
    # ==================== Training ====================
    
    def _update_single(self, feature_name, experience, goal_xy):
        """
        Update a specific feature's WVF using TD learning for a specific goal.
        
        Args:
            feature_name: Which feature network to update
            experience: [state_features_dict, action, next_state_features_dict, reward_dict, done]
            goal_xy: The goal position
            
        Returns:
            loss value
        """
        state_features = experience[0][feature_name]
        action = experience[1]
        next_state_features = experience[2][feature_name]
        reward = experience[3][feature_name]  # Feature-specific reward
        done = experience[4]
        
        norm_goal = self._normalize_goal(goal_xy)
        
        # Convert to tensors
        state_tensor = torch.tensor(
            state_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(
            next_state_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(
            norm_goal, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        model = self.wvf_models[feature_name]
        target_model = self.target_models[feature_name]
        optimizer = self.optimizers[feature_name]
        
        # Get current Q-value
        model.train()
        current_q_values = model(state_tensor, goal_tensor)
        current_q = current_q_values[0, action]
        
        # Compute TD target using target network
        with torch.no_grad():
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32).to(self.device)
            else:
                next_q_values = target_model(next_state_tensor, goal_tensor)
                max_next_q = torch.max(next_q_values)
                td_target = reward + self.gamma * max_next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, td_target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        self.update_counts[feature_name] += 1
        loss_val = loss.item()
        self.loss_history[feature_name].append(loss_val)
        
        # Update target network periodically
        if self.update_counts[feature_name] % self.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        
        return loss_val
    
    def update_for_feature(self, feature_name, experience):
        """
        Update a specific feature's WVF for all its detected goals.
        
        Args:
            feature_name: Which feature to update
            experience: Experience tuple with feature-specific components
            
        Returns:
            Average loss across all goals
        """
        goals = self._get_goals_for_feature(feature_name)
        
        if len(goals) == 0:
            return 0.0
        
        total_loss = 0.0
        for goal in goals:
            loss = self._update_single(feature_name, experience, goal)
            total_loss += loss
        
        return total_loss / len(goals)
    
    def update_all_features(self, experience):
        """
        Update all 4 feature networks.
        
        Args:
            experience: [state_features_dict, action, next_state_features_dict, reward_dict, done]
            
        Returns:
            Dict of losses per feature
        """
        losses = {}
        for feature_name in self.feature_names:
            losses[feature_name] = self.update_for_feature(feature_name, experience)
        return losses
    
    # ==================== Feature-Specific Rewards ====================
    
    def compute_feature_rewards(self, info):
        """
        Compute rewards for each feature based on what object was contacted.
        
        Each feature gets reward=1 if the contacted object has that feature.
        E.g., touching red_box gives reward to both 'red' and 'box' networks.
        
        Args:
            info: Environment info dict containing 'contacted_object'
            
        Returns:
            Dict mapping feature names to reward values (0.0 or 1.0)
        """
        contacted_object = info.get('contacted_object', None)
        
        rewards = {feature: 0.0 for feature in self.feature_names}
        
        if contacted_object is not None and contacted_object in self.object_to_features:
            # Give reward to all features that this object satisfies
            for feature in self.object_to_features[contacted_object]:
                rewards[feature] = 1.0
        
        return rewards
    
    # ==================== Environment ====================
    
    def _get_agent_pos_from_env(self):
        """Get agent position from environment."""
        try:
            x = int(round(self.env.agent.pos[0] / self.env.grid_size))
            z = int(round(self.env.agent.pos[2] / self.env.grid_size))
            # Clamp to grid bounds
            x = np.clip(x, 0, self.grid_size - 1)
            z = np.clip(z, 0, self.grid_size - 1)
            return (x, z)
        except Exception:
            return (0, 0)
    
    def _get_agent_dir_from_env(self):
        """
        Get agent direction from environment.
        
        Returns:
            direction: 0=East, 1=South, 2=West, 3=North
        """
        try:
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
        except Exception:
            return 0
    
    def get_state_index(self):
        """Convert current grid position to flat state index."""
        x, z = self._get_agent_pos_from_env()
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def reset(self):
        """Reset for new episode."""
        # Reset all feature reward maps
        for feature in self.feature_names:
            self.feature_reward_maps[feature] = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.float32
            )
            self.feature_ego_obs[feature] = np.zeros((13, 13), dtype=np.float32)
        
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.current_q_values = None
        self.last_detection = None
    
    # ==================== Helpers ====================
    
    def get_composed_reward_map(self, task):
        """
        Get the composed reward map for a task (for visualization).
        
        Args:
            task: Task dict with 'features' list
            
        Returns:
            Composed reward map as numpy array
        """
        features = task.get('features', [])
        
        if len(features) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        if len(features) == 1:
            return self.feature_reward_maps[features[0]].copy()
        
        # AND logic via minimum
        maps = [self.feature_reward_maps[f] for f in features]
        return np.minimum.reduce(maps)
    
    def get_feature_ego_obs(self, feature_name):
        """Get egocentric observation for a specific feature."""
        return self.feature_ego_obs[feature_name].copy()
    
    def get_all_feature_reward_maps(self):
        """Get all feature reward maps as a dict."""
        return {f: self.feature_reward_maps[f].copy() for f in self.feature_names}