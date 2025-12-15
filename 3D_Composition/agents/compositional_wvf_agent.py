"""Compositional WVF Agent with SHARED state representation (FIXED).

Key Fix: All feature networks now receive the SAME state input containing
ALL object information. They differ ONLY in their reward signals.

This ensures:
- All networks learn the same navigation dynamics (G*)
- Composition via min() is mathematically valid
- Q_red and Q_box can be meaningfully combined

Features: 'red', 'blue', 'box', 'sphere'

Each feature has its own:
    - WVF network: Q_feature(s, g, a) where s is SHARED across all features
    - Target network
    - Optimizer
    - Reward signal (feature-specific)

Composition: For a task like ['red', 'box'], we compose:
    Q_task(s, g, a) = min(Q_red(s, g, a), Q_box(s, g, a))
    
This works because:
    Q_red(s, g, a) = G*(s, g, a) + R_red(g)
    Q_box(s, g, a) = G*(s, g, a) + R_box(g)
    
    min(Q_red, Q_box) = G* + min(R_red, R_box)
    
Where G* is IDENTICAL because both networks see the same world.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CompositionalWVFAgent:
    """
    Compositional WVF Agent with SHARED state representation.
    
    THE KEY DIFFERENCE FROM ORIGINAL:
    All networks receive identical state input containing all object types.
    They only differ in what reward signal they're trained on.
    """

    def __init__(
        self,
        env,
        wvf_model_class,
        model_kwargs,
        lr=0.0005,
        gamma=0.99,
        device="cpu",
        grid_size=10,
        target_update_freq=100,
        confidence_threshold=0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.grid_size = grid_size
        self.target_update_freq = target_update_freq
        self.confidence_threshold = confidence_threshold

        # Actions
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3

        # Feature names
        self.feature_names = ["red", "blue", "box", "sphere"]

        # Mapping object → features
        self.object_to_features = {
            "red_box": ["red", "box"],
            "blue_box": ["blue", "box"],
            "red_sphere": ["red", "sphere"],
            "blue_sphere": ["blue", "sphere"],
        }

        # Reverse mapping: feature → objects
        self.feature_to_objects = {
            "red": ["red_box", "red_sphere"],
            "blue": ["blue_box", "blue_sphere"],
            "box": ["red_box", "blue_box"],
            "sphere": ["red_sphere", "blue_sphere"],
        }

        # Create feature-specific networks (but they'll all use SHARED state)
        self.wvf_models = {}
        self.target_models = {}
        self.optimizers = {}

        for feature in self.feature_names:
            model = wvf_model_class(**model_kwargs).to(device)
            target = wvf_model_class(**model_kwargs).to(device)
            target.load_state_dict(model.state_dict())
            target.eval()

            self.wvf_models[feature] = model
            self.target_models[feature] = target
            self.optimizers[feature] = optim.Adam(model.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

        # Feature-specific reward maps (global, grid_size x grid_size)
        # These track WHERE each feature type has been observed
        self.feature_reward_maps = {
            feature: np.zeros((grid_size, grid_size), dtype=np.float32)
            for feature in self.feature_names
        }

        # Feature-specific egocentric observations (13x13)
        # These are used to BUILD the shared state, not as separate inputs
        self.feature_ego_obs = {
            feature: np.zeros((13, 13), dtype=np.float32)
            for feature in self.feature_names
        }

        # Shared visited map
        self.visited_positions = np.zeros((grid_size, grid_size), dtype=bool)

        # Stats
        self.update_counts = {f: 0 for f in self.feature_names}
        self.loss_history = {f: [] for f in self.feature_names}

        # Visualization cache
        self.current_q_values = None

        # Vision debug info
        self.last_detection = None

        # Confidence parameters
        self.confidence_boost = 0.4
        self.decay_factor = 0.95

    # ==================== SHARED State Features (THE FIX) ====================

    def _get_shared_state_features(self):
        """
        Construct state features that ALL networks will use.
        
        THIS IS THE KEY FIX: All networks see the SAME world state,
        containing information about ALL object types.
        
        Structure:
            - 4 channels of egocentric observations (red, blue, box, sphere)
            - Each channel is 13x13 = 169
            - Total ego obs: 4 * 169 = 676
            - Agent position: 2 (normalized x, z)
            - Agent direction: 4 (one-hot)
            - Total: 676 + 2 + 4 = 682 dimensions
            
        Returns:
            np.array of shape (682,) containing full world state
        """
        # Stack ALL feature channels into multi-channel observation
        # This ensures every network sees every object type
        all_channels = np.stack([
            self.feature_ego_obs["red"],
            self.feature_ego_obs["blue"],
            self.feature_ego_obs["box"],
            self.feature_ego_obs["sphere"]
        ], axis=0)  # Shape: (4, 13, 13)
        
        view_flat = all_channels.flatten().astype(np.float32)  # 676 dims

        # Agent position (normalized to [0, 1])
        pos_x, pos_z = self._get_agent_pos_from_env()
        position = np.array(
            [
                pos_x / max(self.grid_size - 1, 1),
                pos_z / max(self.grid_size - 1, 1),
            ],
            dtype=np.float32,
        )

        # Agent direction (one-hot encoded)
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0

        # Concatenate: ego_obs (676) + position (2) + direction (4) = 682
        return np.concatenate([view_flat, position, direction_onehot])

    def get_all_state_features(self):
        """
        Get state features for all feature networks.
        
        THE FIX: All features get the SAME shared state.
        This ensures they all learn the same navigation dynamics.
        
        Returns:
            dict: {feature_name: shared_state_features} where all values are identical
        """
        shared_state = self._get_shared_state_features()
        
        # ALL features receive the SAME state representation
        return {feature: shared_state.copy() for feature in self.feature_names}

    # ==================== Vision Integration ====================

    def update_from_detection(self, detection_result):
        """
        Update internal maps from vision detection results.
        
        This populates the feature_ego_obs arrays which are then
        combined into the shared state representation.
        """
        self.last_detection = detection_result

        # Decay existing confidence (objects not seen recently fade)
        for f in self.feature_names:
            self.feature_reward_maps[f] *= self.decay_factor

        positions = detection_result.get("positions", {})
        probabilities = detection_result.get("probabilities", {})

        # Reset egocentric obs (these are per-step, not accumulated)
        for f in self.feature_names:
            self.feature_ego_obs[f] = np.zeros((13, 13), dtype=np.float32)

        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()

        for obj_name, pos in positions.items():
            if pos is None:
                continue

            conf = probabilities.get(obj_name, 1.0)
            if conf < self.confidence_threshold:
                continue

            dx, dz = pos
            dx = int(round(dx))
            dz = int(round(dz))

            # Update ego observations for ALL features this object has
            for feature in self.object_to_features.get(obj_name, []):
                self._place_in_ego_obs(feature, dx, dz)

            # Convert to global coordinates for reward maps
            gx, gz = self._ego_to_global(dx, dz, agent_x, agent_z, agent_dir)
            if 0 <= gx < self.grid_size and 0 <= gz < self.grid_size:
                for feature in self.object_to_features.get(obj_name, []):
                    self.feature_reward_maps[feature][gz, gx] = min(
                        1.0,
                        self.feature_reward_maps[feature][gz, gx] + self.confidence_boost
                    )

    def _place_in_ego_obs(self, feature_name, dx, dz):
        """Place detected object in egocentric observation grid."""
        ego_center_x = 6
        ego_center_z = 12

        ego_x = ego_center_x + dx
        ego_z = ego_center_z + dz

        if 0 <= ego_x < 13 and 0 <= ego_z < 13:
            self.feature_ego_obs[feature_name][ego_z, ego_x] = 1.0

    def _ego_to_global(self, dx, dz, ax, az, d):
        """Convert egocentric coordinates to global grid coordinates."""
        if d == 3:  # North
            gx = ax + dx
            gz = az + dz
        elif d == 0:  # East
            gx = ax - dz
            gz = az + dx
        elif d == 1:  # South
            gx = ax - dx
            gz = az - dz
        elif d == 2:  # West
            gx = ax + dz
            gz = az - dx
        else:
            gx = ax + dx
            gz = az + dz

        return int(round(gx)), int(round(gz))

    # ==================== Goal Extraction ====================
    
    def _get_goals_for_feature(self, feature_name):
        """Get all goal positions where a feature has been observed."""
        goals = []
        rm = self.feature_reward_maps[feature_name]
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                if rm[z, x] > self.confidence_threshold:
                    goals.append((x, z))
        return goals

    def _get_goals_for_task(self, task):
        """
        Get goal positions that satisfy ALL task features (intersection).
        
        For task ['red', 'box'], returns positions where BOTH
        red things AND box things have been observed.
        """
        features = task.get("features", [])
        if len(features) == 0:
            return []

        goal_sets = [set(self._get_goals_for_feature(f)) for f in features]

        if any(len(gs) == 0 for gs in goal_sets):
            return []

        # Intersection: positions satisfying ALL features
        valid = goal_sets[0]
        for gs in goal_sets[1:]:
            valid = valid.intersection(gs)

        return list(valid)

    # ==================== Q-Value Computation ====================

    def _normalize_goal(self, goal_xy):
        """Normalize goal position to [0, 1] range."""
        x, z = goal_xy
        return (
            x / max(self.grid_size - 1, 1),
            z / max(self.grid_size - 1, 1),
        )

    def get_q_values_for_goal(self, feature_name, goal_xy, state_features=None):
        """
        Get Q-values for a specific feature network and goal.
        
        Args:
            feature_name: Which feature network to query
            goal_xy: Goal position (x, z)
            state_features: Pre-computed shared state (optional)
            
        Returns:
            np.array of shape (num_actions,) with Q-values
        """
        if state_features is None:
            state_features = self._get_shared_state_features()

        model = self.wvf_models[feature_name]
        model.eval()

        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(
            self.device
        )
        goal_tensor = torch.tensor(self._normalize_goal(goal_xy), dtype=torch.float32).unsqueeze(
            0
        ).to(self.device)

        with torch.no_grad():
            q = model(state_tensor, goal_tensor)
            return q.squeeze(0).cpu().numpy()

    def get_composed_q_values_for_goal(self, task, goal_xy):
        """
        Get composed Q-values for a task via min over feature Q-values.
        
        THIS NOW WORKS CORRECTLY because all feature networks
        computed their Q-values from the SAME shared state.
        
        min(Q_red, Q_box) = min(G* + R_red, G* + R_box)
                         = G* + min(R_red, R_box)
        
        Args:
            task: Task dict with 'features' list
            goal_xy: Goal position
            
        Returns:
            np.array of shape (num_actions,) with composed Q-values
        """
        features = task.get("features", [])
        if len(features) == 0:
            return np.zeros(self.action_size)

        # Get SHARED state once (same for all features)
        shared_state = self._get_shared_state_features()

        # Compute Q-values for each feature using the SAME state
        q_list = []
        for f in features:
            q_list.append(self.get_q_values_for_goal(f, goal_xy, shared_state))

        # Composition via min (conjunction in Boolean algebra)
        return np.min(np.stack(q_list), axis=0)

    def get_all_q_values_for_feature(self, feature_name):
        """
        Get Q-values for all (state, goal) pairs for a feature.
        Used for visualization.
        
        Returns:
            np.array of shape (grid_size, grid_size, action_size)
        """
        q_map = np.zeros((self.grid_size, self.grid_size, self.action_size))
        
        # Use shared state
        state_features = self._get_shared_state_features()
        
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                goal_xy = (x, z)
                q_values = self.get_q_values_for_goal(feature_name, goal_xy, state_features)
                q_map[z, x, :] = q_values
        
        return q_map

    def get_composed_reward_map(self, task):
        """
        Get composed reward map for a task (intersection of feature maps).
        Used for visualization.
        
        Args:
            task: Task dict with 'features' list
            
        Returns:
            np.array of shape (grid_size, grid_size)
        """
        features = task.get("features", [])
        if len(features) == 0:
            return np.zeros((self.grid_size, self.grid_size))
        
        # Start with first feature's map
        composed = self.feature_reward_maps[features[0]].copy()
        
        # AND with other features (element-wise minimum)
        for f in features[1:]:
            composed = np.minimum(composed, self.feature_reward_maps[f])
        
        return composed

    # ==================== Action Selection ====================

    def sample_action_with_wvf(self, obs, task, epsilon=0.0):
        """
        Select action using composed WVF for current task.
        
        Args:
            obs: Current observation (not used directly, state from internal maps)
            task: Current task dict
            epsilon: Exploration probability
            
        Returns:
            int: Selected action
        """
        if np.random.uniform() < epsilon:
            return np.random.randint(self.action_size)

        goals = self._get_goals_for_task(task)
        if len(goals) == 0:
            return np.random.randint(self.action_size)

        best_q = -1e10
        best_a = None

        for g in goals:
            q = self.get_composed_q_values_for_goal(task, g)
            m = np.max(q)
            if m > best_q:
                best_q = m
                best_a = np.argmax(q)

        if best_a is None:
            return np.random.randint(self.action_size)

        return int(best_a)

    # ==================== Training ====================

    def _update_single(self, feature_name, experience, goal_xy):
        """
        Single TD update for one feature network on one goal.
        
        Uses SHARED state representation from experience tuple.
        """
        # Experience now contains shared state for all features
        state_features = experience[0][feature_name]  # But all are identical now
        action = experience[1]
        next_state_features = experience[2][feature_name]
        reward = experience[3][feature_name]  # Only rewards differ per feature
        done = experience[4]

        norm_goal = self._normalize_goal(goal_xy)

        model = self.wvf_models[feature_name]
        target_model = self.target_models[feature_name]
        optimizer = self.optimizers[feature_name]

        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(
            self.device
        )
        next_state_tensor = torch.tensor(
            next_state_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(norm_goal, dtype=torch.float32).unsqueeze(0).to(
            self.device
        )

        model.train()
        current_q_values = model(state_tensor, goal_tensor)
        current_q = current_q_values[0, action]

        with torch.no_grad():
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32).to(self.device)
            else:
                next_q = target_model(next_state_tensor, goal_tensor)
                td_target = reward + self.gamma * torch.max(next_q)

        loss = self.loss_fn(current_q, td_target)
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        self.update_counts[feature_name] += 1
        self.loss_history[feature_name].append(loss.item())

        if self.update_counts[feature_name] % self.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        return loss.item()

    def update_for_feature(self, feature_name, experience):
        """Update a feature network on all its known goals."""
        goals = self._get_goals_for_feature(feature_name)
        if len(goals) == 0:
            return 0.0

        total = 0.0
        for g in goals:
            total += self._update_single(feature_name, experience, g)

        return total / len(goals)

    def update_all_features(self, experience):
        """Update ALL feature networks."""
        return {
            feature: self.update_for_feature(feature, experience)
            for feature in self.feature_names
        }

    def update_selected_features(self, experience, features_to_train):
        """
        Update only SELECTED feature networks.
        
        Args:
            experience: Tuple of (state_features, action, next_state_features, 
                                  feature_rewards, done)
            features_to_train: List of feature names to train
            
        Returns:
            Dict of feature -> loss for trained features
        """
        losses = {}
        
        for feature in features_to_train:
            if feature in self.feature_names:
                losses[feature] = self.update_for_feature(feature, experience)
        
        # Return zeros for non-trained features (for logging)
        for feature in self.feature_names:
            if feature not in losses:
                losses[feature] = 0.0
        
        return losses

    # ==================== Rewards ====================

    def compute_feature_rewards(self, info):
        """
        Compute reward for each feature based on contacted object.
        
        This is the ONLY thing that differs between features.
        All features see the same world, but value different outcomes.
        """
        contacted = info.get("contacted_object", None)
        rewards = {f: 0.0 for f in self.feature_names}

        if contacted in self.object_to_features:
            for f in self.object_to_features[contacted]:
                rewards[f] = 1.0

        return rewards

    # ==================== Environment Helpers ====================

    def _get_agent_pos_from_env(self):
        """Get agent position from environment."""
        try:
            x = int(round(self.env.agent.pos[0] / self.env.grid_size))
            z = int(round(self.env.agent.pos[2] / self.env.grid_size))
            return (np.clip(x, 0, self.grid_size - 1), np.clip(z, 0, self.grid_size - 1))
        except Exception:
            return (0, 0)

    def _get_agent_dir_from_env(self):
        """Get agent direction from environment."""
        try:
            angle = self.env.agent.dir
            deg = np.degrees(angle) % 360
            if deg < 45 or deg >= 315:
                return 0  # East
            elif 45 <= deg < 135:
                return 3  # North
            elif 135 <= deg < 225:
                return 2  # West
            else:
                return 1  # South
        except Exception:
            return 0

    def get_state_index(self):
        """Get flattened state index."""
        x, z = self._get_agent_pos_from_env()
        return z * self.grid_size + x

    # ==================== Reset ====================

    def reset(self):
        """Reset agent state for new episode."""
        for f in self.feature_names:
            self.feature_reward_maps[f] = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.float32
            )
            self.feature_ego_obs[f] = np.zeros((13, 13), dtype=np.float32)
        self.visited_positions = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.current_q_values = None
        self.last_detection = None