"""
Compositional WVF Agent v3 - With Allocentric Memory

Key Change: Network now receives the ALLOCENTRIC (global) reward maps as input,
not just the egocentric observations. This gives the agent memory of objects
it has seen but are no longer in view.

State representation:
    - allocentric_maps: 4 feature maps × 10×10 = 400 dims (MEMORY)
    - agent_position: 2 dims (normalized x, z)
    - agent_direction: 4 dims (one-hot)
    - goal_position: 2 dims (normalized x, z) - passed separately to network
    
Total state_dim = 400 + 2 + 4 = 406
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class WVF_MLP_v3(nn.Module):
    """
    WVF MLP that takes allocentric state + goal.
    
    Input: 
        - state: allocentric_maps (400) + position (2) + direction (4) = 406
        - goal: normalized (x, z) = 2
    Output:
        - Q-values for 3 actions
    """
    
    def __init__(self, state_dim=406, num_actions=3, hidden_dim=128):
        super(WVF_MLP_v3, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Input: state (406) + goal (2) = 408
        input_dim = state_dim + 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(0.1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features, goal_xy):
        """
        Args:
            state_features: (batch, 406) - allocentric maps + position + direction
            goal_xy: (batch, 2) - normalized goal position
        Returns:
            q_values: (batch, 3)
        """
        x = torch.cat([state_features, goal_xy], dim=-1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values


class CompositionalWVFAgent:
    """
    Compositional WVF Agent with Allocentric Memory.
    
    Key difference from v2:
    - State includes ALLOCENTRIC maps (persistent memory of seen objects)
    - Agent remembers objects even after turning away
    - All 4 networks see the same allocentric state
    
    Memory flow:
    1. Agent sees red_box at egocentric (dx=2, dz=-3)
    2. Convert to global coords (e.g., x=5, z=7)  
    3. Mark feature_reward_maps['red'][7, 5] = 1.0
    4. Mark feature_reward_maps['box'][7, 5] = 1.0
    5. These maps PERSIST and are fed to network as state
    6. When agent turns around, maps still show the object!
    """
    
    def __init__(self, env, lr=0.0005, gamma=0.99, device='cpu', grid_size=10,
                 target_update_freq=100, confidence_threshold=0.5,
                 selective_training=True, hidden_dim=128):
        
        self.env = env
        self.gamma = gamma
        self.device = device
        self.grid_size = grid_size
        self.target_update_freq = target_update_freq
        self.confidence_threshold = confidence_threshold
        self.selective_training = selective_training
        
        self.action_size = 3
        self.feature_names = ['red', 'blue', 'box', 'sphere']
        
        self.object_to_features = {
            'red_box': ['red', 'box'],
            'blue_box': ['blue', 'box'],
            'red_sphere': ['red', 'sphere'],
            'blue_sphere': ['blue', 'sphere'],
        }
        
        # State dim: 4 allocentric maps (4 * grid_size^2) + position (2) + direction (4)
        # For grid_size=10: 4*100 + 2 + 4 = 406
        self.state_dim = 4 * grid_size * grid_size + 2 + 4
        
        # Create 4 WVF networks (one per feature)
        self.wvf_models = {}
        self.target_models = {}
        self.optimizers = {}
        
        for feature in self.feature_names:
            self.wvf_models[feature] = WVF_MLP_v3(
                state_dim=self.state_dim,
                num_actions=self.action_size,
                hidden_dim=hidden_dim
            ).to(device)
            
            self.target_models[feature] = WVF_MLP_v3(
                state_dim=self.state_dim,
                num_actions=self.action_size,
                hidden_dim=hidden_dim
            ).to(device)
            self.target_models[feature].load_state_dict(
                self.wvf_models[feature].state_dict()
            )
            self.target_models[feature].eval()
            
            self.optimizers[feature] = optim.Adam(
                self.wvf_models[feature].parameters(), lr=lr
            )
        
        self.loss_fn = nn.MSELoss()
        
        # ALLOCENTRIC reward maps - these are the MEMORY
        # They persist across steps within an episode
        self.feature_reward_maps = {
            feature: np.zeros((grid_size, grid_size), dtype=np.float32)
            for feature in self.feature_names
        }
        
        # Egocentric observations (still used for detection, but not as network input)
        self.feature_ego_obs = {
            feature: np.zeros((13, 13), dtype=np.float32)
            for feature in self.feature_names
        }
        
        # Tracking
        self.update_counts = {feature: 0 for feature in self.feature_names}
        self.loss_history = {feature: [] for feature in self.feature_names}
        self.current_task = None
        self.last_detection = None
    
    # ==================== State Features (WITH MEMORY) ====================
    
    def _get_allocentric_state(self):
        """
        Get state features using ALLOCENTRIC maps.
        
        This is the key change - the network sees persistent global maps,
        not ephemeral egocentric views.
        
        State = [red_map | blue_map | box_map | sphere_map | position | direction]
             = [  100   |   100    |   100   |    100     |    2     |    4     ]
             = 406 dimensions
        """
        # Stack all 4 allocentric maps: shape (4, grid_size, grid_size)
        allocentric_maps = np.stack([
            self.feature_reward_maps['red'],
            self.feature_reward_maps['blue'],
            self.feature_reward_maps['box'],
            self.feature_reward_maps['sphere']
        ], axis=0)
        
        # Flatten: 4 * 100 = 400
        maps_flat = allocentric_maps.flatten().astype(np.float32)
        
        # Normalized agent position (2)
        pos_x, pos_z = self._get_agent_pos_from_env()
        position = np.array([
            pos_x / max(self.grid_size - 1, 1),
            pos_z / max(self.grid_size - 1, 1)
        ], dtype=np.float32)
        
        # One-hot direction (4)
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._get_agent_dir_from_env()] = 1.0
        
        # Concatenate: 400 + 2 + 4 = 406
        state = np.concatenate([maps_flat, position, direction_onehot])
        
        return state
    
    def get_all_state_features(self):
        """
        Get state features for all networks (they all see the same state).
        """
        allocentric_state = self._get_allocentric_state()
        return {feature: allocentric_state for feature in self.feature_names}
    
    # ==================== Vision Integration ====================
    
    def update_from_detection(self, detection_result):
        """
        Update allocentric maps from vision detection.
        
        This is where MEMORY is created:
        - See object → convert to global coords → mark on persistent map
        """
        self.last_detection = detection_result
        positions = detection_result.get('positions', {})
        probabilities = detection_result.get('probabilities', {})
        
        # Reset egocentric observations (these are temporary)
        for feature in self.feature_names:
            self.feature_ego_obs[feature] = np.zeros((13, 13), dtype=np.float32)
        
        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()
        
        for obj_name, pos in positions.items():
            if pos is None:
                continue
            
            confidence = probabilities.get(obj_name, 1.0)
            if confidence < self.confidence_threshold:
                continue
            
            dx, dz = pos
            dx = int(round(dx))
            dz = int(round(dz))
            
            # Update egocentric (temporary, for debugging)
            for feature in self.object_to_features.get(obj_name, []):
                self._place_in_ego_obs(feature, dx, dz)
            
            # Update ALLOCENTRIC maps (PERSISTENT MEMORY!)
            global_x, global_z = self._ego_to_global(dx, dz, agent_x, agent_z, agent_dir)
            
            if 0 <= global_x < self.grid_size and 0 <= global_z < self.grid_size:
                for feature in self.object_to_features.get(obj_name, []):
                    # This persists! Memory of seen objects
                    self.feature_reward_maps[feature][global_z, global_x] = 1.0
    
    def _place_in_ego_obs(self, feature_name, dx, dz):
        """Place detection in egocentric observation (for debugging/visualization)."""
        ego_x = 6 + dx
        ego_z = 12 + dz
        if 0 <= ego_x < 13 and 0 <= ego_z < 13:
            self.feature_ego_obs[feature_name][ego_z, ego_x] = 1.0
    
    def _ego_to_global(self, dx, dz, agent_x, agent_z, agent_dir):
        """Convert egocentric offset to global coordinates."""
        if agent_dir == 3:  # North
            return agent_x + dx, agent_z + dz
        elif agent_dir == 0:  # East
            return agent_x - dz, agent_z + dx
        elif agent_dir == 1:  # South
            return agent_x - dx, agent_z - dz
        elif agent_dir == 2:  # West
            return agent_x + dz, agent_z - dx
        return agent_x + dx, agent_z + dz
    
    # ==================== Goal Extraction ====================
    
    def _get_goals_for_feature(self, feature_name):
        """Get goal positions from allocentric map."""
        goals = []
        reward_map = self.feature_reward_maps[feature_name]
        
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                if reward_map[z, x] > 0.5:
                    goals.append((x, z))
        
        return goals
    
    def _get_goals_for_task(self, task):
        """Get goals satisfying ALL task features (intersection)."""
        features = task.get('features', [])
        
        if len(features) == 0:
            return []
        
        goal_sets = [set(self._get_goals_for_feature(f)) for f in features]
        
        if any(len(gs) == 0 for gs in goal_sets):
            return []
        
        valid_goals = goal_sets[0]
        for gs in goal_sets[1:]:
            valid_goals = valid_goals.intersection(gs)
        
        return list(valid_goals)
    
    def _normalize_goal(self, goal_xy):
        """Normalize goal to [0, 1] range."""
        x, z = goal_xy
        return (x / max(self.grid_size - 1, 1), z / max(self.grid_size - 1, 1))
    
    # ==================== Q-Value Computation ====================
    
    def get_q_values_for_goal(self, feature_name, goal_xy, state_features=None):
        """Get Q-values for reaching a goal using a feature network."""
        if state_features is None:
            state_features = self._get_allocentric_state()
        
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
    
    def get_composed_q_values_for_goal(self, task, goal_xy, state_features=None):
        """Get composed Q-values via min() across task features."""
        features = task.get('features', [])
        
        if len(features) == 0:
            return np.zeros(self.action_size)
        
        if state_features is None:
            state_features = self._get_allocentric_state()
        
        q_values_list = []
        for feature in features:
            q = self.get_q_values_for_goal(feature, goal_xy, state_features)
            q_values_list.append(q)
        
        # Compose via minimum (AND logic)
        return np.min(np.stack(q_values_list), axis=0)
    
    # ==================== Action Selection ====================
    
    def sample_action_with_wvf(self, obs, task, epsilon=0.0):
        """Sample action using composed WVF with epsilon-greedy."""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        valid_goals = self._get_goals_for_task(task)
        
        if len(valid_goals) == 0:
            # No goals found in memory - explore
            return np.random.randint(self.action_size)
        
        # Get state once (includes allocentric memory)
        state_features = self._get_allocentric_state()
        
        best_q = float('-inf')
        best_action = None
        
        for goal in valid_goals:
            composed_q = self.get_composed_q_values_for_goal(task, goal, state_features)
            max_q = np.max(composed_q)
            
            if max_q > best_q:
                best_q = max_q
                best_action = np.argmax(composed_q)
        
        return int(best_action) if best_action is not None else np.random.randint(self.action_size)
    
    # ==================== Training ====================
    
    def set_current_task(self, task):
        """Set current task for selective training."""
        self.current_task = task
    
    def _update_single(self, feature_name, experience, goal_xy):
        """Update a feature network using TD learning."""
        state_features = experience[0][feature_name]
        action = experience[1]
        next_state_features = experience[2][feature_name]
        reward = experience[3][feature_name]
        done = experience[4]
        
        norm_goal = self._normalize_goal(goal_xy)
        
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
        
        model.train()
        current_q_values = model(state_tensor, goal_tensor)
        current_q = current_q_values[0, action]
        
        with torch.no_grad():
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32).to(self.device)
            else:
                next_q_values = target_model(next_state_tensor, goal_tensor)
                max_next_q = torch.max(next_q_values)
                td_target = reward + self.gamma * max_next_q
        
        loss = self.loss_fn(current_q, td_target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        self.update_counts[feature_name] += 1
        loss_val = loss.item()
        self.loss_history[feature_name].append(loss_val)
        
        if self.update_counts[feature_name] % self.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        
        return loss_val
    
    def update_for_feature(self, feature_name, experience):
        """Update a feature network for detected goals."""
        goals = self._get_goals_for_feature(feature_name)
        
        if len(goals) == 0:
            return 0.0
        
        # Limit goals per update for speed
        max_goals = 3
        if len(goals) > max_goals:
            indices = np.random.choice(len(goals), max_goals, replace=False)
            goals = [goals[i] for i in indices]
        
        total_loss = 0.0
        for goal in goals:
            loss = self._update_single(feature_name, experience, goal)
            total_loss += loss
        
        return total_loss / len(goals)
    
    def update_all_features(self, experience):
        """Update networks (selectively if enabled)."""
        losses = {}
        
        if self.selective_training and self.current_task is not None:
            features_to_train = self.current_task.get('features', self.feature_names)
        else:
            features_to_train = self.feature_names
        
        for feature_name in self.feature_names:
            if feature_name in features_to_train:
                losses[feature_name] = self.update_for_feature(feature_name, experience)
            else:
                losses[feature_name] = 0.0
        
        return losses
    
    # ==================== Rewards ====================
    
    def compute_feature_rewards(self, info):
        """Compute per-feature rewards based on contacted object."""
        contacted_object = info.get('contacted_object', None)
        
        rewards = {feature: 0.0 for feature in self.feature_names}
        
        if contacted_object is not None and contacted_object in self.object_to_features:
            for feature in self.object_to_features[contacted_object]:
                rewards[feature] = 1.0
        
        return rewards
    
    # ==================== Environment Helpers ====================
    
    def _get_agent_pos_from_env(self):
        """Get agent grid position."""
        try:
            x = int(round(self.env.agent.pos[0] / self.env.grid_size))
            z = int(round(self.env.agent.pos[2] / self.env.grid_size))
            return (np.clip(x, 0, self.grid_size - 1), 
                    np.clip(z, 0, self.grid_size - 1))
        except:
            return (0, 0)
    
    def _get_agent_dir_from_env(self):
        """Get agent direction (0=E, 1=S, 2=W, 3=N)."""
        try:
            angle = self.env.agent.dir
            degrees = np.degrees(angle) % 360
            if degrees < 45 or degrees >= 315:
                return 0
            elif 45 <= degrees < 135:
                return 3
            elif 135 <= degrees < 225:
                return 2
            else:
                return 1
        except:
            return 0
    
    def reset(self):
        """Reset for new episode - CLEARS MEMORY."""
        for feature in self.feature_names:
            self.feature_reward_maps[feature] = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.float32
            )
            self.feature_ego_obs[feature] = np.zeros((13, 13), dtype=np.float32)
        
        self.last_detection = None
    
    def get_composed_reward_map(self, task):
        """Get composed allocentric map for visualization."""
        features = task.get('features', [])
        
        if len(features) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        if len(features) == 1:
            return self.feature_reward_maps[features[0]].copy()
        
        maps = [self.feature_reward_maps[f] for f in features]
        return np.minimum.reduce(maps)


# ==================== Usage Example ====================

if __name__ == "__main__":
    print("""
    CompositionalWVFAgentV3 - With Allocentric Memory
    
    Key changes from v2:
    1. Network input is now ALLOCENTRIC maps (10x10 per feature = 400 dims)
    2. Objects persist in memory after turning away
    3. Goal is still passed as normalized (x, z) coordinates
    
    State representation:
        [red_map(100) | blue_map(100) | box_map(100) | sphere_map(100) | pos(2) | dir(4)]
        = 406 dimensions
    
    Usage:
    
        agent = CompositionalWVFAgentV3(
            env=env,
            lr=0.0005,
            gamma=0.99,
            device='cuda',
            grid_size=10,
            selective_training=True,
            hidden_dim=128
        )
        
        # The agent now remembers objects it has seen!
        # When it sees red_box, turns around, the red_box is still
        # marked in feature_reward_maps['red'] and ['box']
        
        # These maps are fed directly to the network as state
    """)