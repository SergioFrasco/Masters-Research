import numpy as np

class RandomAgentWithSR:
    """Random agent that learns Successor Representation"""
    
    def __init__(self, env, learning_rate=0.01, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # Grid setup - MiniWorld uses continuous coordinates
        self.grid_size = env.size  # e.g., 10 for 10x10 room
        self.state_size = self.grid_size * self.grid_size
        
        # Initialize SR matrix: M[action, from_state, to_state]
        # M[a,s,s'] = expected future occupancy of state s' from state s under action a
        self.M = np.zeros((self.action_size, self.state_size, self.state_size))
        # self.M += np.random.normal(0, 0.01, self.M.shape)  # Small random initialization
        
        # Path integration - track internal position for state indexing
        self.internal_pos = None  # (grid_x, grid_z)
        self.internal_dir = None  # Cardinal direction 0-3
        self.initialized = False
        
        # Store previous experience for TD update
        self.prev_state = None
        self.prev_action = None
    
    def initialize_path_integration(self):
        """Initialize internal state from environment"""
        if not self.initialized:
            # Convert MiniWorld continuous position to discrete grid
            x = int(round(self.env.agent.pos[0]))  # x coordinate
            z = int(round(self.env.agent.pos[2]))  # z coordinate (not y, which is height)
            self.internal_pos = (x, z)
            
            # Convert angle to cardinal direction
            self.internal_dir = self._angle_to_cardinal(self.env.agent.dir)
            self.initialized = True
            print(f"Initialized: pos={self.internal_pos}, dir={self.internal_dir}")
    
    def _angle_to_cardinal(self, angle):
        """Convert MiniWorld angle (radians) to cardinal direction"""
        # 0=East(+X), 1=South(+Z), 2=West(-X), 3=North(-Z)
        degrees = (np.degrees(angle) % 360)
        if degrees < 45 or degrees >= 315:
            return 0  # East
        elif 45 <= degrees < 135:
            return 1  # South
        elif 135 <= degrees < 225:
            return 2  # West
        else:
            return 3  # North
    
    def get_state_index(self):
        """Convert grid position to flat state index"""
        if not self.initialized:
            self.initialize_path_integration()
        
        x, z = self.internal_pos
        # Clamp to valid range [0, grid_size-1]
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def update_internal_state(self, action):
        """Update internal position based on action (path integration)"""
        if not self.initialized:
            self.initialize_path_integration()
        
        x, z = self.internal_pos
        
        if action == self.TURN_LEFT:
            self.internal_dir = (self.internal_dir - 1) % 4
        elif action == self.TURN_RIGHT:
            self.internal_dir = (self.internal_dir + 1) % 4
        elif action == self.MOVE_FORWARD:
            # Move in current direction
            if self.internal_dir == 0:  # East
                new_x, new_z = x + 1, z
            elif self.internal_dir == 1:  # South
                new_x, new_z = x, z + 1
            elif self.internal_dir == 2:  # West
                new_x, new_z = x - 1, z
            else:  # North
                new_x, new_z = x, z - 1
            
            # Check bounds
            if 0 <= new_x < self.grid_size and 0 <= new_z < self.grid_size:
                self.internal_pos = (new_x, new_z)
    
    
    def update_sr(self, s, action, s_next, next_action, done):
        """
        Update SR matrix using TD learning with next action (SARSA-style)
        Only update on MOVE_FORWARD actions (actual state transitions)
        """
        # Skip turns - they don't change state
        if action != self.MOVE_FORWARD:
            return 0.0
        
        # Skip if we didn't actually move
        if s == s_next and not done:
            return 0.0
        
        # One-hot encoding of current state
        I = np.zeros(self.state_size)
        I[s] = 1.0
        
        if done:
            # Terminal state: no future occupancy expected
            td_target = I
        else:
            # Use the NEXT action for bootstrapping (SARSA-style)
            if next_action == self.MOVE_FORWARD:
                # Next action will transition states, use it for bootstrapping
                td_target = I + self.gamma * self.M[next_action, s_next, :]
            else:
                # Next action is a turn - doesn't change state but takes time
                # Use MOVE_FORWARD as default (no gamma discount for turns)
                td_target = I + self.M[self.MOVE_FORWARD, s_next, :]
        
        # TD error and update
        td_error = td_target - self.M[action, s, :]
        self.M[action, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))
    
    def select_action(self):
        """Select a random action"""
        if not self.initialized:
            self.initialize_path_integration()
        return np.random.randint(self.action_size)
    
    def reset(self):
        """Reset for new episode"""
        self.initialized = False
        self.internal_pos = None
        self.internal_dir = None
        self.prev_state = None
        self.prev_action = None