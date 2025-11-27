import torch
import torch.nn as nn
import torch.nn.functional as F


class WVF_MLP(nn.Module):
    """
    MLP for World Value Functions (WVF).
    
    Matches DQN baseline structure exactly, but adds goal (x, y) as input.
    
    DQN baseline uses:
        Input: ego_obs (13*13=169) + position (2) + direction_onehot (4) = 175 dims
        Output: Q(s, a) for 3 actions
    
    WVF adds goal conditioning:
        Input: ego_obs (169) + position (2) + direction (4) + goal_xy (2) = 177 dims
        Output: Q(s, g, a) for 3 actions
    
    As suggested by Geraud Nangue Tasse:
    "Say you have the state (s) be the same one you are using for the DQN 
    (be it the reward map or feature vector from resnet, etc). Then concatenate 
    that with the goal vector (x,y position) to get the input vector (s,g) for your mlp."
    """
    
    def __init__(self, state_dim=175, num_actions=3, hidden_dim=128):
        """
        Args:
            state_dim: Dimension of state features (default 175 to match DQN)
                      = 13*13 (ego_obs) + 2 (position) + 4 (direction) = 175
            num_actions: Number of actions (default 3: turn left, turn right, forward)
            hidden_dim: Hidden layer size
        """
        super(WVF_MLP, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Input: state features (175) + goal (x, y) normalized (2) = 177
        input_dim = state_dim + 2
        
        # Match DQN architecture: 4 layers with dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(0.1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features, goal_xy):
        """
        Forward pass.
        
        Args:
            state_features: (batch_size, state_dim) 
                           = ego_obs (169) + position (2) + direction (4)
            goal_xy: (batch_size, 2) - normalized goal position (x, y) in [0, 1]
            
        Returns:
            q_values: (batch_size, num_actions) - Q(s, g, a) for each action
        """
        # Concatenate state and goal
        x = torch.cat([state_features, goal_xy], dim=-1)
        
        # Forward through MLP (matching DQN architecture)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values