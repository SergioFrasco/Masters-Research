"""
MLP Models for World Value Functions (WVF) - FIXED VERSION.

Key Change: Input dimension updated to accommodate shared state representation.

Old (broken): 175 dims = 169 (single feature ego) + 2 (pos) + 4 (dir)
New (fixed):  682 dims = 676 (4 feature channels * 169) + 2 (pos) + 4 (dir)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WVF_MLP(nn.Module):
    """
    MLP for World Value Functions with SHARED state representation.
    
    Input structure:
        - Egocentric observations: 4 channels × 13×13 = 676 dims
          (red, blue, box, sphere - ALL visible to ALL networks)
        - Agent position: 2 dims (normalized x, z)
        - Agent direction: 4 dims (one-hot)
        - Goal position: 2 dims (normalized x, z)
        
        Total: 676 + 2 + 4 + 2 = 684 dims
        
    Output:
        - Q(s, g, a) for each action: 3 dims
    """
    
    def __init__(self, state_dim=682, num_actions=3, hidden_dim=128):
        """
        Args:
            state_dim: Dimension of SHARED state features (default 682)
                      = 4*13*13 (all ego channels) + 2 (position) + 4 (direction)
            num_actions: Number of actions (default 3)
            hidden_dim: Hidden layer size
        """
        super(WVF_MLP, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Input: shared state features (682) + goal (x, y) normalized (2) = 684
        input_dim = state_dim + 2
        
        # Architecture: 4 layers with dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(0.1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features, goal_xy):
        """
        Forward pass.
        
        Args:
            state_features: (batch_size, state_dim) - SHARED state representation
                           Contains ALL object channels, not just one feature
            goal_xy: (batch_size, 2) - normalized goal position (x, y) in [0, 1]
            
        Returns:
            q_values: (batch_size, num_actions) - Q(s, g, a) for each action
        """
        # Concatenate state and goal
        x = torch.cat([state_features, goal_xy], dim=-1)
        
        # Forward through MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values


class WVF_MLP_LargerFixed(nn.Module):
    """
    Larger MLP variant for more complex environments.
    
    Since we now have 4x the input channels (676 vs 169 for ego obs),
    we may benefit from a slightly larger network.
    """
    
    def __init__(self, state_dim=682, num_actions=3, hidden_dim=256):
        super(WVF_MLP_LargerFixed, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        input_dim = state_dim + 2  # + goal
        
        # Larger architecture for richer state representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_actions)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features, goal_xy):
        x = torch.cat([state_features, goal_xy], dim=-1)
        
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values


# For backward compatibility, keep the old model available
class WVF_MLP(nn.Module):
    """
    Original MLP (DEPRECATED - use WVF_MLP_Fixed instead).
    
    This version uses feature-specific state input which breaks composition.
    Kept for comparison purposes only.
    """
    
    def __init__(self, state_dim=175, num_actions=3, hidden_dim=128):
        super(WVF_MLP, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
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
        x = torch.cat([state_features, goal_xy], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values