import torch
import torch.nn as nn
import torch.nn.functional as F


class WVF_CNN(nn.Module):
    """
    Convolutional Neural Network for World Value Functions (WVF).
    
    Uses CNNs instead of MLPs to leverage spatial inductive bias:
    - Neighboring cells naturally share features
    - "Reward nearby = high value" is easier to learn
    - Generalizes better to different reward configurations
    
    Takes as input:
    - Reward map (grid_size x grid_size) - where goals are located
    - Agent position index - converted to a position map
    
    Outputs:
    - Q-values for all state-action pairs (grid_size x grid_size x num_actions)
    
    Architecture processes reward map and agent position as 2-channel image,
    then uses convolutions to propagate value information spatially.
    """
    
    def __init__(self, grid_size=10, num_actions=3, hidden_channels=64):
        super(WVF_CNN, self).__init__()
        
        self.grid_size = grid_size
        self.num_actions = num_actions
        
        # Encoder: Extract features from reward map + agent position
        # Input: 2 channels (reward_map, agent_position_map)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Value propagation layers with larger receptive field
        # These layers learn to propagate value from reward locations outward
        self.value_propagation = nn.Sequential(
            # Dilated convolutions to capture long-range dependencies
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Output head: Q-values for each action at each location
        self.q_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_actions, kernel_size=1),  # 1x1 conv to get Q-values per action
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with small weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, reward_map, agent_pos_idx):
        """
        Forward pass through the CNN.
        
        Args:
            reward_map: (batch_size, grid_size, grid_size) - reward locations
            agent_pos_idx: (batch_size,) - flat state indices of agent positions
            
        Returns:
            q_values: (batch_size, grid_size, grid_size, num_actions)
        """
        batch_size = reward_map.shape[0]
        device = reward_map.device
        
        # Create agent position map (same shape as reward map)
        agent_pos_map = torch.zeros(batch_size, self.grid_size, self.grid_size, device=device)
        
        # Convert flat index to 2D coordinates and set position
        rows = agent_pos_idx // self.grid_size
        cols = agent_pos_idx % self.grid_size
        
        for b in range(batch_size):
            agent_pos_map[b, rows[b], cols[b]] = 1.0
        
        # Stack as 2-channel input: [reward_map, agent_position_map]
        # Shape: (batch, 2, grid_size, grid_size)
        x = torch.stack([reward_map, agent_pos_map], dim=1)
        
        # Encode
        features = self.encoder(x)
        
        # Propagate values spatially
        features = self.value_propagation(features)
        
        # Get Q-values
        # Shape: (batch, num_actions, grid_size, grid_size)
        q_values = self.q_head(features)
        
        # Permute to (batch, grid_size, grid_size, num_actions) to match expected format
        q_values = q_values.permute(0, 2, 3, 1)
        
        return q_values


class WVF_CNN_ResNet(nn.Module):
    """
    Alternative: ResNet-style WVF with skip connections.
    
    Skip connections help with:
    - Gradient flow during training
    - Preserving spatial information across layers
    - Faster convergence
    """
    
    def __init__(self, grid_size=10, num_actions=3, hidden_channels=64):
        super(WVF_CNN_ResNet, self).__init__()
        
        self.grid_size = grid_size
        self.num_actions = num_actions
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Residual blocks for value propagation
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_channels, hidden_channels, dilation=1),
            ResBlock(hidden_channels, hidden_channels, dilation=2),
            ResBlock(hidden_channels, hidden_channels, dilation=4),
            ResBlock(hidden_channels, hidden_channels, dilation=2),
            ResBlock(hidden_channels, hidden_channels, dilation=1),
        ])
        
        # Output head
        self.q_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_actions, kernel_size=1),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, reward_map, agent_pos_idx):
        batch_size = reward_map.shape[0]
        device = reward_map.device
        
        # Create agent position map
        agent_pos_map = torch.zeros(batch_size, self.grid_size, self.grid_size, device=device)
        rows = agent_pos_idx // self.grid_size
        cols = agent_pos_idx % self.grid_size
        
        for b in range(batch_size):
            agent_pos_map[b, rows[b], cols[b]] = 1.0
        
        # Stack inputs
        x = torch.stack([reward_map, agent_pos_map], dim=1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Q-values
        q_values = self.q_head(x)
        q_values = q_values.permute(0, 2, 3, 1)
        
        return q_values


class ResBlock(nn.Module):
    """Residual block with optional dilation for larger receptive field"""
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1)
        self.relu = nn.ReLU()
        
        # Skip connection projection if channels differ
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        
        return out


# For backwards compatibility / easy swapping
WVF_Model = WVF_CNN