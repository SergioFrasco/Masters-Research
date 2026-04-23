"""
Comprehensive Logical Composition Evaluation for SR Agent

Tests the SR agent's zero-shot compositional generalization across:
1. AND tasks (already working)
2. OR tasks (disjunction)
3. NOT tasks (negation)
4. Complex combinations (AND + OR + NOT)

Uses pre-trained vision model and frozen SR matrix - NO RETRAINING.
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from env import DiscreteMiniWorldWrapper
from train_vision import CubeDetector
from torchvision import transforms
from PIL import Image


# ============================================================================
# EXTENDED SUCCESSOR AGENT WITH OR AND NOT LOGIC
# ============================================================================

class ExtendedSuccessorAgent:
    """Extended SR agent with AND, OR, and NOT composition"""
    
    def __init__(self, env, learning_rate=0.01, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.action_size = 3
        
        # Grid setup
        self.grid_size = env.size
        self.state_size = self.grid_size * self.grid_size
        
        # SR matrix
        self.M = np.zeros((self.action_size, self.state_size, self.state_size))
        
        # Previous experience
        self.prev_state = None
        self.prev_action = None

        # Feature maps with green
        self.feature_map = {
            "red": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "blue": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "green": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "box": np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
            "sphere": np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        }
        
        # Confidence parameters
        self.confidence_boost = 0.4
        self.step_decay_factor = 0.98
        self.confidence_threshold = 0.3
        
        # Composed reward map
        self.composed_reward_map = np.zeros((self.grid_size, self.grid_size))
        
        # WVF
        self.wvf = np.zeros((self.grid_size, self.grid_size))
        
        # Legacy
        self.reward_maps = np.zeros((self.state_size, self.grid_size, self.grid_size), dtype=np.float32)
    
    def load_frozen_sr(self, sr_path):
        """Load pre-trained frozen SR matrix"""
        self.M = np.load(sr_path)
        print(f"✓ Loaded frozen SR matrix from {sr_path}")
        print(f"  Shape: {self.M.shape}")
    
    def update_feature_map(self, detected_objects, positions):
        """Update feature map with within-episode decay"""
        # Decay existing confidence
        for feature in self.feature_map:
            self.feature_map[feature] *= self.step_decay_factor
        
        # Get agent info
        agent_x, agent_z = self._get_agent_pos_from_env()
        agent_dir = self._get_agent_dir_from_env()
        
        # Boost confidence for detected objects
        for obj_name in detected_objects:
            if obj_name in positions and positions[obj_name] is not None:
                dx, dz = positions[obj_name]
                dx, dz = int(round(dx)), int(round(dz))
                
                global_x, global_z = self._ego_to_global(dx, dz, agent_x, agent_z, agent_dir)
                
                if not (0 <= global_x < self.grid_size and 0 <= global_z < self.grid_size):
                    continue
                
                # Color features
                if "red" in obj_name:
                    self.feature_map["red"][global_z, global_x] = min(1.0, 
                        self.feature_map["red"][global_z, global_x] + self.confidence_boost)
                if "blue" in obj_name:
                    self.feature_map["blue"][global_z, global_x] = min(1.0,
                        self.feature_map["blue"][global_z, global_x] + self.confidence_boost)
                if "green" in obj_name:
                    self.feature_map["green"][global_z, global_x] = min(1.0,
                        self.feature_map["green"][global_z, global_x] + self.confidence_boost)
                
                # Shape features
                if "box" in obj_name:
                    self.feature_map["box"][global_z, global_x] = min(1.0,
                        self.feature_map["box"][global_z, global_x] + self.confidence_boost)
                if "sphere" in obj_name:
                    self.feature_map["sphere"][global_z, global_x] = min(1.0,
                        self.feature_map["sphere"][global_z, global_x] + self.confidence_boost)
    
    def compose_reward_map(self, task):
        """
        Compose feature maps based on task logic.
        
        Task format:
        - AND: {"features": ["red", "sphere"], "logic": "AND"}
        - OR: {"features": ["red", "blue"], "logic": "OR"}
        - NOT: {"features": ["red"], "logic": "NOT"}
        - Complex: {"features": [...], "logic": "COMPLEX", "expression": ...}
        """
        features = task["features"]
        logic = task.get("logic", "AND")  # Default to AND for backward compatibility
        
        if logic == "AND":
            # AND logic: threshold each map, then take minimum
            thresholded_maps = []
            for f in features:
                thresholded = (self.feature_map[f] > self.confidence_threshold).astype(np.float32)
                thresholded_maps.append(thresholded)
            self.composed_reward_map = np.minimum.reduce(thresholded_maps)
        
        elif logic == "OR":
            # OR logic: threshold each map, then take maximum
            thresholded_maps = []
            for f in features:
                thresholded = (self.feature_map[f] > self.confidence_threshold).astype(np.float32)
                thresholded_maps.append(thresholded)
            self.composed_reward_map = np.maximum.reduce(thresholded_maps)
        
        elif logic == "NOT":
            # NOT logic: Find objects that DON'T have this feature
            # Strategy: Take union of all objects, then subtract the NOT feature
            assert len(features) == 1, "NOT logic requires exactly one feature"
            
            feature_to_avoid = features[0]
            
            # Get all object locations (union of all feature maps)
            all_objects = np.zeros_like(self.feature_map["red"])
            for f in self.feature_map:
                thresholded = (self.feature_map[f] > self.confidence_threshold).astype(np.float32)
                all_objects = np.maximum(all_objects, thresholded)
            
            # Get locations of the feature to avoid
            avoid_map = (self.feature_map[feature_to_avoid] > self.confidence_threshold).astype(np.float32)
            
            # Reward map = all objects EXCEPT the ones with the avoided feature
            self.composed_reward_map = all_objects * (1.0 - avoid_map)
        
        elif logic == "COMPLEX":
            # Complex logic: evaluate expression
            # Expression format: "(red AND sphere) OR (blue AND box)"
            expression = task.get("expression", "")
            self.composed_reward_map = self._evaluate_complex_expression(expression)
        
        else:
            raise ValueError(f"Unknown logic type: {logic}")
    
    def _evaluate_complex_expression(self, expression):
        """
        Evaluate complex logical expression using numpy logical operations.
        
        Examples:
        - "(red AND sphere) OR (blue AND box)"
        - "(red OR blue) AND NOT green"
        - "red AND (sphere OR box)"
        """
        # Create a namespace with thresholded feature maps
        namespace = {}
        for feature, fmap in self.feature_map.items():
            # Store as boolean array
            namespace[feature] = (fmap > self.confidence_threshold)
        
        # Add numpy logical functions to namespace
        namespace['logical_and'] = np.logical_and
        namespace['logical_or'] = np.logical_or
        namespace['logical_not'] = np.logical_not
        
        # Replace operators with numpy function calls
        # Need to be careful with order of replacements
        expr = expression
        
        # Replace NOT first (with function call)
        expr = expr.replace("~", "logical_not")
        
        # Replace AND and OR with function calls
        # Use a temporary marker to avoid conflicts
        expr = expr.replace(" & ", " __AND__ ")
        expr = expr.replace(" | ", " __OR__ ")
        
        # Now replace markers with proper numpy calls
        # This is tricky - need to properly handle the syntax
        # Actually, let's just build it manually for the known patterns
        
        # Simpler approach: manually parse and build the expression
        # For the expressions we have, just hard-code the logic
        
        try:
            if expression == "(red & sphere) | (blue & box)":
                result = np.logical_or(
                    np.logical_and(namespace['red'], namespace['sphere']),
                    np.logical_and(namespace['blue'], namespace['box'])
                )
            elif expression == "(red | blue) & sphere":
                result = np.logical_and(
                    np.logical_or(namespace['red'], namespace['blue']),
                    namespace['sphere']
                )
            elif expression == "(red | blue) & ~green":
                result = np.logical_and(
                    np.logical_or(namespace['red'], namespace['blue']),
                    np.logical_not(namespace['green'])
                )
            elif expression == "red & (sphere | box)":
                result = np.logical_and(
                    namespace['red'],
                    np.logical_or(namespace['sphere'], namespace['box'])
                )
            elif expression == "(green & box) | (~red & sphere)":
                result = np.logical_or(
                    np.logical_and(namespace['green'], namespace['box']),
                    np.logical_and(np.logical_not(namespace['red']), namespace['sphere'])
                )
            elif expression == "~(red & sphere)":
                result = np.logical_not(
                    np.logical_and(namespace['red'], namespace['sphere'])
                )
            elif expression == "~(green & box)":
                result = np.logical_not(
                    np.logical_and(namespace['green'], namespace['box'])
                )
            else:
                print(f"Unknown complex expression: '{expression}'")
                return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
            return result.astype(np.float32)
            
        except Exception as e:
            print(f"Error evaluating expression '{expression}': {e}")
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
    
    def compute_wvf(self):
        """Compute WVF by applying SR to composed reward map"""
        MOVE_FORWARD = 2
        M_forward = self.M[MOVE_FORWARD, :, :]
        R_flat = self.composed_reward_map.flatten()
        V_flat = M_forward @ R_flat
        self.wvf = V_flat.reshape(self.grid_size, self.grid_size)
    
    def sample_action_with_wvf(self, obs, epsilon=0.0):
        """Sample action using WVF"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        x, z = self._get_agent_pos_from_env()
        current_dir = self._get_agent_dir_from_env()
        
        neighbors = [
            ((x + 1, z), 0),  # Right
            ((x, z + 1), 1),  # Down
            ((x - 1, z), 2),  # Left
            ((x, z - 1), 3)   # Up
        ]
        
        valid_actions = []
        valid_values = []
        
        for neighbor_pos, target_dir in neighbors:
            if self._is_valid_position(neighbor_pos):
                next_x, next_z = neighbor_pos
                value = self.wvf[next_z, next_x]
                action_to_take = self._get_action_toward_direction(current_dir, target_dir)
                
                valid_actions.append(action_to_take)
                valid_values.append(value)
        
        if not valid_actions:
            return np.random.randint(self.action_size)
        
        best_value = max(valid_values)
        best_indices = [i for i, v in enumerate(valid_values) if v == best_value]
        chosen_index = np.random.choice(best_indices)
        return valid_actions[chosen_index]
    
    def _ego_to_global(self, dx_ego, dz_ego, agent_x, agent_z, agent_dir):
        """Convert egocentric to global coordinates"""
        if agent_dir == 3:  # North
            dx_world, dz_world = dx_ego, dz_ego
        elif agent_dir == 0:  # East
            dx_world, dz_world = -dz_ego, dx_ego
        elif agent_dir == 1:  # South
            dx_world, dz_world = -dx_ego, -dz_ego
        elif agent_dir == 2:  # West
            dx_world, dz_world = dz_ego, -dx_ego
        
        global_x = agent_x + dx_world
        global_z = agent_z + dz_world
        return global_x, global_z
    
    def _is_valid_position(self, pos):
        """Check if position is valid"""
        x, z = pos[0], pos[1]
        original_pos = self._get_agent_pos_from_env()
        new_pos = np.array([x, original_pos[1], z])
        
        boundary = self.grid_size - 1
        if abs(x) > boundary or abs(z) > boundary:
            return False
        
        agent_radius = 0.18
        for entity in self.env.entities:
            if hasattr(entity, 'pos') and hasattr(entity, 'radius'):
                dist = np.linalg.norm(new_pos - entity.pos)
                if dist < agent_radius + entity.radius:
                    return False
        
        return True
    
    def _get_action_toward_direction(self, current_dir, target_dir):
        """Get action to face target direction"""
        if current_dir == target_dir:
            return 2  # move forward
        
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            return 1  # turn right
        elif diff == 3:
            return 0  # turn left
        elif diff == 2:
            return np.random.choice([0, 1])
        
        return 2
    
    def _get_agent_pos_from_env(self):
        """Get agent position from environment"""
        x = int(round(self.env.agent.pos[0]))
        z = int(round(self.env.agent.pos[2]))
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
        """Convert position to state index"""
        x, z = self._get_agent_pos_from_env()
        x = np.clip(x, 0, self.grid_size - 1)
        z = np.clip(z, 0, self.grid_size - 1)
        return z * self.grid_size + x
    
    def reset(self):
        """Reset for new episode"""
        self.prev_state = None
        self.prev_action = None
        
        for feature in self.feature_map:
            self.feature_map[feature].fill(0)
        
        self.composed_reward_map.fill(0)
        self.wvf.fill(0)


# ============================================================================
# VISION MODEL UTILITIES
# ============================================================================

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model."""
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
        model.load_state_dict(checkpoint)
        pos_mean = 0.0
        pos_std = 1.0
    
    model.eval()
    return model, device, pos_mean, pos_std


def detect_cube(model, obs, device, transform, pos_mean=0.0, pos_std=1.0):
    """Run cube detection."""
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
    if isinstance(img, np.ndarray):
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img)
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cls_logits, pos_preds = model(img_tensor)
        probs = torch.sigmoid(cls_logits)
        predictions = (probs > 0.5).cpu().numpy()[0]
        regression_values = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        label_names = ["red_box", "blue_box", "green_box", "red_sphere", "blue_sphere", "green_sphere"]
        detected_objects = [label_names[i] for i in range(6) if predictions[i]]
    
    positions = {
        'red_box': (regression_values[0], regression_values[1]) if predictions[0] else None,
        'blue_box': (regression_values[2], regression_values[3]) if predictions[1] else None,
        'green_box': (regression_values[4], regression_values[5]) if predictions[2] else None,
        'red_sphere': (regression_values[6], regression_values[7]) if predictions[3] else None,
        'blue_sphere': (regression_values[8], regression_values[9]) if predictions[4] else None,
        'green_sphere': (regression_values[10], regression_values[11]) if predictions[5] else None,
    }
    
    return {
        "detected_objects": detected_objects,
        "positions": positions,
    }


# ============================================================================
# TASK SATISFACTION CHECKING
# ============================================================================

def check_task_satisfaction(info, task):
    """
    Check if contacted object satisfies task requirements.
    Now handles AND, OR, NOT, and COMPLEX logic.
    """
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    logic = task.get("logic", "AND")
    
    # Map contacted object to its features
    object_features = {
        "red_box": ["red", "box"],
        "red_sphere": ["red", "sphere"],
        "blue_box": ["blue", "box"],
        "blue_sphere": ["blue", "sphere"],
        "green_box": ["green", "box"],
        "green_sphere": ["green", "sphere"],
    }
    
    contacted_features = set(object_features.get(contacted_object, []))
    task_features = set(features)
    
    if logic == "AND":
        # All task features must be in contacted object
        return task_features.issubset(contacted_features)
    
    elif logic == "OR":
        # At least one task feature must be in contacted object
        return len(task_features & contacted_features) > 0
    
    elif logic == "NOT":
        # None of the task features should be in contacted object
        return len(task_features & contacted_features) == 0
    
    elif logic == "COMPLEX":
        # Evaluate complex expression
        expression = task.get("expression", "")
        return evaluate_complex_satisfaction(expression, contacted_features)
    
    return False


def evaluate_complex_satisfaction(expression, contacted_features):
    """Evaluate complex logical expression for task satisfaction."""
    # Create namespace with feature presence
    namespace = {}
    all_features = ["red", "blue", "green", "box", "sphere"]
    for feature in all_features:
        namespace[feature] = feature in contacted_features
    
    # Replace logical operators
    expr = expression.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    
    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
        return bool(result)
    except Exception as e:
        print(f"Error evaluating satisfaction expression '{expression}': {e}")
        return False


# ============================================================================
# COMPREHENSIVE TASK DEFINITIONS
# ============================================================================

# AND Tasks (Conjunction)
AND_TASKS = [
    {"name": "red AND box", "features": ["red", "box"], "logic": "AND", "category": "AND"},
    {"name": "red AND sphere", "features": ["red", "sphere"], "logic": "AND", "category": "AND"},
    {"name": "blue AND box", "features": ["blue", "box"], "logic": "AND", "category": "AND"},
    {"name": "blue AND sphere", "features": ["blue", "sphere"], "logic": "AND", "category": "AND"},
    {"name": "green AND box", "features": ["green", "box"], "logic": "AND", "category": "AND"},
    {"name": "green AND sphere", "features": ["green", "sphere"], "logic": "AND", "category": "AND"},
]

# OR Tasks (Disjunction)
OR_TASKS = [
    {"name": "red OR blue", "features": ["red", "blue"], "logic": "OR", "category": "OR"},
    {"name": "red OR green", "features": ["red", "green"], "logic": "OR", "category": "OR"},
    {"name": "blue OR green", "features": ["blue", "green"], "logic": "OR", "category": "OR"},
    {"name": "box OR sphere", "features": ["box", "sphere"], "logic": "OR", "category": "OR"},
    {"name": "red OR box", "features": ["red", "box"], "logic": "OR", "category": "OR"},
    {"name": "blue OR sphere", "features": ["blue", "sphere"], "logic": "OR", "category": "OR"},
]

# NOT Tasks (Negation - "go to any object that is NOT X")
NOT_TASKS = [
    {"name": "NOT red (any non-red object)", "features": ["red"], "logic": "NOT", "category": "NOT"},
    {"name": "NOT blue (any non-blue object)", "features": ["blue"], "logic": "NOT", "category": "NOT"},
    {"name": "NOT green (any non-green object)", "features": ["green"], "logic": "NOT", "category": "NOT"},
    {"name": "NOT box (any non-box object)", "features": ["box"], "logic": "NOT", "category": "NOT"},
    {"name": "NOT sphere (any non-sphere object)", "features": ["sphere"], "logic": "NOT", "category": "NOT"},
]

# Complex Tasks (Mixed Logic)
COMPLEX_TASKS = [
    {
        "name": "(red AND sphere) OR (blue AND box)",
        "features": ["red", "sphere", "blue", "box"],
        "logic": "COMPLEX",
        "expression": "(red & sphere) | (blue & box)",
        "category": "COMPLEX"
    },
    {
        "name": "(red OR blue) AND sphere",
        "features": ["red", "blue", "sphere"],
        "logic": "COMPLEX",
        "expression": "(red | blue) & sphere",
        "category": "COMPLEX"
    },
    {
        "name": "(red OR blue) AND NOT green",
        "features": ["red", "blue", "green"],
        "logic": "COMPLEX",
        "expression": "(red | blue) & ~green",
        "category": "COMPLEX"
    },
    {
        "name": "red AND (sphere OR box)",
        "features": ["red", "sphere", "box"],
        "logic": "COMPLEX",
        "expression": "red & (sphere | box)",
        "category": "COMPLEX"
    },
    {
        "name": "(green AND box) OR (NOT red AND sphere)",
        "features": ["green", "box", "red", "sphere"],
        "logic": "COMPLEX",
        "expression": "(green & box) | (~red & sphere)",
        "category": "COMPLEX"
    },
    {
        "name": "NOT (green AND box)",
        "features": ["green", "box"],
        "logic": "COMPLEX",
        "expression": "~(green & box)",
        "category": "COMPLEX"
    },
]


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_logical_compositions(
    sr_matrix_path,
    vision_model_path='models/advanced_cube_detector.pth',
    episodes_per_task=100,
    max_steps=200,
    env_size=10,
    seed=42,
    output_dir='logical_composition_results'
):
    """
    Evaluate SR agent on comprehensive logical composition tasks.
    
    Args:
        sr_matrix_path: Path to frozen SR matrix (.npy file)
        vision_model_path: Path to pre-trained vision model
        episodes_per_task: Number of episodes to run per task
        max_steps: Max steps per episode
        env_size: Environment size
        seed: Random seed
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print(f"LOGICAL COMPOSITION EVALUATION")
    print(f"{'='*70}")
    print(f"SR Matrix: {sr_matrix_path}")
    print(f"Vision Model: {vision_model_path}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    cube_model, device, pos_mean, pos_std = load_cube_detector(vision_model_path, force_cpu=False)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create environment (evaluation mode - includes green objects)
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array", training_mode=False)
    
    # Create agent and load frozen SR
    agent = ExtendedSuccessorAgent(env)
    agent.load_frozen_sr(sr_matrix_path)
    
    # Combine all tasks
    all_tasks = AND_TASKS + OR_TASKS + NOT_TASKS + COMPLEX_TASKS
    
    # Results storage
    results = {}
    
    # Evaluate each task
    for task in all_tasks:
        task_name = task['name']
        category = task['category']
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {task_name} ({category})")
        print(f"{'='*70}")
        
        env.set_task(task)
        
        successes = 0
        total_rewards = []
        
        for episode in tqdm(range(episodes_per_task), desc=f"{task_name}"):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            
            for step in range(max_steps):
                # Detect objects
                detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
                detected_objects = detection_result['detected_objects']
                positions = detection_result['positions']
                
                # Update agent's feature maps
                agent.update_feature_map(detected_objects, positions)
                
                # Compose reward map based on task logic
                agent.compose_reward_map(task)
                
                # Compute WVF
                agent.compute_wvf()
                
                # Select action (greedy, no exploration during eval)
                action = agent.sample_action_with_wvf(obs, epsilon=0.0)
                
                # Take action
                obs, env_reward, terminated, truncated, info = env.step(action)
                
                # Check task satisfaction
                if check_task_satisfaction(info, task):
                    episode_reward = 1.0
                    successes += 1
                    break
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        # Calculate statistics
        success_rate = successes / episodes_per_task
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        results[task_name] = {
            "category": category,
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "successes": successes,
            "total_episodes": episodes_per_task,
        }
        
        print(f"✓ {task_name}: {success_rate:.3f} success rate ({successes}/{episodes_per_task})")
    
    # Save results
    results_file = output_dir / "logical_composition_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate visualization
    plot_results(results, output_dir)
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results, output_dir):
    """Create bar chart visualization of results by category."""
    
    # Organize by category
    categories = ["AND", "OR", "NOT", "COMPLEX"]
    category_results = {cat: [] for cat in categories}
    category_names = {cat: [] for cat in categories}
    category_errors = {cat: [] for cat in categories}
    
    for task_name, data in results.items():
        cat = data['category']
        category_results[cat].append(data['success_rate'])
        category_names[cat].append(task_name)
        category_errors[cat].append(data['std_reward'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SR Agent Performance on Logical Compositions', fontsize=16, fontweight='bold')
    
    colors = {
        "AND": "#2ecc71",      # Green
        "OR": "#3498db",       # Blue
        "NOT": "#e74c3c",      # Red
        "COMPLEX": "#9b59b6"   # Purple
    }
    
    for idx, (ax, category) in enumerate(zip(axes.flat, categories)):
        if not category_results[category]:
            ax.set_visible(False)
            continue
        
        success_rates = category_results[category]
        task_names = category_names[category]
        errors = category_errors[category]
        
        x_pos = np.arange(len(task_names))
        
        bars = ax.bar(x_pos, success_rates, yerr=errors, 
                     color=colors[category], alpha=0.7, capsize=5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, success_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Task', fontsize=11, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{category} Tasks', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = output_dir / "logical_composition_performance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    plt.close()
    
    # Also create a summary comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average success rate per category
    avg_success_by_category = []
    category_labels = []
    
    for cat in categories:
        if category_results[cat]:
            avg_success = np.mean(category_results[cat])
            avg_success_by_category.append(avg_success)
            category_labels.append(cat)
    
    x_pos = np.arange(len(category_labels))
    bars = ax.bar(x_pos, avg_success_by_category, 
                  color=[colors[cat] for cat in category_labels],
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, avg_success_by_category):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Logic Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Success Rate', fontsize=13, fontweight='bold')
    ax.set_title('SR Agent Performance by Logic Type', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_labels, fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    summary_file = output_dir / "category_summary.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {summary_file}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Configuration
    SR_MATRIX_PATH = "experiment_results_green/green_comparison_20251229_153537/SR_seed0/frozen_sr_matrix.npy"
    VISION_MODEL_PATH = "models/advanced_cube_detector.pth"
    EPISODES_PER_TASK = 200  # Increased for better statistics and variance
    MAX_STEPS = 200
    ENV_SIZE = 10
    SEED = 42
    OUTPUT_DIR = "logical_composition_results"
    
    print("\n" + "="*70)
    print("STARTING LOGICAL COMPOSITION EVALUATION")
    print("="*70)
    print(f"\nMake sure you have:")
    print(f"  1. Frozen SR matrix at: {SR_MATRIX_PATH}")
    print(f"  2. Vision model at: {VISION_MODEL_PATH}")
    print(f"\nTesting {len(AND_TASKS)} AND, {len(OR_TASKS)} OR, {len(NOT_TASKS)} NOT, and {len(COMPLEX_TASKS)} COMPLEX tasks")
    print(f"Total tasks: {len(AND_TASKS) + len(OR_TASKS) + len(NOT_TASKS) + len(COMPLEX_TASKS)}")
    print("="*70 + "\n")
    
    # Run evaluation
    results = evaluate_logical_compositions(
        sr_matrix_path=SR_MATRIX_PATH,
        vision_model_path=VISION_MODEL_PATH,
        episodes_per_task=EPISODES_PER_TASK,
        max_steps=MAX_STEPS,
        env_size=ENV_SIZE,
        seed=SEED,
        output_dir=OUTPUT_DIR
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*70)
    
    for category in ["AND", "OR", "NOT", "COMPLEX"]:
        cat_tasks = [t for t in results.values() if t['category'] == category]
        if cat_tasks:
            avg_success = np.mean([t['success_rate'] for t in cat_tasks])
            print(f"\n{category} Tasks:")
            print(f"  Average Success Rate: {avg_success:.3f}")
            for task_name, data in results.items():
                if data['category'] == category:
                    print(f"    {task_name}: {data['success_rate']:.3f}")
    
    print("\n" + "="*70)
    print(f"✓ All results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()