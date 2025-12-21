"""
Experiment Utilities

Training and evaluation functions for all algorithms.
All algorithms trained uniformly (random primitive tasks) and evaluated on compositional tasks.
"""

import os

# Set environment variables for headless mode
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Removed duplicate
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
import json

from env import DiscreteMiniWorldWrapper
from agents import (
    SuccessorAgent,
    UnifiedDQNAgent,
    UnifiedLSTMDQNAgent3D,
    UnifiedWorldValueFunctionAgent
)
from train_vision import CubeDetector
from torchvision import transforms
from PIL import Image


# ============================================================================
# TASK DEFINITIONS (shared across all algorithms)
# ============================================================================

PRIMITIVE_TASKS = [
    {"name": "red", "features": ["red"], "type": "primitive"},
    {"name": "blue", "features": ["blue"], "type": "primitive"},
    {"name": "box", "features": ["box"], "type": "primitive"},
    {"name": "sphere", "features": ["sphere"], "type": "primitive"},
]

COMPOSITIONAL_TASKS = [
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    
    # Single feature tasks
    if len(features) == 1:
        feature = features[0]
        
        if feature == "blue":
            return contacted_object in ["blue_box", "blue_sphere"]
        elif feature == "red":
            return contacted_object in ["red_box", "red_sphere"]
        elif feature == "box":
            return contacted_object in ["blue_box", "red_box"]
        elif feature == "sphere":
            return contacted_object in ["blue_sphere", "red_sphere"]
    
    # Compositional tasks (2 features - AND logic)
    elif len(features) == 2:
        if set(features) == {"blue", "sphere"}:
            return contacted_object == "blue_sphere"
        elif set(features) == {"red", "sphere"}:
            return contacted_object == "red_sphere"
        elif set(features) == {"blue", "box"}:
            return contacted_object == "blue_box"
        elif set(features) == {"red", "box"}:
            return contacted_object == "red_box"
    
    return False


# ============================================================================
# SUCCESSOR REPRESENTATION AGENT
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
        
        label_names = ["red_box", "blue_box", "red_sphere", "blue_sphere"]
        detected_objects = [label_names[i] for i in range(4) if predictions[i]]
    
    positions = {
        'red_box': (regression_values[0], regression_values[1]) if predictions[0] else None,
        'blue_box': (regression_values[2], regression_values[3]) if predictions[1] else None,
        'red_sphere': (regression_values[4], regression_values[5]) if predictions[2] else None,
        'blue_sphere': (regression_values[6], regression_values[7]) if predictions[3] else None,
    }
    
    return {
        "detected_objects": detected_objects,
        "positions": positions,
    }


def train_sr_agent(seed, training_episodes, eval_episodes_per_task, max_steps, env_size,
                   sr_freeze_episode, output_dir):
    """
    Train Successor Representation agent.
    
    CRITICAL: Freeze SR matrix at sr_freeze_episode (3000), continue training with frozen SR.
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING SR AGENT (Seed={seed})")
    print(f"{'='*70}")
    print(f"Training episodes: {training_episodes}")
    print(f"SR freeze episode: {sr_freeze_episode}")
    print(f"Eval episodes per task: {eval_episodes_per_task}")
    print(f"{'='*70}\n")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create environment and agent
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    agent = SuccessorAgent(env)
    
    # Load cube detector
    cube_model, device, pos_mean, pos_std = load_cube_detector(force_cpu=False)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tracking
    all_rewards = []  # All episodes (training + eval)
    episode_labels = []  # Which task each episode
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    sr_frozen = False
    frozen_sr_matrix = None
    
    # ===== TRAINING PHASE =====
    print("Starting training phase...")
    
    for episode in tqdm(range(training_episodes), desc="Training SR"):
        # Randomly sample primitive task
        current_task = random.choice(PRIMITIVE_TASKS)
        env.set_task(current_task)
        
        obs, info = env.reset()
        agent.reset()
        
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Detection
            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            detected_objects = detection_result['detected_objects']
            positions = detection_result['positions']
            
            # Update feature map
            agent.update_feature_map(detected_objects, positions)
            agent.compose_reward_map(current_task)
            agent.compute_wvf()
            
            # Step environment
            obs, env_reward, terminated, truncated, info = env.step(current_action)
            
            # Check task satisfaction
            task_satisfied = check_task_satisfaction(info, current_task)
            if task_satisfied:
                episode_reward += env_reward
            
            # Get next state
            next_state = agent.get_state_index()
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            done = terminated or truncated
            
            # Update SR ONLY if not frozen
            if not sr_frozen:
                agent.update_sr(current_state, current_action, next_state, next_action, done)
            
            current_state = next_state
            current_action = next_action
            
            if done:
                break
        
        # FREEZE SR at specified episode
        if episode == sr_freeze_episode - 1:  # -1 because 0-indexed
            print(f"\n⚠️  FREEZING SR MATRIX at episode {episode + 1}")
            frozen_sr_matrix = agent.M.copy()
            sr_frozen = True
            
            # Save frozen SR
            sr_save_path = output_dir / "frozen_sr_matrix.npy"
            np.save(sr_save_path, frozen_sr_matrix)
            print(f"✓ Frozen SR saved to: {sr_save_path}\n")
        
        all_rewards.append(episode_reward)
        episode_labels.append(current_task['name'])
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # ===== EVALUATION PHASE =====
    print(f"\nStarting evaluation phase (using frozen SR from episode {sr_freeze_episode})...")
    
    # Load frozen SR for evaluation
    if frozen_sr_matrix is not None:
        agent.M = frozen_sr_matrix.copy()
    
    eval_task_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        
        print(f"Evaluating {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            
            for step in range(max_steps):
                # Detection
                detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
                detected_objects = detection_result['detected_objects']
                positions = detection_result['positions']
                
                # Update feature map and compose
                agent.update_feature_map(detected_objects, positions)
                agent.compose_reward_map(comp_task)
                agent.compute_wvf()
                
                # Select action
                action = agent.sample_action_with_wvf(obs, epsilon=0.0)  # Greedy
                
                obs, env_reward, terminated, truncated, info = env.step(action)
                
                task_satisfied = check_task_satisfaction(info, comp_task)
                if task_satisfied:
                    episode_reward += env_reward
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            eval_task_labels.append(task_name)
    
    # Combine labels
    all_labels = episode_labels + eval_task_labels
    
    print(f"\n✓ SR Agent training complete (seed={seed})")
    print(f"  Training episodes: {training_episodes}")
    print(f"  Eval episodes: {len(eval_task_labels)}")
    print(f"  SR frozen at episode: {sr_freeze_episode}")
    
    return {
        "algorithm": "SR",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "eval_episodes": len(eval_task_labels),
        "sr_freeze_episode": sr_freeze_episode,
    }


# ============================================================================
# UNIFIED DQN AGENT
# ============================================================================

def train_unified_dqn(seed, training_episodes, eval_episodes_per_task, max_steps, env_size,
                      learning_rate, gamma, epsilon_decay, output_dir):
    """Train Unified DQN agent."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED DQN (Seed={seed})")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    agent = UnifiedDQNAgent(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=100000,
        batch_size=64,
        hidden_size=256,
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    all_rewards = []
    episode_labels = []
    
    # ===== TRAINING =====
    print("Starting training phase...")
    
    for episode in tqdm(range(training_episodes), desc="Training DQN"):
        task = random.choice(PRIMITIVE_TASKS)
        env.set_task(task)
        
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, task['name'])
            next_obs, _, terminated, truncated, info = env.step(action)
            
            task_satisfied = check_task_satisfaction(info, task)
            reward = 1.0 if task_satisfied else (-0.1 if info.get('contacted_object') else -0.005)
            
            agent.remember(obs, task['name'], action, reward, next_obs, terminated or truncated)
            
            loss = agent.train_step()
            
            if task_satisfied:
                episode_reward = 1.0
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        episode_labels.append(task['name'])
    
    # Save model
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # ===== EVALUATION =====
    print("\nStarting evaluation phase...")
    
    eval_task_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        print(f"Evaluating {comp_task['name']}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Use compositional encoding
                action = agent.select_action(obs, comp_task['features'], epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward = 1.0
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            eval_task_labels.append(comp_task['name'])
    
    all_labels = episode_labels + eval_task_labels
    
    print(f"\n✓ DQN training complete (seed={seed})")
    
    return {
        "algorithm": "DQN",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "eval_episodes": len(eval_task_labels),
    }


# ============================================================================
# UNIFIED LSTM-DQN AGENT
# ============================================================================

def train_unified_lstm_dqn(seed, training_episodes, eval_episodes_per_task, max_steps, env_size,
                           learning_rate, gamma, epsilon_decay, output_dir):
    """Train Unified LSTM-DQN agent."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED LSTM-DQN (Seed={seed})")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    agent = UnifiedLSTMDQNAgent3D(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=2000,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    all_rewards = []
    episode_labels = []
    
    # ===== TRAINING =====
    print("Starting training phase...")
    
    for episode in tqdm(range(training_episodes), desc="Training LSTM-DQN"):
        task = random.choice(PRIMITIVE_TASKS)
        env.set_task(task)
        
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs, task['name'])
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            next_stacked_obs = agent.step_episode(next_obs)
            
            task_satisfied = check_task_satisfaction(info, task)
            reward = 1.0 if task_satisfied else (-0.1 if info.get('contacted_object') else -0.005)
            
            agent.remember(stacked_obs, action, reward, next_stacked_obs, terminated or truncated)
            
            if step % 4 == 0:
                agent.train_step()
            
            if task_satisfied:
                episode_reward = 1.0
            
            stacked_obs = next_stacked_obs
            
            if terminated or truncated:
                break
        
        agent.decay_epsilon()
        agent.update_task_success(task['name'], episode_reward > 0)
        
        all_rewards.append(episode_reward)
        episode_labels.append(task['name'])
    
    # Save model
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # ===== EVALUATION =====
    print("\nStarting evaluation phase...")
    
    eval_task_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        print(f"Evaluating {comp_task['name']}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            stacked_obs = agent.reset_episode(obs, comp_task['name'])
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(stacked_obs, epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                stacked_obs = agent.step_episode(obs)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward = 1.0
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            eval_task_labels.append(comp_task['name'])
    
    all_labels = episode_labels + eval_task_labels
    
    print(f"\n✓ LSTM-DQN training complete (seed={seed})")
    
    return {
        "algorithm": "LSTM",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "eval_episodes": len(eval_task_labels),
    }


# ============================================================================
# UNIFIED WVF AGENT
# ============================================================================


def train_unified_wvf_improved(seed, training_episodes, eval_episodes_per_task, max_steps, env_size,
                                learning_rate, gamma, epsilon_decay, output_dir,
                                composition_mode='softmin',
                                softmin_temperature=0.1,
                                normalize_q_values=True):
    """
    Train Improved WVF agent with softmin composition and Q-value normalization.
    
    New parameters:
        composition_mode: 'softmin' (default), 'min', or 'normalized_min'
        softmin_temperature: Temperature for softmin (lower = more like hard min)
        normalize_q_values: Whether to normalize Q-values before composition
    """
    
    from env import DiscreteMiniWorldWrapper
    
    print(f"\n{'='*70}")
    print(f"TRAINING IMPROVED WVF (Seed={seed})")
    print(f"Composition mode: {composition_mode}")
    print(f"Softmin temperature: {softmin_temperature}")
    print(f"Normalize Q-values: {normalize_q_values}")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    agent = UnifiedWorldValueFunctionAgent(
        env,
        k_frames=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        memory_size=2000,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        tau=0.005,
        grad_clip=10.0,
        r_correct=1.0,
        r_wrong=-0.1,
        step_penalty=-0.005,
        # NEW parameters
        composition_mode=composition_mode,
        softmin_temperature=softmin_temperature,
        normalize_q_values=normalize_q_values
    )
    
    all_rewards = []
    episode_labels = []
    
    # ===== TRAINING PHASE =====
    print("Starting training phase...")
    print("Training on primitive tasks: red, blue, box, sphere\n")
    
    for episode in tqdm(range(training_episodes), desc="Training WVF (Improved)"):
        current_task = agent.sample_task()
        task_idx = agent.TASK_TO_IDX[current_task]
        
        task_config = {"name": current_task, "features": [current_task], "type": "primitive"}
        env.set_task(task_config)
        
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs, current_task)
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, task_idx)
            
            next_obs, _, terminated, truncated, info = env.step(action)
            next_stacked_obs = agent.step_episode(next_obs)
            
            reward, goal_reached = agent.compute_reward(info, current_task)
            
            if reward > 0:
                episode_reward = 1.0
            
            done = goal_reached or terminated or truncated
            agent.remember(stacked_obs, task_idx, action, reward, next_stacked_obs, done)
            
            if step % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.train_step()
            
            stacked_obs = next_stacked_obs
            
            if done:
                break
        
        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        episode_labels.append(current_task)
    
    # Save model
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # ===== EVALUATION PHASE =====
    print(f"\nStarting evaluation phase...")
    print(f"Composition mode: {composition_mode}")
    if composition_mode == 'softmin':
        print(f"Temperature: {softmin_temperature}")
    print(f"Q-value normalization: {normalize_q_values}\n")
    
    eval_task_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        features = comp_task['features']
        
        print(f"Evaluating {task_name} = compose(Q_{features[0]}, Q_{features[1]})")
        
        task_successes = 0
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            stacked_obs = agent.reset_episode(obs)
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action_composed(stacked_obs, features)
                
                obs, _, terminated, truncated, info = env.step(action)
                stacked_obs = agent.step_episode(obs)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward = 1.0
                    task_successes += 1
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            eval_task_labels.append(task_name)
        
        success_rate = task_successes / eval_episodes_per_task
        print(f"  {task_name}: {success_rate:.1%} success rate ({task_successes}/{eval_episodes_per_task})")
    
    all_labels = episode_labels + eval_task_labels
    
    print(f"\n✓ WVF (Improved) training complete (seed={seed})")
    print(f"  Training episodes: {training_episodes}")
    print(f"  Eval episodes: {len(eval_task_labels)}")
    
    return {
        "algorithm": "WVF_improved",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "eval_episodes": len(eval_task_labels),
        "composition_mode": composition_mode,
        "softmin_temperature": softmin_temperature,
        "normalize_q_values": normalize_q_values,
    }