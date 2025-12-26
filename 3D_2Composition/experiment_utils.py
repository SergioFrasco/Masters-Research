"""
Experiment Utilities

Training and evaluation functions for all algorithms.
All algorithms trained uniformly (random primitive tasks) and evaluated on compositional tasks.
"""

import os

# Set environment variables for headless mode
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
    """Train Successor Representation agent."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING SR AGENT (Seed={seed})")
    print(f"{'='*70}")
    print(f"Training episodes: {training_episodes}")
    print(f"SR freeze episode: {sr_freeze_episode}")
    print(f"Eval episodes per task: {eval_episodes_per_task}")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    agent = SuccessorAgent(env)
    
    cube_model, device, pos_mean, pos_std = load_cube_detector(force_cpu=False)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_rewards = []
    episode_labels = []
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    sr_frozen = False
    frozen_sr_matrix = None
    
    print("Starting training phase...")
    
    for episode in tqdm(range(training_episodes), desc="Training SR"):
        current_task = random.choice(PRIMITIVE_TASKS)
        env.set_task(current_task)
        
        obs, info = env.reset()
        agent.reset()
        
        current_state = agent.get_state_index()
        current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
        
        episode_reward = 0
        
        for step in range(max_steps):
            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            detected_objects = detection_result['detected_objects']
            positions = detection_result['positions']
            
            agent.update_feature_map(detected_objects, positions)
            agent.compose_reward_map(current_task)
            agent.compute_wvf()
            
            obs, env_reward, terminated, truncated, info = env.step(current_action)
            
            task_satisfied = check_task_satisfaction(info, current_task)
            if task_satisfied:
                episode_reward += env_reward
            
            next_state = agent.get_state_index()
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            done = terminated or truncated
            
            if not sr_frozen:
                agent.update_sr(current_state, current_action, next_state, next_action, done)
            
            current_state = next_state
            current_action = next_action
            
            if done:
                break
        
        if episode == sr_freeze_episode - 1:
            print(f"\n⚠️  FREEZING SR MATRIX at episode {episode + 1}")
            frozen_sr_matrix = agent.M.copy()
            sr_frozen = True
            sr_save_path = output_dir / "frozen_sr_matrix.npy"
            np.save(sr_save_path, frozen_sr_matrix)
            print(f"✓ Frozen SR saved to: {sr_save_path}\n")
        
        all_rewards.append(episode_reward)
        episode_labels.append(current_task['name'])
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    if frozen_sr_matrix is not None:
        agent.M = frozen_sr_matrix.copy()
    
    # PRIMITIVE EVALUATION
    print(f"\nStarting PRIMITIVE evaluation phase...")
    primitive_eval_labels = []
    
    for prim_task in PRIMITIVE_TASKS:
        env.set_task(prim_task)
        task_name = prim_task['name']
        print(f"Evaluating primitive task: {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            agent.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
                agent.update_feature_map(detection_result['detected_objects'], detection_result['positions'])
                agent.compose_reward_map(prim_task)
                agent.compute_wvf()
                
                action = agent.sample_action_with_wvf(obs, epsilon=0.0)
                obs, env_reward, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, prim_task):
                    episode_reward += env_reward
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            primitive_eval_labels.append(f"eval_primitive_{task_name}")
    
    # COMPOSITIONAL EVALUATION
    print(f"\nStarting COMPOSITIONAL evaluation phase...")
    comp_eval_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        print(f"Evaluating compositional task: {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            agent.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
                agent.update_feature_map(detection_result['detected_objects'], detection_result['positions'])
                agent.compose_reward_map(comp_task)
                agent.compute_wvf()
                
                action = agent.sample_action_with_wvf(obs, epsilon=0.0)
                obs, env_reward, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward += env_reward
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            comp_eval_labels.append(f"eval_comp_{task_name}")
    
    all_labels = episode_labels + primitive_eval_labels + comp_eval_labels
    
    print(f"\n✓ SR Agent training complete (seed={seed})")
    
    return {
        "algorithm": "SR",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "primitive_eval_episodes": len(primitive_eval_labels),
        "comp_eval_episodes": len(comp_eval_labels),
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
            agent.train_step()
            
            if task_satisfied:
                episode_reward = 1.0
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        episode_labels.append(task['name'])
    
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # PRIMITIVE EVALUATION
    print("\nStarting PRIMITIVE evaluation phase...")
    primitive_eval_labels = []
    
    for prim_task in PRIMITIVE_TASKS:
        env.set_task(prim_task)
        task_name = prim_task['name']
        print(f"Evaluating primitive task: {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(obs, prim_task['features'], epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, prim_task):
                    episode_reward = 1.0
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            primitive_eval_labels.append(f"eval_primitive_{task_name}")
    
    # COMPOSITIONAL EVALUATION
    print("\nStarting COMPOSITIONAL evaluation phase...")
    comp_eval_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        print(f"Evaluating compositional task: {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(obs, comp_task['features'], epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward = 1.0
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            comp_eval_labels.append(f"eval_comp_{task_name}")
    
    all_labels = episode_labels + primitive_eval_labels + comp_eval_labels
    
    print(f"\n✓ DQN training complete (seed={seed})")
    
    return {
        "algorithm": "DQN",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "primitive_eval_episodes": len(primitive_eval_labels),
        "comp_eval_episodes": len(comp_eval_labels),
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
    
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # PRIMITIVE EVALUATION
    print("\nStarting PRIMITIVE evaluation phase...")
    primitive_eval_labels = []
    
    for prim_task in PRIMITIVE_TASKS:
        env.set_task(prim_task)
        task_name = prim_task['name']
        print(f"Evaluating primitive task: {task_name}...")
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            stacked_obs = agent.reset_episode(obs, prim_task['name'])
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(stacked_obs, epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                stacked_obs = agent.step_episode(obs)
                
                if check_task_satisfaction(info, prim_task):
                    episode_reward = 1.0
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            primitive_eval_labels.append(f"eval_primitive_{task_name}")
    
    # COMPOSITIONAL EVALUATION
    print("\nStarting COMPOSITIONAL evaluation phase...")
    comp_eval_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        print(f"Evaluating compositional task: {task_name}...")
        
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
            comp_eval_labels.append(f"eval_comp_{task_name}")
    
    all_labels = episode_labels + primitive_eval_labels + comp_eval_labels
    
    print(f"\n✓ LSTM-DQN training complete (seed={seed})")
    
    return {
        "algorithm": "LSTM",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "primitive_eval_episodes": len(primitive_eval_labels),
        "comp_eval_episodes": len(comp_eval_labels),
    }


# ============================================================================
# UNIFIED WVF AGENT (Option A - Pure Task Conditioning)
# ============================================================================

def train_unified_wvf(seed, training_episodes, eval_episodes_per_task, max_steps, env_size,
                      learning_rate, gamma, epsilon_decay, output_dir):
    """
    Train Unified WVF agent (Option A - Pure Task Conditioning).
    
    Key approach:
    1. Learn Q(s, a, task) for each primitive task
    2. Simple reward: +1 for valid object, -0.1 for wrong, small step penalty
    3. Composition via min(Q_task1, Q_task2) for AND operations
    4. Uses target network for stable evaluation
    """
    
    from env import DiscreteMiniWorldWrapper
    
    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED WVF - OPTION A (Seed={seed})")
    print(f"Pure Task Conditioning (No Goal Conditioning)")
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
        step_penalty=-0.005
    )
    
    all_rewards = []
    episode_labels = []
    
    print("Starting training phase...")
    print("Training on primitive tasks: red, blue, box, sphere\n")
    
    for episode in tqdm(range(training_episodes), desc="Training WVF"):
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
        
        if (episode + 1) % 500 == 0:
            recent_rewards = all_rewards[-500:]
            print(f"  Episode {episode+1}: Recent success rate = {np.mean(recent_rewards):.2%}")
    
    model_path = output_dir / "model.pt"
    agent.save_model(str(model_path))
    
    # PRIMITIVE EVALUATION
    print(f"\nStarting PRIMITIVE evaluation phase...")
    print("Using target network for stable Q-estimates\n")
    
    primitive_eval_labels = []
    
    for prim_task in PRIMITIVE_TASKS:
        env.set_task(prim_task)
        task_name = prim_task['name']
        
        print(f"Evaluating primitive task: {task_name}")
        
        task_successes = 0
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            stacked_obs = agent.reset_episode(obs)
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action_primitive(stacked_obs, task_name, use_target=True)
                
                obs, _, terminated, truncated, info = env.step(action)
                stacked_obs = agent.step_episode(obs)
                
                if check_task_satisfaction(info, prim_task):
                    episode_reward = 1.0
                    task_successes += 1
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            primitive_eval_labels.append(f"eval_primitive_{task_name}")
        
        success_rate = task_successes / eval_episodes_per_task
        print(f"  {task_name}: {success_rate:.1%} success rate ({task_successes}/{eval_episodes_per_task})")
    
    # COMPOSITIONAL EVALUATION
    print(f"\nStarting COMPOSITIONAL evaluation phase...")
    print("Using min(Q_task1, Q_task2) composition with target network\n")
    
    comp_eval_labels = []
    
    for comp_task in COMPOSITIONAL_TASKS:
        env.set_task(comp_task)
        task_name = comp_task['name']
        features = comp_task['features']
        
        print(f"Evaluating {task_name} = min(Q_{features[0]}, Q_{features[1]})")
        
        task_successes = 0
        
        for ep in range(eval_episodes_per_task):
            obs, info = env.reset()
            stacked_obs = agent.reset_episode(obs)
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action_composed(stacked_obs, features, use_target=True)
                
                obs, _, terminated, truncated, info = env.step(action)
                stacked_obs = agent.step_episode(obs)
                
                if check_task_satisfaction(info, comp_task):
                    episode_reward = 1.0
                    task_successes += 1
                    break
                
                if terminated or truncated:
                    break
            
            all_rewards.append(episode_reward)
            comp_eval_labels.append(f"eval_comp_{task_name}")
        
        success_rate = task_successes / eval_episodes_per_task
        print(f"  {task_name}: {success_rate:.1%} success rate ({task_successes}/{eval_episodes_per_task})")
    
    all_labels = episode_labels + primitive_eval_labels + comp_eval_labels
    
    print(f"\n✓ WVF (Option A) training complete (seed={seed})")
    print(f"  Training episodes: {training_episodes}")
    print(f"  Primitive eval episodes: {len(primitive_eval_labels)}")
    print(f"  Compositional eval episodes: {len(comp_eval_labels)}")
    
    return {
        "algorithm": "WVF",
        "seed": seed,
        "all_rewards": np.array(all_rewards),
        "episode_labels": all_labels,
        "training_episodes": training_episodes,
        "primitive_eval_episodes": len(primitive_eval_labels),
        "comp_eval_episodes": len(comp_eval_labels),
    }