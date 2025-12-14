"""
Experiment Runner - Compare All Baseline Algorithms

Runs 4 baseline algorithms across multiple seeds:
1. DQN (Separate Models)
2. LSTM-DQN (Separate Models with Frame Stacking)
3. Successor Agent (Modified for consistency)
4. World Value Functions (WVF)

Training: 4 simple tasks (red, blue, box, sphere) - 2000 episodes each
Evaluation: 4 compositional tasks - 500 episodes each

Generates:
- Individual algorithm plots (saved to their respective paths)
- Comparison plots (rewards, success rates, bar charts)
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import gc
import torch
from collections import defaultdict
import time

from env import DiscreteMiniWorldWrapper
from agents import DQNAgent3D, LSTMDQNAgent3D, SuccessorAgent, WorldValueFunctionAgent
from utils import generate_save_path


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

SIMPLE_TASKS = [
    {"name": "red", "features": ["red"], "type": "simple"},
    {"name": "blue", "features": ["blue"], "type": "simple"},
    {"name": "box", "features": ["box"], "type": "simple"},
    {"name": "sphere", "features": ["sphere"], "type": "simple"},
]

COMPOSITIONAL_TASKS = [
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies task requirements."""
    contacted_object = info.get('contacted_object', None)
    
    if contacted_object is None:
        return False
    
    features = task["features"]
    
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


def clear_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# ALGORITHM 1: DQN (Separate Models)
# ============================================================================

def train_dqn_single_task(env, task, episodes=2000, max_steps=200):
    """Train DQN on a single task."""
    from dqn_baseline import train_single_task_dqn
    
    agent, history = train_single_task_dqn(
        env, task,
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        verbose=False,
        step_penalty=-0.005,
        wrong_object_penalty=-0.1
    )
    
    return agent, history


def evaluate_dqn_simple(env, agent, task, episodes=100, max_steps=200):
    """Evaluate DQN on simple task."""
    env.set_task(task)
    
    # Move agent to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        agent.q_network = agent.q_network.to(device)
        agent.target_network = agent.target_network.to(device)
        agent.device = device
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    # Move back to CPU
    agent.q_network = agent.q_network.cpu()
    agent.target_network = agent.target_network.cpu()
    agent.device = torch.device("cpu")
    clear_memory()
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "episode_rewards": episode_rewards
    }


def evaluate_dqn_compositional(env, trained_agents, task, episodes=100, max_steps=200):
    """Evaluate DQN on compositional task using color model."""
    features = task['features']
    color_feature = [f for f in features if f in ['red', 'blue']][0]
    agent = trained_agents[color_feature]
    
    env.set_task(task)
    
    # Move to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        agent.q_network = agent.q_network.to(device)
        agent.target_network = agent.target_network.to(device)
        agent.device = device
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    # Move back to CPU
    agent.q_network = agent.q_network.cpu()
    agent.target_network = agent.target_network.cpu()
    agent.device = torch.device("cpu")
    clear_memory()
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "episode_rewards": episode_rewards,
        "model_used": color_feature
    }


def run_dqn_experiment(env, episodes_per_task=2000, eval_episodes=100, max_steps=200, seed=42):
    """Run full DQN experiment."""
    print("\n" + "="*70)
    print("RUNNING: DQN (SEPARATE MODELS)")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Training
    trained_agents = {}
    all_training_rewards = []
    all_training_successes = []
    all_training_lengths = []
    
    for task in SIMPLE_TASKS:
        clear_memory()
        agent, history = train_dqn_single_task(env, task, episodes_per_task, max_steps)
        trained_agents[task['name']] = agent
        
        # Convert rewards to success rate (1 if reward > 0)
        successes = [1 if r > 0 else 0 for r in history['episode_rewards']]
        all_training_rewards.extend(history['episode_rewards'])
        all_training_successes.extend(successes)
        all_training_lengths.extend(history['episode_lengths'])
    
    # Evaluation on simple tasks
    all_eval_simple_rewards = []
    all_eval_simple_successes = []
    
    for task in SIMPLE_TASKS:
        agent = trained_agents[task['name']]
        results = evaluate_dqn_simple(env, agent, task, eval_episodes, max_steps)
        all_eval_simple_rewards.extend(results['episode_rewards'])
        all_eval_simple_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    # Evaluation on compositional tasks
    all_eval_comp_rewards = []
    all_eval_comp_successes = []
    
    for task in COMPOSITIONAL_TASKS:
        results = evaluate_dqn_compositional(env, trained_agents, task, eval_episodes, max_steps)
        all_eval_comp_rewards.extend(results['episode_rewards'])
        all_eval_comp_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    # Combine all rewards
    all_rewards = all_training_rewards + all_eval_simple_rewards + all_eval_comp_rewards
    all_successes = all_training_successes + all_eval_simple_successes + all_eval_comp_successes
    
    results = {
        "algorithm": "DQN",
        "all_rewards": all_rewards,
        "all_successes": all_successes,
        "training_rewards": all_training_rewards,
        "training_successes": all_training_successes,
        "eval_simple_rewards": all_eval_simple_rewards,
        "eval_simple_successes": all_eval_simple_successes,
        "eval_comp_rewards": all_eval_comp_rewards,
        "eval_comp_successes": all_eval_comp_successes,
        "task_boundaries": {
            "simple_tasks_end": len(all_training_rewards),
            "eval_simple_end": len(all_training_rewards) + len(all_eval_simple_rewards),
            "eval_comp_end": len(all_rewards)
        }
    }
    
    clear_memory()
    return results


# ============================================================================
# ALGORITHM 2: LSTM-DQN
# ============================================================================

def train_lstm_dqn_single_task(env, task, episodes=2000, max_steps=200):
    """Train LSTM-DQN on a single task."""
    from dqn_lstm_baseline import train_single_task_lstm_dqn
    
    agent, history = train_single_task_lstm_dqn(
        env, task,
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        verbose=False,
        step_penalty=-0.005,
        wrong_object_penalty=-0.1
    )
    
    return agent, history


def evaluate_lstm_dqn_simple(env, agent, task, episodes=100, max_steps=200):
    """Evaluate LSTM-DQN on simple task."""
    env.set_task(task)
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "episode_rewards": episode_rewards
    }


def evaluate_lstm_dqn_compositional(env, trained_agents, task, episodes=100, max_steps=200):
    """Evaluate LSTM-DQN on compositional task."""
    features = task['features']
    color_feature = [f for f in features if f in ['red', 'blue']][0]
    agent = trained_agents[color_feature]
    
    env.set_task(task)
    
    successes = []
    lengths = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        stacked_obs = agent.reset_episode(obs)
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(stacked_obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = agent.step_episode(obs)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "episode_rewards": episode_rewards,
        "model_used": color_feature
    }


def run_lstm_dqn_experiment(env, episodes_per_task=2000, eval_episodes=100, max_steps=200, seed=42):
    """Run full LSTM-DQN experiment."""
    print("\n" + "="*70)
    print("RUNNING: LSTM-DQN (FRAME STACKING)")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    trained_agents = {}
    all_training_rewards = []
    all_training_successes = []
    
    for task in SIMPLE_TASKS:
        clear_memory()
        agent, history = train_lstm_dqn_single_task(env, task, episodes_per_task, max_steps)
        trained_agents[task['name']] = agent
        
        successes = [1 if r > 0 else 0 for r in history['episode_rewards']]
        all_training_rewards.extend(history['episode_rewards'])
        all_training_successes.extend(successes)
    
    # Evaluation
    all_eval_simple_rewards = []
    all_eval_simple_successes = []
    
    for task in SIMPLE_TASKS:
        agent = trained_agents[task['name']]
        results = evaluate_lstm_dqn_simple(env, agent, task, eval_episodes, max_steps)
        all_eval_simple_rewards.extend(results['episode_rewards'])
        all_eval_simple_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    all_eval_comp_rewards = []
    all_eval_comp_successes = []
    
    for task in COMPOSITIONAL_TASKS:
        results = evaluate_lstm_dqn_compositional(env, trained_agents, task, eval_episodes, max_steps)
        all_eval_comp_rewards.extend(results['episode_rewards'])
        all_eval_comp_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    all_rewards = all_training_rewards + all_eval_simple_rewards + all_eval_comp_rewards
    all_successes = all_training_successes + all_eval_simple_successes + all_eval_comp_successes
    
    results = {
        "algorithm": "LSTM-DQN",
        "all_rewards": all_rewards,
        "all_successes": all_successes,
        "training_rewards": all_training_rewards,
        "training_successes": all_training_successes,
        "eval_simple_rewards": all_eval_simple_rewards,
        "eval_simple_successes": all_eval_simple_successes,
        "eval_comp_rewards": all_eval_comp_rewards,
        "eval_comp_successes": all_eval_comp_successes,
        "task_boundaries": {
            "simple_tasks_end": len(all_training_rewards),
            "eval_simple_end": len(all_training_rewards) + len(all_eval_simple_rewards),
            "eval_comp_end": len(all_rewards)
        }
    }
    
    clear_memory()
    return results


# ============================================================================
# ALGORITHM 3: SUCCESSOR AGENT (Modified)
# ============================================================================

def train_successor_single_task(env, agent, task, episodes=2000, max_steps=200):
    """Train Successor Agent on a single task."""
    env.set_task(task)
    
    episode_rewards = []
    episode_successes = []
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.999
    
    for episode in tqdm(range(episodes), desc=f"Training Successor '{task['name']}'", leave=False):
        obs, info = env.reset()
        agent.reset()
        ep_reward = 0
        
        current_state = agent.get_state_index()
        
        for step in range(max_steps):
            action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            next_obs, _, terminated, truncated, info = env.step(action)
            
            next_state = agent.get_state_index()
            done = terminated or truncated
            
            # Update SR
            agent.update_sr(current_state, action, next_state, action, done)
            
            # Check task satisfaction
            if check_task_satisfaction(info, task):
                ep_reward = 1
                break
            
            current_state = next_state
            obs = next_obs
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)
        episode_successes.append(1 if ep_reward > 0 else 0)
    
    return {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes
    }


def evaluate_successor_simple(env, agent, task, episodes=100, max_steps=200):
    """Evaluate Successor Agent on simple task."""
    env.set_task(task)
    
    successes = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        agent.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.sample_action_with_wvf(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                break
        else:
            successes.append(0)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "episode_rewards": episode_rewards
    }


def evaluate_successor_compositional(env, agent, task, episodes=100, max_steps=200):
    """Evaluate Successor Agent on compositional task."""
    env.set_task(task)
    
    successes = []
    episode_rewards = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        agent.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.sample_action_with_wvf(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                ep_reward = 1
                break
            
            if terminated or truncated:
                successes.append(0)
                break
        else:
            successes.append(0)
        
        episode_rewards.append(ep_reward)
    
    return {
        "success_rate": np.mean(successes),
        "episode_rewards": episode_rewards
    }


def run_successor_experiment(env, episodes_per_task=2000, eval_episodes=100, max_steps=200, seed=42):
    """Run full Successor Agent experiment."""
    print("\n" + "="*70)
    print("RUNNING: SUCCESSOR AGENT")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    agent = SuccessorAgent(env)
    
    all_training_rewards = []
    all_training_successes = []
    
    # Train on each simple task
    for task in SIMPLE_TASKS:
        history = train_successor_single_task(env, agent, task, episodes_per_task, max_steps)
        all_training_rewards.extend(history['episode_rewards'])
        all_training_successes.extend(history['episode_successes'])
    
    # Evaluation on simple tasks
    all_eval_simple_rewards = []
    all_eval_simple_successes = []
    
    for task in SIMPLE_TASKS:
        results = evaluate_successor_simple(env, agent, task, eval_episodes, max_steps)
        all_eval_simple_rewards.extend(results['episode_rewards'])
        all_eval_simple_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    # Evaluation on compositional tasks
    all_eval_comp_rewards = []
    all_eval_comp_successes = []
    
    for task in COMPOSITIONAL_TASKS:
        results = evaluate_successor_compositional(env, agent, task, eval_episodes, max_steps)
        all_eval_comp_rewards.extend(results['episode_rewards'])
        all_eval_comp_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    all_rewards = all_training_rewards + all_eval_simple_rewards + all_eval_comp_rewards
    all_successes = all_training_successes + all_eval_simple_successes + all_eval_comp_successes
    
    results = {
        "algorithm": "Successor",
        "all_rewards": all_rewards,
        "all_successes": all_successes,
        "training_rewards": all_training_rewards,
        "training_successes": all_training_successes,
        "eval_simple_rewards": all_eval_simple_rewards,
        "eval_simple_successes": all_eval_simple_successes,
        "eval_comp_rewards": all_eval_comp_rewards,
        "eval_comp_successes": all_eval_comp_successes,
        "task_boundaries": {
            "simple_tasks_end": len(all_training_rewards),
            "eval_simple_end": len(all_training_rewards) + len(all_eval_simple_rewards),
            "eval_comp_end": len(all_rewards)
        }
    }
    
    clear_memory()
    return results


# ============================================================================
# ALGORITHM 4: WORLD VALUE FUNCTIONS (WVF)
# ============================================================================

def train_wvf_single_primitive(env, agent, primitive, episodes=2000, max_steps=200):
    """Train WVF on a single primitive task."""
    from wvf_baseline2 import train_primitive_wvf, PRIMITIVE_TASKS
    
    task = PRIMITIVE_TASKS[primitive]
    history = train_primitive_wvf(
        env, agent, primitive,
        episodes=episodes,
        max_steps=max_steps,
        train_every=4,
        verbose=False
    )
    
    return history


def evaluate_wvf_simple(env, agent, primitive, episodes=100, max_steps=200):
    """Evaluate WVF on simple/primitive task."""
    from wvf_baseline2 import evaluate_primitive_wvf, PRIMITIVE_TASKS
    
    task = PRIMITIVE_TASKS[primitive]
    results = evaluate_primitive_wvf(env, agent, primitive, episodes, max_steps)
    
    return results


def evaluate_wvf_compositional(env, agent, task_name, episodes=100, max_steps=200):
    """Evaluate WVF on compositional task."""
    from wvf_baseline2 import evaluate_compositional_wvf, COMPOSITIONAL_TASKS
    
    task = COMPOSITIONAL_TASKS[task_name]
    results = evaluate_compositional_wvf(env, agent, task_name, episodes, max_steps)
    
    return results


def run_wvf_experiment(env, episodes_per_task=2000, eval_episodes=100, max_steps=200, seed=42):
    """Run full WVF experiment."""
    print("\n" + "="*70)
    print("RUNNING: WORLD VALUE FUNCTIONS (WVF)")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    agent = WorldValueFunctionAgent(
        env,
        k_frames=4,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        memory_size=2000,
        batch_size=16,
        seq_len=4,
        hidden_size=128,
        lstm_size=64,
        tau=0.005,
        grad_clip=10.0,
        r_min=-10.0,
        r_correct=1.0,
        r_wrong=-1.0,
        step_penalty=-0.01
    )
    
    all_training_rewards = []
    all_training_successes = []
    
    # Train on primitives
    primitives = ['red', 'blue', 'box', 'sphere']
    for primitive in primitives:
        clear_memory()
        history = train_wvf_single_primitive(env, agent, primitive, episodes_per_task, max_steps)
        all_training_rewards.extend(history['episode_rewards'])
        all_training_successes.extend([1 if r > 0 else 0 for r in history['episode_rewards']])
    
    # Evaluation on primitives
    all_eval_simple_rewards = []
    all_eval_simple_successes = []
    
    for primitive in primitives:
        results = evaluate_wvf_simple(env, agent, primitive, eval_episodes, max_steps)
        all_eval_simple_rewards.extend(results['episode_rewards'])
        all_eval_simple_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    # Evaluation on compositional
    all_eval_comp_rewards = []
    all_eval_comp_successes = []
    
    comp_task_names = ['red_box', 'red_sphere', 'blue_box', 'blue_sphere']
    for task_name in comp_task_names:
        results = evaluate_wvf_compositional(env, agent, task_name, eval_episodes, max_steps)
        all_eval_comp_rewards.extend(results['episode_rewards'])
        all_eval_comp_successes.extend([1 if r > 0 else 0 for r in results['episode_rewards']])
    
    all_rewards = all_training_rewards + all_eval_simple_rewards + all_eval_comp_rewards
    all_successes = all_training_successes + all_eval_simple_successes + all_eval_comp_successes
    
    results = {
        "algorithm": "WVF",
        "all_rewards": all_rewards,
        "all_successes": all_successes,
        "training_rewards": all_training_rewards,
        "training_successes": all_training_successes,
        "eval_simple_rewards": all_eval_simple_rewards,
        "eval_simple_successes": all_eval_simple_successes,
        "eval_comp_rewards": all_eval_comp_rewards,
        "eval_comp_successes": all_eval_comp_successes,
        "task_boundaries": {
            "simple_tasks_end": len(all_training_rewards),
            "eval_simple_end": len(all_training_rewards) + len(all_eval_simple_rewards),
            "eval_comp_end": len(all_rewards)
        }
    }
    
    clear_memory()
    return results


# ============================================================================
# COMPARISON PLOTTING
# ============================================================================

def plot_comparison_rewards(all_results, save_path, window=100):
    """
    Plot rewards across episodes for all algorithms with task boundaries.
    
    Shows:
    - Training on 4 simple tasks (vertical lines between tasks)
    - Evaluation on simple tasks (vertical line before)
    - Evaluation on compositional tasks (vertical line before)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    colors = {
        'DQN': '#1f77b4',
        'LSTM-DQN': '#ff7f0e',
        'Successor': '#2ca02c',
        'WVF': '#d62728'
    }
    
    # Get boundaries from first result to calculate episodes_per_task dynamically
    first_result = list(all_results.values())[0][0]
    boundaries = first_result['task_boundaries']
    
    # Calculate episodes per task from the actual data (4 simple tasks during training)
    episodes_per_task = boundaries['simple_tasks_end'] // 4
    
    # Plot 1: Raw rewards with smoothing
    for algo_name, results_list in all_results.items():
        all_algo_rewards = []
        
        for results in results_list:
            all_algo_rewards.append(results['all_rewards'])
        
        # Average across seeds
        mean_rewards = np.mean(all_algo_rewards, axis=0)
        std_rewards = np.std(all_algo_rewards, axis=0)
        
        # Smooth
        if len(mean_rewards) >= window:
            smoothed = pd.Series(mean_rewards).rolling(window, min_periods=1).mean()
            std_smoothed = pd.Series(std_rewards).rolling(window, min_periods=1).mean()
        else:
            smoothed = pd.Series(mean_rewards)
            std_smoothed = pd.Series(std_rewards)
        
        x = range(len(smoothed))
        ax1.plot(x, smoothed, label=algo_name, color=colors[algo_name], linewidth=2.5, alpha=0.9)
        ax1.fill_between(x, smoothed - std_smoothed, smoothed + std_smoothed, 
                         color=colors[algo_name], alpha=0.2)
    
    # Add vertical lines for task boundaries during training (4 tasks)
    for i in range(1, 4):
        boundary = i * episodes_per_task
        ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.text(boundary, ax1.get_ylim()[1], f'Task {i+1}', 
                rotation=90, va='top', ha='right', fontsize=9, color='gray')
    
    # Evaluation boundary (training -> eval simple)
    eval_start = boundaries['simple_tasks_end']
    ax1.axvline(x=eval_start, color='red', linestyle='-', alpha=0.7, linewidth=2.5)
    ax1.text(eval_start, ax1.get_ylim()[1], 'EVAL SIMPLE', 
            rotation=90, va='top', ha='right', fontsize=10, fontweight='bold', color='red')
    
    # Compositional evaluation boundary
    comp_start = boundaries['eval_simple_end']
    ax1.axvline(x=comp_start, color='purple', linestyle='-', alpha=0.7, linewidth=2.5)
    ax1.text(comp_start, ax1.get_ylim()[1], 'EVAL COMP', 
            rotation=90, va='top', ha='right', fontsize=10, fontweight='bold', color='purple')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'Rewards Across Episodes (Smoothed, window={window})', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    
    # Plot 2: Success rates
    for algo_name, results_list in all_results.items():
        all_algo_successes = []
        
        for results in results_list:
            all_algo_successes.append(results['all_successes'])
        
        mean_successes = np.mean(all_algo_successes, axis=0)
        std_successes = np.std(all_algo_successes, axis=0)
        
        if len(mean_successes) >= window:
            smoothed = pd.Series(mean_successes).rolling(window, min_periods=1).mean()
            std_smoothed = pd.Series(std_successes).rolling(window, min_periods=1).mean()
        else:
            smoothed = pd.Series(mean_successes)
            std_smoothed = pd.Series(std_successes)
        
        x = range(len(smoothed))
        ax2.plot(x, smoothed, label=algo_name, color=colors[algo_name], linewidth=2.5, alpha=0.9)
        ax2.fill_between(x, smoothed - std_smoothed, smoothed + std_smoothed,
                         color=colors[algo_name], alpha=0.2)
    
    # Add same boundaries
    for i in range(1, 4):
        boundary = i * episodes_per_task
        ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax2.axvline(x=eval_start, color='red', linestyle='-', alpha=0.7, linewidth=2.5)
    ax2.axvline(x=comp_start, color='purple', linestyle='-', alpha=0.7, linewidth=2.5)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title(f'Success Rate Across Episodes (Smoothed, window={window})', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison rewards plot saved to: {save_path}")


def plot_comparison_bars(all_results, save_path):
    """
    Plot bar charts comparing performance on simple vs compositional tasks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algorithms = list(all_results.keys())
    colors = {
        'DQN': '#1f77b4',
        'LSTM-DQN': '#ff7f0e',
        'Successor': '#2ca02c',
        'WVF': '#d62728'
    }
    
    # Compute averages
    simple_means = []
    simple_stds = []
    comp_means = []
    comp_stds = []
    
    for algo_name in algorithms:
        results_list = all_results[algo_name]
        
        simple_rates = []
        comp_rates = []
        
        for results in results_list:
            simple_rate = np.mean(results['eval_simple_successes'])
            comp_rate = np.mean(results['eval_comp_successes'])
            simple_rates.append(simple_rate)
            comp_rates.append(comp_rate)
        
        simple_means.append(np.mean(simple_rates))
        simple_stds.append(np.std(simple_rates))
        comp_means.append(np.mean(comp_rates))
        comp_stds.append(np.std(comp_rates))
    
    # Plot 1: Simple tasks
    x = np.arange(len(algorithms))
    bars1 = ax1.bar(x, simple_means, yerr=simple_stds, 
                    color=[colors[a] for a in algorithms],
                    edgecolor='black', linewidth=1.5, capsize=5)
    
    for i, (bar, mean, std) in enumerate(zip(bars1, simple_means, simple_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Simple Tasks Evaluation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.set_ylim([0, 1.15])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Compositional tasks
    bars2 = ax2.bar(x, comp_means, yerr=comp_stds,
                    color=[colors[a] for a in algorithms],
                    edgecolor='black', linewidth=1.5, capsize=5)
    
    for i, (bar, mean, std) in enumerate(zip(bars2, comp_means, comp_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title('Compositional Tasks Evaluation (Zero-Shot)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=11)
    ax2.set_ylim([0, 1.15])
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random baseline (1/4)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison bars plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_all_experiments(num_seeds=3, episodes_per_task=2000, eval_episodes=500, max_steps=200):
    """
    Run all baseline experiments across multiple seeds.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("="*70)
    print(f"Number of seeds: {num_seeds}")
    print(f"Episodes per simple task: {episodes_per_task}")
    print(f"Evaluation episodes per task: {eval_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print("="*70)
    
    # Store results across seeds
    all_results = defaultdict(list)
    
    for seed in range(num_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{num_seeds}")
        print(f"{'='*70}")
        
        # Create fresh environment for each seed
        env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")
        
        # Run DQN
        start_time = time.time()
        dqn_results = run_dqn_experiment(env, episodes_per_task, eval_episodes, max_steps, seed)
        all_results['DQN'].append(dqn_results)
        print(f"DQN completed in {time.time() - start_time:.1f}s")
        clear_memory()
        
        # Run LSTM-DQN
        start_time = time.time()
        lstm_results = run_lstm_dqn_experiment(env, episodes_per_task, eval_episodes, max_steps, seed)
        all_results['LSTM-DQN'].append(lstm_results)
        print(f"LSTM-DQN completed in {time.time() - start_time:.1f}s")
        clear_memory()
        
        # Run Successor
        start_time = time.time()
        successor_results = run_successor_experiment(env, episodes_per_task, eval_episodes, max_steps, seed)
        all_results['Successor'].append(successor_results)
        print(f"Successor completed in {time.time() - start_time:.1f}s")
        clear_memory()
        
        # Run WVF
        start_time = time.time()
        wvf_results = run_wvf_experiment(env, episodes_per_task, eval_episodes, max_steps, seed)
        all_results['WVF'].append(wvf_results)
        print(f"WVF completed in {time.time() - start_time:.1f}s")
        clear_memory()
        
        # Clean up environment
        env.close()
        del env
        clear_memory()
    
    # Generate comparison plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plot_comparison_rewards(all_results, generate_save_path("comparison_rewards.png"), window=100)
    plot_comparison_bars(all_results, generate_save_path("comparison_bars.png"))
    
    # Save results to JSON
    results_summary = {}
    for algo_name, results_list in all_results.items():
        simple_rates = [np.mean(r['eval_simple_successes']) for r in results_list]
        comp_rates = [np.mean(r['eval_comp_successes']) for r in results_list]
        
        results_summary[algo_name] = {
            "simple_mean": float(np.mean(simple_rates)),
            "simple_std": float(np.std(simple_rates)),
            "comp_mean": float(np.mean(comp_rates)),
            "comp_std": float(np.std(comp_rates)),
            "generalization_gap": float(np.mean(simple_rates) - np.mean(comp_rates))
        }
    
    summary_path = generate_save_path("comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for algo_name, summary in results_summary.items():
        print(f"\n{algo_name}:")
        print(f"  Simple Tasks:        {summary['simple_mean']:.1%} ± {summary['simple_std']:.1%}")
        print(f"  Compositional Tasks: {summary['comp_mean']:.1%} ± {summary['comp_std']:.1%}")
        print(f"  Generalization Gap:  {summary['generalization_gap']:.1%}")
    print("="*70)
    
    return all_results, results_summary


if __name__ == "__main__":
    all_results, summary = run_all_experiments(
        num_seeds=3,
        episodes_per_task=2500,
        eval_episodes=500,
        max_steps=200
    )