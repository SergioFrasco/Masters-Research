"""
Unified DQN Training with Unseen Goal Evaluation

Key modifications:
1. Training ONLY on red/blue objects (4 tasks)
2. Evaluation on UNSEEN green objects using compositional encoding
3. Tests true zero-shot compositional generalization
4. TIMELINE PLOTTING with clear phase demarcation
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm
import json
import gc
import torch

# Import environment and agent
from env import DiscreteMiniWorldWrapper
from agents import UnifiedDQNAgent
from utils import generate_save_path
from utils import plot_experiment_timeline


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# TRAINING TASKS - Only red and blue objects
TRAINING_TASKS = [
    {"name": "red", "features": ["red"], "type": "simple"},
    {"name": "blue", "features": ["blue"], "type": "simple"},
    {"name": "box", "features": ["box"], "type": "simple"},
    {"name": "sphere", "features": ["sphere"], "type": "simple"},
]

# SEEN COMPOSITIONAL (for sanity check)
SEEN_COMPOSITIONAL = [
    {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
    {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
    {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
]

# UNSEEN TASKS - Green objects (never seen during training!)
UNSEEN_SIMPLE_TASKS = [
    {"name": "green", "features": ["green"], "type": "simple_unseen"},
]

UNSEEN_COMPOSITIONAL_TASKS = [
    {"name": "green_box", "features": ["green", "box"], "type": "compositional_unseen"},
    {"name": "green_sphere", "features": ["green", "sphere"], "type": "compositional_unseen"},
]


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies current task requirements."""
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
        elif feature == "green":
            return contacted_object in ["green_box", "green_sphere"]
        elif feature == "box":
            return contacted_object in ["blue_box", "red_box", "green_box"]
        elif feature == "sphere":
            return contacted_object in ["blue_sphere", "red_sphere", "green_sphere"]
    
    # Compositional tasks (2 features)
    elif len(features) == 2:
        feature_set = set(features)
        
        # Define all possible combinations
        mappings = {
            frozenset({"blue", "sphere"}): "blue_sphere",
            frozenset({"red", "sphere"}): "red_sphere",
            frozenset({"green", "sphere"}): "green_sphere",
            frozenset({"blue", "box"}): "blue_box",
            frozenset({"red", "box"}): "red_box",
            frozenset({"green", "box"}): "green_box",
        }
        
        expected_object = mappings.get(frozenset(feature_set))
        return contacted_object == expected_object
    
    return False


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# TRAINING FUNCTION (NO GREEN OBJECTS)
# ============================================================================

def train_unified_dqn(env, episodes=8000, max_steps=200,
                      learning_rate=0.0001, gamma=0.99,
                      epsilon_start=1.0, epsilon_end=0.05,
                      epsilon_decay=0.9995, verbose=True,
                      step_penalty=-0.005, wrong_object_penalty=-0.1):
    """
    Train unified DQN ONLY on red/blue objects.
    Green objects are excluded from training entirely.
    """
    
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFIED DQN (RED/BLUE ONLY)")
    print(f"{'='*60}")
    print(f"  Total episodes: {episodes}")
    print(f"  Training tasks: {[t['name'] for t in TRAINING_TASKS]}")
    print(f"  EXCLUDED: All green objects")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"{'='*60}")
    
    # Create unified agent
    agent = UnifiedDQNAgent(
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=100000,
        batch_size=64,
        target_update_freq=1,
        hidden_size=256,
        use_dueling=True,
        tau=0.005,
        use_double_dqn=True,
        grad_clip=10.0
    )
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_epsilons = []
    episode_tasks = []
    
    task_rewards = defaultdict(list)
    task_lengths = defaultdict(list)
    task_counts = defaultdict(int)
    
    for episode in tqdm(range(episodes), desc="Training (Red/Blue Only)"):
        # Sample only from TRAINING_TASKS (no green!)
        task = np.random.choice(TRAINING_TASKS)
        task_name = task['name']
        task_counts[task_name] += 1
        
        env.set_task(task)
        
        obs, info = env.reset()
        true_reward_total = 0
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(obs, task_name)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            contacted_object = info.get('contacted_object', None)
            task_satisfied = check_task_satisfaction(info, task)
            
            # TRUE REWARD
            true_reward = 1.0 if task_satisfied else 0.0
            
            # SHAPED REWARD
            if task_satisfied:
                shaped_reward = 1.0
            elif contacted_object is not None:
                shaped_reward = wrong_object_penalty
            else:
                shaped_reward = step_penalty
            
            agent.remember(obs, task_name, action, shaped_reward, next_obs, done)
            
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            true_reward_total += true_reward
            obs = next_obs
            
            if done:
                break
        
        episode_epsilons.append(agent.epsilon)
        agent.decay_epsilon()
        
        episode_rewards.append(true_reward_total)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        episode_tasks.append(task_name)
        
        task_rewards[task_name].append(true_reward_total)
        task_lengths[task_name].append(step + 1)
        
        if verbose and (episode + 1) % 500 == 0:
            recent_success = np.mean([r > 0 for r in episode_rewards[-500:]])
            recent_length = np.mean(episode_lengths[-500:])
            
            task_stats = []
            for t in TRAINING_TASKS:
                tname = t['name']
                if tname in task_rewards and len(task_rewards[tname]) > 0:
                    recent_task_rewards = task_rewards[tname][-100:]
                    task_success = np.mean([r > 0 for r in recent_task_rewards]) if recent_task_rewards else 0
                    task_stats.append(f"{tname}={task_success:.1%}")
            
            print(f"  Episode {episode+1}: Success={recent_success:.1%}, "
                  f"Epsilon={agent.epsilon:.3f}")
            print(f"    Per-task: {', '.join(task_stats)}")
            
            clear_gpu_memory()
    
    model_path = generate_save_path("unified_dqn_no_green_model.pt")
    agent.save_model(model_path)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE (RED/BLUE ONLY)")
    print(f"{'='*60}")
    
    final_success = np.mean([r > 0 for r in episode_rewards[-100:]])
    
    # Calculate per-task final success for plotting
    per_task_final_success = {}
    for t in TRAINING_TASKS:
        tname = t['name']
        if tname in task_rewards and len(task_rewards[tname]) > 0:
            recent = task_rewards[tname][-min(100, len(task_rewards[tname])):]
            per_task_final_success[tname] = np.mean([r > 0 for r in recent])
    
    print(f"  Final success rate: {final_success:.1%}")
    print(f"  Model saved: {model_path}")
    print(f"{'='*60}")
    
    return agent, {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_epsilons": episode_epsilons,
        "episode_tasks": episode_tasks,
        "task_rewards": dict(task_rewards),
        "task_lengths": dict(task_lengths),
        "task_counts": dict(task_counts),
        "final_epsilon": agent.epsilon,
        "final_success_rate": final_success,
        "per_task_final_success": per_task_final_success,  # Added for plotting
        "model_path": model_path
    }


# ============================================================================
# EVALUATION FUNCTIONS (WITH EPISODE DATA COLLECTION)
# ============================================================================

def evaluate_agent_on_task(env, agent, task, episodes=100, max_steps=200):
    """
    Evaluate agent on a single task.
    Returns both summary stats AND per-episode rewards for timeline plotting.
    """
    
    task_name = task['name']
    features = task.get('features', [task_name])
    
    env.set_task(task)
    
    successes = []
    lengths = []
    episode_rewards = []  # NEW: Track each episode for timeline plot
    
    for _ in range(episodes):
        obs, info = env.reset()
        
        for step in range(max_steps):
            # Use compositional encoding if multiple features
            if len(features) > 1:
                action = agent.select_action(obs, features, epsilon=0.0)
            else:
                action = agent.select_action(obs, task_name, epsilon=0.0)
            
            obs, _, terminated, truncated, info = env.step(action)
            
            if check_task_satisfaction(info, task):
                successes.append(1)
                lengths.append(step + 1)
                episode_rewards.append(1.0)  # Success
                break
            
            if terminated or truncated:
                successes.append(0)
                lengths.append(step + 1)
                episode_rewards.append(0.0)  # Failure
                break
        else:
            successes.append(0)
            lengths.append(max_steps)
            episode_rewards.append(0.0)  # Timeout
    
    return {
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "features": features,
        "episode_rewards": episode_rewards  # NEW: For timeline plotting
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_with_unseen_goals(
    env_size=10,
    total_episodes=8000,
    eval_episodes=100,
    max_steps=200,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995,
    seed=42
):
    """Run experiment: train on red/blue, evaluate on green (unseen)."""
    
    print("\n" + "="*70)
    print("UNSEEN GOAL GENERALIZATION EXPERIMENT")
    print("="*70)
    print("Training: red/blue objects only")
    print("Evaluation: includes GREEN objects (never seen during training)")
    print("="*70 + "\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    env = DiscreteMiniWorldWrapper(size=env_size, render_mode="rgb_array")
    
    # ===== PHASE 1: TRAINING (NO GREEN) =====
    print("\n" + "="*60)
    print("PHASE 1: TRAINING ON RED/BLUE OBJECTS")
    print("="*60)
    
    agent, history = train_unified_dqn(
        env,
        episodes=total_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        verbose=True
    )
    
    # ===== PHASE 2: EVALUATE ON SEEN TASKS =====
    print("\n" + "="*60)
    print("PHASE 2: EVALUATING ON SEEN TASKS (RED/BLUE)")
    print("="*60)
    
    seen_simple_results = {}
    for task in TRAINING_TASKS:
        print(f"  Evaluating {task['name']}...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        seen_simple_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%}")
    
    seen_comp_results = {}
    for task in SEEN_COMPOSITIONAL:
        print(f"  Evaluating {task['name']}...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        seen_comp_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%}")
    
    # Combine seen results for plotting
    seen_results = {**seen_simple_results, **seen_comp_results}
    
    # ===== PHASE 3: EVALUATE ON UNSEEN TASKS (GREEN) =====
    print("\n" + "="*60)
    print("PHASE 3: ZERO-SHOT EVALUATION ON UNSEEN GREEN OBJECTS")
    print("="*60)
    print("NOTE: Model has NEVER seen green objects during training!")
    print("Testing if learned representations generalize to novel color")
    print("="*60)
    
    unseen_simple_results = {}
    for task in UNSEEN_SIMPLE_TASKS:
        print(f"  Evaluating {task['name']} (UNSEEN)...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        unseen_simple_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%} (UNSEEN COLOR)")
    
    unseen_comp_results = {}
    for task in UNSEEN_COMPOSITIONAL_TASKS:
        print(f"  Evaluating {task['name']} (UNSEEN)...")
        results = evaluate_agent_on_task(env, agent, task, eval_episodes, max_steps)
        unseen_comp_results[task['name']] = results
        print(f"    → {results['success_rate']:.1%} (UNSEEN COMPOSITION)")
    
    # Combine unseen results for plotting
    unseen_results = {**unseen_simple_results, **unseen_comp_results}
    
    # ===== GENERATE TIMELINE PLOT =====
    print("\n" + "="*60)
    print("GENERATING COMPLETE TIMELINE PLOT")
    print("="*60)
    
    timeline_path = generate_save_path("complete_experiment_timeline.png")
    plot_experiment_timeline(
        training_history=history,
        seen_eval_results=seen_results,
        unseen_eval_results=unseen_results,
        save_path=timeline_path,
        eval_episodes_per_task=eval_episodes
    )
    
    # ===== SAVE RESULTS =====
    all_results = {
        "training": {
            "total_episodes": total_episodes,
            "final_success_rate": history["final_success_rate"],
            "training_tasks": [t['name'] for t in TRAINING_TASKS],
            "per_task_final_success": history["per_task_final_success"],
            "model_path": history["model_path"]
        },
        "evaluation_seen_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in seen_simple_results.items()},
        "evaluation_seen_compositional": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"]
        } for t, r in seen_comp_results.items()},
        "evaluation_unseen_simple": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"],
            "note": "ZERO-SHOT: Never saw green during training"
        } for t, r in unseen_simple_results.items()},
        "evaluation_unseen_compositional": {t: {
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"],
            "note": "ZERO-SHOT: Novel green+shape composition"
        } for t, r in unseen_comp_results.items()},
        "summary": {
            "avg_seen_simple": np.mean([r["success_rate"] for r in seen_simple_results.values()]),
            "avg_seen_comp": np.mean([r["success_rate"] for r in seen_comp_results.values()]),
            "avg_unseen_simple": np.mean([r["success_rate"] for r in unseen_simple_results.values()]),
            "avg_unseen_comp": np.mean([r["success_rate"] for r in unseen_comp_results.values()]),
        },
        "plots": {
            "timeline": timeline_path
        }
    }
    
    results_path = generate_save_path("unseen_goal_experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    print(f"Timeline plot saved to: {timeline_path}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE - GENERALIZATION ANALYSIS")
    print("="*70)
    print(f"SEEN (Red/Blue):")
    print(f"  Simple tasks:        {all_results['summary']['avg_seen_simple']:.1%}")
    print(f"  Compositional tasks: {all_results['summary']['avg_seen_comp']:.1%}")
    print(f"\nUNSEEN (Green - Zero-Shot):")
    print(f"  Simple tasks:        {all_results['summary']['avg_unseen_simple']:.1%}")
    print(f"  Compositional tasks: {all_results['summary']['avg_unseen_comp']:.1%}")
    print(f"\nGeneralization Gap:")
    print(f"  Seen→Unseen Simple:  {all_results['summary']['avg_seen_simple'] - all_results['summary']['avg_unseen_simple']:.1%} drop")
    print(f"  Seen→Unseen Comp:    {all_results['summary']['avg_seen_comp'] - all_results['summary']['avg_unseen_comp']:.1%} drop")
    print(f"\nPlots:")
    print(f"  Timeline: {timeline_path}")
    print("="*70)
    
    return all_results, agent


if __name__ == "__main__":
    results, agent = run_experiment_with_unseen_goals(
        env_size=10,
        total_episodes=3000,
        eval_episodes=400,
        max_steps=200,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.9995,
        seed=42
    )