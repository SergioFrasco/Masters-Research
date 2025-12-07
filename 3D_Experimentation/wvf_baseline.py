"""
Training script for Compositional WVF Agent v3 (with Allocentric Memory).

This version:
- Uses allocentric maps as network input (memory of seen objects)
- Supports curriculum learning (simple tasks first)
- Selective training for speed
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import miniworld
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from agents import CompositionalWVFAgent

# You'll need to import these from your codebase:
# from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
# from train_vision import CubeDetector


# ==================== Task Scheduling ====================

def create_curriculum_schedule(total_episodes, simple_ratio=0.6):
    """
    Curriculum: Train simple tasks FIRST, then compositional.
    """
    simple_tasks = [
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"},
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
    ]
    
    compositional_tasks = [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]
    
    simple_episodes = int(total_episodes * simple_ratio)
    compositional_episodes = total_episodes - simple_episodes
    
    simple_eps_per_task = simple_episodes // len(simple_tasks)
    comp_eps_per_task = compositional_episodes // len(compositional_tasks)
    
    schedule = []
    
    # Simple tasks first
    for task in simple_tasks:
        t = task.copy()
        t["duration"] = simple_eps_per_task
        schedule.append(t)
    
    # Then compositional
    for task in compositional_tasks:
        t = task.copy()
        t["duration"] = comp_eps_per_task
        schedule.append(t)
    
    return schedule


def create_interleaved_schedule(total_episodes):
    """
    Interleaved: Alternate simple and compositional (original approach).
    """
    simple_tasks = [
        {"name": "blue", "features": ["blue"], "type": "simple"},
        {"name": "red", "features": ["red"], "type": "simple"},
        {"name": "box", "features": ["box"], "type": "simple"},
        {"name": "sphere", "features": ["sphere"], "type": "simple"},
    ]
    
    compositional_tasks = [
        {"name": "blue_sphere", "features": ["blue", "sphere"], "type": "compositional"},
        {"name": "red_sphere", "features": ["red", "sphere"], "type": "compositional"},
        {"name": "blue_box", "features": ["blue", "box"], "type": "compositional"},
        {"name": "red_box", "features": ["red", "box"], "type": "compositional"},
    ]
    
    interleaved = []
    for i in range(4):
        interleaved.append(simple_tasks[i])
        interleaved.append(compositional_tasks[i])
    
    eps_per_task = total_episodes // len(interleaved)
    for task in interleaved:
        task["duration"] = eps_per_task
    
    return interleaved


def check_task_satisfaction(info, task):
    """Check if contacted object satisfies task."""
    contacted = info.get('contacted_object', None)
    if contacted is None:
        return False
    
    features = task["features"]
    
    if len(features) == 1:
        f = features[0]
        if f == "blue":
            return contacted in ["blue_box", "blue_sphere"]
        elif f == "red":
            return contacted in ["red_box", "red_sphere"]
        elif f == "box":
            return contacted in ["blue_box", "red_box"]
        elif f == "sphere":
            return contacted in ["blue_sphere", "red_sphere"]
    
    elif len(features) == 2:
        if set(features) == {"blue", "sphere"}:
            return contacted == "blue_sphere"
        elif set(features) == {"red", "sphere"}:
            return contacted == "red_sphere"
        elif set(features) == {"blue", "box"}:
            return contacted == "blue_box"
        elif set(features) == {"red", "box"}:
            return contacted == "red_box"
    
    return False


# ==================== Vision Model ====================

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load pretrained vision model."""
    # Import here to avoid circular imports
    from train_vision import CubeDetector
    
    device = torch.device('cpu') if force_cpu else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = CubeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
        model.load_state_dict(checkpoint)
        pos_mean, pos_std = 0.0, 1.0
    
    model.eval()
    print(f"✓ Vision model loaded on {device}")
    return model, device, pos_mean, pos_std


def detect_objects(model, obs, device, transform, pos_mean=0.0, pos_std=1.0):
    """Run object detection on observation."""
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
    if isinstance(img, np.ndarray):
        if img.shape[0] in [3, 4]:
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
        preds = (probs > 0.5).cpu().numpy()[0]
        reg_vals = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        labels = ["red_box", "blue_box", "red_sphere", "blue_sphere"]
        detected = [labels[i] for i in range(4) if preds[i]]
    
    positions = {
        'red_box': (reg_vals[0], reg_vals[1]) if preds[0] else None,
        'blue_box': (reg_vals[2], reg_vals[3]) if preds[1] else None,
        'red_sphere': (reg_vals[4], reg_vals[5]) if preds[2] else None,
        'blue_sphere': (reg_vals[6], reg_vals[7]) if preds[3] else None,
    }
    
    probabilities = {labels[i]: float(probs[0, i]) for i in range(4)}
    
    return {
        "detected_objects": detected,
        "predictions": preds,
        "probabilities": probabilities,
        "positions": positions,
    }


# ==================== Main Training ====================

def train_v3_agent(
    env,
    max_episodes=2000,
    max_steps=200,
    schedule_type='curriculum',
    simple_ratio=0.6,
    lr=0.0005,
    gamma=0.99,
    hidden_dim=128,
    selective_training=True,
    save_dir='results_v3'
):
    """
    Train CompositionalWVFAgentV3.
    
    Args:
        env: MiniWorld environment
        max_episodes: Total training episodes
        max_steps: Max steps per episode
        schedule_type: 'curriculum' or 'interleaved'
        simple_ratio: Ratio of simple task episodes (for curriculum)
        lr: Learning rate
        gamma: Discount factor
        hidden_dim: Hidden layer size
        selective_training: Only train relevant networks
        save_dir: Directory for saving results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agent
    agent = CompositionalWVFAgent(
        env=env,
        lr=lr,
        gamma=gamma,
        device=device,
        grid_size=env.size,
        selective_training=selective_training,
        hidden_dim=hidden_dim
    )
    
    print(f"\n✓ Agent created with state_dim={agent.state_dim}")
    print(f"  Networks: {list(agent.wvf_models.keys())}")
    print(f"  Selective training: {selective_training}")
    
    # Load vision model
    vision_model, vision_device, pos_mean, pos_std = load_cube_detector()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create task schedule
    if schedule_type == 'curriculum':
        tasks = create_curriculum_schedule(max_episodes, simple_ratio)
        print(f"\n✓ Curriculum schedule: {int(simple_ratio*100)}% simple first")
    else:
        tasks = create_interleaved_schedule(max_episodes)
        print(f"\n✓ Interleaved schedule")
    
    # Print schedule
    print("\nTask schedule:")
    cumulative = 0
    for i, task in enumerate(tasks):
        dur = task['duration']
        print(f"  {i}: {task['name']:15} eps {cumulative:4} - {cumulative + dur:4}")
        cumulative += dur
    
    # Tracking
    episode_rewards = []
    episode_task_rewards = []
    episode_lengths = []
    feature_losses = {f: [] for f in agent.feature_names}
    task_log = []
    
    # Exploration
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    # Task tracking
    current_task_idx = 0
    eps_in_task = 0
    current_task = tasks[0]
    
    obs, info = env.reset()
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    for episode in tqdm(range(max_episodes), desc="Training"):
        # Check task switch
        if eps_in_task >= current_task['duration'] and current_task_idx < len(tasks) - 1:
            current_task_idx += 1
            current_task = tasks[current_task_idx]
            eps_in_task = 0
            print(f"\n>>> Task: {current_task['name']} (ep {episode})")
        
        # Set task
        agent.set_current_task(current_task)
        agent.reset()
        
        # Initial detection
        detection = detect_objects(vision_model, obs, vision_device, transform, pos_mean, pos_std)
        agent.update_from_detection(detection)
        
        current_state = agent.get_all_state_features()
        
        ep_reward = 0
        ep_task_reward = 0
        ep_losses = {f: [] for f in agent.feature_names}
        
        for step in range(max_steps):
            # Action
            action = agent.sample_action_with_wvf(obs, current_task, epsilon=epsilon)
            
            # Step
            obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Rewards
            feature_rewards = agent.compute_feature_rewards(info)
            if check_task_satisfaction(info, current_task):
                ep_task_reward += env_reward
            ep_reward += env_reward
            
            # Update detection
            detection = detect_objects(vision_model, obs, vision_device, transform, pos_mean, pos_std)
            agent.update_from_detection(detection)
            
            # Next state
            next_state = agent.get_all_state_features()
            
            # Experience
            experience = [current_state, action, next_state, feature_rewards, done]
            
            # Train
            losses = agent.update_all_features(experience)
            for f, loss in losses.items():
                if loss > 0:
                    ep_losses[f].append(loss)
            
            current_state = next_state
            
            if done:
                break
        
        # Episode done
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)
        episode_task_rewards.append(ep_task_reward)
        episode_lengths.append(step + 1)
        task_log.append((episode, ep_task_reward, ep_reward, current_task['name']))
        eps_in_task += 1
        
        for f in agent.feature_names:
            feature_losses[f].append(np.mean(ep_losses[f]) if ep_losses[f] else 0.0)
        
        # Visualize periodically
        if episode % 500 == 0 or episode == max_episodes - 1:
            save_visualizations(agent, current_task, feature_losses, episode, save_dir)
        
        obs, info = env.reset()
    
    print("\n✓ Training complete!")
    
    # Final plots
    plot_results(episode_rewards, episode_task_rewards, episode_lengths, 
                 feature_losses, tasks, save_dir)
    
    # Save models
    for feature in agent.feature_names:
        path = os.path.join(save_dir, f"wvf_{feature}.pth")
        torch.save({
            'model_state_dict': agent.wvf_models[feature].state_dict(),
            'optimizer_state_dict': agent.optimizers[feature].state_dict(),
        }, path)
        print(f"  ✓ Saved {feature}: {path}")
    
    return {
        'rewards': episode_rewards,
        'task_rewards': episode_task_rewards,
        'lengths': episode_lengths,
        'feature_losses': feature_losses,
        'task_log': task_log,
        'tasks': tasks,
    }


# ==================== Visualization ====================

def save_visualizations(agent, task, feature_losses, episode, save_dir):
    """Save feature maps and loss plots."""
    # Feature maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, feature in enumerate(agent.feature_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        im = ax.imshow(agent.feature_reward_maps[feature], cmap='viridis', origin='lower')
        ax.set_title(f'{feature} (allocentric)')
        plt.colorbar(im, ax=ax)
    
    # Composed map
    composed = agent.get_composed_reward_map(task)
    ax = axes[1, 2]
    im = ax.imshow(composed, cmap='viridis', origin='lower')
    ax.set_title(f'Composed: {task["name"]}')
    plt.colorbar(im, ax=ax)
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, f'Episode {episode}\nTask: {task["name"]}',
                    ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'maps_ep{episode}.png'), dpi=150)
    plt.close()


def plot_results(rewards, task_rewards, lengths, feature_losses, tasks, save_dir):
    """Plot final training results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 50
    
    # Calculate task boundaries
    boundaries = []
    cumulative = 0
    for task in tasks:
        cumulative += task['duration']
        boundaries.append(cumulative)
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        ax.plot(pd.Series(rewards).rolling(window).mean(), linewidth=2, label='Smoothed')
    for b in boundaries[:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Task rewards
    ax = axes[0, 1]
    ax.plot(task_rewards, alpha=0.3, label='Raw')
    if len(task_rewards) >= window:
        ax.plot(pd.Series(task_rewards).rolling(window).mean(), linewidth=2, label='Smoothed')
    for b in boundaries[:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Task Reward')
    ax.set_title('Task-Specific Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Lengths
    ax = axes[1, 0]
    ax.plot(lengths, alpha=0.3, label='Raw')
    if len(lengths) >= window:
        ax.plot(pd.Series(lengths).rolling(window).mean(), linewidth=2, label='Smoothed')
    for b in boundaries[:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Losses
    ax = axes[1, 1]
    colors = ['red', 'blue', 'brown', 'purple']
    for idx, (feature, losses) in enumerate(feature_losses.items()):
        if len(losses) >= window:
            smoothed = pd.Series(losses).rolling(window).mean()
            ax.plot(smoothed, label=feature, color=colors[idx], linewidth=2)
    for b in boundaries[:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Feature Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_results.png'), dpi=300)
    plt.close()
    print(f"✓ Results saved to {save_dir}/final_results.png")


if __name__ == "__main__":
    # Import your environment
    from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
    
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode=None)
    
    # Train!
    results = train_v3_agent(
        env=env,
        max_episodes=2500,
        max_steps=200,
        schedule_type='curriculum',  # 'curriculum' or 'interleaved'
        simple_ratio=0.6,            # 60% simple tasks first
        lr=0.0005,
        gamma=0.99,
        hidden_dim=128,
        selective_training=True,     # Speed optimization
        save_dir='results_v3'
    )
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✓ Done!")