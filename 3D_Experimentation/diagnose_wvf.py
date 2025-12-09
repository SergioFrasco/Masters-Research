"""
Diagnostic Script for WVF Composition

Run this during evaluation to understand why composition is failing.
"""

import numpy as np
import torch


def diagnose_composition(agent, task, obs, info):
    """
    Comprehensive diagnostic for compositional task execution.
    
    Call this at the start of each evaluation episode.
    """
    print("\n" + "="*60)
    print(f"DIAGNOSTIC: Task = {task['name']}")
    print("="*60)
    
    features = task.get("features", [])
    print(f"Features required: {features}")
    
    # 1. Check feature reward maps
    print("\n--- Feature Reward Maps ---")
    for f in features:
        rm = agent.feature_reward_maps[f]
        num_goals = np.sum(rm > agent.confidence_threshold)
        max_val = np.max(rm)
        print(f"  {f}: {num_goals} goals with confidence > {agent.confidence_threshold}, max = {max_val:.3f}")
        
        # Show goal positions
        goals = agent._get_goals_for_feature(f)
        if len(goals) <= 5:
            print(f"    Positions: {goals}")
        else:
            print(f"    Positions: {goals[:5]} ... ({len(goals)} total)")
    
    # 2. Check intersection (compositional goals)
    print("\n--- Compositional Goals (Intersection) ---")
    comp_goals = agent._get_goals_for_task(task)
    print(f"  Found {len(comp_goals)} goals satisfying ALL features")
    if len(comp_goals) == 0:
        print("  ⚠ NO VALID GOALS! This is likely the problem.")
        print("  Check: Are objects being detected? Is vision working?")
        return
    
    if len(comp_goals) <= 5:
        print(f"  Positions: {comp_goals}")
    
    # 3. Check Q-values for compositional goals
    print("\n--- Q-Values at Compositional Goals ---")
    state_features = agent.get_all_state_features()
    
    for goal in comp_goals[:3]:  # Check first 3 goals
        print(f"\n  Goal {goal}:")
        
        q_per_feature = {}
        for f in features:
            q = agent.get_q_values_for_goal(f, goal, state_features[f])
            q_per_feature[f] = q
            print(f"    Q_{f}: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}]")
        
        # Composed Q-values
        q_composed = agent.get_composed_q_values_for_goal(task, goal)
        print(f"    Q_composed (min): [{q_composed[0]:.3f}, {q_composed[1]:.3f}, {q_composed[2]:.3f}]")
        
        # Which feature is limiting?
        for a in range(3):
            limiting = min(features, key=lambda f: q_per_feature[f][a])
            print(f"    Action {a}: limited by {limiting}")
    
    # 4. Check action selection
    print("\n--- Action Selection ---")
    best_q = -1e10
    best_goal = None
    best_action = None
    
    for goal in comp_goals:
        q = agent.get_composed_q_values_for_goal(task, goal)
        max_q = np.max(q)
        if max_q > best_q:
            best_q = max_q
            best_goal = goal
            best_action = np.argmax(q)
    
    action_names = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]
    print(f"  Selected goal: {best_goal}")
    print(f"  Selected action: {action_names[best_action]} (Q = {best_q:.3f})")
    
    # 5. Check agent position relative to goal
    agent_pos = agent._get_agent_pos_from_env()
    print(f"\n  Agent position: {agent_pos}")
    if best_goal:
        dx = best_goal[0] - agent_pos[0]
        dz = best_goal[1] - agent_pos[1]
        print(f"  Delta to goal: ({dx}, {dz})")
    
    print("\n" + "="*60)


def diagnose_value_maps(agent, task):
    """
    Visualize the value maps for each feature and the composed map.
    """
    import matplotlib.pyplot as plt
    
    features = task.get("features", [])
    state_features = agent.get_all_state_features()
    grid_size = agent.grid_size
    
    # Compute value map for each feature
    value_maps = {}
    for f in features:
        v_map = np.zeros((grid_size, grid_size))
        for z in range(grid_size):
            for x in range(grid_size):
                q = agent.get_q_values_for_goal(f, (x, z), state_features[f])
                v_map[z, x] = np.max(q)
        value_maps[f] = v_map
    
    # Compute composed value map
    composed_map = np.min(np.stack([value_maps[f] for f in features]), axis=0)
    
    # Plot
    n_plots = len(features) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    for i, f in enumerate(features):
        ax = axes[i]
        im = ax.imshow(value_maps[f], cmap='viridis', origin='lower')
        ax.set_title(f'V_{f}(g)')
        plt.colorbar(im, ax=ax)
        
        # Mark goals for this feature
        goals = agent._get_goals_for_feature(f)
        for (x, z) in goals:
            ax.plot(x, z, 'r*', markersize=10)
    
    # Composed map
    ax = axes[-1]
    im = ax.imshow(composed_map, cmap='viridis', origin='lower')
    ax.set_title(f'V_composed = min({", ".join(features)})')
    plt.colorbar(im, ax=ax)
    
    # Mark compositional goals
    comp_goals = agent._get_goals_for_task(task)
    for (x, z) in comp_goals:
        ax.plot(x, z, 'g*', markersize=15)
    
    # Mark agent position
    agent_pos = agent._get_agent_pos_from_env()
    for ax in axes:
        ax.plot(agent_pos[0], agent_pos[1], 'wo', markersize=10, markeredgecolor='black')
    
    plt.tight_layout()
    plt.savefig('value_maps_diagnostic.png', dpi=150)
    plt.close()
    print("✓ Value maps saved to value_maps_diagnostic.png")


def check_q_value_scales(agent):
    """
    Check if Q-values from different feature networks are on comparable scales.
    """
    print("\n--- Q-Value Scale Check ---")
    
    state_features = agent.get_all_state_features()
    
    for f in agent.feature_names:
        goals = agent._get_goals_for_feature(f)
        if len(goals) == 0:
            print(f"  {f}: No goals found")
            continue
        
        all_q = []
        for g in goals[:10]:  # Sample up to 10 goals
            q = agent.get_q_values_for_goal(f, g, state_features[f])
            all_q.extend(q.tolist())
        
        all_q = np.array(all_q)
        print(f"  {f}: mean={np.mean(all_q):.3f}, std={np.std(all_q):.3f}, "
              f"min={np.min(all_q):.3f}, max={np.max(all_q):.3f}")


def check_training_quality(train_results):
    """
    Analyze training results to see if networks converged.
    """
    print("\n--- Training Quality Check ---")
    
    # Check final success rates per task
    rewards = train_results["task_rewards"]
    tasks = train_results["tasks"]
    
    cumulative = 0
    for task in tasks:
        task_rewards = rewards[cumulative:cumulative + task["duration"]]
        success_rate = np.mean([1 if r > 0 else 0 for r in task_rewards])
        
        # Check improvement over time
        first_half = task_rewards[:len(task_rewards)//2]
        second_half = task_rewards[len(task_rewards)//2:]
        
        first_success = np.mean([1 if r > 0 else 0 for r in first_half])
        second_success = np.mean([1 if r > 0 else 0 for r in second_half])
        
        print(f"  {task['name']:10}: overall={success_rate*100:.1f}%, "
              f"first_half={first_success*100:.1f}%, second_half={second_success*100:.1f}%")
        
        if second_success < first_success:
            print(f"    ⚠ Performance DECREASED - possible overfitting or instability")
        
        cumulative += task["duration"]
    
    # Check feature losses
    print("\n  Final losses per feature:")
    for f, losses in train_results["feature_losses"].items():
        if len(losses) > 100:
            final_loss = np.mean(losses[-100:])
            print(f"    {f}: {final_loss:.4f}")


# Usage example:
"""
# Add this to your evaluation loop:

for task in compositional_tasks:
    agent.reset()
    obs, info = env.reset()
    
    # Update agent's observation
    detection = detect_cube(...)
    agent.update_from_detection(detection)
    
    # Run diagnostic
    diagnose_composition(agent, task, obs, info)
    
    # Optionally visualize value maps
    diagnose_value_maps(agent, task)
    
    # Then continue with normal evaluation...
"""