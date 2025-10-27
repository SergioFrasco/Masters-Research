# collect_3d_dataset.py
import os
import csv
import math
from pathlib import Path
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

# Import your wrapper (adjust path if needed)
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper

# ---------------------------
# Geometry: world -> agent frame
# ---------------------------
def world_to_agent_relative(agent_pos, agent_dir, obj_pos):
    """
    Convert world (x, z) difference to agent-relative (dx, dz)
    
    agent_pos: (x, y, z)
    agent_dir: angle in radians
    obj_pos:   (x, y, z)

    Returns (dx_local, dz_local) where:
      - dx_local positive means object is to the agent's right
      - dz_local negative means object is in front (forward = negative dz)
    """
    dx_global = float(obj_pos[0] - agent_pos[0])
    dz_global = float(obj_pos[2] - agent_pos[2])

    # Rotate global difference into agent frame
    cos_t = math.cos(agent_dir)
    sin_t = math.sin(agent_dir)

    dx_local = dx_global * cos_t + dz_global * sin_t
    dz_local = -dx_global * sin_t + dz_global * cos_t

    return dx_local, dz_local

def is_object_in_fov(agent_pos, agent_dir, obj_pos, fov_angle=90, max_distance=20):
    """
    Check if an object is within the agent's field of view using pure geometry.
    Rotated 90 degrees clockwise to match MiniWorld coordinate system.
    
    agent_pos: (x, y, z)
    agent_dir: angle in radians (direction agent is facing)
    obj_pos: (x, y, z)
    fov_angle: field of view in degrees (default 90)
    max_distance: maximum distance to consider object visible
    
    Returns True if object is in FOV
    """
    # Rotate the agent direction by 90 degrees clockwise (subtract π/2)
    rotated_dir = agent_dir - math.pi / 2
    
    dx_local, dz_local = world_to_agent_relative(agent_pos, rotated_dir, obj_pos)
    
    # Calculate distance
    distance = math.sqrt(dx_local**2 + dz_local**2)
    
    if distance > max_distance or distance < 0.1:
        return False
    
    # Object must be in front of agent (positive dz in agent frame means forward)
    if dz_local < 0:  # Object is behind the agent
        return False
    
    # Calculate angle from forward direction
    angle_to_object = math.atan2(dx_local, dz_local)
    
    # Check if within FOV
    half_fov = math.radians(fov_angle / 2)
    return abs(angle_to_object) <= half_fov

# ---------------------------
# Dataset collector
# ---------------------------
def collect_dataset(
    env,
    num_images=5000,
    out_dir="dataset_3d",
    test_fraction=0.2,
    max_steps_per_episode=50,
    img_prefix="img",
    random_seed=0,
    fov_angle=60,
    max_distance=20,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.csv"
    tmp_rows = []

    images_collected = 0
    step = 0
    episode = 0

    obs, info = env.reset()
    step = 0

    pbar = tqdm(total=num_images, desc="Collecting images", unit="img")
    while images_collected < num_images:
        frame = env.render()

        if frame is not None:
            try:
                agent_pos = env.agent.pos
                agent_dir = getattr(env.agent, "dir", 0.0)

                # Check each object purely geometrically
                red_box_visible = False
                blue_box_visible = False
                red_sphere_visible = False
                blue_sphere_visible = False

                if hasattr(env, "box_red"):
                    red_box_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.box_red.pos, 
                        fov_angle=fov_angle, max_distance=max_distance
                    )
                
                if hasattr(env, "box_blue"):
                    blue_box_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.box_blue.pos,
                        fov_angle=fov_angle, max_distance=max_distance
                    )
                
                if hasattr(env, "sphere_red"):
                    red_sphere_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.sphere_red.pos,
                        fov_angle=fov_angle, max_distance=max_distance
                    )
                
                if hasattr(env, "sphere_blue"):
                    blue_sphere_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.sphere_blue.pos,
                        fov_angle=fov_angle, max_distance=max_distance
                    )

                # Create binary flags
                box_flag = 1 if (red_box_visible or blue_box_visible) else 0
                sphere_flag = 1 if (red_sphere_visible or blue_sphere_visible) else 0
                red_flag = 1 if (red_box_visible or red_sphere_visible) else 0
                blue_flag = 1 if (blue_box_visible or blue_sphere_visible) else 0

                # Compute relative positions for visible objects
                red_box_dx = red_box_dz = ""
                blue_box_dx = blue_box_dz = ""
                red_sphere_dx = red_sphere_dz = ""
                blue_sphere_dx = blue_sphere_dz = ""

                if red_box_visible:
                    rdx, rdz = world_to_agent_relative(agent_pos, agent_dir, env.box_red.pos)
                    red_box_dx, red_box_dz = f"{rdx:.5f}", f"{rdz:.5f}"

                if blue_box_visible:
                    bdx, bdz = world_to_agent_relative(agent_pos, agent_dir, env.box_blue.pos)
                    blue_box_dx, blue_box_dz = f"{bdx:.5f}", f"{bdz:.5f}"

                if red_sphere_visible:
                    rsdx, rsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_red.pos)
                    red_sphere_dx, red_sphere_dz = f"{rsdx:.5f}", f"{rsdz:.5f}"

                if blue_sphere_visible:
                    bsdx, bsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_blue.pos)
                    blue_sphere_dx, blue_sphere_dz = f"{bsdx:.5f}", f"{bsdz:.5f}"

            except Exception as e:
                print(f"\nWarning computing visibility/positions: {e}")
                import traceback
                traceback.print_exc()
                box_flag = sphere_flag = red_flag = blue_flag = 0
                red_box_dx = red_box_dz = ""
                blue_box_dx = blue_box_dz = ""
                red_sphere_dx = red_sphere_dz = ""
                blue_sphere_dx = blue_sphere_dz = ""

            # Save image
            filename = f"{img_prefix}_{images_collected:05d}.png"
            tmp_folder = out_dir / "tmp"
            tmp_folder.mkdir(parents=True, exist_ok=True)
            img_path = tmp_folder / filename

            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
            else:
                img = frame
            img.save(img_path)

            tmp_rows.append(
                {
                    "filename": filename,
                    "box": box_flag,
                    "sphere": sphere_flag,
                    "red": red_flag,
                    "blue": blue_flag,
                    "red_box_dx": red_box_dx,
                    "red_box_dz": red_box_dz,
                    "blue_box_dx": blue_box_dx,
                    "blue_box_dz": blue_box_dz,
                    "red_sphere_dx": red_sphere_dx,
                    "red_sphere_dz": red_sphere_dz,
                    "blue_sphere_dx": blue_sphere_dx,
                    "blue_sphere_dz": blue_sphere_dz,
                }
            )

            images_collected += 1
            pbar.update(1)

            if images_collected % 100 == 0:
                box_count = sum(1 for r in tmp_rows[-100:] if r["box"] == 1)
                sphere_count = sum(1 for r in tmp_rows[-100:] if r["sphere"] == 1)
                red_count = sum(1 for r in tmp_rows[-100:] if r["red"] == 1)
                blue_count = sum(1 for r in tmp_rows[-100:] if r["blue"] == 1)
                none_count = sum(1 for r in tmp_rows[-100:] if r["box"] == 0 and r["sphere"] == 0)
                print(f"\nLast 100 -> Box:{box_count} Sphere:{sphere_count} Red:{red_count} Blue:{blue_count} None:{none_count}")

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if terminated or truncated or step >= max_steps_per_episode:
            episode += 1
            obs, info = env.reset()
            step = 0

    pbar.close()
    print(f"\nCollected {len(tmp_rows)} images in tmp folder: {out_dir/'tmp'}")
    print("Shuffling and splitting into train/test...")

    random.shuffle(tmp_rows)
    split_index = int((1.0 - test_fraction) * len(tmp_rows))
    train_rows = tmp_rows[:split_index]
    test_rows = tmp_rows[split_index:]

    with open(labels_path, "w", newline="") as csvfile:
        fieldnames = [
            "filename", "box", "sphere", "red", "blue",
            "red_box_dx", "red_box_dz", "blue_box_dx", "blue_box_dz",
            "red_sphere_dx", "red_sphere_dz", "blue_sphere_dx", "blue_sphere_dz",
            "split"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for rows, subdir, splitname in [(train_rows, train_dir, "train"), (test_rows, test_dir, "test")]:
            for r in rows:
                src = out_dir / "tmp" / r["filename"]
                dst = subdir / r["filename"]
                src.replace(dst)
                r_out = r.copy()
                r_out["split"] = splitname
                writer.writerow(r_out)

    tmp = out_dir / "tmp"
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()

    print(f"\nDone. Dataset saved to: {out_dir}")
    print(f"  train: {len(train_rows)} images -> {train_dir}")
    print(f"  test : {len(test_rows)} images -> {test_dir}")
    print(f"  labels: {labels_path}")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")

    collect_dataset(
        env, 
        num_images=10, 
        out_dir="dataset/dataset_3d", 
        test_fraction=0.2,
        fov_angle=60,
        max_distance=20,
    )