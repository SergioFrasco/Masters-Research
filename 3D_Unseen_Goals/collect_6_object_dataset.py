import os
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'  # Optional: for matplotlib

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
    following MiniWorld's convention:
    
    agent_pos: (x, y, z)
    agent_dir: angle in radians
    obj_pos:   (x, y, z)

    Returns (dx_local, dz_local) where:
      - dx_local positive means object is to the agent's right (+X in world when dir=0)
      - dz_local negative means object is in front (-Z in world when dir=0)
    """
    dx_global = float(obj_pos[0] - agent_pos[0])
    dz_global = float(obj_pos[2] - agent_pos[2])

    # Rotate global difference into agent frame
    cos_t = math.cos(-agent_dir)
    sin_t = math.sin(-agent_dir)

    # Agent frame: +X is right, -Z is forward
    dx_local = dx_global * cos_t + dz_global * sin_t  # Right
    dz_local = -dx_global * sin_t + dz_global * cos_t  # Forward (negative is ahead)

    return dx_local, dz_local

def is_object_in_fov(agent_pos, agent_dir, obj_pos, fov_angle=90, max_distance=20):
    """
    Check if an object is within the agent's field of view using pure geometry.
    Uses MiniWorld convention: -Z is forward, +X is right.
    
    agent_pos: (x, y, z)
    agent_dir: angle in radians (direction agent is facing)
    obj_pos: (x, y, z)
    fov_angle: field of view in degrees (default 90)
    max_distance: maximum distance to consider object visible
    
    Returns True if object is in FOV
    """
    dx_local, dz_local = world_to_agent_relative(agent_pos, agent_dir, obj_pos)
    
    # Calculate distance
    distance = math.sqrt(dx_local**2 + dz_local**2)
    
    if distance > max_distance or distance < 0.1:
        return False
    
    # Object must be in front of agent (negative dz_local)
    if dz_local >= 0:  # Object is behind or too close
        return False
    
    # Calculate angle from forward direction
    # atan2(right, -forward) gives angle from forward axis
    angle_to_object = math.atan2(dx_local, -dz_local)
    
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
                agent_dir = env.agent.dir - math.pi / 2  # Adjust to MiniWorld's convention

                # Check each object purely geometrically
                red_box_visible = False
                blue_box_visible = False
                green_box_visible = False
                red_sphere_visible = False
                blue_sphere_visible = False
                green_sphere_visible = False

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
                
                if hasattr(env, "box_green"):
                    green_box_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.box_green.pos,
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
                
                if hasattr(env, "sphere_green"):
                    green_sphere_visible = is_object_in_fov(
                        agent_pos, agent_dir, env.sphere_green.pos,
                        fov_angle=fov_angle, max_distance=max_distance
                    )

                # Label ALL visible objects (multiple can be 1)
                # If no objects visible, only None=1
                
                # Initialize all flags to 0
                red_box_flag = 0
                blue_box_flag = 0
                green_box_flag = 0
                red_sphere_flag = 0
                blue_sphere_flag = 0
                green_sphere_flag = 0
                none_flag = 0
                
                # Initialize all positions to 0
                red_box_dx = red_box_dz = 0
                blue_box_dx = blue_box_dz = 0
                green_box_dx = green_box_dz = 0
                red_sphere_dx = red_sphere_dz = 0
                blue_sphere_dx = blue_sphere_dz = 0
                green_sphere_dx = green_sphere_dz = 0
                
                # Set flag and position for each visible object
                if red_box_visible:
                    red_box_flag = 1
                    rdx, rdz = world_to_agent_relative(agent_pos, agent_dir, env.box_red.pos)
                    red_box_dx, red_box_dz = rdx, rdz
                
                if blue_box_visible:
                    blue_box_flag = 1
                    bdx, bdz = world_to_agent_relative(agent_pos, agent_dir, env.box_blue.pos)
                    blue_box_dx, blue_box_dz = bdx, bdz
                
                if green_box_visible:
                    green_box_flag = 1
                    gdx, gdz = world_to_agent_relative(agent_pos, agent_dir, env.box_green.pos)
                    green_box_dx, green_box_dz = gdx, gdz
                
                if red_sphere_visible:
                    red_sphere_flag = 1
                    rsdx, rsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_red.pos)
                    red_sphere_dx, red_sphere_dz = rsdx, rsdz
                
                if blue_sphere_visible:
                    blue_sphere_flag = 1
                    bsdx, bsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_blue.pos)
                    blue_sphere_dx, blue_sphere_dz = bsdx, bsdz
                
                if green_sphere_visible:
                    green_sphere_flag = 1
                    gsdx, gsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_green.pos)
                    green_sphere_dx, green_sphere_dz = gsdx, gsdz
                
                # If NO objects are visible, set None flag
                if not (red_box_visible or blue_box_visible or green_box_visible or 
                        red_sphere_visible or blue_sphere_visible or green_sphere_visible):
                    none_flag = 1
                    # All positions remain 0

            except Exception as e:
                print(f"\nWarning computing visibility/positions: {e}")
                import traceback
                traceback.print_exc()
                # Set all to 0 and none_flag to 1 on error
                red_box_flag = blue_box_flag = green_box_flag = 0
                red_sphere_flag = blue_sphere_flag = green_sphere_flag = 0
                none_flag = 1
                red_box_dx = red_box_dz = 0
                blue_box_dx = blue_box_dz = 0
                green_box_dx = green_box_dz = 0
                red_sphere_dx = red_sphere_dz = 0
                blue_sphere_dx = blue_sphere_dz = 0
                green_sphere_dx = green_sphere_dz = 0

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

            # Store label format with all 6 objects
            tmp_rows.append(
                {
                    "filename": filename,
                    "red_box": red_box_flag,
                    "blue_box": blue_box_flag,
                    "green_box": green_box_flag,
                    "red_sphere": red_sphere_flag,
                    "blue_sphere": blue_sphere_flag,
                    "green_sphere": green_sphere_flag,
                    "None": none_flag,
                    "red_box_dx": f"{red_box_dx:.5f}" if isinstance(red_box_dx, float) else red_box_dx,
                    "red_box_dz": f"{red_box_dz:.5f}" if isinstance(red_box_dz, float) else red_box_dz,
                    "blue_box_dx": f"{blue_box_dx:.5f}" if isinstance(blue_box_dx, float) else blue_box_dx,
                    "blue_box_dz": f"{blue_box_dz:.5f}" if isinstance(blue_box_dz, float) else blue_box_dz,
                    "green_box_dx": f"{green_box_dx:.5f}" if isinstance(green_box_dx, float) else green_box_dx,
                    "green_box_dz": f"{green_box_dz:.5f}" if isinstance(green_box_dz, float) else green_box_dz,
                    "red_sphere_dx": f"{red_sphere_dx:.5f}" if isinstance(red_sphere_dx, float) else red_sphere_dx,
                    "red_sphere_dz": f"{red_sphere_dz:.5f}" if isinstance(red_sphere_dz, float) else red_sphere_dz,
                    "blue_sphere_dx": f"{blue_sphere_dx:.5f}" if isinstance(blue_sphere_dx, float) else blue_sphere_dx,
                    "blue_sphere_dz": f"{blue_sphere_dz:.5f}" if isinstance(blue_sphere_dz, float) else blue_sphere_dz,
                    "green_sphere_dx": f"{green_sphere_dx:.5f}" if isinstance(green_sphere_dx, float) else green_sphere_dx,
                    "green_sphere_dz": f"{green_sphere_dz:.5f}" if isinstance(green_sphere_dz, float) else green_sphere_dz,
                }
            )

            images_collected += 1
            pbar.update(1)

            # Updated progress reporting with all 6 objects
            if images_collected % 100 == 0:
                red_box_count = sum(1 for r in tmp_rows[-100:] if r["red_box"] == 1)
                blue_box_count = sum(1 for r in tmp_rows[-100:] if r["blue_box"] == 1)
                green_box_count = sum(1 for r in tmp_rows[-100:] if r["green_box"] == 1)
                red_sphere_count = sum(1 for r in tmp_rows[-100:] if r["red_sphere"] == 1)
                blue_sphere_count = sum(1 for r in tmp_rows[-100:] if r["blue_sphere"] == 1)
                green_sphere_count = sum(1 for r in tmp_rows[-100:] if r["green_sphere"] == 1)
                none_count = sum(1 for r in tmp_rows[-100:] if r["None"] == 1)
                print(f"\nLast 100 -> RedBox:{red_box_count} BlueBox:{blue_box_count} GreenBox:{green_box_count} "
                      f"RedSphere:{red_sphere_count} BlueSphere:{blue_sphere_count} GreenSphere:{green_sphere_count} None:{none_count}")

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

    # Updated CSV fieldnames with all 6 objects
    with open(labels_path, "w", newline="") as csvfile:
        fieldnames = [
            "filename", 
            "red_box", "blue_box", "green_box", 
            "red_sphere", "blue_sphere", "green_sphere", 
            "None",
            "red_box_dx", "red_box_dz", 
            "blue_box_dx", "blue_box_dz",
            "green_box_dx", "green_box_dz",
            "red_sphere_dx", "red_sphere_dz", 
            "blue_sphere_dx", "blue_sphere_dz",
            "green_sphere_dx", "green_sphere_dz",
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
        num_images=30000, 
        out_dir="dataset/dataset_3d", 
        test_fraction=0.2,
        fov_angle=90,
        max_distance=20,
    )