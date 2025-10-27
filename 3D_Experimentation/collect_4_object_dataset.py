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
# Color-detection utilities
# ---------------------------
def detect_color_mask(img_arr, color="red", threshold=100, ratio=1.5, min_pixels=30):
    """
    Return True if `color` is detected in img_arr (numpy HxWx3 uint8).
    color: "red" or "blue"
    """
    if img_arr is None or not isinstance(img_arr, np.ndarray):
        return False

    r = img_arr[:, :, 0].astype(np.int32)
    g = img_arr[:, :, 1].astype(np.int32)
    b = img_arr[:, :, 2].astype(np.int32)

    if color == "red":
        is_col = (r > threshold) & (r > ratio * np.maximum(g, b))
    elif color == "blue":
        is_col = (b > threshold) & (b > ratio * np.maximum(r, g))
    else:
        raise ValueError("color must be 'red' or 'blue'")

    n = np.sum(is_col)
    return n >= min_pixels

# ---------------------------
# Geometry: world -> agent frame
# ---------------------------
def world_to_agent_relative(agent_pos, agent_dir, obj_pos):
    """
    Convert world (x, z) difference to agent-relative (dx, dz)
    using the rotation formula we discussed.

    agent_pos: (x, y, z)
    agent_dir: angle in radians
    obj_pos:   (x, y, z)

    Returns (dx_local, dz_local) where:
      - dx_local positive means object is to the agent's right
      - dz_local negative means object is in front (forward = negative dz)
    """
    dx_global = float(obj_pos[0] - agent_pos[0])
    dz_global = float(obj_pos[2] - agent_pos[2])

    # Rotate global difference into agent frame (same formulas we discussed)
    cos_t = math.cos(agent_dir)
    sin_t = math.sin(agent_dir)

    dx_local = dx_global * cos_t - dz_global * sin_t
    dz_local = dx_global * sin_t + dz_global * cos_t

    return dx_local, dz_local

def is_object_in_fov(agent_pos, agent_dir, obj_pos, fov_angle=60, max_distance=15):
    """
    Check if an object is within the agent's field of view.
    
    agent_pos: (x, y, z)
    agent_dir: angle in radians (direction agent is facing)
    obj_pos: (x, y, z)
    fov_angle: field of view in degrees (default 60, adjust based on MiniWorld's camera)
    max_distance: maximum distance to consider object visible
    
    Returns True if object is in FOV
    """
    dx_local, dz_local = world_to_agent_relative(agent_pos, agent_dir, obj_pos)
    
    # Calculate distance
    distance = math.sqrt(dx_local**2 + dz_local**2)
    if distance > max_distance or distance < 0.1:
        return False
    
    # Calculate angle from forward direction
    # In agent frame: forward is negative dz, so we use atan2(-dz_local, dx_local)
    angle_to_object = math.atan2(dx_local, -dz_local)  # angle from forward axis
    
    # Check if within FOV (half angle on each side)
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
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.csv"
    tmp_rows = []  # collect rows first, then split and move files

    images_collected = 0
    step = 0
    episode = 0

    obs, info = env.reset()
    step = 0

    pbar = tqdm(total=num_images, desc="Collecting images", unit="img")
    while images_collected < num_images:
        frame = env.render()  # expected to be HxWx3 uint8 numpy array

        if frame is not None:
            # Step 1: Detect colors in the rendered image (visual detection)
            red_seen = detect_color_mask(frame, color="red")
            blue_seen = detect_color_mask(frame, color="blue")

            # Step 2: Use geometry to determine which specific objects are in FOV
            try:
                agent_pos = env.agent.pos
                agent_dir = getattr(env.agent, "dir", 0.0)  # angle in radians

                # Check which objects are actually in the field of view
                red_box_visible = False
                blue_box_visible = False
                red_sphere_visible = False
                blue_sphere_visible = False

                if hasattr(env, "box_red"):
                    red_box_visible = is_object_in_fov(agent_pos, agent_dir, env.box_red.pos)
                if hasattr(env, "box_blue"):
                    blue_box_visible = is_object_in_fov(agent_pos, agent_dir, env.box_blue.pos)
                if hasattr(env, "sphere_red"):
                    red_sphere_visible = is_object_in_fov(agent_pos, agent_dir, env.sphere_red.pos)
                if hasattr(env, "sphere_blue"):
                    blue_sphere_visible = is_object_in_fov(agent_pos, agent_dir, env.sphere_blue.pos)

                # Step 3: Cross-check visual detection with geometric visibility
                # Only mark objects as visible if BOTH color is detected AND geometry says it's in FOV
                
                # If red is seen, it's either red box or red sphere (whichever is in FOV)
                if red_seen:
                    if not red_box_visible:
                        red_box_visible = False
                    if not red_sphere_visible:
                        red_sphere_visible = False
                else:
                    # If red not detected visually, neither red object is visible
                    red_box_visible = False
                    red_sphere_visible = False
                
                # Same for blue
                if blue_seen:
                    if not blue_box_visible:
                        blue_box_visible = False
                    if not blue_sphere_visible:
                        blue_sphere_visible = False
                else:
                    blue_box_visible = False
                    blue_sphere_visible = False

                # Create binary flags
                box_flag = 1 if (red_box_visible or blue_box_visible) else 0
                sphere_flag = 1 if (red_sphere_visible or blue_sphere_visible) else 0
                red_flag = 1 if (red_box_visible or red_sphere_visible) else 0
                blue_flag = 1 if (blue_box_visible or blue_sphere_visible) else 0

                # Compute relative positions for each object (only if visible)
                red_box_dx = red_box_dz = ""
                blue_box_dx = blue_box_dz = ""
                red_sphere_dx = red_sphere_dz = ""
                blue_sphere_dx = blue_sphere_dz = ""

                if red_box_visible and hasattr(env, "box_red"):
                    rdx, rdz = world_to_agent_relative(agent_pos, agent_dir, env.box_red.pos)
                    red_box_dx, red_box_dz = f"{rdx:.5f}", f"{rdz:.5f}"

                if blue_box_visible and hasattr(env, "box_blue"):
                    bdx, bdz = world_to_agent_relative(agent_pos, agent_dir, env.box_blue.pos)
                    blue_box_dx, blue_box_dz = f"{bdx:.5f}", f"{bdz:.5f}"

                if red_sphere_visible and hasattr(env, "sphere_red"):
                    rsdx, rsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_red.pos)
                    red_sphere_dx, red_sphere_dz = f"{rsdx:.5f}", f"{rsdz:.5f}"

                if blue_sphere_visible and hasattr(env, "sphere_blue"):
                    bsdx, bsdz = world_to_agent_relative(agent_pos, agent_dir, env.sphere_blue.pos)
                    blue_sphere_dx, blue_sphere_dz = f"{bsdx:.5f}", f"{bsdz:.5f}"

            except Exception as e:
                print("Warning computing visibility/positions:", e)
                box_flag = sphere_flag = red_flag = blue_flag = 0
                red_box_dx = red_box_dz = ""
                blue_box_dx = blue_box_dz = ""
                red_sphere_dx = red_sphere_dz = ""
                blue_sphere_dx = blue_sphere_dz = ""

            # filename (temporary, will move to train/test after split)
            filename = f"{img_prefix}_{images_collected:05d}.png"
            tmp_folder = out_dir / "tmp"
            tmp_folder.mkdir(parents=True, exist_ok=True)
            img_path = tmp_folder / filename

            # Save image
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
                # basic reporting
                box_count = sum(1 for r in tmp_rows[-100:] if r["box"] == 1)
                sphere_count = sum(1 for r in tmp_rows[-100:] if r["sphere"] == 1)
                red_count = sum(1 for r in tmp_rows[-100:] if r["red"] == 1)
                blue_count = sum(1 for r in tmp_rows[-100:] if r["blue"] == 1)
                none_count = sum(1 for r in tmp_rows[-100:] if r["box"] == 0 and r["sphere"] == 0)
                print(f"Last 100 -> Box:{box_count} Sphere:{sphere_count} Red:{red_count} Blue:{blue_count} None:{none_count}")

        # step environment with random action for exploration
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

    # Shuffle and split
    random.shuffle(tmp_rows)
    split_index = int((1.0 - test_fraction) * len(tmp_rows))
    train_rows = tmp_rows[:split_index]
    test_rows = tmp_rows[split_index:]

    # Move files into train/test and write CSV
    with open(labels_path, "w", newline="") as csvfile:
        fieldnames = [
            "filename", "box", "sphere", "red", "blue",
            "red_box_dx", "red_box_dz", "blue_box_dx", "blue_box_dz",
            "red_sphere_dx", "red_sphere_dz", "blue_sphere_dx", "blue_sphere_dz",
            "split"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # helper to move files
        for rows, subdir, splitname in [(train_rows, train_dir, "train"), (test_rows, test_dir, "test")]:
            for r in rows:
                src = out_dir / "tmp" / r["filename"]
                dst = subdir / r["filename"]
                # move file
                src.replace(dst)
                r_out = r.copy()
                r_out["split"] = splitname
                writer.writerow(r_out)

    # clean up tmp folder if empty
    tmp = out_dir / "tmp"
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()

    print(f"Done. Dataset saved to: {out_dir}")
    print(f"  train: {len(train_rows)} images -> {train_dir}")
    print(f"  test : {len(test_rows)} images -> {test_dir}")
    print(f"  labels: {labels_path}")


if __name__ == "__main__":
    # Adjust size/resolution exactly as you used previously
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")

    # Collect dataset (default 5000; change if you want more/less)
    collect_dataset(env, num_images=100, out_dir="dataset/dataset_3d", test_fraction=0.2)