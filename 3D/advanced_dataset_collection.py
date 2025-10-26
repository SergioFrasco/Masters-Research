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
            # detect colors in the rendered image
            red_seen = detect_color_mask(frame, color="red")
            blue_seen = detect_color_mask(frame, color="blue")

            if red_seen and blue_seen:
                label = "Both"
            elif red_seen:
                label = "Red_Cube"
            elif blue_seen:
                label = "Blue_Cube"
            else:
                label = "None"

            # filename (temporary, will move to train/test after split)
            filename = f"{img_prefix}_{images_collected:05d}.png"
            # save to a temporary folder under out_dir/tmp
            tmp_folder = out_dir / "tmp"
            tmp_folder.mkdir(parents=True, exist_ok=True)
            img_path = tmp_folder / filename

            # Save image
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
            else:
                img = frame
            img.save(img_path)

            # compute relative positions using env attributes (wrapper has box_red, box_blue)
            red_dx = red_dz = ""
            blue_dx = blue_dz = ""

            # Only include relative coords for objects WE DETECTED as visible in image.
            # This avoids labeling invisible objects as visible.
            try:
                agent_pos = env.agent.pos
                agent_dir = getattr(env.agent, "dir", 0.0)  # angle in radians

                if red_seen and hasattr(env, "box_red"):
                    rdx, rdz = world_to_agent_relative(agent_pos, agent_dir, env.box_red.pos)
                    red_dx, red_dz = f"{rdx:.5f}", f"{rdz:.5f}"

                if blue_seen and hasattr(env, "box_blue"):
                    bdx, bdz = world_to_agent_relative(agent_pos, agent_dir, env.box_blue.pos)
                    blue_dx, blue_dz = f"{bdx:.5f}", f"{bdz:.5f}"
            except Exception as e:
                # If anything about env attributes changes, we fallback to empty coords
                # but keep running so dataset still collects images.
                print("Warning computing relative pos:", e)

            tmp_rows.append(
                {
                    "filename": filename,
                    "label": label,
                    "red_dx": red_dx,
                    "red_dz": red_dz,
                    "blue_dx": blue_dx,
                    "blue_dz": blue_dz,
                }
            )

            images_collected += 1
            pbar.update(1)

            if images_collected % 100 == 0:
                # basic reporting
                counts = {"Both":0,"Red_Cube":0,"Blue_Cube":0,"None":0}
                for r in tmp_rows[-100:]:
                    counts[r["label"]] += 1
                print(f"Last 100 -> Both:{counts['Both']} Red:{counts['Red_Cube']} Blue:{counts['Blue_Cube']} None:{counts['None']}")

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
        fieldnames = ["filename", "label", "red_dx", "red_dz", "blue_dx", "blue_dz", "split"]
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

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Adjust size/resolution exactly as you used previously
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")

    # Collect dataset (default 5000; change if you want more/less)
    collect_dataset(env, num_images=10000, out_dir="dataset/dataset_3d", test_fraction=0.2)
