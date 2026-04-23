"""
Comprehensive Logical Composition Evaluation for ALL Algorithms

Evaluates SR, DQN, LSTM, and WVF agents on zero-shot compositional
generalization using AND, OR, NOT, and COMPLEX logical operators.

Composition strategy per algorithm:
─────────────────────────────────────────────────────────────────────
  SR:   Spatial feature-map composition
        (AND = min of thresholded maps, OR = max, NOT = all_objects − avoided)
        Then SR matrix converts spatial reward → value → action.

  DQN:  Q-value composition via get_q_values(raw_obs, primitive_name)
        Each call handles its own preprocessing (RGB + task-channel tiling).
        Compose the returned Q-arrays with min/max/NOT algebra.

  LSTM: Q-value composition by swapping the last 5 task-channels
        in the stacked observation (shape k*3+5, H, W).
        Run the network once per primitive with swapped channels.

  WVF:  Q-value composition via task-conditioned network.
        State = stacked frames (12,H,W); task = one-hot (5,).
        Cleanest comparator — just swap the one-hot per primitive.
─────────────────────────────────────────────────────────────────────

Usage:
    python eval_all_logical_compositions.py
"""

import os
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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

# ── Fix for PyTorch 2.6+ weights_only default change ────────────────
# The LSTM/WVF checkpoints contain collections.deque (from task tracking).
import collections
try:
    torch.serialization.add_safe_globals([collections.deque])
except AttributeError:
    pass  # Older PyTorch versions don't have this

# Monkey-patch load_model on all agents to use weights_only=False
# so checkpoints saved with older PyTorch load cleanly.
def _patched_load_lstm(self, filepath):
    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epsilon = checkpoint['epsilon']
    self.training_steps = checkpoint.get('training_steps', 0)
    print(f"Model loaded from {filepath}")

def _patched_load_dqn(self, filepath):
    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epsilon = checkpoint['epsilon']
    self.training_steps = checkpoint.get('training_steps', 0)
    print(f"Model loaded from {filepath}")

def _patched_load_wvf(self, filepath):
    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
    self.hidden_size = checkpoint.get('hidden_size', 128)
    self.lstm_size = checkpoint.get('lstm_size', 64)
    self.q_network.load_state_dict(checkpoint['q_network_state'])
    self.target_network.load_state_dict(checkpoint['target_network_state'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f"WVF model loaded from {filepath}")

UnifiedDQNAgent.load_model = _patched_load_dqn
UnifiedLSTMDQNAgent3D.load_model = _patched_load_lstm
UnifiedWorldValueFunctionAgent.load_model = _patched_load_wvf


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_DIR = Path("experiment_results_green/green_comparison_20251229_153537")
VISION_MODEL_PATH = "models/advanced_cube_detector.pth"
SEED_TO_USE = 1

EPISODES_PER_TASK = 50
MAX_STEPS = 200
ENV_SIZE = 10
EVAL_SEED = 42

OUTPUT_DIR = Path("logical_composition_all_algos")

SR_MATRIX_PATH  = EXPERIMENT_DIR / f"SR_seed{SEED_TO_USE}" / "frozen_sr_matrix.npy"
DQN_MODEL_PATH  = EXPERIMENT_DIR / f"DQN_seed{SEED_TO_USE}" / "model.pt"
LSTM_MODEL_PATH = EXPERIMENT_DIR / f"LSTM_seed{SEED_TO_USE}" / "model.pt"
WVF_MODEL_PATH  = EXPERIMENT_DIR / f"WVF_seed{SEED_TO_USE}" / "model.pt"

# All primitives the agents know about (green is unseen during training)
ALL_PRIMITIVES = ["red", "blue", "green", "box", "sphere"]


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

AND_TASKS = [
    {"name": "red AND box",     "features": ["red", "box"],     "logic": "AND", "category": "AND"},
    {"name": "red AND sphere",  "features": ["red", "sphere"],  "logic": "AND", "category": "AND"},
    {"name": "blue AND box",    "features": ["blue", "box"],    "logic": "AND", "category": "AND"},
    {"name": "blue AND sphere", "features": ["blue", "sphere"], "logic": "AND", "category": "AND"},
    {"name": "green AND box",   "features": ["green", "box"],   "logic": "AND", "category": "AND"},
    {"name": "green AND sphere","features": ["green", "sphere"],"logic": "AND", "category": "AND"},
]

OR_TASKS = [
    {"name": "red OR blue",     "features": ["red", "blue"],    "logic": "OR", "category": "OR"},
    {"name": "red OR green",    "features": ["red", "green"],   "logic": "OR", "category": "OR"},
    {"name": "blue OR green",   "features": ["blue", "green"],  "logic": "OR", "category": "OR"},
    {"name": "box OR sphere",   "features": ["box", "sphere"],  "logic": "OR", "category": "OR"},
    {"name": "red OR box",      "features": ["red", "box"],     "logic": "OR", "category": "OR"},
    {"name": "blue OR sphere",  "features": ["blue", "sphere"], "logic": "OR", "category": "OR"},
]

NOT_TASKS = [
    {"name": "NOT red",    "features": ["red"],    "logic": "NOT", "category": "NOT"},
    {"name": "NOT blue",   "features": ["blue"],   "logic": "NOT", "category": "NOT"},
    {"name": "NOT green",  "features": ["green"],  "logic": "NOT", "category": "NOT"},
    {"name": "NOT box",    "features": ["box"],    "logic": "NOT", "category": "NOT"},
    {"name": "NOT sphere", "features": ["sphere"], "logic": "NOT", "category": "NOT"},
]

COMPLEX_TASKS = [
    {
        "name": "(red AND sphere) OR (blue AND box)",
        "features": ["red", "sphere", "blue", "box"],
        "logic": "COMPLEX",
        "expression": "(red & sphere) | (blue & box)",
        "category": "COMPLEX",
    },
    {
        "name": "(red OR blue) AND sphere",
        "features": ["red", "blue", "sphere"],
        "logic": "COMPLEX",
        "expression": "(red | blue) & sphere",
        "category": "COMPLEX",
    },
    {
        "name": "red AND (sphere OR box)",
        "features": ["red", "sphere", "box"],
        "logic": "COMPLEX",
        "expression": "red & (sphere | box)",
        "category": "COMPLEX",
    },
    {
        "name": "(green AND box) OR (NOT red AND sphere)",
        "features": ["green", "box", "red", "sphere"],
        "logic": "COMPLEX",
        "expression": "(green & box) | (~red & sphere)",
        "category": "COMPLEX",
    },
]

ALL_TASKS = AND_TASKS + OR_TASKS + NOT_TASKS + COMPLEX_TASKS


# ============================================================================
# TASK SATISFACTION
# ============================================================================

OBJECT_FEATURES = {
    "red_box":      {"red", "box"},
    "red_sphere":   {"red", "sphere"},
    "blue_box":     {"blue", "box"},
    "blue_sphere":  {"blue", "sphere"},
    "green_box":    {"green", "box"},
    "green_sphere": {"green", "sphere"},
}


def check_task_satisfaction(info, task):
    contacted = info.get("contacted_object", None)
    if contacted is None:
        return False
    obj_feats = OBJECT_FEATURES.get(contacted, set())
    task_feats = set(task["features"])
    logic = task.get("logic", "AND")

    if logic == "AND":
        return task_feats.issubset(obj_feats)
    elif logic == "OR":
        return len(task_feats & obj_feats) > 0
    elif logic == "NOT":
        return len(task_feats & obj_feats) == 0
    elif logic == "COMPLEX":
        ns = {f: (f in obj_feats) for f in ALL_PRIMITIVES}
        expr = task["expression"].replace("&", " and ").replace("|", " or ").replace("~", " not ")
        try:
            return bool(eval(expr, {"__builtins__": {}}, ns))
        except Exception:
            return False
    return False


# ============================================================================
# VISION UTILITIES  (used by SR agent)
# ============================================================================

def load_cube_detector(model_path=VISION_MODEL_PATH, force_cpu=False):
    device = torch.device("cpu") if force_cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = CubeDetector().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        pos_mean = ckpt.get("pos_mean", 0.0)
        pos_std  = ckpt.get("pos_std", 1.0)
    else:
        model.load_state_dict(ckpt)
        pos_mean, pos_std = 0.0, 1.0
    model.eval()
    return model, device, pos_mean, pos_std


def detect_cube(model, obs, device, transform, pos_mean=0.0, pos_std=1.0):
    img = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    if isinstance(img, np.ndarray):
        if img.shape[0] in (3, 4):
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
        reg = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        names = ["red_box", "blue_box", "green_box",
                 "red_sphere", "blue_sphere", "green_sphere"]
        detected = [names[i] for i in range(6) if preds[i]]
    positions = {}
    for i, n in enumerate(names):
        positions[n] = (reg[2*i], reg[2*i+1]) if preds[i] else None
    return {"detected_objects": detected, "positions": positions}


# ============================================================================
# Q-VALUE COMPOSITION  (shared by DQN, LSTM, WVF)
# ============================================================================

def compose_q_values(q_per_prim, task, all_prims):
    """
    Compose per-primitive Q-value arrays → single composed Q-array.

    q_per_prim:  {prim_name: np.array(n_actions,)}
    Returns np.array(n_actions,) or None.
    """
    feats = task["features"]
    logic = task.get("logic", "AND")

    if logic == "AND":
        qs = [q_per_prim[f] for f in feats if f in q_per_prim]
        return np.minimum.reduce(qs) if qs else None

    elif logic == "OR":
        qs = [q_per_prim[f] for f in feats if f in q_per_prim]
        return np.maximum.reduce(qs) if qs else None

    elif logic == "NOT":
        avoid = feats[0]
        others = [q_per_prim[f] for f in all_prims
                  if f != avoid and f in q_per_prim]
        if not others:
            return None
        avg_other = np.mean(others, axis=0)
        avoid_q = q_per_prim.get(avoid)
        return avg_other - avoid_q if avoid_q is not None else avg_other

    elif logic == "COMPLEX":
        return _compose_complex_q(task.get("expression", ""), q_per_prim)

    return None


def _compose_complex_q(expr, qs):
    all_avg = np.mean(list(qs.values()), axis=0)
    def _not(q):  return all_avg - q
    def _and(*a): return np.minimum.reduce(a)
    def _or(*a):  return np.maximum.reduce(a)
    try:
        if expr == "(red & sphere) | (blue & box)":
            return _or(_and(qs["red"], qs["sphere"]),
                       _and(qs["blue"], qs["box"]))
        elif expr == "(red | blue) & sphere":
            return _and(_or(qs["red"], qs["blue"]), qs["sphere"])
        elif expr == "red & (sphere | box)":
            return _and(qs["red"], _or(qs["sphere"], qs["box"]))
        elif expr == "(green & box) | (~red & sphere)":
            g = qs.get("green")
            if g is not None:
                return _or(_and(g, qs["box"]),
                           _and(_not(qs["red"]), qs["sphere"]))
            return _and(_not(qs["red"]), qs["sphere"])
    except KeyError:
        pass
    return None


# ============================================================================
# SR EVALUATOR — spatial feature-map composition
# ============================================================================

class SRLogicalEvaluator:
    """
    SuccessorAgent with extended compose_reward_map for AND/OR/NOT/COMPLEX
    over spatial feature maps, then SR × reward → WVF → action.
    """

    def __init__(self, env, sr_path, cube_model, dev, tf, pm, ps):
        self.env = env
        self.agent = SuccessorAgent(env)
        self.agent.M = np.load(str(sr_path))
        self.cube_model, self.device = cube_model, dev
        self.transform, self.pos_mean, self.pos_std = tf, pm, ps
        print(f"  SR: loaded frozen SR from {sr_path}")

    def _compose_extended(self, task):
        a = self.agent
        feats  = task["features"]
        logic  = task.get("logic", "AND")
        thresh = a.confidence_threshold

        if logic == "AND":
            maps = [(a.feature_map[f] > thresh).astype(np.float32) for f in feats]
            a.composed_reward_map = np.minimum.reduce(maps)

        elif logic == "OR":
            maps = [(a.feature_map[f] > thresh).astype(np.float32) for f in feats]
            a.composed_reward_map = np.maximum.reduce(maps)

        elif logic == "NOT":
            all_obj = np.zeros_like(a.feature_map["red"])
            for f in a.feature_map:
                all_obj = np.maximum(all_obj,
                    (a.feature_map[f] > thresh).astype(np.float32))
            avoid = (a.feature_map[feats[0]] > thresh).astype(np.float32)
            a.composed_reward_map = all_obj * (1.0 - avoid)

        elif logic == "COMPLEX":
            a.composed_reward_map = self._eval_complex_map(task.get("expression", ""))

    def _eval_complex_map(self, expr):
        a = self.agent
        t = a.confidence_threshold
        ns = {f: (a.feature_map[f] > t) for f in a.feature_map}
        try:
            if expr == "(red & sphere) | (blue & box)":
                r = np.logical_or(np.logical_and(ns["red"], ns["sphere"]),
                                  np.logical_and(ns["blue"], ns["box"]))
            elif expr == "(red | blue) & sphere":
                r = np.logical_and(np.logical_or(ns["red"], ns["blue"]),
                                   ns["sphere"])
            elif expr == "red & (sphere | box)":
                r = np.logical_and(ns["red"],
                                   np.logical_or(ns["sphere"], ns["box"]))
            elif expr == "(green & box) | (~red & sphere)":
                r = np.logical_or(
                    np.logical_and(ns["green"], ns["box"]),
                    np.logical_and(np.logical_not(ns["red"]), ns["sphere"]))
            else:
                return np.zeros((a.grid_size, a.grid_size), dtype=np.float32)
            return r.astype(np.float32)
        except Exception:
            return np.zeros((a.grid_size, a.grid_size), dtype=np.float32)

    def evaluate_episode(self, task):
        obs, info = self.env.reset()
        self.agent.reset()
        for _ in range(MAX_STEPS):
            det = detect_cube(self.cube_model, obs, self.device,
                              self.transform, self.pos_mean, self.pos_std)
            self.agent.update_feature_map(det["detected_objects"], det["positions"])
            self._compose_extended(task)
            self.agent.compute_wvf()
            action = self.agent.sample_action_with_wvf(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = self.env.step(action)
            if check_task_satisfaction(info, task):
                return 1.0
            if terminated or truncated:
                break
        return 0.0


# ============================================================================
# DQN EVALUATOR — get_q_values(raw_obs, prim) handles preprocessing
# ============================================================================

class DQNLogicalEvaluator:
    """
    UnifiedDQNAgent.get_q_values(obs, task_name) takes a RAW observation
    and a task identifier (str), internally builds the (8,H,W) tensor
    (3 RGB + 5 task channels), and returns np.array(3,).

    For composition we call it once per primitive, then combine.
    """

    def __init__(self, env, model_path):
        self.env = env
        self.agent = UnifiedDQNAgent(
            env, learning_rate=1e-4, gamma=0.99,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
            memory_size=100, batch_size=64, hidden_size=256,
            use_dueling=True, tau=0.005, use_double_dqn=True, grad_clip=10.0)
        self.agent.load_model(str(model_path))
        self.agent.epsilon = 0.0
        print(f"  DQN: loaded from {model_path}")

    def evaluate_episode(self, task):
        obs, info = self.env.reset()
        for _ in range(MAX_STEPS):
            # One forward pass per primitive, using raw obs each time
            q_per_prim = {}
            for prim in ALL_PRIMITIVES:
                q_per_prim[prim] = self.agent.get_q_values(obs, prim)  # np(3,)

            composed = compose_q_values(q_per_prim, task, ALL_PRIMITIVES)
            if composed is not None:
                action = int(np.argmax(composed))
            else:
                action = self.agent.select_action(obs, task["features"], epsilon=0.0)

            obs, _, terminated, truncated, info = self.env.step(action)
            if check_task_satisfaction(info, task):
                return 1.0
            if terminated or truncated:
                break
        return 0.0


# ============================================================================
# LSTM EVALUATOR — swap the last 5 task-channels in the stacked obs
# ============================================================================

class LSTMLogicalEvaluator:
    """
    UnifiedLSTMDQNAgent3D bakes the task into the LAST 5 channels of the
    stacked observation: shape (k*3 + 5, H, W) = (17, 60, 80).

    get_q_values(stacked_obs) runs the network on whatever task channels
    are currently in the array — it takes NO separate task argument.

    Strategy: copy the stacked obs, overwrite the last 5 channels with the
    desired primitive's one-hot tiled spatially, then call get_q_values.
    """

    TASK_ENCODINGS = {
        'red':    [1., 0., 0., 0., 0.],
        'blue':   [0., 1., 0., 0., 0.],
        'green':  [0., 0., 1., 0., 0.],
        'box':    [0., 0., 0., 1., 0.],
        'sphere': [0., 0., 0., 0., 1.],
    }

    def __init__(self, env, model_path):
        self.env = env
        self.agent = UnifiedLSTMDQNAgent3D(
            env, k_frames=4,
            learning_rate=1e-4, gamma=0.99,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
            memory_size=100, batch_size=16, seq_len=4,
            hidden_size=128, lstm_size=64,
            use_dueling=True, tau=0.005, use_double_dqn=True, grad_clip=10.0)
        self.agent.load_model(str(model_path))
        self.agent.epsilon = 0.0
        print(f"  LSTM: loaded from {model_path}")

    def _swap_task(self, stacked_obs, prim):
        """Return a copy with the last 5 channels set to prim's one-hot."""
        out = stacked_obs.copy()
        enc = self.TASK_ENCODINGS[prim]
        H, W = out.shape[1], out.shape[2]
        for i in range(5):
            out[-(5 - i), :, :] = enc[i]
        return out

    def _get_q_for_prim(self, stacked_obs, prim):
        """Forward pass with task channels swapped to `prim`."""
        swapped = self._swap_task(stacked_obs, prim)
        t = torch.FloatTensor(swapped).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q, _ = self.agent.q_network(t, self.agent.current_hidden)
        return q.cpu().numpy().flatten()

    def evaluate_episode(self, task):
        obs, info = self.env.reset()
        # Initialise frame stack with first feature (just to fill buffers)
        stacked = self.agent.reset_episode(obs, task["features"][0])

        for _ in range(MAX_STEPS):
            # Get Q-values per primitive by swapping task channels
            q_per_prim = {p: self._get_q_for_prim(stacked, p) for p in ALL_PRIMITIVES}

            composed = compose_q_values(q_per_prim, task, ALL_PRIMITIVES)
            if composed is not None:
                action = int(np.argmax(composed))
            else:
                action = self.agent.select_action(stacked, epsilon=0.0)

            obs, _, terminated, truncated, info = self.env.step(action)
            stacked = self.agent.step_episode(obs)

            # Advance LSTM hidden state (use first feature for consistency)
            s = self._swap_task(stacked, task["features"][0])
            t = torch.FloatTensor(s).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                _, h = self.agent.q_network(t, self.agent.current_hidden)
                self.agent.current_hidden = (h[0].detach(), h[1].detach())

            if check_task_satisfaction(info, task):
                return 1.0
            if terminated or truncated:
                break
        return 0.0


# ============================================================================
# WVF EVALUATOR — task-conditioned network (state and task are separate)
# ============================================================================

class WVFLogicalEvaluator:
    """
    UnifiedWorldValueFunctionAgent's network signature is:
        q_values, hidden = network(state, task_one_hot, hidden)
    where state = stacked frames (12,H,W) and task_one_hot = (1, 5).

    This is the cleanest architecture for composition because we just
    swap the one-hot vector per primitive — no channel surgery needed.
    """

    def __init__(self, env, model_path):
        self.env = env
        self.agent = UnifiedWorldValueFunctionAgent(
            env, k_frames=4,
            learning_rate=1e-4, gamma=0.99,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
            memory_size=100, batch_size=16, seq_len=4,
            hidden_size=128, lstm_size=64,
            tau=0.005, grad_clip=10.0,
            r_correct=1.0, r_wrong=-0.1, step_penalty=-0.005)
        self.agent.load_model(str(model_path))
        self.agent.epsilon = 0.0
        print(f"  WVF: loaded from {model_path}")

    def _get_q_for_prim(self, stacked_obs, prim, use_target=True):
        """Forward pass with a specific task one-hot."""
        state = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.agent.device)
        idx   = self.agent.TASK_TO_IDX[prim]
        oh    = self.agent.get_task_one_hot(idx)                    # (1, 5)
        net   = self.agent.target_network if use_target else self.agent.q_network
        with torch.no_grad():
            q, _ = net(state, oh, self.agent.current_hidden)
        return q.cpu().numpy().flatten()

    def evaluate_episode(self, task):
        obs, info = self.env.reset()
        stacked = self.agent.reset_episode(obs)

        for _ in range(MAX_STEPS):
            q_per_prim = {p: self._get_q_for_prim(stacked, p) for p in ALL_PRIMITIVES}

            composed = compose_q_values(q_per_prim, task, ALL_PRIMITIVES)
            if composed is not None:
                action = int(np.argmax(composed))
            else:
                # Fallback: agent's own AND composition
                try:
                    action = self.agent.select_action_composed(
                        stacked, task["features"], use_target=True)
                except Exception:
                    action = self.agent.select_action_primitive(
                        stacked, task["features"][0], use_target=True)

            obs, _, terminated, truncated, info = self.env.step(action)
            stacked = self.agent.step_episode(obs)

            # Advance hidden state with first feature
            first = task["features"][0]
            s_t = torch.FloatTensor(stacked).unsqueeze(0).to(self.agent.device)
            oh  = self.agent.get_task_one_hot(self.agent.TASK_TO_IDX[first])
            with torch.no_grad():
                _, h = self.agent.target_network(s_t, oh, self.agent.current_hidden)
                self.agent.current_hidden = (h[0].detach(), h[1].detach())

            if check_task_satisfaction(info, task):
                return 1.0
            if terminated or truncated:
                break
        return 0.0


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_evaluation():
    print("\n" + "=" * 70)
    print("LOGICAL COMPOSITION EVALUATION — ALL ALGORITHMS")
    print("=" * 70)
    print(f"Tasks:  {len(ALL_TASKS)} "
          f"({len(AND_TASKS)} AND, {len(OR_TASKS)} OR, "
          f"{len(NOT_TASKS)} NOT, {len(COMPLEX_TASKS)} COMPLEX)")
    print(f"Episodes/task: {EPISODES_PER_TASK}   Max steps: {MAX_STEPS}")
    print(f"Eval seed: {EVAL_SEED}   Model seed: {SEED_TO_USE}")
    print("=" * 70 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    random.seed(EVAL_SEED)

    # Vision model (shared by SR)
    print("Loading vision model …")
    cube_model, vis_dev, pm, ps = load_cube_detector()
    vis_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Environment (eval mode → green objects spawned)
    env = DiscreteMiniWorldWrapper(size=ENV_SIZE, render_mode="rgb_array",
                                   training_mode=False)

    # Build evaluators for every algorithm whose model exists
    print("\nLoading agents …")
    evaluators = {}
    if SR_MATRIX_PATH.exists():
        evaluators["SR"] = SRLogicalEvaluator(
            env, SR_MATRIX_PATH, cube_model, vis_dev, vis_tf, pm, ps)
    else:
        print(f"  ⚠ SR not found: {SR_MATRIX_PATH}")

    if DQN_MODEL_PATH.exists():
        evaluators["DQN"] = DQNLogicalEvaluator(env, DQN_MODEL_PATH)
    else:
        print(f"  ⚠ DQN not found: {DQN_MODEL_PATH}")

    if LSTM_MODEL_PATH.exists():
        evaluators["LSTM"] = LSTMLogicalEvaluator(env, LSTM_MODEL_PATH)
    else:
        print(f"  ⚠ LSTM not found: {LSTM_MODEL_PATH}")

    if WVF_MODEL_PATH.exists():
        evaluators["WVF"] = WVFLogicalEvaluator(env, WVF_MODEL_PATH)
    else:
        print(f"  ⚠ WVF not found: {WVF_MODEL_PATH}")

    print(f"\nActive algorithms: {list(evaluators.keys())}\n")

    # ── Run evaluation ──────────────────────────────────────────────
    results = {a: {} for a in evaluators}

    for task in ALL_TASKS:
        tname = task["name"]
        cat   = task["category"]
        env.set_task(task)

        print(f"\n{'─'*60}")
        print(f"Task: {tname}  ({cat})")
        print(f"{'─'*60}")

        for algo, ev in evaluators.items():
            ep_results = []
            for _ in tqdm(range(EPISODES_PER_TASK), desc=f"  {algo}", leave=False):
                ep_results.append(ev.evaluate_episode(task))
            sr = np.mean(ep_results)
            results[algo][tname] = {
                "category": cat,
                "rewards": ep_results,
                "success_rate": float(sr),
                "n_success": int(sum(ep_results)),
                "n_episodes": EPISODES_PER_TASK,
            }
            print(f"  {algo:5s}: {sr:.3f}  "
                  f"({int(sum(ep_results))}/{EPISODES_PER_TASK})")

    # ── Save ────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "all_algo_logical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {OUTPUT_DIR / 'all_algo_logical_results.json'}")
    return results


# ============================================================================
# PLOTTING
# ============================================================================

ALGO_COLORS = {"SR": "#2ecc71", "DQN": "#3498db",
               "LSTM": "#e67e22", "WVF": "#9b59b6"}
CATEGORIES  = ["AND", "OR", "NOT", "COMPLEX"]


def plot_all(results, out):
    out = Path(out)
    algos = list(results.keys())

    # ── 1. Box plots per category ────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle("Logical Composition — All Algorithms",
                 fontsize=17, fontweight="bold", y=1.02)

    for ax, cat in zip(axes, CATEGORIES):
        data, labels, cols = [], [], []
        for algo in algos:
            rates = [v["success_rate"]
                     for v in results[algo].values() if v["category"] == cat]
            if rates:
                data.append(rates); labels.append(algo)
                cols.append(ALGO_COLORS.get(algo, "#95a5a6"))
        if not data:
            ax.set_visible(False); continue

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c); p.set_alpha(0.7)
        for i, (d, c) in enumerate(zip(data, cols)):
            j = np.random.normal(0, 0.04, len(d))
            ax.scatter(np.full(len(d), i+1)+j, d, color=c,
                       edgecolor="black", lw=.5, s=40, zorder=5, alpha=.8)
        ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_title(f"{cat}", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Success Rate" if cat == "AND" else "", fontsize=12)
        ax.axhline(.5, color="gray", ls="--", lw=1, alpha=.6)
        ax.grid(axis="y", alpha=.25, ls="--")

    plt.tight_layout()
    p1 = out / "boxplot_by_category.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight"); plt.close()
    print(f"✓ {p1}")

    # ── 2. Overall box plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    data, labels, cols = [], [], []
    for algo in algos:
        data.append([v["success_rate"] for v in results[algo].values()])
        labels.append(algo); cols.append(ALGO_COLORS.get(algo, "#95a5a6"))

    bp = ax.boxplot(data, patch_artist=True, widths=.55,
                    medianprops=dict(color="black", linewidth=2))
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.7)
    for i, (d, c) in enumerate(zip(data, cols)):
        j = np.random.normal(0, 0.05, len(d))
        ax.scatter(np.full(len(d), i+1)+j, d, color=c,
                   edgecolor="black", lw=.5, s=50, zorder=5, alpha=.8)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.set_ylabel("Success Rate", fontsize=13, fontweight="bold")
    ax.set_title("Overall Logical Composition Performance\n"
                 "(all task types combined)", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(.5, color="gray", ls="--", lw=1.5, alpha=.6, label="50 % baseline")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=.25, ls="--")
    plt.tight_layout()
    p2 = out / "boxplot_overall.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight"); plt.close()
    print(f"✓ {p2}")

    # ── 3. Grouped bar chart per task ────────────────────────────────
    fig, ax = plt.subplots(figsize=(24, 8))
    tnames = [t["name"] for t in ALL_TASKS]
    n_t, n_a = len(tnames), len(algos)
    bw = 0.8 / n_a
    xb = np.arange(n_t)

    for i, algo in enumerate(algos):
        rates = [results[algo].get(t, {}).get("success_rate", 0) for t in tnames]
        ax.bar(xb + (i - n_a/2 + .5)*bw, rates, bw,
               color=ALGO_COLORS.get(algo, "#95a5a6"), alpha=.8,
               edgecolor="black", lw=.5, label=algo)

    ax.set_xticks(xb)
    ax.set_xticklabels(tnames, rotation=55, ha="right", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=13, fontweight="bold")
    ax.set_title("Per-Task Success Rate — All Algorithms",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(.5, color="gray", ls="--", lw=1, alpha=.6)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=.25, ls="--")

    cum = 0
    for ct in [AND_TASKS, OR_TASKS, NOT_TASKS, COMPLEX_TASKS]:
        if cum > 0:
            ax.axvline(cum - .5, color="black", lw=1.5, alpha=.4)
        cum += len(ct)

    plt.tight_layout()
    p3 = out / "bar_per_task.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight"); plt.close()
    print(f"✓ {p3}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_evaluation()
    
    plot_all(results, OUTPUT_DIR)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    algos = list(results.keys())
    for cat in CATEGORIES:
        print(f"\n{cat}:")
        for algo in algos:
            rates = [v["success_rate"]
                     for v in results[algo].values() if v["category"] == cat]
            if rates:
                print(f"  {algo:5s}  mean={np.mean(rates):.3f}  "
                      f"std={np.std(rates):.3f}")
    print("\n" + "=" * 70)
    print(f"All outputs in: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")