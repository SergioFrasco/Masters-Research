# Compositional Reinforcement Learning with Zero-Shot Generalisation

**Sergio Frasco** | Masters Research, University of the Witwatersrand
Supervisors: **Devon Jarvis** and **Geraud Nangue Tasse**

---

## What This Project Is About

The study of decision-making in biological agents reveals a fascinating interplay be-
tween vision and spatial reasoning. While animals can navigate complex environments
and locate rewards without exhaustive exploration, traditional reinforcement learning
approaches often require extensive trial-and-error. Much like how humans can visu-
ally identify a coffee cup and plan a path to reach it without physically bumping into
every object in the room, We propose VISTR (Vision Integration for Successor Task
Representations), a framework that combines visual understanding with learned spa-
tial representations.

Our approach integrates vision-based reward identification with successor representa-
tions to construct adaptive world value functions. By using vision to compress the envi-
ronment into discrete 2D reward maps and combining these with successor representation-
derived occupancy predictions, we create a structured representation that enables log-
ical composition of tasks and zero-shot behaviour. This marriage of visual and spa-
tial learning mirrors biological systems: while the successor representation component
learns hard spatial relationships through experience, the vision component enables im-
mediate reward identification and task planning without direct interaction.
We demonstrate this system’s effectiveness by transitioning from allocentric to egocen-
tric representations, reflecting the biological basis of mammalian spatial processing. By
grounding our approach in biological plausibility, this research addresses key limita-
tions of traditional RL models, improving their ability to learn efficiently and flexibly
with minimal environmental interaction, however, this comes with a trade-off: the vi-
sion system is pretrained at certain stages, meaning it does not learn entirely from
scratch within the environment. While spatial reasoning remains computationally dif-
ficult, our approach makes this process more transferable across tasks, significantly
reducing the spatial exploration needed when adapting to new reward configurations.
We anticipate our findings will contribute to developing more adaptive and general-
izable artificial agents, bridging the gap between biological intelligence and artificial
reinforcement learning systems.

---

## Core Ideas

A few concepts come up throughout the project. It helps to understand these before diving into the code.

**Successor Representations (SR).** Rather than learning a direct mapping from state to reward, SR agents learn to predict which states they are likely to visit in the future. This separates the structure of the environment from the specific goal the agent is pursuing. When the goal changes, the agent does not need to relearn how to navigate, it just updates what it cares about. This makes SR agents naturally suited to compositional tasks.

**World Value Functions (WVF).** A neural network approach where the agent learns Q-values conditioned on the current task. The task is provided as part of the input, so the network learns a family of policies simultaneously. At test time, the agent can combine value functions from primitive tasks to construct a solution to a composite task, without retraining.

**Compositional tasks.** The environment contains objects defined by two features: colour (red, blue, green) and shape (box, sphere). A primitive task is something like "find the red object" or "find the box". A compositional task adds a logical constraint, for example "find the red sphere", which requires satisfying both conditions simultaneously. Composition is implemented using the AND operator by taking the minimum of feature confidence maps or Q-values across primitives.

**Zero-shot generalisation.** The agents are never trained on any task involving the colour green. After training, they are evaluated on green tasks. Success here means the agent has learned representations that genuinely abstract over colour, rather than memorising specific object appearances.

---

## Algorithms Compared

Four algorithms are compared across the experiments.

| Algorithm | Type | Key Characteristic |
|-----------|------|--------------------|
| **SR** | Tabular | Learns expected future state occupancy; composes via spatial feature maps and the min operator |
| **DQN** | Neural network | Standard deep Q-network with a CNN visual backbone |
| **LSTM-DQN** | Neural network | DQN extended with an LSTM for temporal reasoning over stacked frames |
| **WVF** | Neural network | Task-conditioned Q-network that composes value functions directly at inference time |

The SR agent is the primary agent of interest in this project. The others serve as baselines to contextualise its performance.

---

## Repository Structure

The experiments are numbered to reflect the order in which they were developed. Each directory is largely self-contained with its own agents, environment wrappers, models, and results.

```
Masters-Research/
├── 2D/                         # Early 2D grid experiments (foundational work)
├── 2D_Partial_Observability/   # 2D with partial observation, path integration, and a vision model
├── 3D/                         # Initial 3D environment exploration
├── 3D_1Experimentation/        # First full 3D experiment with all four algorithms
├── 3D_2Composition/            # 3D compositional task learning with WITS cluster support
├── 3D_3Unseen_Goals/           # 3D zero-shot generalisation to unseen green objects (main results)
├── requirements.txt            # Python dependencies
└── ActivateVenv.txt            # Virtual environment activation note from development machine
```

### How Each Experiment Folder Is Organised

The pattern is consistent across all the major experiment directories. Each one contains:

- `agents/` - Agent class implementations specific to that experiment
- `env/` - Environment wrapper(s)
- `models/` - Vision models and saved weights
- `results/` or `experiment_results*/` - Output from training runs (plots, JSON files, saved models)
- `utils/` - Plotting and utility functions

The general workflow within each experiment is also consistent:

1. Individual baseline scripts (`main.py`, `dqn_baseline.py`, etc.) let you test a single algorithm in isolation and see that it is working.
2. An experiment runner (`experiment_runner.py` or `run_experiments.py`) runs all algorithms head to head, either locally or submitted to the WITS cluster.
3. An aggregation script (`aggregate_and_plot*.py`) collects results from multiple runs and seeds, then generates comparison plots and summary statistics.

---

## Experiment Progression

### 2D

The starting point. A 10x10 grid world where the agent navigates to goal objects with full observability of the environment. This is where the SR agent was first implemented and tested, and where the grid cell analysis tooling was built. There is no neural vision model at this stage.

### 2D Partial Observability

The same 2D environment, but the agent can only see a 7x7 first-person view of the grid rather than the full map. This makes the problem significantly harder. To compensate, agents at this stage incorporate **path integration**, which means maintaining an internal estimate of position by integrating velocity over time. This is similar to how animals use dead reckoning to navigate in the dark. A visual detection model is also introduced here, making this experiment considerably more end-to-end than the pure 2D work.

### 3D (Exploratory)

Initial work porting the problem to gym-miniworld, which is a continuous 3D first-person environment. This directory is mostly exploratory and not a full experiment in its own right.

### 3D Experimentation

The first full 3D experiment with all four algorithms compared together. The environment contains four objects: red box, blue box, red sphere, and blue sphere. This is where the 3D training infrastructure, vision model pipeline, and comparison plotting tools were solidified.

### 3D Composition

Extends 3D Experimentation with explicit compositional task evaluation. Agents are trained on primitive tasks (colour only, or shape only) and then evaluated on compositional tasks that require satisfying both colour and shape constraints simultaneously. This experiment introduces SLURM cluster submission via `submitit` for parallel multi-seed runs on the WITS cluster.

### 3D Unseen Goals

The main experiment. A green object is added to the environment (green box and green sphere), but the agent is never trained on any green task. After training on red and blue primitives, the agent is evaluated on four increasingly difficult scenarios:

1. Primitive tasks that were seen during training
2. Compositional AND tasks using seen colour and shape combinations
3. Green primitive tasks (unseen colour, zero-shot)
4. Green compositional tasks (unseen colour combined with a shape constraint, also zero-shot)

This directory also contains the **logical composition experiments**, which test whether agents can compose using OR and NOT operators in addition to AND. These evaluations are run against already-trained models rather than requiring a new training run.

---

## Environment and Dependencies

The project uses Python 3.10 or 3.11 with a virtual environment. All dependencies are pinned in `requirements.txt`. The key packages are:

| Package | Purpose |
|---------|---------|
| `miniworld`, `gymnasium` | 3D first-person environment |
| `torch`, `torchvision` | Neural network agents (DQN, LSTM, WVF) |
| `tensorflow-cpu` | Used in some earlier 2D agents |
| `opencv-python`, `pillow` | Image processing and the vision detector |
| `submitit` | SLURM cluster job submission |
| `matplotlib`, `seaborn` | Plotting and result visualisation |
| `numpy`, `pandas`, `scipy` | Numerical work and data handling |
| `pygame`, `pyglet`, `PyOpenGL` | Headless rendering for the 3D environment |

To set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA GPU and want hardware-accelerated training, replace the `torch` and `torchvision` lines in `requirements.txt` with the CUDA wheel for your driver version before running pip install. The CPU versions work fine for development and smaller runs.

The 3D experiments require either a display or a headless rendering setup. On a server or cluster without a monitor, OpenGL rendering is handled via OSMesa. The experiment scripts set all the necessary environment variables automatically before importing any graphics libraries, so you do not need to configure this manually.

The `ActivateVenv.txt` file in the root is a leftover from development on the original machine. You can ignore it.

---

## Where to Go Next

For a full walkthrough of how to run each experiment, replicate the results, and find where all the outputs are saved, see [EXPERIMENTS.md](EXPERIMENTS.md).
