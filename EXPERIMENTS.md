# Replicating the Experiments

This guide walks through every experiment in the repository from start to finish: what to run, in what order, and where to find the results when it is done. It assumes you have already set up the Python environment as described in [README.md](README.md).

If you are new to the project, read the README first. It explains what each experiment is trying to achieve, which will help you understand why the steps below are structured the way they are.

---

## Table of Contents

1. [Before You Start](#before-you-start)
2. [A Note on Where Things Are Stored](#a-note-on-where-things-are-stored)
3. [Experiment 1: 2D Grid](#experiment-1-2d-grid)
4. [Experiment 2: 2D Partial Observability](#experiment-2-2d-partial-observability)
5. [Experiment 3: 3D Baseline](#experiment-3-3d-baseline-3d_1experimentation)
6. [Experiment 4: 3D Composition](#experiment-4-3d-composition-3d_2composition)
7. [Experiment 5: 3D Unseen Goals](#experiment-5-3d-unseen-goals-3d_3unseen_goals)
8. [Running on the WITS Cluster](#running-on-the-wits-cluster)
9. [Quick Reference: Result Locations](#quick-reference-result-locations)

---

## Before You Start

### Step 1: Clone the repository

```bash
git clone https://github.com/SergioFrasco/Masters-Research.git
cd Masters-Research
```

### Step 2: Create a virtual environment and install dependencies

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install everything from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The installation may take a few minutes. The two largest downloads are PyTorch and TensorFlow.

If you are on a machine with a CUDA-capable GPU and want GPU support for PyTorch, replace the torch and torchvision lines in `requirements.txt` with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

### Step 3: Verify the environment

A quick sanity check to make sure the key packages are importable:

```bash
python -c "import torch; import gymnasium; import miniworld; print('All good')"
```

If that prints "All good" without errors, you are ready to run experiments.

### Step 4: Note on working directories

All scripts must be run from inside the relevant experiment directory, not from the root of the repository. Each experiment folder is self-contained and imports from its own local `agents/`, `env/`, and `models/` subdirectories. Running from the wrong location will cause import errors.

```bash
# For example, to run the 3D Unseen Goals experiment:
cd 3D_3Unseen_Goals
python run_experiments.py
```

---

## A Note on Where Things Are Stored

The repository structure can be a little inconsistent because it grew organically across several months. Here are the things worth knowing before you start digging around for outputs.

**Results from local runs** (running a single algorithm on your own machine) are almost always saved into a `results/` subfolder inside the experiment directory. For example, `3D_1Experimentation/results/`.

**Results from cluster runs** (multi-seed, multi-algorithm comparisons submitted to SLURM) are saved into a timestamped folder. In `3D_3Unseen_Goals`, this is `experiment_results_green/green_comparison_YYYYMMDD_HHMMSS/`. In `3D_2Composition`, it is `experiment_results/comparison_YYYYMMDD_HHMMSS/`. Each algorithm and seed gets its own subfolder inside that timestamp folder.

**Trained models** are saved alongside their results. For neural network agents (DQN, LSTM, WVF), a `model.pt` file is written into the algorithm's seed subfolder. For the SR agent, the learned SR matrix is saved as a `.npy` file in the same location.

**SLURM logs** from cluster runs are saved into a `slurm_logs/` subfolder within the timestamped experiment folder. If a job fails, this is the first place to look.

**Vision models** live in the `models/` folder within each experiment directory. The main file you need is `models/advanced_cube_detector.pth` (in the 3D experiments). This is a pre-trained checkpoint. You only need to retrain it if you are changing the object set or the visual environment.

---

## Experiment 1: 2D Grid

**Directory:** `2D/`

This is the earliest and simplest experiment. The environment is a fully observable 10x10 grid with four coloured objects. There is no vision model; the agent receives the ground truth state directly.

### Testing a single agent

To run the SR agent on its own:

```bash
cd 2D
python main.py
```

### Comparing all agents

To run all agents against each other and generate comparison plots:

```bash
cd 2D
python experiment_runner.py
```

### Grid cell analysis

After training, you can analyse the learned state representations:

```bash
python analyse_grid_cells.py
```

### Where results go

Plots and reward curves are saved to `2D/results/`. Grid cell analysis images go to `2D/grid_cell_analysis/` and `2D/grid_cell_images/`.

---

## Experiment 2: 2D Partial Observability

**Directory:** `2D_Partial_Observability/`

This experiment uses the same 10x10 grid but the agent can only see a 7x7 first-person window, not the full map. The agents here also incorporate a visual detection model and path integration (an internal estimate of position built up by tracking movement over time). This makes it considerably more complex than the pure 2D work.

### Testing individual agents

Each algorithm has its own standalone script:

```bash
cd 2D_Partial_Observability

# SR with Q-learning and path integration
python main_q_learning.py

# SR with SARSA
python main_sarsa.py

# Standard DQN
python dqn_baseline.py

# LSTM-DQN (uses frame stacking for temporal context)
python dqn_lstm_baseline.py

# World Value Function with LSTM
python wvf_lstm_baseline.py
```

Each of these runs a training loop and saves reward curves and any relevant visualisations to `2D_Partial_Observability/results/`.

### Comparing all agents

To run all five agents with multiple seeds and generate a unified comparison:

```bash
python experiment_runner.py
```

This will run sequentially on your local machine. It may take a while depending on the number of seeds configured at the top of the file (`num_seeds`). Results land in `2D_Partial_Observability/results/`.

### Grid cell analysis

To analyse the representations learned by the SR agent after training:

```bash
python analyse_grid_cells.py
```

Results go to `2D_Partial_Observability/grid_cell_analysis_random_walk/`.

### Hyperparameter sweeps

If you want to reproduce the hyperparameter tuning, the sweep configurations and logs are in `2D_Partial_Observability/hyperparamater_sweeps/`. The SLURM submission logs in `hyperparamater_sweeps/submitit_logs/` show exactly what was submitted to the cluster during tuning.

---

## Experiment 3: 3D Baseline (3D_1Experimentation)

**Directory:** `3D_1Experimentation/`

This is the first full 3D experiment. The environment is a MiniWorld room mapped to a 10x10 discrete grid, with four objects: red box, blue box, red sphere, and blue sphere. The agent receives a 60x80 RGB image as its observation.

### The vision model

The 3D agents rely on a pre-trained `CubeDetector` model that processes raw images and identifies which objects are present and roughly where they are. The checkpoint lives at `3D_1Experimentation/models/`. If you need to retrain it (for example, if you change the object set), first collect a dataset and then train:

```bash
cd 3D_1Experimentation
python collect_4_object_dataset.py
python train_vision.py
```

This is not necessary if you are using the existing checkpoint.

### Testing a single agent

```bash
cd 3D_1Experimentation
python main.py
```

The script at the top has configuration variables for which algorithm to run, number of episodes, and so on.

### Comparing all agents

```bash
python experiment_runner.py
```

This runs all four algorithms sequentially and saves comparison plots to `3D_1Experimentation/results/`.

### Debugging the WVF agent

If the WVF agent is behaving unexpectedly, there is a dedicated diagnostic script:

```bash
python diagnose_wvf.py
```

---

## Experiment 4: 3D Composition (3D_2Composition)

**Directory:** `3D_2Composition/`

This experiment introduces explicit compositional evaluation. Agents are trained on primitive tasks (find anything red, find any sphere, etc.) and then tested on tasks requiring both constraints (find the red sphere). The setup is otherwise the same as `3D_1Experimentation`.

### Running locally

```bash
cd 3D_2Composition
python main.py
```

### Running on the WITS cluster (recommended)

The multi-seed comparison is designed to run on the WITS SLURM cluster using `submitit`. From the cluster login node:

```bash
cd 3D_2Composition
python run_experiments.py
```

This submits one job per algorithm per seed to the `bigbatch` partition. Each job is allowed up to 72 hours. The script will print the job IDs and wait for completion. If a job fails (for example, due to a graphics initialisation issue on a particular node), it retries automatically up to five times.

Results accumulate in `3D_2Composition/experiment_results/comparison_YYYYMMDD_HHMMSS/`. Each job writes its output to a subfolder named `{ALGORITHM}_seed{N}/`.

### Generating plots after cluster runs

Once all jobs have finished:

```bash
python aggregate_and_plot.py experiment_results/comparison_YYYYMMDD_HHMMSS
```

Replace the timestamp with the actual folder name from your run. This generates reward curve plots and a bar chart comparing performance across algorithms and seeds.

---

## Experiment 5: 3D Unseen Goals (3D_3Unseen_Goals)

**Directory:** `3D_3Unseen_Goals/`

This is the main experiment and the most important one to understand. The environment now contains six objects by adding a green box and green sphere. The agents are never trained on any task involving green. After training, they are tested on four evaluation phases:

- Phase 1: Primitive tasks (seen during training)
- Phase 2: Compositional AND tasks (seen colour-shape combinations)
- Phase 3: Green primitive tasks (unseen colour, zero-shot)
- Phase 4: Green compositional tasks (unseen colour with shape constraint, zero-shot)

### The vision model

The pre-trained vision checkpoint for this experiment is at `3D_3Unseen_Goals/models/advanced_cube_detector.pth`. It was trained to detect all six objects including green.

If you ever need to retrain it from scratch:

```bash
cd 3D_3Unseen_Goals
python collect_6_object_dataset.py   # collects training images from the environment
python train_vision.py               # trains and saves the detector
```

This takes a while and should ideally be done on the cluster. The dataset is saved to `3D_3Unseen_Goals/dataset/`.

### Running the main experiment locally

For a quick single-algorithm test:

```bash
cd 3D_3Unseen_Goals
python main.py
```

The configuration at the top of `main.py` lets you pick the algorithm, number of episodes, and so on.

### Running on the WITS cluster (recommended for full results)

The full multi-seed comparison should be run on the cluster. From the login node:

```bash
cd 3D_3Unseen_Goals
python run_experiments.py
```

This submits one SLURM job per algorithm per seed to the `bigbatch` partition. The default configuration is:

- Algorithms: SR, DQN, LSTM, WVF
- Seeds: 2
- Training episodes: 20,000
- Evaluation episodes per task: 1,500
- Max steps per episode: 200
- Time limit per job: 72 hours

The script handles automatic retries if a job fails (up to 5 attempts). It also prints live progress as jobs complete.

Results are written to a new timestamped folder:

```
3D_3Unseen_Goals/experiment_results_green/green_comparison_YYYYMMDD_HHMMSS/
```

Inside that folder, each algorithm and seed gets its own directory:

```
green_comparison_YYYYMMDD_HHMMSS/
├── SR_seed0/
│   ├── config.json          # hyperparameters used for this run
│   └── results.json         # episode rewards across all phases
├── DQN_seed0/
│   ├── config.json
│   ├── results.json
│   └── model.pt             # trained network weights
├── LSTM_seed0/
│   └── ...
├── WVF_seed0/
│   └── ...
├── experiment_metadata.json # algorithms, seeds, task definitions
└── slurm_logs/              # SLURM submission and output logs per job
```

### Generating comparison plots after the cluster run

Once all jobs are done:

```bash
cd 3D_3Unseen_Goals
python aggregate_and_plot_green.py experiment_results_green/green_comparison_YYYYMMDD_HHMMSS
```

This script loads every `results.json` file, averages across seeds, and writes the following to the same experiment folder:

- `green_comparison_plot.png` - Full reward curves for all phases, with shaded standard deviation bands
- `green_bar_comparison.png` - Bar chart comparing mean performance per algorithm per evaluation phase
- `green_summary_statistics.txt` - Numerical summary (means, standard deviations) for every algorithm and phase

---

### Logical Composition Experiments

These experiments evaluate the trained models on a much wider range of logical tasks beyond AND, including OR, NOT, and complex nested combinations. No retraining is involved; the scripts load the weights from an existing experiment run and evaluate them directly.

There are two scripts, one for the SR agent only and one for all four agents.

#### SR agent only (AND, OR, NOT, complex)

```bash
cd 3D_3Unseen_Goals
python more_operators.py
```

Before running, open the file and check the path at the top points to your experiment directory and the seed you want to use. The relevant lines look like this:

```python
EXPERIMENT_DIR = Path("experiment_results_green/green_comparison_YYYYMMDD_HHMMSS")
SEED_TO_USE = 1
```

Update these to match your run. Results are written to `3D_3Unseen_Goals/logical_composition_results/`.

#### All four agents (AND, OR, NOT, complex)

```bash
python more_operators_all_agents.py
```

Again, open the file first and update the `EXPERIMENT_DIR` and `SEED_TO_USE` variables at the top to point to your experiment. The model paths are constructed automatically from those two variables:

```python
DQN_MODEL_PATH  = EXPERIMENT_DIR / f"DQN_seed{SEED_TO_USE}" / "model.pt"
LSTM_MODEL_PATH = EXPERIMENT_DIR / f"LSTM_seed{SEED_TO_USE}" / "model.pt"
WVF_MODEL_PATH  = EXPERIMENT_DIR / f"WVF_seed{SEED_TO_USE}" / "model.pt"
```

Results are written to `3D_3Unseen_Goals/logical_composition_all_algos/`.

#### Plotting the logical composition results

Once either (or both) of the above scripts have run and written their JSON output:

```bash
python plot_logical_compositions_all_agents.py
```

By default this reads from `logical_composition_all_algos/all_algo_logical_results.json`. You can pass a different path as an argument if needed:

```bash
python plot_logical_compositions_all_agents.py path/to/results.json
```

This generates three figures:

- Box plots per operator category (AND, OR, NOT, complex)
- An overall box plot across all tasks
- A grouped bar chart per individual task

All plots are saved into whichever results directory the JSON came from.

---

## Running on the WITS Cluster

The `3D_2Composition` and `3D_3Unseen_Goals` experiments are designed to run on the WITS SLURM cluster. Here is what happens under the hood when you run `python run_experiments.py`.

The script uses the `submitit` library, which wraps SLURM job submission in Python. When you run it, `submitit` constructs a shell script for each job, submits it to SLURM via `sbatch`, and then monitors the job queue for you. You do not need to write any SLURM scripts manually.

Each job runs a single (algorithm, seed) combination in isolation, which means up to 8 jobs run in parallel (4 algorithms x 2 seeds by default). The SLURM partition is `bigbatch` and each job is allowed 72 hours.

Because the cluster nodes do not have a display, all graphics rendering is done in software using OSMesa. The scripts set the required environment variables at the very top before importing any libraries:

```python
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["SDL_VIDEODRIVER"] = "dummy"
```

If a job crashes on a particular node (which sometimes happens due to graphics driver issues), the runner script catches the failure and resubmits automatically. It will try up to five times before giving up.

SLURM submission scripts and stdout logs are saved to `slurm_logs/` inside the timestamped experiment folder. If something goes wrong, checking `slurm_logs/{job_id}_submission.sh` and `slurm_logs/{job_id}.out` is usually the fastest way to diagnose it.

To run on the cluster, `ssh` into the login node and then:

```bash
# Navigate to the experiment
cd /path/to/Masters-Research/3D_3Unseen_Goals

# Activate the virtual environment
source ../.venv/bin/activate

# Submit all jobs
python run_experiments.py
```

Leave the terminal session alive (or use `tmux` or `screen`) while the jobs run. The script will poll for job completion and print updates. Once everything finishes, the results will be in `experiment_results_green/green_comparison_YYYYMMDD_HHMMSS/` as described above.

---

## Quick Reference: Result Locations

| Experiment | What you run | Where results go |
|------------|-------------|-----------------|
| 2D | `experiment_runner.py` | `2D/results/` |
| 2D grid cell analysis | `analyse_grid_cells.py` | `2D/grid_cell_analysis/` |
| 2D Partial Observability | `experiment_runner.py` | `2D_Partial_Observability/results/` |
| 2D PO grid cell analysis | `analyse_grid_cells.py` | `2D_Partial_Observability/grid_cell_analysis_random_walk/` |
| 3D Baseline | `experiment_runner.py` | `3D_1Experimentation/results/` |
| 3D Composition (cluster) | `run_experiments.py` then `aggregate_and_plot.py` | `3D_2Composition/experiment_results/comparison_TIMESTAMP/` |
| 3D Unseen Goals (cluster) | `run_experiments.py` then `aggregate_and_plot_green.py` | `3D_3Unseen_Goals/experiment_results_green/green_comparison_TIMESTAMP/` |
| Logical composition (SR only) | `more_operators.py` | `3D_3Unseen_Goals/logical_composition_results/` |
| Logical composition (all agents) | `more_operators_all_agents.py` | `3D_3Unseen_Goals/logical_composition_all_algos/` |
| Logical composition plots | `plot_logical_compositions_all_agents.py` | Same folder as the input JSON |
| Vision model dataset (3D) | `collect_6_object_dataset.py` | `3D_3Unseen_Goals/dataset/` |
| Trained vision model | `train_vision.py` | `3D_3Unseen_Goals/models/advanced_cube_detector.pth` |
| Trained agent weights | (written automatically during training) | `{ALGO}_seed{N}/model.pt` inside the experiment timestamp folder |
