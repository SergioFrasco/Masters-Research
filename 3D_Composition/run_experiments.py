"""
Parallel Experiment Runner using Submitit

Trains all algorithms (SR, DQN, LSTM-DQN, WVF) across multiple seeds
using SLURM cluster parallelization.

Usage:
    python run_parallel_experiments.py
    
After all jobs complete:
    python aggregate_and_plot.py
"""

import os

# Set environment variables for headless mode
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Removed duplicate
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]


import sys
import submitit
from pathlib import Path
import json
from datetime import datetime




import numpy as np
import torch

# Import training functions
from experiment_utils import (
    train_sr_agent,
    train_unified_dqn,
    train_unified_lstm_dqn,
    train_unified_wvf
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Easily adjustable number of seeds
NUM_SEEDS = 2

# Algorithms to compare
# ALGORITHMS = ["SR", "DQN", "LSTM", "WVF"]
ALGORITHMS = ["WVF"]

# Training configuration
TRAINING_CONFIG = {
    "training_episodes": 10000,
    "eval_episodes_per_task": 500,  # 300 per compositional task
    "max_steps": 200,
    "env_size": 10,
    
    # SR-specific
    "sr_freeze_episode": 3000,  # Freeze SR matrix at this episode
    
    # Shared hyperparameters
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
}

# SLURM configuration (based on your script)
SLURM_CONFIG = {
    "partition": "bigbatch",
    "time": 72 * 60,  # 72 hours in minutes
    "ntasks": 1,
    # Note: mem and cpus_per_task removed - cluster doesn't allow these specifications
}

# Output directory structure
BASE_OUTPUT_DIR = Path("experiment_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = BASE_OUTPUT_DIR / f"comparison_{TIMESTAMP}"


# ============================================================================
# TRAINING WRAPPER FUNCTION
# ============================================================================

def run_single_experiment(algorithm: str, seed: int, config: dict, output_dir: Path):
    """
    Train a single algorithm with a specific seed.
    
    This function will be submitted as a separate SLURM job for each
    (algorithm, seed) combination.
    
    Args:
        algorithm: One of ["SR", "DQN", "LSTM", "WVF"]
        seed: Random seed for reproducibility
        config: Training configuration dictionary
        output_dir: Where to save results
        
    Returns:
        dict: Results containing rewards, model paths, etc.
    """
    
    
    print(f"\n{'='*70}")
    print(f"STARTING JOB: Algorithm={algorithm}, Seed={seed}")
    print(f"{'='*70}\n")

    # Flush immediately
    sys.stdout.flush()
    
    # Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory for this specific run
    run_dir = output_dir / f"{algorithm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(run_dir / "config.json", 'w') as f:
        json.dump({
            "algorithm": algorithm,
            "seed": seed,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Train based on algorithm type
    try:
        if algorithm == "SR":
            results = train_sr_agent(
                seed=seed,
                training_episodes=config["training_episodes"],
                eval_episodes_per_task=config["eval_episodes_per_task"],
                max_steps=config["max_steps"],
                env_size=config["env_size"],
                sr_freeze_episode=config["sr_freeze_episode"],
                output_dir=run_dir
            )
        
        elif algorithm == "DQN":
            results = train_unified_dqn(
                seed=seed,
                training_episodes=config["training_episodes"],
                eval_episodes_per_task=config["eval_episodes_per_task"],
                max_steps=config["max_steps"],
                env_size=config["env_size"],
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                epsilon_decay=config["epsilon_decay"],
                output_dir=run_dir
            )
        
        elif algorithm == "LSTM":
            results = train_unified_lstm_dqn(
                seed=seed,
                training_episodes=config["training_episodes"],
                eval_episodes_per_task=config["eval_episodes_per_task"],
                max_steps=config["max_steps"],
                env_size=config["env_size"],
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                epsilon_decay=config["epsilon_decay"],
                output_dir=run_dir
            )
        
        elif algorithm == "WVF":
            results = train_unified_wvf(
                seed=seed,
                training_episodes=config["training_episodes"],
                eval_episodes_per_task=config["eval_episodes_per_task"],
                max_steps=config["max_steps"],
                env_size=config["env_size"],
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                epsilon_decay=config["epsilon_decay"],
                output_dir=run_dir
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Save results
        results_file = run_dir / "results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"JOB COMPLETE: Algorithm={algorithm}, Seed={seed}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*70}\n")
        
        return {
            "algorithm": algorithm,
            "seed": seed,
            "status": "success",
            "output_dir": str(run_dir),
            "results_file": str(results_file)
        }
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"JOB FAILED: Algorithm={algorithm}, Seed={seed}")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        
        # Save error information
        with open(run_dir / "error.txt", 'w') as f:
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Error: {str(e)}\n")
            
            import traceback
            f.write("\nTraceback:\n")
            f.write(traceback.format_exc())
        
        return {
            "algorithm": algorithm,
            "seed": seed,
            "status": "failed",
            "error": str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to launch all parallel experiments.
    """
    
    print(f"\n{'='*80}")
    print(f"PARALLEL EXPERIMENT LAUNCHER")
    print(f"{'='*80}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Seeds: {list(range(NUM_SEEDS))}")
    print(f"Total jobs: {len(ALGORITHMS) * NUM_SEEDS}")
    print(f"Output directory: {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")
    
    # Create experiment directory
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        "timestamp": TIMESTAMP,
        "algorithms": ALGORITHMS,
        "num_seeds": NUM_SEEDS,
        "seeds": list(range(NUM_SEEDS)),
        "training_config": TRAINING_CONFIG,
        "slurm_config": SLURM_CONFIG,
    }
    
    with open(EXPERIMENT_DIR / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Set up submitit executor
    log_dir = EXPERIMENT_DIR / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    executor = submitit.SlurmExecutor(folder=log_dir)
    
    # Update SLURM parameters (only what your cluster allows)
    executor.update_parameters(
        partition=SLURM_CONFIG["partition"],
        time=SLURM_CONFIG["time"],
        ntasks_per_node=SLURM_CONFIG["ntasks"],
        stderr_to_stdout=True,
    )
    
    print("Submitting jobs to SLURM cluster...")
    print(f"Logs will be saved to: {log_dir}\n")
    
    # Submit all jobs
    jobs = []
    job_info = []
    
    for algorithm in ALGORITHMS:
        for seed in range(NUM_SEEDS):
            job = executor.submit(
                run_single_experiment,
                algorithm=algorithm,
                seed=seed,
                config=TRAINING_CONFIG,
                output_dir=EXPERIMENT_DIR
            )
            
            jobs.append(job)
            job_info.append({
                "job_id": job.job_id,
                "algorithm": algorithm,
                "seed": seed
            })
            
            print(f"  ✓ Submitted: {algorithm} (seed={seed}) - Job ID: {job.job_id}")
    
    # Save job information
    with open(EXPERIMENT_DIR / "submitted_jobs.json", 'w') as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ALL JOBS SUBMITTED!")
    print(f"{'='*80}")
    print(f"Total jobs submitted: {len(jobs)}")
    print(f"Job IDs: {[job.job_id for job in jobs]}")
    print(f"\nTo check job status:")
    print(f"  squeue -u $USER")
    print(f"\nTo monitor a specific job:")
    print(f"  tail -f {log_dir}/<job_id>_0_log.out")
    print(f"\nAfter all jobs complete, run:")
    print(f"  python aggregate_and_plot.py {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")
    
    # Optional: Wait for all jobs to complete
    print("Waiting for jobs to complete...")
    print("(You can Ctrl+C to exit - jobs will continue running)")
    print("(Or keep this running to auto-generate plots when done)\n")
    
    try:
        results = []
        for i, job in enumerate(jobs):
            algo = job_info[i]["algorithm"]
            seed = job_info[i]["seed"]
            print(f"Waiting for {algo} (seed={seed}) - Job {job.job_id}...")
            
            result = job.result()  # Blocks until job completes
            results.append(result)
            
            if result["status"] == "success":
                print(f"  ✓ {algo} (seed={seed}) completed successfully")
            else:
                print(f"  ✗ {algo} (seed={seed}) FAILED: {result.get('error', 'Unknown error')}")
        
        # Save final results
        with open(EXPERIMENT_DIR / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"ALL JOBS COMPLETED!")
        print(f"{'='*80}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{len(results)}")
        
        # Automatically run aggregation
        print("\nRunning aggregation and plotting...")
        from aggregate_and_plot import aggregate_and_plot
        aggregate_and_plot(EXPERIMENT_DIR)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Jobs are still running on cluster.")
        print(f"Run aggregation later with:")
        print(f"  python aggregate_and_plot.py {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()