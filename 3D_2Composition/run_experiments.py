"""
Parallel Experiment Runner using Submitit (WITH AUTOMATIC RETRY)

Trains all algorithms (SR, DQN, LSTM-DQN, WVF) across multiple seeds
using SLURM cluster parallelization.

This version automatically retries failed jobs (e.g., due to node graphics issues).

Usage:
    python run_parallel_experiments_with_retry.py
    
After all jobs complete:
    python aggregate_and_plot.py
"""

import os
import sys
import submitit
from pathlib import Path
import json
from datetime import datetime
import time
import numpy as np
import torch

# Delay between job submissions (seconds) to avoid race conditions
SUBMISSION_DELAY = 3

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Easily adjustable number of seeds
NUM_SEEDS = 2

# Algorithms to compare
ALGORITHMS = ["SR", "DQN", "LSTM", "WVF"]

# Retry configuration
MAX_RETRIES = 5  # Maximum number of times to retry a failed job
RETRY_DELAY = 30  # Seconds to wait before resubmitting

# Training configuration
TRAINING_CONFIG = {
    "training_episodes": 20000,
    "eval_episodes_per_task": 1500,
    "max_steps": 200,
    "env_size": 10,
    "sr_freeze_episode": 3000,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
}

# SLURM configuration
SLURM_CONFIG = {
    "partition": "bigbatch",
    "time": 72 * 60,  # 72 hours in minutes
    "ntasks": 1,
}

# Output directory structure
BASE_OUTPUT_DIR = Path("experiment_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = BASE_OUTPUT_DIR / f"comparison_{TIMESTAMP}"


# ============================================================================
# WRAPPER FUNCTION THAT SETS ENV VARS BEFORE ANY IMPORTS
# ============================================================================

def run_single_experiment(algorithm: str, seed: int, config: dict, output_dir: Path):
    """
    Train a single algorithm with a specific seed.
    
    IMPORTANT: This function sets environment variables BEFORE importing
    any graphics-related modules to ensure headless mode works.
    """
    import os
    
    # Set environment variables FIRST, before any imports
    os.environ["MINIWORLD_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "True"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Remove DISPLAY if it exists
    if "DISPLAY" in os.environ:
        del os.environ["DISPLAY"]
    
    # Also try setting EGL as fallback
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    # Now do the imports
    import sys
    import numpy as np
    import torch
    from pathlib import Path
    import json
    from datetime import datetime
    
    # Import training functions (these will import the graphics libraries)
    from experiment_utils import (
        train_sr_agent,
        train_unified_dqn,
        train_unified_lstm_dqn,
        train_unified_wvf
    )
    
    print(f"\n{'='*70}")
    print(f"STARTING JOB: Algorithm={algorithm}, Seed={seed}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    
    # Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Convert output_dir to Path if it's a string
    output_dir = Path(output_dir)
    
    # Create output directory for this specific run
    run_dir = output_dir / f"{algorithm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(run_dir / "config.json", 'w') as f:
        json.dump({
            "algorithm": algorithm,
            "seed": seed,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "node": os.environ.get('SLURMD_NODENAME', 'unknown')
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
        
        import traceback
        
        with open(run_dir / "error.txt", 'w') as f:
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("\nTraceback:\n")
            f.write(traceback.format_exc())
        
        # Re-raise the exception so submitit knows the job failed
        raise


# ============================================================================
# JOB MANAGEMENT WITH RETRY LOGIC
# ============================================================================

class JobManager:
    """Manages job submission and automatic retries."""
    
    def __init__(self, experiment_dir, slurm_config, max_retries=5, retry_delay=30):
        self.experiment_dir = Path(experiment_dir)
        self.slurm_config = slurm_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.log_dir = self.experiment_dir / "slurm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track job attempts
        self.job_attempts = {}  # key: (algorithm, seed), value: attempt count
        self.failed_nodes = set()  # Track nodes that have failed
    
    def create_executor(self):
        """Create a new submitit executor."""
        executor = submitit.SlurmExecutor(folder=self.log_dir)
        executor.update_parameters(
            partition=self.slurm_config["partition"],
            time=self.slurm_config["time"],
            ntasks_per_node=self.slurm_config["ntasks"],
            stderr_to_stdout=True,
        )
        return executor
    
    def submit_job(self, algorithm, seed, config, output_dir):
        """Submit a single job."""
        key = (algorithm, seed)
        self.job_attempts[key] = self.job_attempts.get(key, 0) + 1
        
        executor = self.create_executor()
        
        # If we have failed nodes, try to exclude them
        if self.failed_nodes:
            try:
                exclude_str = ",".join(self.failed_nodes)
                executor.update_parameters(exclude=exclude_str)
                print(f"  Excluding nodes: {exclude_str}")
            except Exception as e:
                print(f"  Warning: Could not exclude nodes: {e}")
        
        job = executor.submit(
            run_single_experiment,
            algorithm=algorithm,
            seed=seed,
            config=config,
            output_dir=output_dir
        )
        
        return job
    
    def run_with_retries(self, algorithms, seeds, config, output_dir):
        """
        Submit all jobs and automatically retry failed ones.
        
        Returns:
            list: Final results for all jobs
        """
        # Initial submission
        pending_jobs = {}  # job_id -> (job, algorithm, seed)
        completed_results = []
        
        print("Submitting initial jobs...")
        for algorithm in algorithms:
            for seed in seeds:
                job = self.submit_job(algorithm, seed, config, output_dir)
                pending_jobs[job.job_id] = (job, algorithm, seed)
                print(f"  ✓ Submitted: {algorithm} (seed={seed}) - Job ID: {job.job_id}")
                
                # Stagger submissions to avoid race conditions during initialization
                time.sleep(SUBMISSION_DELAY)
        
        print(f"\nMonitoring {len(pending_jobs)} jobs...")
        print("(Ctrl+C to exit - but jobs will continue running)\n")
        
        while pending_jobs:
            jobs_to_remove = []
            jobs_to_retry = []
            
            for job_id, (job, algorithm, seed) in pending_jobs.items():
                try:
                    # Check if job is done (non-blocking check)
                    if job.done():
                        try:
                            result = job.result()
                            print(f"✓ {algorithm} (seed={seed}) completed successfully")
                            completed_results.append(result)
                            jobs_to_remove.append(job_id)
                        except Exception as e:
                            error_str = str(e)
                            print(f"✗ {algorithm} (seed={seed}) FAILED: {error_str[:100]}...")
                            
                            # Try to extract the node name from the error
                            if "SLURMD_NODENAME" in error_str or "node" in error_str.lower():
                                # Try to parse node name from logs
                                try:
                                    log_files = list(self.log_dir.glob(f"{job_id}*"))
                                    for log_file in log_files:
                                        content = log_file.read_text()
                                        if "SLURMD_NODENAME" in content:
                                            # Extract node name
                                            import re
                                            match = re.search(r'SLURMD_NODENAME[=:]\s*(\S+)', content)
                                            if match:
                                                failed_node = match.group(1)
                                                self.failed_nodes.add(failed_node)
                                                print(f"  Added {failed_node} to exclusion list")
                                except:
                                    pass
                            
                            # Check if we should retry
                            key = (algorithm, seed)
                            attempts = self.job_attempts.get(key, 1)
                            
                            if attempts < self.max_retries:
                                jobs_to_retry.append((algorithm, seed))
                            else:
                                print(f"  Max retries ({self.max_retries}) reached for {algorithm} (seed={seed})")
                                completed_results.append({
                                    "algorithm": algorithm,
                                    "seed": seed,
                                    "status": "failed",
                                    "error": str(e)
                                })
                            
                            jobs_to_remove.append(job_id)
                            
                except Exception as e:
                    print(f"Error checking job {job_id}: {e}")
            
            # Remove completed/failed jobs
            for job_id in jobs_to_remove:
                del pending_jobs[job_id]
            
            # Retry failed jobs
            for algorithm, seed in jobs_to_retry:
                key = (algorithm, seed)
                attempt = self.job_attempts.get(key, 0) + 1
                print(f"\n🔄 Retrying {algorithm} (seed={seed}) - Attempt {attempt}/{self.max_retries}")
                time.sleep(self.retry_delay)
                
                job = self.submit_job(algorithm, seed, config, output_dir)
                pending_jobs[job.job_id] = (job, algorithm, seed)
                print(f"  ✓ Resubmitted: Job ID {job.job_id}")
            
            if pending_jobs:
                # Wait before checking again
                time.sleep(30)
        
        return completed_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to launch all parallel experiments with retry logic."""
    
    print(f"\n{'='*80}")
    print(f"PARALLEL EXPERIMENT LAUNCHER (WITH AUTOMATIC RETRY)")
    print(f"{'='*80}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Seeds: {list(range(NUM_SEEDS))}")
    print(f"Total jobs: {len(ALGORITHMS) * NUM_SEEDS}")
    print(f"Max retries per job: {MAX_RETRIES}")
    print(f"Submission delay: {SUBMISSION_DELAY} seconds between jobs")
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
        "max_retries": MAX_RETRIES,
    }
    
    with open(EXPERIMENT_DIR / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create job manager
    job_manager = JobManager(
        experiment_dir=EXPERIMENT_DIR,
        slurm_config=SLURM_CONFIG,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY
    )
    
    try:
        # Run all jobs with automatic retries
        results = job_manager.run_with_retries(
            algorithms=ALGORITHMS,
            seeds=list(range(NUM_SEEDS)),
            config=TRAINING_CONFIG,
            output_dir=EXPERIMENT_DIR
        )
        
        # Save final results
        with open(EXPERIMENT_DIR / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"\n{'='*80}")
        print(f"ALL JOBS COMPLETED!")
        print(f"{'='*80}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        
        if job_manager.failed_nodes:
            print(f"Problematic nodes: {job_manager.failed_nodes}")
        
        # Automatically run aggregation if we have successful results
        if successful > 0:
            print("\nRunning aggregation and plotting...")
            try:
                from aggregate_and_plot import aggregate_and_plot
                aggregate_and_plot(EXPERIMENT_DIR)
            except ImportError:
                print("aggregate_and_plot.py not found. Run it manually:")
                print(f"  python aggregate_and_plot.py {EXPERIMENT_DIR}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Some jobs may still be running on the cluster.")
        print(f"Run aggregation later with:")
        print(f"  python aggregate_and_plot.py {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()