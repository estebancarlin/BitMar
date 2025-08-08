# evaluation_runners/run_all_fast_evals.py
# filepath: evals-2025/evaluation_runners/run_all_fast_evals.py
"""
Run fast evaluations on all checkpoints using existing pipeline scripts
"""
import subprocess
import os
from pathlib import Path

def run_fast_eval_checkpoint(model_path, revision, architecture="causal"):
    """Run fast evaluation for a single checkpoint"""
    eval_script = Path("../evaluation-pipeline-2025/eval_zero_shot_fast.sh")
    
    cmd = [
        str(eval_script),
        model_path,
        revision,
        architecture
    ]
    
    print(f"Running fast eval for {model_path} revision {revision}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return False
    return True

def batch_run_fast_evals(model_path, architecture="causal"):
    """Run fast evals on all required checkpoints"""
    checkpoints = []
    # Add all required checkpoint revisions
    checkpoints.extend([f"chck_{i}M" for i in range(1, 11)])
    checkpoints.extend([f"chck_{i}M" for i in range(20, 101, 10)])
    # Add 100M+ if not strict-small track
    checkpoints.extend([f"chck_{i}M" for i in range(200, 1001, 100)])
    
    for checkpoint in checkpoints:
        success = run_fast_eval_checkpoint(model_path, checkpoint, architecture)
        if not success:
            print(f"Failed to evaluate checkpoint {checkpoint}")
            break

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    architecture = sys.argv[2] if len(sys.argv) > 2 else "causal"
    batch_run_fast_evals(model_path, architecture)