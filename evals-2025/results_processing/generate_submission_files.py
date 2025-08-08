# results_processing/generate_submission_files.py
# filepath: evals-2025/results_processing/generate_submission_files.py
"""
Generate submission files using existing collate_preds script
"""
import subprocess
from pathlib import Path
import os

def generate_submission(model_name, backend, fast=False):
    """Generate submission file using evaluation_pipeline.collate_preds"""
    os.chdir("../evaluation-pipeline-2025")
    
    cmd = [
        "python", "-m", "evaluation_pipeline.collate_preds",
        f"--model_path_or_name={model_name}",
        f"--backend={backend}"
    ]
    
    if fast:
        cmd.append("--fast")
        
    print(f"Generating {'fast' if fast else 'full'} submission for {model_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Submission file generated successfully!")
    else:
        print(f"Error: {result.stderr}")

def generate_all_submissions(model_name, backend="hf"):
    """Generate both fast and full submission files"""
    # Generate fast submission for checkpoints
    generate_submission(model_name, backend, fast=True)
    
    # Generate full submission for final model
    generate_submission(model_name, backend, fast=False)

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "hf"
    generate_all_submissions(model_name, backend)