#!/usr/bin/env python3
"""
Run multimodal evaluations for BabyLM 2025
"""
import subprocess
import os
import sys
from pathlib import Path
import argparse

def run_multimodal_evaluation(model_path, architecture="causal"):
    """Run multimodal evaluations"""
    eval_dir = Path("../evaluation-pipeline-2025")
    
    if not eval_dir.exists():
        print("Error: evaluation-pipeline-2025 directory not found")
        return False
    
    os.chdir(eval_dir)
    
    success = True
    
    # Run multimodal evaluations if scripts exist
    multimodal_scripts = [
        "eval_winoground.sh",
        "eval_vqa.sh"
    ]
    
    for script in multimodal_scripts:
        script_path = eval_dir / script
        if script_path.exists():
            print(f"Running {script}...")
            try:
                cmd = [str(script_path), model_path, architecture]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✓ {script} completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ {script} failed: {e}")
                print(f"stderr: {e.stderr}")
                success = False
        else:
            print(f"Script {script} not found, skipping...")
    
    # Run text-only tasks compatible with multimodal models
    print("Running text-only evaluations for multimodal model...")
    try:
        cmd = ["./eval_zero_shot_fast.sh", model_path, "main", architecture]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Text-only evaluations completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Text-only evaluations failed: {e}")
        success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Run BabyLM 2025 multimodal evaluations")
    parser.add_argument("model_path", help="Path to multimodal model")
    parser.add_argument("--architecture", default="causal", help="Model architecture")
    
    args = parser.parse_args()
    
    print(f"Starting multimodal evaluation for: {args.model_path}")
    success = run_multimodal_evaluation(args.model_path, args.architecture)
    
    if success:
        print("Multimodal evaluation completed successfully!")
    else:
        print("Multimodal evaluation completed with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()