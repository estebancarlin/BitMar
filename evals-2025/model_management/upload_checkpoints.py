#!/usr/bin/env python3
"""
Upload model checkpoints to HuggingFace Hub with proper branching structure
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
import argparse

def get_checkpoint_list(track="strict-small"):
    """Get list of required checkpoints based on track"""
    checkpoints = []
    
    # 1M-10M checkpoints (all tracks)
    checkpoints.extend([f"chck_{i}M" for i in range(1, 11)])
    
    # 20M-100M checkpoints (all tracks)  
    checkpoints.extend([f"chck_{i}M" for i in range(20, 101, 10)])
    
    # 200M-1000M checkpoints (only for non-strict tracks)
    if track != "strict-small":
        checkpoints.extend([f"chck_{i}M" for i in range(200, 1001, 100)])
    
    return checkpoints

def upload_checkpoint(local_path, repo_id, branch_name):
    """Upload a single checkpoint to specific branch"""
    if not os.path.exists(local_path):
        print(f"Warning: Checkpoint path {local_path} does not exist")
        return False
    
    api = HfApi()
    
    try:
        print(f"Uploading {local_path} to {repo_id} branch {branch_name}")
        upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            revision=branch_name,
            commit_message=f"Upload checkpoint {branch_name}",
            ignore_patterns=["*.git*", "__pycache__", "*.pyc"]
        )
        print(f"Successfully uploaded {branch_name}")
        return True
    except Exception as e:
        print(f"Error uploading {branch_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload BabyLM checkpoints to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repository ID (username/repo-name)")
    parser.add_argument("--checkpoints_dir", required=True, help="Local directory containing checkpoints")
    parser.add_argument("--track", default="strict-small", choices=["strict-small", "strict", "multimodal"], 
                       help="BabyLM track type")
    
    args = parser.parse_args()
    
    checkpoints = get_checkpoint_list(args.track)
    checkpoints_dir = Path(args.checkpoints_dir)
    
    success_count = 0
    for checkpoint in checkpoints:
        checkpoint_path = checkpoints_dir / checkpoint
        if upload_checkpoint(str(checkpoint_path), args.repo_id, checkpoint):
            success_count += 1
    
    print(f"Upload complete: {success_count}/{len(checkpoints)} checkpoints uploaded successfully")

if __name__ == "__main__":
    main()