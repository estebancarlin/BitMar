# model_management/create_hf_repo_structure.py
# filepath: evals-2025/model_management/create_hf_repo_structure.py
"""
Create HuggingFace repository with branches for each checkpoint
Following: https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-mixed
"""
import os
from huggingface_hub import HfApi, create_repo

def create_babylm_repo(repo_name, private=False):
    """Create repo with required checkpoint branches"""
    checkpoints = []
    # 1M-10M checkpoints
    checkpoints.extend([f"chck_{i}M" for i in range(1, 11)])
    # 10M-100M checkpoints  
    checkpoints.extend([f"chck_{i}M" for i in range(20, 101, 10)])
    # 100M-1000M checkpoints (if not strict-small)
    checkpoints.extend([f"chck_{i}M" for i in range(200, 1001, 100)])
    
    api = HfApi()
    repo_id = create_repo(repo_name, private=private)
    
    for checkpoint in checkpoints:
        api.create_branch(repo_id, branch=checkpoint)
    
    print(f"Repository {repo_id} created with {len(checkpoints)} checkpoint branches")
    return repo_id