#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

def download_bitmar_model():
    """Download the BitMar attention multimodal model"""
    
    model_repo = "euhidaman/bitmar-attention-multimodal"
    cache_dir = "/work/u3191895/hf_models"
    local_dir = "/work/u3191895/babylm/BitMar/models/bitmar-attention-multimodal"
    
    print(f"Downloading {model_repo}...")
    print(f"Cache directory: {cache_dir}")
    print(f"Local directory: {local_dir}")
    
    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    
    # Download model
    model_path = snapshot_download(
        repo_id=model_repo,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Use actual files, not symlinks
    )
    
    print(f"Model successfully downloaded to: {model_path}")
    print("Contents:")
    os.system(f"ls -la {local_dir}")

if __name__ == "__main__":
    download_bitmar_model()
