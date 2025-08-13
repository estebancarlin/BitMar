#!/usr/bin/env python3
"""
Download specific epochs/commits from HuggingFace model repository
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_commit(repo_id: str, commit_hash: str, local_dir: str, epoch_name: str = None):
    """
    Download a specific commit from HuggingFace model repository
    
    Args:
        repo_id: HuggingFace repository ID
        commit_hash: Git commit hash or tag name
        local_dir: Local directory to save the model
        epoch_name: Optional name for the epoch (for folder naming)
    """
    
    if epoch_name:
        folder_name = f"epoch_{epoch_name}"
    else:
        folder_name = f"commit_{commit_hash[:8]}"
    
    download_path = os.path.join(local_dir, folder_name)
    
    logger.info(f"📥 Downloading {repo_id} @ {commit_hash} to {download_path}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=commit_hash,
            local_dir=download_path,
            local_dir_use_symlinks=False,
            cache_dir="/work/u3191895/hf_models"
        )
        
        logger.info(f"✅ Successfully downloaded to {download_path}")
        return download_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download {commit_hash}: {e}")
        return None

def main():
    repo_id = "euhidaman/bitmar-attention-multimodal"
    base_dir = "/work/u3191895/babylm/BitMar/models"
    
    # BitMar model epochs with actual commit hashes
    epochs_to_download = [
        {"commit": "f209f3a", "epoch": "10", "tokens": "996,862,184"},
        {"commit": "f5153ad", "epoch": "9", "tokens": "897,175,985"},
        {"commit": "cb7594d", "epoch": "8", "tokens": "797,489,669"},
        {"commit": "b404af1", "epoch": "7", "tokens": "697,803,570"},
        {"commit": "e2f4d65", "epoch": "6", "tokens": "598,117,273"},
        {"commit": "ce78b62", "epoch": "5", "tokens": "498,431,160"},
        {"commit": "97621f5", "epoch": "4", "tokens": "398,744,846"},
        {"commit": "9c651d9", "epoch": "3", "tokens": "299,058,679"},
        {"commit": "fa7c80f", "epoch": "2", "tokens": "199,372,346"},
        {"commit": "5be3c56", "epoch": "1", "tokens": "99,686,013"}
    ]
    
    logger.info(f"🚀 Starting download of {len(epochs_to_download)} epochs from {repo_id}")
    logger.info("📊 Epochs to download:")
    for epoch_info in epochs_to_download:
        logger.info(f"   Epoch {epoch_info['epoch']}: {epoch_info['tokens']} tokens ({epoch_info['commit']})")
    
    print("=" * 60)
    
    successful_downloads = []
    failed_downloads = []
    
    for i, epoch_info in enumerate(epochs_to_download, 1):
        commit = epoch_info["commit"]
        epoch = epoch_info["epoch"]
        tokens = epoch_info["tokens"]
        
        print(f"\n[{i}/{len(epochs_to_download)}] Processing Epoch {epoch}")
        print(f"Tokens processed: {tokens}")
        print(f"Commit hash: {commit}")
        
        download_path = download_model_commit(
            repo_id=repo_id,
            commit_hash=commit,
            local_dir=base_dir,
            epoch_name=epoch
        )
        
        if download_path:
            # Verify the download
            config_path = os.path.join(download_path, "config.json")
            weights_path = os.path.join(download_path, "pytorch_model.bin")
            
            if os.path.exists(config_path) and os.path.exists(weights_path):
                logger.info(f"✅ Epoch {epoch} verified: config.json and pytorch_model.bin found")
                successful_downloads.append(epoch)
                
                # Check file sizes
                config_size = os.path.getsize(config_path)
                weights_size = os.path.getsize(weights_path)
                logger.info(f"   Files: config.json ({config_size:,} bytes), pytorch_model.bin ({weights_size:,} bytes)")
                
            else:
                logger.warning(f"⚠️  Epoch {epoch} incomplete: missing files")
                failed_downloads.append(epoch)
        else:
            failed_downloads.append(epoch)
        
        print("-" * 50)
    
    # Summary
    print("\n" + "=" * 60)
    logger.info("🎉 Download process completed!")
    logger.info(f"✅ Successfully downloaded: {len(successful_downloads)} epochs")
    if successful_downloads:
        logger.info(f"   Epochs: {', '.join(successful_downloads)}")
    
    if failed_downloads:
        logger.warning(f"❌ Failed downloads: {len(failed_downloads)} epochs")
        logger.warning(f"   Epochs: {', '.join(failed_downloads)}")
    
    # Show final directory structure
    logger.info(f"\n📁 Final directory structure in {base_dir}:")
    try:
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                logger.info(f"   📂 {item}/")
    except Exception as e:
        logger.warning(f"Could not list directory: {e}")

if __name__ == "__main__":
    main()