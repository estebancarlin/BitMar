from huggingface_hub import HfApi

def upload_fixed_modeling_file():
    api = HfApi()
    repo_id = "estebancarlin/bitmar-attention-multimodal"
    
    epochs = ["epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5", 
              "epoch-6", "epoch-7", "epoch-8", "epoch-9", "epoch-10"]
    
    print("🔄 Uploading fixed modeling_bitmar.py to all epochs...")
    
    for epoch in epochs:
        try:
            api.upload_file(
                path_or_fileobj="modeling_bitmar.py",
                path_in_repo="modeling_bitmar.py", 
                repo_id=repo_id,
                commit_message=f"Fix BitMarModel class and tensor shapes for {epoch}",
                revision=epoch
            )
            print(f"✅ Updated {epoch}")
        except Exception as e:
            print(f"❌ Failed to update {epoch}: {e}")
    
    # Also upload to main branch
    try:
        api.upload_file(
            path_or_fileobj="modeling_bitmar.py",
            path_in_repo="modeling_bitmar.py",
            repo_id=repo_id,
            commit_message="Fix BitMarModel class and tensor shapes for main",
            revision="main"
        )
        print("✅ Updated main branch")
    except Exception as e:
        print(f"❌ Failed to update main: {e}")
    
    print("\n🎉 All uploads complete!")

if __name__ == "__main__":
    upload_fixed_modeling_file()