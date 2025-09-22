import torch
import os
import json
import yaml

# ------------------- Paths -------------------
checkpoint_path = "/work/u3191895/babylm/BitMar/checkpoints_100M_dataset/latest_checkpoint.pt"
config_bitmar_path = "/work/u3191895/babylm/BitMar/configs/bitmar_100M_tokens_no_memory.yaml"
output_dir = "hf_no_memory"

# ------------------- Load Config -------------------
with open(config_bitmar_path, "r") as f:
    raw_config = yaml.safe_load(f)

config_model = raw_config["model"]

# ------------------- Load Checkpoint -------------------
checkpoint = torch.load(checkpoint_path, map_location="cpu")

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# ------------------- Save Hugging Face Format -------------------
os.makedirs(output_dir, exist_ok=True)

# Save raw weights (avoid shape mismatch)
torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

# Hugging Face config.json
hf_config = {
    "architectures": ["BitMarModel"],
    "model_type": "bitmar",
    "vocab_size": config_model["vocab_size"],
    "text_encoder_dim": config_model["text_encoder_dim"],
    "text_encoder_layers": config_model["text_encoder_layers"],
    "text_encoder_heads": config_model["text_encoder_heads"],
    "vision_encoder_dim": config_model["vision_encoder_dim"],
    "vision_latent_size": config_model["vision_latent_size"],
    "fusion_hidden_size": config_model["fusion_hidden_size"],
    "max_seq_len": config_model["max_seq_len"],
    "dropout": config_model["dropout"],
    "torch_dtype": "float32"
}

with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(hf_config, f, indent=2)

# Training metadata (extra info from checkpoint)
training_metadata = {
    "epoch": checkpoint.get("epoch"),
    "global_step": checkpoint.get("global_step"),
    "tokens_processed": checkpoint.get("tokens_processed"),
    "target_tokens": checkpoint.get("target_tokens"),
    "best_similarity": checkpoint.get("best_similarity"),
    "training_config": raw_config
}

with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
    json.dump(training_metadata, f, indent=2)

print(f"✅ Saved Hugging Face-style model in {output_dir}/")
