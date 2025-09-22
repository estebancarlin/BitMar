# BitMar: Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** designed for the BabyLM Challenge. It combines BitNet-quantized text processing, DiNOv2 vision encoding, and episodic memory mechanisms to achieve efficient multimodal understanding with exactly 100M tokens.

## 🌟 Key Features

- **Token-Constrained Training**: Exactly 100M tokens with perfect alignment
- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **Episodic Memory**: Cross-modal memory system for visual-text associations
- **Comprehensive Logging**: Detailed WandB visualizations and metrics tracking
- **Automatic Evaluation**: Built-in evaluation pipelines for both 2024 and 2025 tracks
- **Hugging Face Integration**: Automatic model uploads after each epoch
- **Carbon Tracking**: Environmental impact monitoring

## 🛠️ Installation

```bash
git clone <your-repo-url>
cd BitMar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Training Commands

### Basic Training (100M Tokens)

```bash
# Standard training with all features enabled
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml

# Training with specific GPU device
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0

# Training with cache rebuild (if dataset changes)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --rebuild_cache

# Updated training without memory module for ablation study
python train_100M_tokens_no_memory.py --config configs/bitmar_100M_tokens_no_memory.yaml
```

### Training with Custom Checkpoint Frequency

```bash
# Save checkpoint every 1000 steps (in addition to epoch-based saves)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 1000

# Save checkpoint every 500 steps for frequent monitoring
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 500
```

### Training with Evaluation Control

```bash
# Enable fast evaluation after each epoch (default: enabled)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --enable_fast_eval

# Disable fast evaluation to speed up training
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_fast_eval

# Enable full evaluation at the end (default: enabled)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --enable_full_eval

# Disable full evaluation to save time
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_full_eval

# Custom evaluation setup
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_fast_eval --enable_full_eval
```

### Environment Variable Control

```bash
# Set evaluation flags via environment variables (useful for bash scripts)
export BITMAR_ENABLE_FAST_EVAL=true
export BITMAR_ENABLE_FULL_EVAL=false
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml
```

### Complete Training Command with All Options

```bash
python train_100M_tokens.py \
    --config configs/bitmar_100M_tokens.yaml \
    --device cuda:0 \
    --save_every_n_steps 1000 \
    --enable_fast_eval \
    --enable_full_eval
```

## 📊 WandB Logging & Visualizations

BitMar includes comprehensive logging to Weights & Biases with detailed visualizations and metrics tracking. Here's what gets logged and how to interpret it:

### 🎯 Training Metrics

**train/loss**
- **What**: Cross-entropy loss during training
- **X-axis**: Training steps
- **Y-axis**: Loss value
- **Interpretation**: Should decrease over time; sudden spikes indicate potential issues

**train/learning_rate**
- **What**: Learning rate schedule (cosine annealing with warm restarts)
- **X-axis**: Training steps
- **Y-axis**: Learning rate value
- **Interpretation**: Shows learning rate cycles; restarts help escape local minima

**train/cross_modal_similarity**
- **What**: Cosine similarity between text and vision features
- **X-axis**: Training steps
- **Y-axis**: Similarity score (-1 to 1)
- **Interpretation**: Higher values = better cross-modal alignment; key metric for multimodal understanding

### 📊 Token Tracking

**tokens/processed**
- **What**: Total number of tokens processed so far
- **X-axis**: Training steps
- **Y-axis**: Token count
- **Interpretation**: Should reach exactly 100M tokens; tracks progress toward target

**tokens/batch_size**
- **What**: Number of tokens in current batch
- **X-axis**: Training steps
- **Y-axis**: Token count per batch
- **Interpretation**: Shows batch size variation; should be relatively consistent

**token_progress/processed** and **token_progress/target**
- **What**: Progress tracking toward 100M token goal
- **X-axis**: Training steps
- **Y-axis**: Token counts
- **Interpretation**: Tracks completion percentage

### 📈 Epoch-Level Metrics

**epoch/train_loss**
- **What**: Average loss per epoch
- **X-axis**: Epoch number
- **Y-axis**: Loss value
- **Interpretation**: Should show steady decrease across epochs

**epoch/cross_modal_similarity**
- **What**: Average cross-modal similarity per epoch
- **X-axis**: Epoch number
- **Y-axis**: Similarity score
- **Interpretation**: Should increase as model learns better alignment

**epoch/tokens_processed** and **epoch/tokens_in_epoch**
- **What**: Token consumption tracking per epoch
- **X-axis**: Epoch number
- **Y-axis**: Token counts
- **Interpretation**: Shows token distribution across epochs

### 🤗 Hugging Face Integration Logs

**huggingface/upload_success**
- **What**: Whether model upload to HF Hub succeeded
- **X-axis**: Training steps
- **Y-axis**: Boolean (True/False)
- **Interpretation**: Tracks upload reliability

**huggingface/repo_url**
- **What**: Link to uploaded model repository
- **Interpretation**: Direct link to view uploaded models

### 🧪 Evaluation Results

**epoch_X/eval_2025_success** and **epoch_X/eval_2024_success**
- **What**: Success status of fast evaluation after each epoch
- **X-axis**: Training steps
- **Y-axis**: Boolean (True/False)
- **Interpretation**: Tracks evaluation pipeline health

**final/eval_2025_success** and **final/eval_2024_success**
- **What**: Success status of full evaluation at training end
- **Interpretation**: Final model evaluation results

### 🔧 Optional Advanced Metrics

The following comprehensive metrics are **now actively logged** to WandB during training:

**Memory Analysis**
- **Memory/Usage_Mean, Memory/Usage_Max, Memory/Usage_Min**: Episodic memory slot utilization statistics
- **Memory/Active_Slots_Percentage**: Percentage of memory slots being actively used  
- **Memory/Analysis_Avg_Similarity**: Average similarity between active memory slots (lower = more diverse)
- **Memory/Top_1_Slot_Access through Memory/Top_5_Slot_Access**: Access frequency for most-used memory slots

**Attention Analysis**
- **Attention/CrossModal_layer_X_Mean, Attention/CrossModal_layer_X_Max**: Cross-modal attention weights by layer
- **Attention/CrossModal_layer_X_Entropy**: Attention distribution entropy (lower = more focused)
- **Attention/Memory_Mean, Attention/Memory_Max, Attention/Memory_Entropy**: Memory attention patterns

**Quantization Metrics** ⚡
- **Quantization/BitNet_Layer_Count**: Number of quantized layers in the model
- **Quantization/Sparsity_Mean**: Average sparsity percentage across all BitNet layers (30-60% typical)
- **Quantization/Distribution_Balance**: Balance between +1/-1 weights (closer to 1.0 = better balance)
- **Quantization/Compression_Effectiveness**: Overall compression achieved through sparsity
- **Quantization/WeightScale_Mean/Std**: BitNet weight scaling factor statistics
- **Quantization/Zeros_Ratio_Mean/Std, Quantization/Ones_Ratio_Mean/Std, Quantization/NegOnes_Ratio_Mean/Std**: Aggregated ternary weight distribution statistics

**Gradient Analysis**
- **Gradients/Total_Norm**: L2 norm of all gradients (monitor for explosion/vanishing)
- **Gradients/Encoder_Norm, Gradients/Decoder_Norm, Gradients/Fusion_Norm, Gradients/Memory_Norm**: Component-wise gradient norms

**Feature Statistics**
- **Features/Text_Mean, Features/Text_Std, Features/Text_Norm**: Text feature representation statistics
- **Features/Vision_Mean, Features/Vision_Std, Features/Vision_Norm**: Vision feature statistics  
- **Features/Episode_Mean, Features/Episode_Std, Features/Episode_Norm**: Episodic memory feature statistics

> **Note**: These comprehensive metrics are logged every 100 training steps alongside the basic training metrics. This provides deep insights into model behavior, quantization efficiency, and memory utilization patterns.
