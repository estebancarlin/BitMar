# BitMar: Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** that combines BitNet-quantized text processing, DiNOv2 vision encoding, and Larimar's episodic memory mechanism for the BabyLM Challenge. The model maintains cross-modal episodic memories to improve zero-shot image-language understanding.

## 🎯 Project Overview

BitMar addresses the challenge of grounding language in visual experience through episodic memory. Unlike traditional vision-language models that process images and text separately, BitMar stores cross-modal episodes (visual-textual experiences) in a persistent memory bank that can be retrieved during inference.

### Core Innovation
- **Episodic Memory Grounding**: Maintains 48-512 episodic memory slots containing cross-modal associations
- **BitNet Quantization**: Uses 1.58-bit quantized weights {-1, 0, +1} for extreme efficiency
- **DiNOv2 Vision**: Leverages pre-computed DiNOv2 features for robust visual understanding
- **Cross-Modal Learning**: Two parallel learning trajectories (text and vision) that converge during training

## 🏗️ Architecture Overview

```text
Text Input (256 tokens) → BitNet Text Encoder (128D, 4 layers, 4 heads)
                                        ↓
Vision Input (DiNOv2) → Learned Compression (64D) → Spatial Pooling (7x7)
                                        ↓
                          Cross-Modal Fusion (128D, 3 layers, 6 heads)
                                        ↓
                            Multimodal Latent (128D)
                                        ↓
                        Episodic Memory (48 slots, 128D each)
                                        ↓
                         BitNet Decoder → Generated Text
```

### Key Components

#### 1. BitNet Text Processing
- **Encoder**: 4-layer transformer with 1.58-bit quantized weights
- **Dimensions**: 128D hidden size, 4 attention heads per layer
- **Quantization**: Ternary weights {-1, 0, +1} with 8-bit activations
- **Efficiency**: ~90% memory reduction compared to full-precision models

#### 2. Vision Processing Pipeline
- **Input**: Pre-computed DiNOv2 features (768D → 64D compression)
- **Spatial Pooling**: 14x14 → 7x7 attention-based pooling
- **Method**: Learned compression with spatial information preservation
- **Output**: 64D vision features ready for cross-modal fusion

#### 3. Cross-Modal Fusion
- **Architecture**: 3-layer transformer with 6 attention heads
- **Purpose**: Align text and vision representations in shared 128D space
- **Innovation**: Dual trajectory learning with text starting lower (~0.2) and vision higher (~0.35)

#### 4. Episodic Memory System
- **Capacity**: 48 memory slots (optimized for efficiency)
- **Content**: Cross-modal episode embeddings (128D each)
- **Access**: Attention-based retrieval during inference
- **Updates**: Gradient-based memory writing with diversity preservation

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended
- Wandb account (for enhanced logging)

## 🚀 Complete Setup Guide

### Step 1: Download Evaluation Pipelines

First, download both BabyLM evaluation pipelines (2024 and 2025):

```bash
# Clone both evaluation pipelines
git clone https://github.com/babylm/evaluation-pipeline-2025.git
git clone https://github.com/babylm/evaluation-pipeline-2024.git

# Setup 2025 pipeline
cd evaluation-pipeline-2025
pip install -r requirements.txt

curl -L "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip=" -o full_eval.zip

# Extract using Python
python3 -c "
import zipfile
import os
os.makedirs('evaluation_data', exist_ok=True)
with zipfile.ZipFile('full_eval.zip', 'r') as zip_ref:
    zip_ref.extractall('evaluation_data/')
print('✅ Evaluation data extracted!')
"
rm full_eval.zip

# Create empty evaluation_data directory for 2024 pipeline (not actually needed)
mkdir -p evaluation-pipeline-2024/evaluation_data

# 5. Try to download EWoK data (optional - may not be needed)
python -m evaluation_pipeline.ewok.dl_and_filter || echo "⚠️  EWoK download failed - this is OK, will be handled during evaluation"

# 6. Install dependencies for 2025 pipeline
pip install -r requirements.txt
cd ..

# 7. Install dependencies for 2024 pipeline
cd evaluation-pipeline-2024
pip install -e .
pip install minicons lm_eval[all]
cd ..

```

### Step 2: Clone BitMar Repository

```bash
# Clone BitMar repository
git clone https://github.com/euhidaman/BitMar.git
git checkout stable1
cd BitMar

# Install BitMar requirements
pip install -r requirements.txt

# Fix pytorch issues,if any exists- PyTorch (make sure to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

```

### Step 3: Download BabyLM Dataset

```bash
# Download BabyLM multimodal dataset
python download_babylm_data.py
```

### Step 4: Verify Setup (Eval part may fail right now)

```bash
# Validate evaluation setup
python validate_evaluation_setup.py
```

### Step 5: Hugging Face and wandb Login (Optional)

```bash
# Hugging Face login
huggingface-cli login

# Set Wandb API key for logging
export WANDB_API_KEY="your_api_key_here"
```

## 📊 Training & Evaluation Pipeline

### Training Options

#### 1. Full 100M Token Training (Recommended)

```bash
# Set Wandb API key for logging
export WANDB_API_KEY="your_api_key_here"

# Train with 100M tokens and enhanced cross-modal tracking
python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml

# Monitor training progress at: https://wandb.ai/babylm-ntust/bitmar-100M-optimized-memory
```

#### 2. Quick Test Training

```bash
# Quick test (1 epoch, small batch)
python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml --max_epochs 1 --batch_size 4
```

#### 3. Adaptive Training with Memory Optimization

```bash
# Train with adaptive training controller
python train_adaptive_bitmar.py --config configs/bitmar_100M_tokens_optimized_memory.yaml --rebuild_cache
```

### Evaluation Commands

#### 1. BitMar Model Evaluation

```bash
# Evaluate trained BitMar model
python evaluate_bitmar.py --model_path outputs/checkpoints/latest_checkpoint.pt --config configs/bitmar_100M_tokens_optimized_memory.yaml

# Run all text benchmarks
python run_all_benchmarks.py

# Quick benchmark testing
python test_quick_benchmarks.py
```

#### 2. BabyLM Challenge Evaluation

```bash
# Convert BitMar to HuggingFace format
python scripts/convert_to_hf.py --checkpoint outputs/checkpoints/latest_checkpoint.pt --output_dir ./hf_model

# Run BabyLM 2025 evaluation pipeline
cd ../evaluation-pipeline-2025

# Fast evaluation (for testing)
python -m evaluation_pipeline.run_evaluation --model_path ../BitMar/hf_model --eval_type fast

# Full evaluation (for submission)
python -m evaluation_pipeline.run_evaluation --model_path ../BitMar/hf_model --eval_type full

# Run BabyLM 2024 evaluation pipeline (if needed)
cd ../evaluation-pipeline-2024
# Follow 2024 pipeline evaluation commands
```

#### 3. Specific Task Evaluations

```bash
# Text-only benchmarks
./eval_zero_shot_fast.sh ../BitMar/hf_model

# Fine-tuning tasks
./eval_finetuning.sh ../BitMar/hf_model

# Age of Acquisition evaluation
./eval_aoa.sh ../BitMar/hf_model

# DevBench evaluation
./eval_devbench.sh ../BitMar/hf_model
```

## 📈 Comprehensive Wandb Visualizations

BitMar creates an extensive set of visualizations and metrics in Wandb, organized into clear categories:

### 1. Training Metrics Dashboard

**Core Training Metrics:**
- `Training/Loss` - Training loss over training steps (X: steps, Y: loss value)
- `Training/Learning_Rate` - Learning rate schedule progression (X: steps, Y: learning rate)
- `Training/Epoch` - Current epoch number
- `Training/Tokens_Processed` - Total tokens processed during training

**How to Interpret:**
- Training loss should generally decrease over time
- Learning rate follows cosine annealing schedule with restarts
- Monitor tokens processed to ensure exact 100M token training

### 2. Cross-Modal Learning Trajectories (NEW!)

**Dual Learning Visualization:**
- `Cross-Modal Trajectories/Text Learning` - Text learning progression (orange line)
- `Cross-Modal Trajectories/Vision Learning` - Vision learning progression (blue line)

**How to Interpret:**
- **Text Learning**: Starts lower (~0.2) and gradually increases, representing text understanding development
- **Vision Learning**: Starts higher (~0.35) and increases, representing visual feature learning
- **Convergence**: Both trajectories should converge upward over training, indicating aligned cross-modal understanding
- **Ideal Pattern**: Text line starts below vision line, both ascend, and gap narrows as training progresses

### 3. Memory Analysis Suite

**Memory Utilization Metrics:**
- `Memory/Usage_Mean` - Average memory slot utilization (0-1 scale)
- `Memory/Active_Slots_Percentage` - Percentage of memory slots actively used
- `Memory/Diversity_Score` - Specialization diversity across memory slots
- `Memory/Slot_Utilization_Distribution` - Distribution of memory slot usage

**Memory Evolution Visualizations:**
- `Memory/Evolution_Heatmap` - Memory slot changes over epochs (X: slot ID, Y: epoch)
- `Memory/Access_Patterns` - Which memory slots are accessed most frequently
- `Memory/Specialization_Trends` - How memory slots become specialized

**How to Interpret:**
- **High Memory Usage (>70%)**: Good utilization of episodic memory capacity
- **High Diversity Score (>0.5)**: Memory slots are specialized for different content types
- **Evolving Heatmap**: Colors should change over training, indicating memory adaptation
- **Access Patterns**: Some slots should be accessed more frequently (hot spots)

### 4. Attention Pattern Analysis

**Attention Head Tracking:**
- `Attention/CrossModal_layer_X_Entropy` - Cross-modal attention entropy by layer
- `Attention/Memory_Mean` - Average attention weights to memory slots
- `Attention/Head_Importance_Rankings` - Most important attention heads over time

**Attention Visualizations:**
- Attention head heatmaps (X: attention heads, Y: importance scores)
- Timeline evolution plots (X: training steps, Y: attention scores)
- Cross-modal attention patterns between text and vision

**How to Interpret:**
- **Low Entropy**: Focused attention patterns (good for specialized tasks)
- **High Entropy**: Distributed attention (good for general understanding)
- **Memory Attention**: Should increase over training as memory becomes more useful
- **Head Rankings**: Important heads should remain stable across training

### 5. Feature Statistics

**Feature Distribution Metrics:**
- `Features/Text_Mean, Text_Std, Text_Norm` - Text feature statistics
- `Features/Vision_Mean, Vision_Std, Vision_Norm` - Vision feature statistics
- `Features/CrossModal_Similarity` - Cosine similarity between text and vision features

**How to Interpret:**
- **Feature Norms**: Should be stable (not exploding or vanishing)
- **Cross-Modal Similarity**: Should increase over training (0.0 to 1.0)
- **Standard Deviation**: Indicates feature distribution spread

### 6. BitNet Quantization Analysis

**Quantization Metrics:**
- `Quantization/WeightScale_encoder, WeightScale_decoder` - BitNet scaling factors
- `Quantization/Sparsity_encoder, Sparsity_decoder` - Ternary weight sparsity
- `Quantization/Compression_Ratio` - Model compression achieved

**Quantization Visualizations:**
- Weight distribution plots (X: weight values {-1, 0, +1}, Y: frequency)
- Compression ratio over training
- Sparsity patterns in different layers

**How to Interpret:**
- **Weight Scales**: Should be stable and not too extreme
- **Sparsity**: Higher sparsity = more efficient model
- **Compression Ratio**: Should achieve ~90% compression vs full precision

### 7. Gradient Flow Analysis

**Gradient Metrics:**
- `Gradients/Total_Norm` - Overall gradient magnitude
- `Gradients/Encoder_Norm, Decoder_Norm, Fusion_Norm` - Component-wise gradients
- `Gradients/Memory_Norm` - Episodic memory gradient flow

**How to Interpret:**
- **Gradient Norms**: Should be stable (not too large or too small)
- **Component Balance**: All components should receive gradients
- **Memory Gradients**: Should increase as memory becomes more important

### 8. Memory-Specific Advanced Metrics

**Memory Evolution Tracking:**
- `memory_diversity_score` - How specialized memory slots are
- `memory_consolidation_index` - Rate of memory consolidation
- `cross_modal_memory_ratio` - Text vs vision memory distribution

**Learning Trajectory Metrics:**
- `text_learning_trajectory` - Text understanding progression
- `vision_learning_trajectory` - Vision understanding progression  
- `trajectory_convergence_rate` - Speed of text-vision alignment
- `cross_modal_alignment_quality` - Quality of text-vision alignment

**How to Interpret:**
- **Diversity Score**: 0.5+ indicates good specialization
- **Consolidation Index**: Lower values = more stable memory
- **Convergence Rate**: Higher values = faster alignment learning

## 🎯 Training Strategy & Phases

### Phase 1: Memory Initialization (Steps 0-2000)
- Memory slots learn basic text-vision associations
- High diversity regularization to prevent early consolidation
- Text and vision trajectories start at different levels

### Phase 2: Cross-Modal Alignment (Steps 2000-8000)
- Cross-modal similarity increases
- Memory specialization emerges
- Attention patterns stabilize

### Phase 3: Episodic Consolidation (Steps 8000+)
- Memory slots become specialized
- Cross-modal trajectories converge
- Model achieves stable performance

### Training Configuration Details

**Optimized Memory Parameters:**
- `memory_size: 48` - Optimized for efficiency vs capacity
- `memory_alpha: 0.12` - Slower, more stable adaptation
- `memory_diversity_regularization: 0.03` - Gentler diversity enforcement
- `memory_temperature: 1.1` - More focused access patterns

**Learning Schedule:**
- `learning_rate: 0.00015` - Conservative for stability
- `warmup_steps: 1500` - Longer warmup for memory initialization
- `scheduler: cosine_with_restarts` - Helps avoid local minima
- `gradient_clip_val: 0.25` - Prevents gradient explosions

## 📊 BabyLM Dataset Details

### Dataset Overview

BitMar is designed for the **BabyLM Challenge Multimodal Track**, processing exactly 100M tokens from two data sources:

### Dataset Components

**Text-only Data (50M tokens):**

- `train_50M.zip` - Clean, child-appropriate text corpus
- Processed through GPT-2 tokenizer for consistent token counting
- Includes diverse text types: stories, descriptions, conversations

**Multimodal Data (50M tokens from captions):**

- `cc_3M_captions.json` - 3 million image-caption pairs from Conceptual Captions
- `cc_3M_dino_v2_states_1of2.npy` + `cc_3M_dino_v2_states_2of2.npy` - Pre-computed DiNOv2 features (768D)
- `local_narr_captions.json` + `local_narr_dino_v2_states.npy` - Localized narratives data

### Data Processing Pipeline

1. **Text Processing**:
   - Tokenization using GPT-2 tokenizer for consistency
   - Sequence length: 256 tokens maximum
   - Exact token counting to ensure 100M total

2. **Vision Processing**:
   - Pre-computed DiNOv2 features (768D) compressed to 64D
   - Spatial pooling from 14x14 to 7x7 patches
   - Attention-based pooling preserves spatial relationships

3. **Image-Caption Alignment**:
   - Perfect 1:1 alignment between captions and visual features
   - Strict validation ensures no misaligned pairs
   - Batch-level consistency checks during training

## 🔧 Technical Implementation Details

### Model Architecture Deep Dive

#### BitNet Quantization Implementation

```python
# 1.58-bit weight quantization
def quantize_weights_1_58_bit(weight):
    scale = weight.abs().mean()
    weight_norm = weight / scale
    
    # Ternary quantization {-1, 0, +1}
    threshold = 2.0 / 3.0
    quantized = torch.zeros_like(weight_norm)
    quantized[weight_norm > threshold] = 1.0
    quantized[weight_norm < -threshold] = -1.0
    
    return quantized * scale
```

**Key Benefits:**
- ~90% memory reduction compared to full precision
- Faster inference on specialized hardware
- Maintains model performance with proper training

#### Episodic Memory Mechanism

```python
# Memory access and updates
def memory_forward(self, query, key, value):
    # Attention-based memory access
    attention_scores = torch.matmul(query, self.memory_keys.T)
    attention_weights = F.softmax(attention_scores / sqrt(d_k), dim=-1)
    
    # Retrieve memory content
    retrieved = torch.matmul(attention_weights, self.memory_values)
    
    # Update memory (gradient-based)
    memory_update = self.memory_alpha * (value - self.memory_values)
    self.memory_values += memory_update
    
    return retrieved
```

**Memory Properties:**
- **Size**: 48 slots × 128D per slot = 6,144 parameters
- **Access**: Soft attention over all slots
- **Updates**: Exponential moving average with α=0.12
- **Diversity**: Regularization prevents slot consolidation

#### Cross-Modal Fusion Architecture

```python
class CrossModalFusion(nn.Module):
    def __init__(self, hidden_size=128, num_heads=6, num_layers=3):
        self.text_projection = BitNetLinear(128, 128)
        self.vision_projection = BitNetLinear(64, 128)
        
        self.fusion_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, text_features, vision_features):
        # Project to common space
        text_proj = self.text_projection(text_features)
        vision_proj = self.vision_projection(vision_features)
        
        # Concatenate and fuse
        fused = torch.cat([text_proj, vision_proj], dim=1)
        
        for layer in self.fusion_layers:
            fused = layer(fused)
            
        return fused
```

### Training Dynamics

#### Learning Rate Schedule

- **Base LR**: 0.00015 (conservative for stability)
- **Schedule**: Cosine annealing with warm restarts
- **T_0**: 800 steps (restart frequency)
- **T_mult**: 2 (restart interval multiplier)
- **Warmup**: 1500 steps (memory initialization)

#### Loss Function Components

```python
total_loss = (
    1.0 * text_generation_loss +           # Standard language modeling
    1.2 * cross_modal_loss +               # Text-vision alignment
    0.10 * memory_regularization_loss +    # Memory diversity
    0.3 * alignment_consistency_loss       # Caption-image consistency
)
```

#### Memory Preservation Strategy

- **Diversity Regularization**: Prevents memory slot collapse
- **Utilization Penalty**: Encourages using all memory slots
- **Temperature Control**: Balances focused vs distributed access
- **Momentum Updates**: Stable memory evolution

### Key Training Files

#### `train_100M_tokens.py` - Main Training Script

**Features:**
- Exact 100M token counting and stopping
- Enhanced cross-modal trajectory tracking
- Memory visualization integration
- Token-aware checkpointing
- Comprehensive Wandb logging

**Key Components:**
- `TokenAwareTrainer` - Handles exact token limits
- `_compute_enhanced_cross_modal_metrics` - Dual trajectory calculation
- `_update_cross_modal_trajectories` - Smooth trajectory updates
- Token counting and progress logging

#### `configs/bitmar_100M_tokens_optimized_memory.yaml` - Configuration

**Optimized Parameters:**
- `memory_size: 48` - Balanced capacity vs efficiency
- `memory_alpha: 0.12` - Stable memory updates
- `memory_diversity_regularization: 0.03` - Gentle diversity preservation
- `batch_size: 48` - Memory-efficient training
- `fusion_num_heads: 6` - Enhanced cross-modal attention

#### `src/model.py` - Core Architecture

**Key Classes:**
- `BitNetLinear` - 1.58-bit quantized linear layers
- `EpisodicMemoryModule` - Memory storage and retrieval
- `CrossModalFusion` - Text-vision alignment
- `BitMarModel` - Complete architecture integration

## 🔍 Advanced Features

### Adaptive Training Controller

**Purpose**: Automatically adjusts training when cross-modal similarity drops

**Parameters:**
- `similarity_window_size: 150` - Rolling window for similarity tracking
- `drop_threshold: 0.06` - Threshold for intervention
- `freeze_duration_steps: 600` - Temporary component freezing
- `loss_rebalance_factor: 2.0` - Dynamic loss reweighting

### Memory Visualization System

**Heatmap Generation:**
- Memory evolution over training epochs
- Slot utilization patterns
- Access frequency distributions
- Specialization emergence

**Integration:**
- Automatic Wandb uploads
- Configurable snapshot frequency
- Error-resistant plotting
- Multi-format output (PNG, SVG)

### Attention Analysis Framework

**Head Tracking:**
- Top-K most important attention heads
- Cross-modal attention patterns
- Timeline evolution visualization
- Layer-wise attention distribution

**Inspired by lo-fit methodology:**
- Head importance scoring
- Attention pattern clustering
- Evolution tracking over training
- Cross-modal attention analysis

## 🚀 Quick Commands Reference

### Essential Setup Commands (In Order)

```bash
# 1. Download both evaluation pipelines first
git clone https://github.com/babylm/evaluation-pipeline-2025.git
git clone https://github.com/babylm/evaluation-pipeline-2024.git

# Setup 2025 pipeline
cd evaluation-pipeline-2025
pip install -r requirements.txt

# Download evaluation data from OSF
curl -L "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip=" -o full_eval.zip
python3 -c "
import zipfile
import os
os.makedirs('evaluation_data', exist_ok=True)
with zipfile.ZipFile('full_eval.zip', 'r') as zip_ref:
    zip_ref.extractall('evaluation_data/')
print('✅ Evaluation data extracted!')
"
rm full_eval.zip

# Setup 2024 pipeline
cd ../evaluation-pipeline-2024
pip install -r requirements.txt
mkdir -p evaluation_data

# 2. Clone BitMar repository
cd ..
git clone https://github.com/euhidaman/BitMar.git
cd BitMar

# 3. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Download datasets
python download_babylm_data.py

# 5. Verify setup
python test_cpu_compatibility.py
python validate_setup.py
```

### Training Commands

```bash
# Set Wandb API key
export WANDB_API_KEY="your_api_key_here"

# Full 100M token training
python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml

# Quick test training
python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml --max_epochs 1

# Adaptive training
python train_adaptive_bitmar.py --config configs/bitmar_100M_tokens_optimized_memory.yaml
```

### Evaluation Commands

```bash
# BitMar evaluation
python evaluate_bitmar.py --model_path outputs/checkpoints/latest_checkpoint.pt

# BabyLM 2025 challenge evaluation
cd ../evaluation-pipeline-2025
python -m evaluation_pipeline.run_evaluation --model_path ../BitMar/hf_model --eval_type full

# BabyLM 2024 challenge evaluation (if needed)
cd ../evaluation-pipeline-2024
# Follow 2024 evaluation instructions

# Specific benchmarks (2025 pipeline)
cd ../evaluation-pipeline-2025
./eval_zero_shot_fast.sh ../BitMar/hf_model
./eval_finetuning.sh ../BitMar/hf_model
```

## 📈 Monitoring Training

### Key Metrics to Watch

**Training Progress:**

- `Training/Loss` - Should decrease over steps
- `Training/Tokens_Processed` - Should reach exactly 100M
- `Training/Learning_Rate` - Cosine schedule with restarts

**Cross-Modal Learning:**

- `Cross-Modal Trajectories/Text Learning` - Orange line starting ~0.2
- `Cross-Modal Trajectories/Vision Learning` - Blue line starting ~0.35
- Both should converge upward over training

**Memory Health:**

- `Memory/Usage_Mean` - Target >70% utilization
- `Memory/Diversity_Score` - Target >0.5 for specialization
- `Memory/Active_Slots_Percentage` - Should increase over training

**Model Stability:**

- `Gradients/Total_Norm` - Should be stable (1-10 range)
- `Features/CrossModal_Similarity` - Should increase (0.0→1.0)
- `Quantization/Compression_Ratio` - Should achieve ~90%

### Warning Signs

**Training Issues:**
- Loss not decreasing after 1000 steps
- Gradient norms >50 or <0.001
- Memory usage <30% after 2000 steps

**Memory Problems:**
- Diversity score <0.2 (memory collapse)
- All slots accessing same content
- Memory attention weights not changing

**Cross-Modal Issues:**
- Text/vision trajectories not converging
- Cross-modal similarity not increasing
- Large gap between text and vision learning

## 🎯 Expected Results

### Training Metrics Targets

- **Final Training Loss**: <2.5
- **Cross-Modal Similarity**: >0.7
- **Memory Utilization**: >70%
- **Memory Diversity**: >0.5
- **Model Compression**: ~90%

### Evaluation Benchmarks

- **BLiMP**: Competitive with baseline models
- **GLUE**: Reasonable performance for 100M tokens
- **Cross-Modal Retrieval**: Improved over text-only models
- **Memory Efficiency**: Better memory utilization than standard transformers

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python train_100M_tokens.py --batch_size 24

# Enable gradient checkpointing
# Set use_mixed_precision: true in config
```

**Dataset Loading Issues:**
```bash
# Rebuild dataset cache
python train_100M_tokens.py --rebuild_cache

# Verify dataset integrity
python test_dataset_compatibility.py
```

**Wandb Logging Problems:**
```bash
# Check API key
wandb login

# Disable wandb for debugging
export WANDB_MODE=offline
```

**Memory Visualization Errors:**
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Use Agg backend for headless systems
export MPLBACKEND=Agg
```

### Performance Optimization

**For Faster Training:**
- Use `batch_size: 64` on high-memory GPUs
- Set `num_workers: 8` for data loading
- Enable `pin_memory: true`

**For Memory Efficiency:**
- Reduce `memory_size` to 32
- Use `use_mixed_precision: true`
- Set `gradient_checkpointing: true`

## 📚 Research Background

### Motivation

Traditional vision-language models process images and text separately without persistent memory of cross-modal associations. BitMar addresses this through:

1. **Episodic Memory**: Stores concrete visual-text associations
2. **BitNet Quantization**: Enables efficient deployment
3. **Dual Learning Trajectories**: Text and vision learn at different rates
4. **Cross-Modal Grounding**: Links language understanding to visual experience

### Key Innovations

- **1.58-bit Quantization**: Extreme efficiency with minimal performance loss
- **Episodic Memory Mechanism**: Persistent cross-modal association storage
- **Dual Trajectory Learning**: Text starts lower, vision higher, both converge
- **Memory Diversity Preservation**: Prevents memory slot consolidation

### Cognitive Alignment

The episodic memory mechanism aligns with cognitive theories where language understanding relies on recalled sensory experiences, making the model more human-like in its learning pattern.

## 📄 File Structure Overview

```
BitMar/
├── train_100M_tokens.py          # Main training script
├── configs/
│   └── bitmar_100M_tokens_optimized_memory.yaml  # Training config
├── src/
│   ├── model.py                  # Core architecture
│   ├── dataset.py                # Data loading
│   ├── wandb_logger.py           # Logging system
│   ├── attention_visualizer.py   # Attention analysis
│   └── memory_visualization_integration.py  # Memory viz
├── download_babylm_data.py       # Dataset downloader
├── download_evaluation_data.py   # Evaluation data
├── evaluate_bitmar.py            # Model evaluation
├── test_*.py                     # Various tests
└── outputs/                      # Training outputs
    ├── checkpoints/              # Model checkpoints
    ├── logs/                     # Training logs
    ├── attention_analysis/       # Attention visualizations
    └── memory_visualization/     # Memory heatmaps
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BabyLM Challenge**: For providing the framework and datasets
- **BitNet**: For the quantization methodology
- **DiNOv2**: For robust vision features
- **Larimar**: For episodic memory inspiration
- **lo-fit**: For attention analysis methodology

---

**Happy Training! 🚀**

For questions or issues, please open a GitHub issue or join the BabyLM Challenge Slack community.
