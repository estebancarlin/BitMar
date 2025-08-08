# Enhanced Cross-Modal Similarity Tracking

## Overview

The BitMar training script now includes **enhanced cross-modal similarity tracking** that visualizes how text and vision learning trajectories converge over time. This provides much better insights into the model's cross-modal alignment compared to a single similarity metric.

## Key Features

### 🎯 **Single Graph Dual Learning Trajectories**
The main visualization shows both text and vision learning on **one graph**:

1. **Text Learning Line** (Orange) - Text encoder learning progression
   - Starts at a lower baseline
   - Gradually improves over training
   - Moves toward convergence with vision line

2. **Vision Learning Line** (Blue) - Vision encoder learning progression
   - May start at different level than text
   - Shows independent vision learning
   - Gradually converges with text line

3. **Convergence Pattern** - Both lines meet and rise together
   - Distance between lines decreases over time
   - Both trajectories trend upward
   - Final convergence indicates successful cross-modal alignment

## WandB Visualization

### **Main Graph: Cross-Modal Learning**
- `Cross-Modal Learning/Text Learning` - Orange line showing text progress
- `Cross-Modal Learning/Vision Learning` - Blue line showing vision progress  
- `Cross-Modal Learning/Trajectory Distance` - Gap between lines (should decrease)
- `Cross-Modal Learning/Convergence Score` - How well aligned they are (should increase)
- `Cross-Modal Learning/Overall Similarity` - Traditional similarity metric
- `learning_trajectories/vision_learning` - Vision learning progression (orange line)
- `learning_trajectories/trajectory_distance` - Distance between trajectories (should decrease)
- `learning_trajectories/trajectory_alignment` - How well aligned the trajectories are (should increase)

### **Supporting Metrics**
- `cross_modal/similarity` - Traditional cross-modal similarity
- `cross_modal/text_consistency` - Internal text representation consistency
- `cross_modal/vision_consistency` - Internal vision representation consistency
- `cross_modal/alignment_convergence` - Overall alignment quality

### **Learning Strength Indicators**
- `learning_strength/text_strength` - How diverse/rich text representations are
- `learning_strength/vision_strength` - How diverse/rich vision representations are

## What to Look For

### **Healthy Training Pattern** ✅

```
Text Learning (Orange):   ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲
Vision Learning (Blue):   ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲  
Gap Distance:             ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
```

- Both orange and blue lines trend upward
- Lines start apart and gradually converge
- Final convergence score > 0.8
- Smooth, non-jagged trajectories

### **Poor Alignment Pattern** ❌

```
Text Learning (Orange):   ▲ ▲ ▲ ▼ ▼ ▼ ▼ ▼
Vision Learning (Blue):   ▼ ▼ ▲ ▲ ▲ ▲ ▲ ▲
Gap Distance:             ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲
```

- Lines diverge instead of converging
- One or both lines decline over time
- Final convergence score < 0.4
- Jagged, unstable patterns

## Implementation Details

### **Smoothing**
- Uses exponential moving average (EMA) with α=0.1 for trajectory smoothing
- Reduces noise while preserving learning trends
- Updated every training step

### **Memory Management**
- Keeps only last 1000 steps of history to avoid memory issues
- Efficiently stores trajectory data for visualization

### **Error Handling**
- Graceful fallback to traditional cross-modal similarity if enhanced tracking fails
- Comprehensive error logging for debugging

## Training Progress Display

The progress bar now shows trajectory convergence:
```
Epoch 1 | Tokens: 50,000 | T↔V: 0.743 | loss: 2.456
```
- `T↔V`: Text-Vision convergence score (0.0-1.0)
- Higher values = better alignment

## Final Training Report

At the end of training, you'll get a comprehensive summary:

```
📊 Cross-Modal Learning Trajectory Summary:
  🔤 Text Learning Trajectory:
    • Initial: 0.234
    • Final: 0.789
    • Improvement: 0.555
    • Stability: 0.045
  👁️  Vision Learning Trajectory:
    • Initial: 0.198
    • Final: 0.812
    • Improvement: 0.614
    • Stability: 0.038
  🤝 Trajectory Convergence:
    • Initial distance: 0.456
    • Final distance: 0.089
    • Convergence rate: 0.834
  ⭐ Cross-Modal Similarity:
    • Initial: 0.123
    • Final: 0.876
    • Peak: 0.891
    • Overall improvement: 0.753
  🎯 Assessment: Excellent: Text and vision learning trajectories are highly aligned
    • Convergence score: 0.911
```

## Troubleshooting

### **If trajectories don't converge:**
1. Increase `cross_modal_loss_weight` in config
2. Reduce learning rate for more stable training
3. Check vision feature quality
4. Verify text-image alignment in dataset

### **If one trajectory is flat:**
1. **Text flat**: Check text encoder learning rate, increase text-specific losses
2. **Vision flat**: Check vision encoder parameters, verify vision features

### **If trajectories are noisy:**
1. Increase `cross_modal_smoothing_alpha` in trainer initialization
2. Reduce batch size for more stable gradients
3. Increase gradient clipping

## Benefits

1. **Better Debugging**: See exactly which modality is struggling
2. **Training Monitoring**: Catch alignment issues early
3. **Hyperparameter Tuning**: Understand impact of changes on each modality
4. **Research Insights**: Visualize cross-modal learning dynamics

## Usage

The enhanced tracking is automatically enabled in the updated training script. Simply run:

```bash
python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml
```

And monitor the new metrics in WandB under:
- `learning_trajectories/*` - Main trajectory visualizations
- `cross_modal/*` - Detailed cross-modal metrics
- `learning_strength/*` - Learning quality indicators

This gives you a much richer understanding of how your model learns to align text and vision representations! 🚀
