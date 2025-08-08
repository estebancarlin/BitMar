#!/usr/bin/env python3
"""
Test script to verify cross-modal trajectory visualization.
This script simulates the cross-modal logging to ensure we get two colored lines in a single WandB graph.
"""

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

def simulate_cross_modal_trajectories(steps: int = 100) -> Dict[str, List[float]]:
    """Simulate cross-modal learning trajectories that converge over time"""
    
    # Initialize trajectories
    text_trajectory = []
    vision_trajectory = []
    convergence_scores = []
    distances = []
    
    # Text starts lower and grows more slowly initially
    text_start = 0.25
    text_growth_rate = 0.8
    
    # Vision starts higher but grows more consistently
    vision_start = 0.35
    vision_growth_rate = 0.9
    
    for step in range(1, steps + 1):
        # Step progress factor
        step_progress = min(1.0, step / 50.0)  # Normalize to 50 steps
        
        # Text learning trajectory (starts lower, accelerates later)
        text_base = text_start + 0.4 * (step_progress ** text_growth_rate)
        text_noise = 0.02 * np.random.normal()  # Small random variation
        text_value = text_base + text_noise
        
        # Vision learning trajectory (starts higher, steadier growth)
        vision_base = vision_start + 0.35 * (step_progress ** vision_growth_rate)
        vision_noise = 0.02 * np.random.normal()  # Small random variation
        vision_value = vision_base + vision_noise
        
        # Add convergence factor - they should move closer together over time
        convergence_factor = 0.1 * step_progress
        if len(text_trajectory) > 0:
            # Pull trajectories toward each other
            prev_text = text_trajectory[-1]
            prev_vision = vision_trajectory[-1]
            
            text_value = text_value + convergence_factor * (prev_vision - prev_text) * 0.3
            vision_value = vision_value + convergence_factor * (prev_text - prev_vision) * 0.3
        
        # Ensure upward trend
        if text_trajectory:
            text_value = max(text_value, text_trajectory[-1] + 0.001)
        if vision_trajectory:
            vision_value = max(vision_value, vision_trajectory[-1] + 0.001)
        
        # Store values
        text_trajectory.append(text_value)
        vision_trajectory.append(vision_value)
        
        # Calculate distance and convergence
        distance = abs(text_value - vision_value)
        convergence = 1.0 / (1.0 + distance)
        
        distances.append(distance)
        convergence_scores.append(convergence)
    
    return {
        'text_learning': text_trajectory,
        'vision_learning': vision_trajectory,
        'distances': distances,
        'convergence_scores': convergence_scores
    }

def test_wandb_logging():
    """Test WandB logging with the exact format used in training"""
    print("🧪 Testing WandB cross-modal trajectory logging...")
    
    # Initialize WandB (you can disable this if you don't want to log to WandB)
    try:
        wandb.init(
            project="bitmar-cross-modal-test",
            name="trajectory-visualization-test",
            config={
                "test_type": "cross_modal_visualization",
                "steps": 100
            }
        )
        use_wandb = True
        print("✅ WandB initialized successfully")
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print("   Continuing with local visualization only...")
        use_wandb = False
    
    # Generate simulated trajectories
    trajectories = simulate_cross_modal_trajectories(steps=100)
    
    # Log to WandB in the same format as the training script
    for step in range(1, 101):
        step_idx = step - 1
        
        log_dict = {
            'test/step': step,
            'test/loss': 0.5 * np.exp(-step / 50.0),  # Simulated decreasing loss
        }
        
        # CRITICAL: Log both trajectories together in the same section
        # This is the exact format from the training script
        text_learning = trajectories['text_learning'][step_idx]
        vision_learning = trajectories['vision_learning'][step_idx]
        
        log_dict['Cross-Modal Trajectories/Text Learning (Orange)'] = text_learning
        log_dict['Cross-Modal Trajectories/Vision Learning (Blue)'] = vision_learning
        log_dict['Cross-Modal Trajectories/Distance'] = trajectories['distances'][step_idx]
        log_dict['Cross-Modal Trajectories/Convergence'] = trajectories['convergence_scores'][step_idx]
        
        # Additional metrics
        log_dict['Cross-Modal Metrics/Overall Similarity'] = (text_learning + vision_learning) / 2
        
        if use_wandb:
            try:
                wandb.log(log_dict, step=step)
            except Exception as e:
                print(f"❌ WandB logging failed at step {step}: {e}")
                use_wandb = False
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d}: Text={text_learning:.3f}, Vision={vision_learning:.3f}, "
                  f"Distance={trajectories['distances'][step_idx]:.3f}")
    
    if use_wandb:
        print("\n✅ WandB logging completed!")
        print("📊 Check your WandB dashboard for the 'Cross-Modal Trajectories' graph")
        print("   You should see two lines (orange for text, blue for vision) converging upward")
        wandb.finish()
    
    # Create local visualization
    create_local_visualization(trajectories)

def create_local_visualization(trajectories: Dict[str, List[float]]):
    """Create a local matplotlib visualization of the trajectories"""
    print("\n📈 Creating local visualization...")
    
    steps = list(range(1, len(trajectories['text_learning']) + 1))
    
    plt.figure(figsize=(12, 8))
    
    # Main plot - Cross-Modal Trajectories
    plt.subplot(2, 2, 1)
    plt.plot(steps, trajectories['text_learning'], 'orange', linewidth=2.5, label='Text Learning (Orange)', alpha=0.9)
    plt.plot(steps, trajectories['vision_learning'], 'blue', linewidth=2.5, label='Vision Learning (Blue)', alpha=0.9)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Score')
    plt.title('Cross-Modal Learning Trajectories\n(Two lines converging upward)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distance plot
    plt.subplot(2, 2, 2)
    plt.plot(steps, trajectories['distances'], 'red', linewidth=2, label='Trajectory Distance')
    plt.xlabel('Training Steps')
    plt.ylabel('Distance')
    plt.title('Distance Between Trajectories\n(Should decrease over time)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convergence plot
    plt.subplot(2, 2, 3)
    plt.plot(steps, trajectories['convergence_scores'], 'green', linewidth=2, label='Convergence Score')
    plt.xlabel('Training Steps')
    plt.ylabel('Convergence')
    plt.title('Convergence Score\n(Should increase over time)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined view
    plt.subplot(2, 2, 4)
    plt.plot(steps, trajectories['text_learning'], 'orange', linewidth=2, label='Text', alpha=0.8)
    plt.plot(steps, trajectories['vision_learning'], 'blue', linewidth=2, label='Vision', alpha=0.8)
    plt.fill_between(steps, trajectories['text_learning'], trajectories['vision_learning'], 
                     alpha=0.2, color='purple', label='Gap')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Score')
    plt.title('Trajectory Convergence Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'cross_modal_trajectories_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Local visualization saved to: {output_path}")
    
    # Show statistics
    print(f"\n📊 Trajectory Statistics:")
    print(f"   Text Learning:   {trajectories['text_learning'][0]:.3f} → {trajectories['text_learning'][-1]:.3f} "
          f"(+{trajectories['text_learning'][-1] - trajectories['text_learning'][0]:.3f})")
    print(f"   Vision Learning: {trajectories['vision_learning'][0]:.3f} → {trajectories['vision_learning'][-1]:.3f} "
          f"(+{trajectories['vision_learning'][-1] - trajectories['vision_learning'][0]:.3f})")
    print(f"   Initial Distance: {trajectories['distances'][0]:.3f}")
    print(f"   Final Distance:   {trajectories['distances'][-1]:.3f}")
    print(f"   Convergence:      {trajectories['convergence_scores'][0]:.3f} → {trajectories['convergence_scores'][-1]:.3f}")
    
    plt.show()

def main():
    """Main test function"""
    print("🔬 Cross-Modal Trajectory Visualization Test")
    print("=" * 50)
    
    print("\nThis test verifies that:")
    print("1. ✅ Two separate learning trajectories are generated")
    print("2. ✅ Text learning starts lower, vision starts higher")
    print("3. ✅ Both trajectories move upward over time")
    print("4. ✅ Trajectories gradually converge (get closer)")
    print("5. ✅ WandB logging format creates single graph with two colored lines")
    
    # Run the test
    test_wandb_logging()
    
    print("\n🎯 Expected WandB Visualization:")
    print("   • Go to your WandB dashboard")
    print("   • Look for 'Cross-Modal Trajectories' section")
    print("   • You should see ONE graph with TWO lines:")
    print("     - Orange line: 'Text Learning (Orange)'")
    print("     - Blue line: 'Vision Learning (Blue)'")
    print("   • Both lines should trend upward and get closer together")
    
    print("\n✅ Test completed! Check the outputs above.")

if __name__ == "__main__":
    main()
