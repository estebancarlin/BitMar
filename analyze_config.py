"""
SD Card Deployment Analysis for BitMar Configuration
Analyzes memory optimization and model proportionality
"""
import yaml

def analyze_config():
    try:
        # Load the configuration
        with open('configs/bitmar_100M_tokens_optimized_memory.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract key parameters
        memory_size = config['model']['memory_size']
        episode_dim = config['model']['episode_dim']
        fusion_hidden_size = config['model']['fusion_hidden_size']
        text_hidden = config['model']['text_hidden_size']
        text_layers = config['model']['text_num_layers']
        vision_hidden = config['model']['vision_hidden_size']
        
        # Memory calculations
        memory_parameters = memory_size * episode_dim
        memory_mb = memory_parameters * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Model size estimation (rough)
        text_params = text_hidden * text_hidden * text_layers * 12  # Transformer estimation
        vision_params = vision_hidden * 768 + fusion_hidden_size * vision_hidden
        fusion_params = fusion_hidden_size * fusion_hidden_size * 3
        total_model_params = text_params + vision_params + fusion_params
        total_model_mb = total_model_params * 4 / (1024 * 1024)
        
        # Calculate ratios
        memory_to_model_ratio = memory_parameters / total_model_params
        total_size_mb = memory_mb + total_model_mb
        
        print("=" * 60)
        print("🔍 SD CARD DEPLOYMENT ANALYSIS")
        print("=" * 60)
        print()
        
        print("📊 EPISODIC MEMORY CONFIGURATION:")
        print(f"  • Memory slots: {memory_size:,}")
        print(f"  • Episode dimension: {episode_dim:,}")
        print(f"  • Total episodic parameters: {memory_parameters:,}")
        print(f"  • Episodic memory size: {memory_mb:.1f} MB")
        print()
        
        print("🤖 MODEL CONFIGURATION:")
        print(f"  • Text hidden size: {text_hidden}")
        print(f"  • Fusion hidden size: {fusion_hidden_size}")
        print(f"  • Est. total model params: {total_model_params:,}")
        print(f"  • Est. model size: {total_model_mb:.1f} MB")
        print()
        
        print("⚖️  MEMORY-TO-MODEL RATIO ANALYSIS:")
        print(f"  • Memory/Model ratio: {memory_to_model_ratio:.2f}")
        if memory_to_model_ratio > 0.8:
            print("  • Status: ✅ EXCELLENT - Memory is substantial part of model")
        elif memory_to_model_ratio > 0.3:
            print("  • Status: ✅ GOOD - Acceptable for episodic learning")
        else:
            print("  • Status: ⚠️  LOW - Memory might be too small")
        print()
        
        print("💾 SD CARD DEPLOYMENT STATUS:")
        print(f"  • Total model + memory: {total_size_mb:.1f} MB")
        if total_size_mb < 32:
            print("  • SD card compatibility: ✅ EXCELLENT (fits on any SD card)")
        elif total_size_mb < 128:
            print("  • SD card compatibility: ✅ GOOD (fits on standard SD cards)")
        elif total_size_mb < 512:
            print("  • SD card compatibility: ⚠️  MODERATE (requires larger SD card)")
        else:
            print("  • SD card compatibility: ❌ POOR (very large SD card needed)")
        print()
        
        print("🚀 FAST FACT EDITING OPTIMIZATION:")
        fast_edit = config['model'].get('fast_fact_editing_mode', False)
        memory_alpha = config['model']['memory_alpha']
        consolidation_rate = config['model'].get('episodic_consolidation_rate', 'Not configured')
        utilization_target = config['model']['min_memory_utilization']
        
        print(f"  • Fast fact editing mode: {'✅ ENABLED' if fast_edit else '❌ DISABLED'}")
        print(f"  • Memory alpha (adaptation rate): {memory_alpha}")
        print(f"  • Episodic consolidation rate: {consolidation_rate}")
        print(f"  • Memory utilization target: {utilization_target * 100:.0f}%")
        print()
        
        print("🎯 OPTIMIZATION STATUS:")
        optimizations = []
        if memory_to_model_ratio > 0.3:
            optimizations.append("✅ Well-proportioned episodic memory")
        if total_size_mb < 128:
            optimizations.append("✅ SD card deployment ready")
        if fast_edit:
            optimizations.append("✅ Fast fact editing enabled")
        if fusion_hidden_size == episode_dim:
            optimizations.append("✅ Fusion-memory dimension alignment")
        
        for opt in optimizations:
            print(f"  {opt}")
        
        if len(optimizations) >= 3:
            print("\n🎉 CONFIGURATION IS WELL OPTIMIZED FOR SD CARD DEPLOYMENT!")
        else:
            print("\n⚠️  Configuration needs additional optimization")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing configuration: {e}")

if __name__ == "__main__":
    analyze_config()
