"""
Setup Validation Script for BabyLM Evaluation Pipelines
Validates that all required components are properly installed and configured
"""

import os
import sys
from pathlib import Path
import subprocess
import yaml

def check_directory_structure(pipeline_path: str, pipeline_name: str, required_files: list):
    """Check if pipeline directory has required structure"""
    print(f"\n📁 Checking {pipeline_name} structure...")
    
    pipeline_dir = Path(pipeline_path)
    if not pipeline_dir.exists():
        print(f"❌ {pipeline_name} directory not found: {pipeline_path}")
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = pipeline_dir / file_name
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            print(f"✅ Found: {file_name}")
    
    if missing_files:
        print(f"❌ Missing files in {pipeline_name}:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print(f"✅ {pipeline_name} structure validation passed")
    return True

def check_python_packages(required_packages: list):
    """Check if required Python packages are installed"""
    print(f"\n📦 Checking Python packages...")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages installed")
    return True

def check_hf_access():
    """Check HuggingFace CLI access"""
    print(f"\n🤗 Checking HuggingFace access...")
    
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ HuggingFace CLI authenticated")
            print(f"   User: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ HuggingFace CLI not authenticated")
            print(f"   Run: huggingface-cli login")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ HuggingFace CLI not found")
        print(f"   Install with: pip install huggingface_hub")
        return False

def check_evaluation_data(pipeline_2024_path: str, pipeline_2025_path: str):
    """Check if evaluation data is downloaded"""
    print(f"\n📊 Checking evaluation data...")
    
    # Check 2025 data with correct structure
    eval_data_2025 = Path(pipeline_2025_path) / "evaluation_data" / "full_eval"
    if eval_data_2025.exists():
        print(f"✅ Pipeline 2025 evaluation data found")
        
        # Check specific directories
        expected_dirs = [
            "blimp_filtered",
            "supplement_filtered", 
            "ewok_filtered",
            "entity_tracking",
            "glue_filtered",
            "winoground_filtered",
            "vqa_filtered",
            "wug_adj_nominalization",
            "wug_past_tense",
            "comps",
            "reading",
            "cdi_childes"
        ]
        
        missing_dirs = []
        for dir_name in expected_dirs:
            dir_path = eval_data_2025 / dir_name
            if dir_path.exists():
                print(f"  ✅ Found: {dir_name}")
            else:
                missing_dirs.append(dir_name)
                print(f"  ❌ Missing: {dir_name}")
        
        if missing_dirs:
            print(f"  ⚠️  Some evaluation directories are missing")
        else:
            print(f"  ✅ All evaluation directories found")
            
    else:
        print(f"❌ Pipeline 2025 evaluation data not found: {eval_data_2025}")
        print(f"   Download from: https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip=")
        print(f"   Extract to: {pipeline_2025_path}/evaluation_data/")
        
    # Check 2024 data
    eval_data_2024 = Path(pipeline_2024_path) / "evaluation_data"
    if eval_data_2024.exists():
        print(f"✅ Pipeline 2024 evaluation data directory found")
        # Check for some expected files
        if (eval_data_2024 / "winoground_filtered").exists() or (eval_data_2024 / "vqa_filtered").exists():
            print(f"  ✅ Found multimodal evaluation data")
        else:
            print(f"  ⚠️  Multimodal evaluation data may be incomplete")
    else:
        print(f"❌ Pipeline 2024 evaluation data not found: {eval_data_2024}")
        print(f"   Download from: https://osf.io/ad7qg/")
    
    return eval_data_2025.exists() and eval_data_2024.exists()

def check_bitmar_config():
    """Check BitMar configuration"""
    print(f"\n⚙️  Checking BitMar configuration...")
    
    config_path = Path("configs/bitmar_100M_tokens.yaml")
    if not config_path.exists():
        print(f"❌ BitMar config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'evaluation' not in config:
            print(f"❌ No evaluation section in config")
            return False
        
        eval_config = config['evaluation']
        
        # Check required evaluation config
        required_keys = ['enabled', 'pipeline_2024_path', 'pipeline_2025_path']
        missing_keys = [key for key in required_keys if key not in eval_config]
        
        if missing_keys:
            print(f"❌ Missing evaluation config keys: {missing_keys}")
            return False
        
        # Check if paths exist
        pipeline_2024_path = eval_config['pipeline_2024_path']
        pipeline_2025_path = eval_config['pipeline_2025_path']
        
        if not Path(pipeline_2024_path).exists():
            print(f"❌ Pipeline 2024 path doesn't exist: {pipeline_2024_path}")
            return False
        
        if not Path(pipeline_2025_path).exists():
            print(f"❌ Pipeline 2025 path doesn't exist: {pipeline_2025_path}")
            return False
        
        print(f"✅ BitMar configuration valid")
        print(f"   Pipeline 2024: {pipeline_2024_path}")
        print(f"   Pipeline 2025: {pipeline_2025_path}")
        
        return True, pipeline_2024_path, pipeline_2025_path
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def main():
    """Main validation function"""
    print("🧪 BabyLM Evaluation Pipeline Setup Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check BitMar config first to get paths
    config_result = check_bitmar_config()
    if isinstance(config_result, tuple):
        _, pipeline_2024_path, pipeline_2025_path = config_result
    else:
        print("❌ Cannot proceed without valid BitMar configuration")
        return False
    
    # Check directory structures
    required_2025_files = [
        "eval_zero_shot.sh",
        "eval_zero_shot_fast.sh", 
        "eval_finetuning.sh",
        "evaluation_pipeline",
        "requirements.txt"
    ]
    
    required_2024_files = [
        "eval_multimodal.sh",
        "eval_blimp.sh",
        "eval_ewok.sh",
        "requirements.txt"
    ]
    
    if not check_directory_structure(pipeline_2025_path, "Pipeline 2025", required_2025_files):
        all_checks_passed = False
    
    if not check_directory_structure(pipeline_2024_path, "Pipeline 2024", required_2024_files):
        all_checks_passed = False
    
    # Check Python packages
    required_packages = [
        "transformers", "torch", "numpy", "pandas", 
        "wandb", "datasets", "sklearn", "statsmodels"
    ]
    
    if not check_python_packages(required_packages):
        all_checks_passed = False
    
    # Check HuggingFace access
    if not check_hf_access():
        print("⚠️  HuggingFace access check failed (optional but recommended)")
    
    # Check evaluation data
    if not check_evaluation_data(pipeline_2024_path, pipeline_2025_path):
        print("⚠️  Evaluation data check failed (required for actual evaluation)")
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ Setup validation PASSED!")
        print("\nYou can now run BitMar training with evaluation integration:")
        print("python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml")
    else:
        print("❌ Setup validation FAILED!")
        print("\nPlease fix the issues above before running evaluation integration.")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
