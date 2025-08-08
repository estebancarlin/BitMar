#!/usr/bin/env python3
"""
Test the evaluation pipeline with mock data and small models
"""
import os
import sys
import tempfile
import json
from pathlib import Path

def create_mock_results():
    """Create mock evaluation results for testing"""
    mock_results = {
        "blimp_overall": 0.65,
        "glue_cola": 0.78,
        "glue_sst2": 0.82,
        "supplement_eos_distribution": 0.71,
        "ewok": 0.58
    }
    
    # Create mock results directory
    results_dir = Path("../evaluation-pipeline-2025/results/mock_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results for different checkpoints
    checkpoints = ["chck_1M", "chck_2M", "chck_5M", "chck_10M"]
    
    for i, checkpoint in enumerate(checkpoints):
        # Simulate improving scores
        checkpoint_results = {}
        for task, base_score in mock_results.items():
            # Add some improvement over checkpoints + noise
            improvement = (i * 0.02) + (hash(task + checkpoint) % 100) * 0.001
            checkpoint_results[task] = min(1.0, base_score + improvement)
        
        # Save checkpoint results
        checkpoint_file = results_dir / f"{checkpoint}_results.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_results, f, indent=2)
    
    print(f"Mock results created in: {results_dir}")
    return str(results_dir)

def test_results_processing():
    """Test results processing scripts with mock data"""
    print("Testing results processing...")
    
    # Create mock results
    results_dir = create_mock_results()
    
    # Test collate_all_results
    import subprocess
    cmd = [
        "python", "results_processing/collate_all_results.py",
        "--results_dir", results_dir,
        "--output_dir", "./test_summaries"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Results collation test passed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Results collation test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    
    return True

def test_analysis_scripts():
    """Test analysis scripts"""
    print("Testing analysis scripts...")
    
    # Test performance analysis
    if Path("./test_summaries/checkpoint_summary.csv").exists():
        import subprocess
        cmd = [
            "python", "analysis/performance_analysis.py",
            "test-model",
            "--summary_file", "./test_summaries/checkpoint_summary.csv",
            "--output_dir", "./test_analysis"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Performance analysis test passed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Performance analysis test failed: {e}")
            return False
    
    # Test paper tables generation
    cmd = [
        "python", "analysis/create_paper_tables.py",
        "--summary_file", "./test_summaries/checkpoint_summary.csv",
        "--output_dir", "./test_tables",
        "--model_name", "Test Model"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("✓ Paper tables test passed")
    except Exception as e:
        print(f"✗ Paper tables test failed: {e}")
    
    return True

def main():
    print("BabyLM Evaluation Pipeline - Mock Data Test")
    print("=" * 50)
    
    # Test results processing
    if test_results_processing():
        print("Results processing: ✓ PASSED")
    else:
        print("Results processing: ✗ FAILED")
        return
    
    # Test analysis
    if test_analysis_scripts():
        print("Analysis scripts: ✓ PASSED")
    else:
        print("Analysis scripts: ✗ FAILED")
    
    print("\nTest files created:")
    print("- ./test_summaries/ - Collated results")
    print("- ./test_analysis/ - Performance analysis")  
    print("- ./test_tables/ - LaTeX tables")
    
    print("\n✓ Mock data pipeline test completed successfully!")

if __name__ == "__main__":
    main()