# evaluation_runners/run_final_full_eval.py  
# filepath: evals-2025/evaluation_runners/run_final_full_eval.py
"""
Run full evaluation (including finetuning) on final model
"""
import subprocess
from pathlib import Path

def run_full_evaluation(model_path, architecture="causal"):
    """Run complete evaluation suite on final model"""
    eval_dir = Path("../evaluation-pipeline-2025")
    
    # Run zero-shot full evaluation
    print("Running full zero-shot evaluation...")
    zero_shot_cmd = [str(eval_dir / "eval_zero_shot.sh"), model_path, architecture]
    subprocess.run(zero_shot_cmd, check=True)
    
    # Run finetuning evaluation  
    print("Running finetuning evaluation...")
    finetune_cmd = [str(eval_dir / "eval_finetuning.sh"), model_path]
    subprocess.run(finetune_cmd, check=True)
    
    print("Full evaluation complete!")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] 
    architecture = sys.argv[2] if len(sys.argv) > 2 else "causal"
    run_full_evaluation(model_path, architecture)