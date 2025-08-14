"""
BitMar BabyLM 2025 Evaluation Runner - HuggingFace Hub Version with BitMar Registration
Uses models directly from HuggingFace Hub with specific commits
"""
import sys
import subprocess
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

# Register BitMar model BEFORE importing evaluation pipeline
try:
    from register_bitmar_model import BitMarConfig, BitMarForCausalLM
    from transformers import AutoConfig, AutoModel
    
    AutoConfig.register("bitmar", BitMarConfig)
    AutoModel.register(BitMarConfig, BitMarForCausalLM)
    print("✅ BitMar model registered with transformers")
except Exception as e:
    print(f"⚠️ Could not register BitMar model: {e}")

# Add evaluation pipeline to path
evaluation_pipeline_path = Path("evaluation-pipeline-2025")
if evaluation_pipeline_path.exists():
    sys.path.insert(0, str(evaluation_pipeline_path))

class BabyLMEvaluationRunner:
    """Main evaluation runner for BabyLM 2025 Challenge using HuggingFace Hub"""
    
    def __init__(self, 
                 hf_repo_id: str,
                 backend: str,
                 track: str,
                 output_dir: str = "evaluation_results",
                 model_type: Optional[str] = None,
                 image_model: Optional[str] = None):
        
        self.hf_repo_id = hf_repo_id
        self.backend = backend
        self.track = track
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.image_model = image_model
        
        # Mapping from epoch to commit hash (your correct commits)
        self.epoch_to_commit = {
            1: "5be3c56",
            2: "fa7c80f", 
            3: "9c651d9",
            4: "97621f5",
            5: "ce78b62",
            6: "e2f4d65",
            7: "b404af1",
            8: "cb7594d",
            9: "f5153ad",
            10: "f209f3a"
        }
        
        # Mapping from epoch to BabyLM checkpoint names (for logging only)
        self.epoch_to_checkpoint = {
            1: "chck_1M",
            2: "chck_2M",
            3: "chck_3M",
            4: "chck_4M",
            5: "chck_5M",
            6: "chck_6M",
            7: "chck_7M",
            8: "chck_8M",
            9: "chck_9M",
            10: "chck_10M"
        }
        
        self.setup_directories()
        self.setup_logging()
        self.validate_inputs()
        self.define_tasks()

    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "logs" / f"bitmar_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 BitMar BabyLM 2025 Evaluation Started")
        self.logger.info(f"📂 Output directory: {self.output_dir}")
        self.logger.info(f"🤗 HF Repository: {self.hf_repo_id}")
        self.logger.info(f"⚙️ Backend: {self.backend}")
        self.logger.info(f"🎯 Track: {self.track}")

    def validate_inputs(self):
        """Validate input parameters"""
        valid_backends = ["causal", "mlm", "mntp", "enc_dec_mask", "enc_dec_prefix"]
        valid_tracks = ["strict-small", "strict", "interaction", "multimodal"]
        
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")
        
        if self.track not in valid_tracks:
            raise ValueError(f"Invalid track: {self.track}. Must be one of {valid_tracks}")
        
        if not evaluation_pipeline_path.exists():
            raise FileNotFoundError(f"Evaluation pipeline not found at {evaluation_pipeline_path}")

    def define_tasks(self):
        """Define evaluation tasks based on track"""
        
        # Fast evaluation tasks
        self.fast_tasks = {
            "zero_shot": [
                ("blimp", "evaluation_data/fast_eval/blimp_fast"),
                ("msgs", "evaluation_data/fast_eval/msgs_fast"),
                ("supplement", "evaluation_data/fast_eval/supplement_fast")
            ],
            "reading": [
                ("ewok", "evaluation_data/fast_eval/ewok_fast")
            ]
        }
        
        # Full evaluation tasks
        self.full_tasks = {
            "zero_shot": [
                ("blimp", "evaluation_data/zero_shot/blimp"),
                ("msgs", "evaluation_data/zero_shot/msgs"),
                ("supplement", "evaluation_data/zero_shot/supplement")
            ],
            "reading": [
                ("ewok", "evaluation_data/reading/ewok")
            ],
            "finetuning": [
                ("glue", "evaluation_data/finetuning/glue"),
                ("ood_generalization", "evaluation_data/finetuning/ood_generalization")
            ],
            "aoa": [
                ("aoa", "evaluation_data/aoa/aoa_words.txt")
            ]
        }
        
        # Multimodal tasks
        if self.track == "multimodal":
            self.full_tasks["multimodal"] = [
                ("multimodal_blimp", "evaluation_data/multimodal/multimodal_blimp"),
                ("visual_reasoning", "evaluation_data/multimodal/visual_reasoning")
            ]

    def run_sentence_zero_shot_task(self, task: str, data_path: str, revision: str = None, 
                                   revision_name: str = None) -> bool:
        """Run a single sentence-level zero-shot task with HF Hub"""
        
        self.logger.info(f"🧪 Running zero-shot task: {task} ({data_path})")
        
        cmd = [
            "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", self.hf_repo_id,
            "--backend", self.backend,
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--trust_remote_code"  # Important for custom models
        ]
        
        # Only use commit hash for revision (don't pass revision_name as it causes conflicts)
        if revision:
            cmd.extend(["--revision", revision])
        
        return self.run_command(cmd, f"zero_shot_{task}")

    def run_reading_task(self, data_path: str, revision: str = None, revision_name: str = None) -> bool:
        """Run reading task with HF Hub"""
        
        self.logger.info(f"📖 Running reading task: {data_path}")
        
        cmd = [
            "python", "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", self.hf_repo_id,
            "--backend", self.backend,
            "--data_path", data_path,
            "--trust_remote_code"  # Important for custom models
        ]
        
        # Only use commit hash for revision
        if revision:
            cmd.extend(["--revision", revision])
        
        return self.run_command(cmd, "reading")

    def run_fast_evaluation(self, tasks: Optional[List[str]] = None, epochs: Optional[List[int]] = None) -> Dict[str, Dict]:
        """Run fast evaluation on specified epochs"""
        
        self.logger.info("⚡ Starting fast evaluation...")
        
        # Default to final model if no epochs specified
        if epochs is None:
            epochs = [10]  # Use final model
        
        self.logger.info(f"Evaluating epochs: {epochs}")
        
        results = {}
        
        # Run for each epoch
        for epoch in epochs:
            revision = self.epoch_to_commit.get(epoch)
            checkpoint_name = self.epoch_to_checkpoint.get(epoch, f"epoch_{epoch}")
            
            self.logger.info(f"🔄 Evaluating epoch {epoch} (commit: {revision}, checkpoint: {checkpoint_name})")
            
            epoch_results = {}
            
            # Zero-shot tasks
            if not tasks or "zero_shot" in tasks:
                for task, data_path in self.fast_tasks["zero_shot"]:
                    result = self.run_sentence_zero_shot_task(task, data_path, revision, checkpoint_name)
                    epoch_results[f"zero_shot_{task}"] = result
            
            # Reading tasks
            if not tasks or "reading" in tasks:
                for task, data_path in self.fast_tasks["reading"]:
                    result = self.run_reading_task(data_path, revision, checkpoint_name)
                    epoch_results[f"reading_{task}"] = result
            
            results[f"{checkpoint_name}"] = epoch_results
        
        # Log summary
        self.log_evaluation_summary("Fast Evaluation", results)
        return results

    def run_command(self, cmd: List[str], task_name: str) -> bool:
        """Execute a command and handle results"""
        
        # Change to evaluation pipeline directory
        original_cwd = Path.cwd()
        evaluation_cwd = evaluation_pipeline_path
        
        try:
            self.logger.info(f"🏃 Executing: {' '.join(cmd)}")
            self.logger.info(f"📁 Working directory: {evaluation_cwd}")
            
            result = subprocess.run(
                cmd,
                cwd=evaluation_cwd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {task_name} completed successfully")
                if result.stdout:
                    self.logger.debug(f"STDOUT: {result.stdout[:500]}...")
                return True
            else:
                self.logger.error(f"❌ {task_name} failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"STDERR: {result.stderr}")
                if result.stdout:
                    self.logger.error(f"STDOUT: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {task_name} timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ {task_name} failed with exception: {e}")
            return False
        finally:
            # Always return to original directory
            pass

    def log_evaluation_summary(self, eval_type: str, results: Dict):
        """Log a summary of evaluation results"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📊 {eval_type} Summary")
        self.logger.info(f"{'='*50}")
        
        for epoch, epoch_results in results.items():
            success_count = sum(1 for result in epoch_results.values() if result)
            total_count = len(epoch_results)
            self.logger.info(f"🎯 {epoch}: {success_count}/{total_count} tasks completed")
            
            for task, success in epoch_results.items():
                status = "✅" if success else "❌"
                self.logger.info(f"  {status} {task}")
        
        self.logger.info(f"{'='*50}\n")

    def run_evaluation(self, eval_types: List[str], tasks: Optional[List[str]] = None,
                      epochs: Optional[List[int]] = None) -> Dict:
        """Main evaluation orchestrator"""
        
        self.logger.info(f"🎬 Starting BitMar evaluation")
        self.logger.info(f"📋 Evaluation types: {eval_types}")
        self.logger.info(f"🎯 Tasks: {tasks or 'all'}")
        self.logger.info(f"📅 Epochs: {epochs or 'default'}")
        
        all_results = {}
        
        # Run fast evaluation
        if "fast" in eval_types:
            fast_results = self.run_fast_evaluation(tasks, epochs)
            all_results["fast"] = fast_results
        
        # Run full evaluation
        if "full" in eval_types:
            # For full evaluation, use the final model by default
            final_revision = self.epoch_to_commit.get(10)  # Final model
            full_results = self.run_full_evaluation(tasks, final_revision)
            all_results["full"] = full_results
        
        return all_results

    def run_full_evaluation(self, tasks: Optional[List[str]] = None, revision: str = None) -> Dict[str, bool]:
        """Run full evaluation (placeholder - implement based on your needs)"""
        self.logger.info("🔄 Full evaluation not yet implemented")
        return {}

def main():
    parser = argparse.ArgumentParser(description="BitMar BabyLM 2025 Evaluation Runner - HF Hub Version")
    
    # Required arguments
    parser.add_argument("--hf_repo_id", required=True, 
                       help="HuggingFace repository ID (e.g., 'username/model-name')")
    parser.add_argument("--backend", required=True,
                       choices=["causal", "mlm", "mntp", "enc_dec_mask", "enc_dec_prefix"],
                       help="Model backend type")
    parser.add_argument("--track", required=True,
                       choices=["strict-small", "strict", "interaction", "multimodal"],
                       help="BabyLM track")
    
    # Optional arguments
    parser.add_argument("--eval_types", nargs="+", choices=["fast", "full"], 
                       default=["fast"], help="Types of evaluation to run")
    parser.add_argument("--tasks", nargs="+", 
                       choices=["zero_shot", "reading", "finetuning", "aoa", "multimodal"],
                       help="Specific tasks to run (default: all available)")
    parser.add_argument("--epochs", nargs="+", type=int, choices=list(range(1, 11)),
                       help="Specific epochs to evaluate (default: final model)")
    parser.add_argument("--model_type", 
                       choices=["git", "flamingo", "llava", "flava", "clip", "blip", "siglip", "bridgetower", "vilt", "cvcl"],
                       help="Vision-language model type for multimodal track")
    parser.add_argument("--image_model", help="Specific image model to use")
    parser.add_argument("--output_dir", default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create runner
    runner = BabyLMEvaluationRunner(
        hf_repo_id=args.hf_repo_id,
        backend=args.backend,
        track=args.track,
        output_dir=args.output_dir,
        model_type=args.model_type,
        image_model=args.image_model
    )
    
    # Run evaluation
    try:
        results = runner.run_evaluation(
            eval_types=args.eval_types,
            tasks=args.tasks,
            epochs=args.epochs
        )
        
        print("\n🎉 BitMar evaluation completed!")
        print(f"📁 Results saved to: {runner.output_dir}")
        
    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()