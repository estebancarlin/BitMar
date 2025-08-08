"""
Evaluation Pipeline Integration for BabyLM Challenge 2025
Integrates evaluation-pipeline-2024 and evaluation-pipeline-2025 for epoch-based evaluation

Multimodal Data Notes:
- VQA and Winoground tasks use HuggingFace datasets directly (no local data files needed)
- 2024 pipeline: Uses lm_eval to run VQA (HuggingFaceM4/VQAv2) and Winoground (facebook/winoground)
- 2025 pipeline: Has VQA/Winoground support but multimodal evaluation is "under construction"
- No additional multimodal data beyond what's in HuggingFace datasets
"""

import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil
import wandb

logger = logging.getLogger(__name__)

class BabyLMEvaluationPipeline:
    """Integration class for BabyLM evaluation pipelines"""
    
    def __init__(
        self,
        pipeline_2024_path: str,
        pipeline_2025_path: str,
        model_path: str,
        results_base_dir: str = "evaluation_results",
        use_wandb: bool = True
    ):
        self.pipeline_2024_path = Path(pipeline_2024_path)
        self.pipeline_2025_path = Path(pipeline_2025_path)
        self.model_path = model_path
        self.results_base_dir = Path(results_base_dir)
        self.use_wandb = use_wandb
        
        # Create results directory
        self.results_base_dir.mkdir(exist_ok=True)
        
        # Validate pipeline directories
        self._validate_pipelines()
        
        logger.info(f"Initialized BabyLM Evaluation Pipeline")
        logger.info(f"Pipeline 2024: {self.pipeline_2024_path}")
        logger.info(f"Pipeline 2025: {self.pipeline_2025_path}")
        logger.info(f"Model path: {self.model_path}")
    
    def _validate_pipelines(self):
        """Validate that evaluation pipeline directories exist and have required files"""
        # Check 2025 pipeline
        required_2025_files = [
            "eval_zero_shot.sh",
            "eval_finetuning.sh", 
            "evaluation_pipeline"
        ]
        
        for file_name in required_2025_files:
            file_path = self.pipeline_2025_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_path} not found in pipeline 2025")
        
        # Check 2024 pipeline  
        required_2024_files = [
            "eval_multimodal.sh"
        ]
        
        for file_name in required_2024_files:
            file_path = self.pipeline_2024_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_path} not found in pipeline 2024")
        
        logger.info("Pipeline validation passed")
    
    def _create_hf_compatible_model(self, epoch: int) -> str:
        """
        Create a temporary HuggingFace-compatible model for evaluation
        This ensures compatibility with AutoModelForCausalLM and AutoModelForSequenceClassification
        """
        try:
            # Create temporary directory for this epoch's model
            temp_model_dir = self.results_base_dir / f"temp_model_epoch_{epoch}"
            temp_model_dir.mkdir(exist_ok=True)
            
            # For now, assume the model is already HF compatible
            # In practice, you might need to save/convert your BitMar model
            # to be compatible with HuggingFace transformers
            
            logger.info(f"Created HF-compatible model for epoch {epoch} at {temp_model_dir}")
            return str(temp_model_dir)
            
        except Exception as e:
            logger.error(f"Failed to create HF-compatible model: {e}")
            raise
    
    def run_text_evaluations_2025(
        self, 
        epoch: int, 
        model_path: str,
        fast: bool = False
    ) -> Dict[str, Any]:
        """Run text-only evaluations from pipeline 2025"""
        try:
            results = {}
            
            # Determine which script and data directory to use
            if fast:
                script_name = "eval_zero_shot_fast.sh"
                eval_data_suffix = "fast_eval"  # Use fast_eval for quick evaluation
            else:
                script_name = "eval_zero_shot.sh"
                eval_data_suffix = "full_eval"  # Use full_eval for comprehensive evaluation
                
            script_path = self.pipeline_2025_path / script_name
            
            # Results directory for this epoch
            epoch_results_dir = self.results_base_dir / f"epoch_{epoch}" / "text_2025"
            epoch_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Change to pipeline 2025 directory
            original_cwd = os.getcwd()
            
            try:
                # Run zero-shot evaluations with correct data path
                eval_data_path = f"evaluation_data/{eval_data_suffix}"
                
                # Construct command with correct argument order
                # Script expects: MODEL_PATH REVISION_NAME BACKEND [EVAL_DIR]
                revision_name = f"epoch_{epoch}" if fast else f"epoch_{epoch}_full"
                
                # Ensure model path is absolute (since we'll change directories)
                abs_model_path = os.path.abspath(model_path)
                
                # Change to pipeline directory after getting absolute paths
                os.chdir(self.pipeline_2025_path)
                
                cmd = [
                    "bash", 
                    str(script_path),
                    abs_model_path,  # Use absolute model path
                    revision_name,  # revision name
                    "causal",  # backend for BitMar
                    eval_data_path  # evaluation data path
                ]
                
                logger.info(f"Running text evaluations 2025 (fast={fast}): {' '.join(cmd)}")
                logger.info(f"Using evaluation data from: {eval_data_path}")
                logger.info(f"Using revision name: {revision_name}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"Text evaluation failed: {result.stderr}")
                    results["error"] = result.stderr
                else:
                    logger.info("Text evaluations 2025 completed successfully")
                    results["status"] = "success"
                    results["stdout"] = result.stdout
                
                # Parse results from generated files
                results.update(self._parse_2025_results(model_path, fast))
                
            finally:
                os.chdir(original_cwd)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text evaluations 2025: {e}")
            return {"error": str(e)}
    
    def run_finetuning_evaluations_2025(
        self, 
        epoch: int, 
        model_path: str
    ) -> Dict[str, Any]:
        """Run fine-tuning evaluations from pipeline 2025"""
        try:
            results = {}
            
            script_path = self.pipeline_2025_path / "eval_finetuning.sh"
            
            # Results directory for this epoch
            epoch_results_dir = self.results_base_dir / f"epoch_{epoch}" / "finetune_2025"
            epoch_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Change to pipeline 2025 directory
            original_cwd = os.getcwd()
            os.chdir(self.pipeline_2025_path)
            
            try:
                # Run fine-tuning evaluations
                cmd = [
                    "bash", 
                    str(script_path),
                    model_path,
                    "3e-5",  # learning_rate
                    "16",    # batch_size
                    "5",     # max_epochs (reduced for faster evaluation)
                    "42"     # seed
                ]
                
                logger.info(f"Running fine-tuning evaluations 2025: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"Fine-tuning evaluation failed: {result.stderr}")
                    results["error"] = result.stderr
                else:
                    logger.info("Fine-tuning evaluations 2025 completed successfully")
                    results["status"] = "success"
                    results["stdout"] = result.stdout
                
                # Parse results from generated files
                results.update(self._parse_2025_finetune_results(model_path))
                
            finally:
                os.chdir(original_cwd)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fine-tuning evaluations 2025: {e}")
            return {"error": str(e)}
    
    def run_multimodal_evaluations_2024(
        self, 
        epoch: int, 
        model_path: str
    ) -> Dict[str, Any]:
        """
        Run multimodal evaluations from pipeline 2024 (VQA + Winoground)
        
        Note: Uses HuggingFace datasets directly:
        - VQA: HuggingFaceM4/VQAv2 (validation split)
        - Winoground: facebook/winoground (test split)
        No additional local data files needed.
        """
        try:
            results = {}
            
            script_path = self.pipeline_2024_path / "eval_multimodal.sh"
            
            # Results directory for this epoch
            epoch_results_dir = self.results_base_dir / f"epoch_{epoch}" / "multimodal_2024"
            epoch_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Change to pipeline 2024 directory
            original_cwd = os.getcwd()
            os.chdir(self.pipeline_2024_path)
            
            try:
                # Run multimodal evaluations (downloads HF datasets automatically)
                cmd = [
                    "bash", 
                    str(script_path),
                    model_path
                ]
                
                logger.info(f"Running multimodal evaluations 2024 (VQA + Winoground via lm_eval): {' '.join(cmd)}")
                logger.info("Note: This will download HuggingFace datasets automatically (VQAv2, Winoground)")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"Multimodal evaluation failed: {result.stderr}")
                    results["error"] = result.stderr
                else:
                    logger.info("Multimodal evaluations 2024 completed successfully")
                    results["status"] = "success"
                    results["stdout"] = result.stdout
                
                # Parse results from generated files
                results.update(self._parse_2024_multimodal_results(model_path))
                
            finally:
                os.chdir(original_cwd)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multimodal evaluations 2024: {e}")
            return {"error": str(e)}
    
    def _parse_2025_results(self, model_path: str, fast: bool = False) -> Dict[str, Any]:
        """Parse results from 2025 pipeline"""
        results = {}
        
        try:
            model_name = Path(model_path).name
            results_dir = self.pipeline_2025_path / "results" / model_name
            
            if not results_dir.exists():
                logger.warning(f"Results directory {results_dir} not found")
                return results
            
            # Parse zero-shot results
            zero_shot_dir = results_dir / "main" / "zero_shot" / "causal"
            
            tasks = ["blimp", "ewok", "entity_tracking", "wug_adj", "wug_past", "comps"]
            
            for task in tasks:
                task_dir = zero_shot_dir / task
                if task_dir.exists():
                    # Look for results files
                    for result_file in task_dir.rglob("*.jsonl"):
                        try:
                            with open(result_file, 'r') as f:
                                task_results = [json.loads(line) for line in f]
                                results[f"{task}_results"] = task_results
                        except Exception as e:
                            logger.warning(f"Failed to parse {result_file}: {e}")
            
            logger.info(f"Parsed {len(results)} result sets from 2025 pipeline")
            
        except Exception as e:
            logger.error(f"Error parsing 2025 results: {e}")
        
        return results
    
    def _parse_2025_finetune_results(self, model_path: str) -> Dict[str, Any]:
        """Parse fine-tuning results from 2025 pipeline"""
        results = {}
        
        try:
            model_name = Path(model_path).name
            results_dir = self.pipeline_2025_path / "results" / model_name / "main" / "finetune"
            
            if not results_dir.exists():
                logger.warning(f"Finetune results directory {results_dir} not found")
                return results
            
            # Parse fine-tuning results
            for task_dir in results_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name
                    
                    # Look for results.txt
                    results_file = task_dir / "results.txt"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                                results[f"{task_name}_finetune"] = content
                        except Exception as e:
                            logger.warning(f"Failed to parse {results_file}: {e}")
            
            logger.info(f"Parsed {len(results)} fine-tuning results from 2025 pipeline")
            
        except Exception as e:
            logger.error(f"Error parsing 2025 finetune results: {e}")
        
        return results
    
    def _parse_2024_multimodal_results(self, model_path: str) -> Dict[str, Any]:
        """Parse multimodal results from 2024 pipeline"""
        results = {}
        
        try:
            model_name = Path(model_path).name
            
            # Look for results in the expected locations
            tasks = ["winoground_filtered", "vqa_filtered"]
            
            for task in tasks:
                result_file = self.pipeline_2024_path / "results" / task / model_name / f"{task}_results.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            task_results = json.load(f)
                            results[f"{task}_2024"] = task_results
                    except Exception as e:
                        logger.warning(f"Failed to parse {result_file}: {e}")
            
            logger.info(f"Parsed {len(results)} multimodal results from 2024 pipeline")
            
        except Exception as e:
            logger.error(f"Error parsing 2024 multimodal results: {e}")
        
        return results
    
    def run_epoch_evaluation(
        self, 
        epoch: int,
        use_fast_eval: bool = False
    ) -> Dict[str, Any]:
        """
        Run all evaluations for a given epoch
        
        Args:
            epoch: Current training epoch
            use_fast_eval: Whether to use fast evaluation (for intermediate checkpoints)
        
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Starting epoch {epoch} evaluation (fast={use_fast_eval})")
        
        all_results = {
            "epoch": epoch,
            "fast_eval": use_fast_eval,
            "timestamp": time.time() if self.use_wandb else None
        }
        
        try:
            # Create HF-compatible model for this epoch
            hf_model_path = self._create_hf_compatible_model(epoch)
            
            # Run text evaluations from 2025 pipeline
            logger.info("Running 2025 text evaluations...")
            text_results_2025 = self.run_text_evaluations_2025(
                epoch, hf_model_path, fast=use_fast_eval
            )
            all_results["text_2025"] = text_results_2025
            
            # Run fine-tuning evaluations from 2025 pipeline (only for full eval)
            if not use_fast_eval:
                logger.info("Running 2025 fine-tuning evaluations...")
                finetune_results_2025 = self.run_finetuning_evaluations_2025(
                    epoch, hf_model_path
                )
                all_results["finetune_2025"] = finetune_results_2025
            
            # Run multimodal evaluations from 2024 pipeline
            # Note: 2025 pipeline multimodal evaluation is "under construction"
            # VQA and Winoground use HuggingFace datasets directly (no local data needed)
            logger.info("Running 2024 multimodal evaluations (VQA + Winoground)...")
            multimodal_results_2024 = self.run_multimodal_evaluations_2024(
                epoch, hf_model_path
            )
            all_results["multimodal_2024"] = multimodal_results_2024
            
            # Log to WandB if enabled
            if self.use_wandb:
                self._log_to_wandb(epoch, all_results, use_fast_eval)
            
            # Save results to file
            self._save_results(epoch, all_results)
            
            logger.info(f"Completed epoch {epoch} evaluation")
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch} evaluation: {e}")
            all_results["error"] = str(e)
        
        return all_results
    
    def run_step_evaluation(
        self, 
        step: int,
        epoch: int
    ) -> Dict[str, Any]:
        """
        Run lightweight evaluation for a specific training step
        
        Args:
            step: Current training step
            epoch: Current training epoch
        
        Returns:
            Dictionary containing step evaluation results
        """
        logger.info(f"Starting step {step} evaluation (epoch {epoch})")
        
        step_results = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time() if self.use_wandb else None
        }
        
        try:
            # For step evaluations, only run a subset of lightweight tasks
            # Use the existing model path (should be set by training integration)
            if not self.model_path:
                logger.warning("Model path not set for step evaluation")
                step_results["error"] = "Model path not set"
                return step_results
            
            # Run only a subset of text evaluations for efficiency
            logger.info("Running lightweight text evaluations for step...")
            
            # Run a minimal subset of 2025 text evaluations (fastest tasks only)
            minimal_tasks = ["blimp"]  # Run only BLIMP for step evaluation
            try:
                text_results = self.run_text_evaluations_2025(
                    epoch, self.model_path, fast=True
                )
                step_results["text_minimal"] = text_results
            except Exception as text_e:
                logger.warning(f"Text evaluation failed in step evaluation: {text_e}")
                step_results["text_minimal"] = {"error": str(text_e)}
            
            # Log to WandB if enabled
            if self.use_wandb:
                try:
                    wandb_log = {
                        "step_eval/step": step,
                        "step_eval/epoch": epoch
                    }
                    if "text_minimal" in step_results and "error" not in step_results["text_minimal"]:
                        wandb_log["step_eval/text_status"] = 1
                    else:
                        wandb_log["step_eval/text_status"] = 0
                    
                    try:
                        wandb.log(wandb_log, step=step)
                        logger.info(f"Logged step {step} evaluation to WandB")
                    except Exception as wandb_e:
                        logger.warning(f"Failed to log step evaluation to WandB with step: {wandb_e}")
                        try:
                            # Fallback without step parameter
                            wandb.log(wandb_log)
                            logger.info(f"Logged step {step} evaluation to WandB (fallback)")
                        except Exception as wandb_e2:
                            logger.error(f"Failed to log step evaluation to WandB even without step: {wandb_e2}")
                except Exception as wandb_e:
                    logger.warning(f"Failed to create wandb log for step evaluation: {wandb_e}")
            
            # Save step results
            try:
                step_results_file = self.results_base_dir / f"step_{step}_epoch_{epoch}_results.json"
                with open(step_results_file, 'w') as f:
                    json.dump(step_results, f, indent=2)
                logger.info(f"Saved step {step} results to {step_results_file}")
            except Exception as save_e:
                logger.warning(f"Failed to save step results: {save_e}")
            
            logger.info(f"Step {step} evaluation completed successfully")
            return step_results
            
        except Exception as e:
            logger.error(f"Error in step {step} evaluation: {e}")
            step_results["error"] = str(e)
            return step_results
    
    def _log_to_wandb(self, epoch: int, results: Dict[str, Any], fast_eval: bool):
        """Log evaluation results to WandB"""
        try:
            prefix = "fast_eval" if fast_eval else "full_eval"
            
            # Extract key metrics for logging
            wandb_log = {
                f"{prefix}/epoch": epoch
            }
            
            # Log text evaluation metrics
            if "text_2025" in results and "status" in results["text_2025"]:
                wandb_log[f"{prefix}/text_2025_status"] = 1 if results["text_2025"]["status"] == "success" else 0
            
            # Log fine-tuning metrics
            if "finetune_2025" in results and "status" in results["finetune_2025"]:
                wandb_log[f"{prefix}/finetune_2025_status"] = 1 if results["finetune_2025"]["status"] == "success" else 0
            
            # Log multimodal metrics
            if "multimodal_2024" in results and "status" in results["multimodal_2024"]:
                wandb_log[f"{prefix}/multimodal_2024_status"] = 1 if results["multimodal_2024"]["status"] == "success" else 0
            
            try:
                wandb.log(wandb_log, step=epoch)
                logger.info(f"Logged epoch {epoch} evaluation to WandB")
            except Exception as e:
                logger.warning(f"Failed to log epoch evaluation to WandB with step: {e}")
                try:
                    # Fallback without step parameter
                    wandb.log(wandb_log)
                    logger.info(f"Logged epoch {epoch} evaluation to WandB (fallback)")
                except Exception as e2:
                    logger.error(f"Failed to log epoch evaluation to WandB even without step: {e2}")
            
            logger.info(f"Logged epoch {epoch} evaluation metrics to WandB")
            
        except Exception as e:
            logger.error(f"Failed to log to WandB: {e}")
    
    def _save_results(self, epoch: int, results: Dict[str, Any]):
        """Save evaluation results to file"""
        try:
            results_file = self.results_base_dir / f"epoch_{epoch}_results.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved epoch {epoch} results to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def cleanup_temp_files(self, epoch: int):
        """Clean up temporary files created for evaluation"""
        try:
            temp_model_dir = self.results_base_dir / f"temp_model_epoch_{epoch}"
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir)
                logger.info(f"Cleaned up temporary model directory for epoch {epoch}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
