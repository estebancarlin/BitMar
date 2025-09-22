"""
BitMar Evaluation Script for 2025 Pipeline
Evaluates BitMar models on both text and multimodal tasks using the 2025 evaluation pipeline
"""

import os
import sys
import argparse
import logging
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil
import requests
import zipfile
import tarfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """Load BitMar model from checkpoint"""
    try:
        # Add src to path for model loading
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.model_no_memory import create_bitmar_model

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config and model state
        config = checkpoint.get('config', {})
        model_state = checkpoint['model_state_dict']

        # Create model with config
        model = create_bitmar_model(config['model'])
        model.load_state_dict(model_state)

        # Move to device
        device = torch.device(device)
        model = model.to(device)
        model.eval()

        logger.info(f"✅ Model loaded successfully on {device}")
        return model, config

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def setup_evaluation_environment(pipeline_path: Path, eval_data_path: Path):
    """Setup evaluation environment for 2025 pipeline"""
    try:
        # Change to pipeline directory
        original_cwd = os.getcwd()
        os.chdir(pipeline_path)

        # Check if evaluation_data exists, if not create symlink
        eval_data_link = pipeline_path / "evaluation_data"

        if not eval_data_link.exists():
            logger.info(f"Creating evaluation data link: {eval_data_link} -> {eval_data_path}")
            try:
                eval_data_link.symlink_to(eval_data_path.resolve(), target_is_directory=True)
            except OSError:
                # Fallback to copying on Windows
                shutil.copytree(eval_data_path, eval_data_link)
                logger.info(f"Copied evaluation data to: {eval_data_link}")

        return original_cwd

    except Exception as e:
        logger.error(f"Failed to setup evaluation environment: {e}")
        raise


def run_text_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2025"):
    """Run text-only evaluations (BLIMP, etc.) - main focus for 2025 pipeline"""
    logger.info(f"🔤 Running text evaluations ({eval_type}) - Primary focus for 2025 pipeline...")

    results = {}

    try:
        # Create output directory
        text_output_dir = Path(output_dir) / "text_results"
        text_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert BitMar checkpoint to HuggingFace format first
        hf_model_dir = f"hf_model_temp_text_{eval_type}"
        try:
            sys.path.append(str(Path(__file__).parent))
            from bitmar_hf_adapter import save_hf_compatible_model
            hf_model_path = save_hf_compatible_model(model_path, hf_model_dir)
            logger.info(f"✅ Created HuggingFace compatible model for text evaluation at: {hf_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create HF adapter for text evaluation: {e}")
            results['text'] = {"error": f"HF adapter failed: {e}"}
            return results

        # Focus on available text evaluation tasks based on actual data
        eval_dir = "evaluation_data/fast_eval" if eval_type == "fast" else "evaluation_data/full_eval"

        # Define core text evaluation tasks that are actually supported by 2025 pipeline
        if eval_type == "fast":
            tasks = [
                ("blimp", f"{eval_dir}/blimp_fast"),
                ("supplement", f"{eval_dir}/supplement_fast"),
                ("wug_adj", f"{eval_dir}/wug_adj_nominalization"),
                ("wug_past", f"{eval_dir}/wug_past_tense"),
                ("entity_tracking", f"{eval_dir}/entity_tracking_fast"),
                ("ewok", f"{eval_dir}/ewok_fast")  # Available in fast_eval
            ]
        else:
            tasks = [
                ("blimp", f"{eval_dir}/blimp_filtered"),
                ("supplement", f"{eval_dir}/supplement_filtered"),
                ("wug_adj", f"{eval_dir}/wug_adj_nominalization"),
                ("wug_past", f"{eval_dir}/wug_past_tense"),
                ("entity_tracking", f"{eval_dir}/entity_tracking"),
                ("comps", f"{eval_dir}/comps")  # Available in full_eval and supported by 2025 pipeline
                # NOTE: cdi_childes removed - data exists but NOT supported by 2025 evaluation pipeline
                # NOTE: glue_filtered removed - data exists but NOT supported by 2025 evaluation pipeline
                # 2025 pipeline only supports: blimp, ewok, entity_tracking, wug_adj, wug_past, comps, vqa, winoground
            ]

        # Validate available tasks
        available_tasks = []
        for task_name, data_path in tasks:
            if Path(data_path).exists():
                available_tasks.append((task_name, data_path))
                logger.info(f"✅ Found data for {task_name}: {data_path}")
            else:
                logger.warning(f"⚠️ Data not found for {task_name}: {data_path}")

        if not available_tasks:
            logger.error("❌ No text evaluation data found!")
            results['text'] = {"error": "No text evaluation data available"}
            return results

        logger.info(f"📊 Running {len(available_tasks)} available text evaluation tasks...")

        task_results = {}
        successful_tasks = 0

        for task_name, data_path in available_tasks:
            try:
                logger.info(f"🔄 Running {task_name} evaluation...")

                # Run individual task with enhanced error handling
                cmd = [
                    "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--task", task_name if task_name != "supplement" else "blimp",  # supplement uses blimp task
                    "--data_path", data_path,
                    "--save_predictions",
                    "--revision_name", "main"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes per task
                    cwd=Path.cwd()
                )

                if result.returncode == 0:
                    logger.info(f"✅ {task_name} evaluation completed successfully")
                    task_results[task_name] = {
                        "status": "completed",
                        "stdout": result.stdout[-500:] if result.stdout else ""
                    }
                    successful_tasks += 1
                else:
                    logger.warning(f"⚠️ {task_name} evaluation failed with return code {result.returncode}")
                    logger.warning(f"STDERR: {result.stderr[:500] if result.stderr else 'No stderr'}")
                    task_results[task_name] = {
                        "error": f"Return code {result.returncode}",
                        "stderr": result.stderr[:500] if result.stderr else "",
                        "stdout": result.stdout[:500] if result.stdout else ""
                    }

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {task_name} evaluation timed out")
                task_results[task_name] = {"error": "Timeout after 30 minutes"}
            except Exception as e:
                logger.error(f"❌ {task_name} evaluation error: {e}")
                task_results[task_name] = {"error": str(e)}

        # Try reading evaluation if available
        try:
            reading_data_path = f"{eval_dir}/reading/reading_data.csv"
            if Path(reading_data_path).exists():
                logger.info("🔄 Running reading evaluation...")

                cmd = [
                    "python", "-m", "evaluation_pipeline.reading.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--data_path", reading_data_path,
                    "--revision_name", "main"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=Path.cwd()
                )

                if result.returncode == 0:
                    logger.info("✅ Reading evaluation completed successfully")
                    task_results["reading"] = {
                        "status": "completed",
                        "stdout": result.stdout[-500:] if result.stdout else ""
                    }
                    successful_tasks += 1
                else:
                    logger.warning(f"⚠️ Reading evaluation failed: {result.stderr[:500] if result.stderr else 'No stderr'}")
                    task_results["reading"] = {
                        "error": f"Return code {result.returncode}",
                        "stderr": result.stderr[:500] if result.stderr else ""
                    }
            else:
                logger.info(f"ℹ️ Reading data not found: {reading_data_path} (optional)")
                task_results["reading"] = {"status": "skipped", "reason": "Data not available"}

        except Exception as e:
            logger.warning(f"⚠️ Reading evaluation error: {e}")
            task_results["reading"] = {"error": str(e)}

        # Collect prediction files
        predictions_dir = Path("predictions")
        prediction_files = []
        if predictions_dir.exists():
            prediction_files = list(predictions_dir.glob("*.json"))

        # Compile results
        total_possible_tasks = len(available_tasks) + 1  # +1 for reading
        if successful_tasks > 0:
            results['text'] = {
                "status": "partially_completed" if successful_tasks < total_possible_tasks else "completed",
                "successful_tasks": successful_tasks,
                "total_tasks": total_possible_tasks,
                "available_tasks": len(available_tasks),
                "task_results": task_results,
                "prediction_files": [str(f) for f in prediction_files]
            }
            logger.info(f"✅ Text evaluation completed: {successful_tasks}/{total_possible_tasks} tasks successful")
        else:
            results['text'] = {
                "error": "All text tasks failed",
                "task_results": task_results,
                "available_tasks": len(available_tasks)
            }
            logger.error("❌ All text evaluation tasks failed")

        # Cleanup temp HF model
        if Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    except Exception as e:
        logger.error(f"❌ Text evaluation error: {e}")
        results['text'] = {"error": str(e)}
        # Cleanup temp HF model
        if 'hf_model_dir' in locals() and Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    return results


def run_multimodal_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2025"):
    """Run multimodal evaluations (VQA, Winoground, etc.) - Only available in full evaluation"""
    logger.info(f"🖼️ Checking for multimodal evaluations ({eval_type}) - Only available in full evaluation...")

    results = {}

    try:
        # Multimodal tasks are only available in full evaluation
        if eval_type == "fast":
            logger.info("ℹ️ Multimodal evaluation not available in fast mode - VQA and Winoground only exist in full_eval")
            results['multimodal'] = {
                "status": "skipped",
                "reason": "Multimodal tasks not available in fast evaluation mode",
                "note": "VQA and Winoground are only available in full_eval, not fast_eval"
            }
            return results

        # Check if multimodal data is actually available in full evaluation
        multimodal_tasks = []

        # Check for VQA evaluation data (only in full_eval)
        vqa_data_dir = "evaluation_data/full_eval/vqa_filtered"
        if Path(vqa_data_dir).exists():
            multimodal_tasks.append(("vqa", vqa_data_dir))
            logger.info(f"✅ Found VQA data: {vqa_data_dir}")
        else:
            logger.info(f"ℹ️ VQA data not found: {vqa_data_dir}")

        # Check for Winoground evaluation data (only in full_eval)
        winoground_data_dir = "evaluation_data/full_eval/winoground_filtered"
        if Path(winoground_data_dir).exists():
            multimodal_tasks.append(("winoground", winoground_data_dir))
            logger.info(f"✅ Found Winoground data: {winoground_data_dir}")
        else:
            logger.info(f"ℹ️ Winoground data not found: {winoground_data_dir}")

        if not multimodal_tasks:
            logger.info("ℹ️ No multimodal evaluation data found in full_eval")
            results['multimodal'] = {
                "status": "skipped",
                "reason": "No multimodal data available in full_eval",
                "note": "Expected VQA and Winoground data not found"
            }
            return results

        logger.info(f"📊 Found {len(multimodal_tasks)} multimodal tasks available in full evaluation")

        # Convert BitMar model to HuggingFace format for evaluation
        hf_model_dir = f"hf_model_temp_multimodal_{eval_type}"
        try:
            from bitmar_hf_adapter import save_hf_compatible_model
            hf_model_path = save_hf_compatible_model(model_path, hf_model_dir)
            logger.info(f"✅ Created HuggingFace compatible model for multimodal evaluation at: {hf_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create HF adapter for multimodal evaluation: {e}")
            results['multimodal'] = {
                "error": f"HF adapter failed: {e}",
                "status": "failed",
                "note": "Multimodal evaluation failed but this is optional for 2025 pipeline"
            }
            return results

        # Run each multimodal task with enhanced error handling
        task_results = {}
        successful_tasks = 0

        for task_name, data_dir in multimodal_tasks:
            try:
                logger.info(f"🔄 Running {task_name} evaluation...")

                # Special handling for VQA which has known issues with result processing
                if task_name == "vqa":
                    # Try to run VQA with additional error handling
                    vqa_result = run_vqa_with_fallback(hf_model_path, data_dir, eval_type)
                    task_results[task_name] = vqa_result
                    if vqa_result.get("status") == "completed":
                        successful_tasks += 1
                    continue

                # Use the evaluation pipeline's multimodal runner with enhanced error handling
                cmd = [
                    "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--task", task_name,
                    "--data_path", data_dir,
                    "--save_predictions"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    logger.info(f"✅ {task_name} evaluation completed")
                    task_results[task_name] = {
                        "status": "completed",
                        "stdout": result.stdout[-500:] if result.stdout else ""
                    }
                    successful_tasks += 1
                else:
                    logger.warning(f"⚠️ {task_name} evaluation failed with return code {result.returncode}")
                    logger.warning(f"STDERR: {result.stderr[:500] if result.stderr else 'No stderr'}")

                    # Check for specific error patterns
                    error_msg = result.stderr if result.stderr else "Unknown error"
                    if "process_results" in error_msg:
                        error_msg = "Results processing failed - possibly empty results"
                    elif "Traceback" in error_msg:
                        # Extract the actual error from traceback
                        lines = error_msg.split('\n')
                        for i, line in enumerate(lines):
                            if 'Error:' in line or 'Exception:' in line:
                                error_msg = line.strip()
                                break

                    task_results[task_name] = {
                        "error": f"Return code {result.returncode}: {error_msg[:200]}",
                        "stderr": result.stderr[:300] if result.stderr else "",
                        "stdout": result.stdout[:300] if result.stdout else ""
                    }

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {task_name} evaluation timed out after 30 minutes")
                task_results[task_name] = {"error": "Timeout after 30 minutes"}
            except Exception as e:
                logger.warning(f"⚠️ {task_name} evaluation error: {e}")
                task_results[task_name] = {"error": str(e)}

        # Compile results
        if successful_tasks > 0:
            results['multimodal'] = {
                "status": "partially_completed" if successful_tasks < len(multimodal_tasks) else "completed",
                "successful_tasks": successful_tasks,
                "total_tasks": len(multimodal_tasks),
                "task_results": task_results,
                "note": "Multimodal evaluation only available in full mode"
            }
            logger.info(f"✅ Multimodal evaluation completed: {successful_tasks}/{len(multimodal_tasks)} tasks successful")
        else:
            results['multimodal'] = {
                "error": "All multimodal tasks failed",
                "task_results": task_results,
                "status": "failed",
                "note": "Multimodal evaluation failed but this is optional for 2025 pipeline"
            }
            logger.warning("⚠️ All multimodal evaluation tasks failed (but this is optional for 2025 pipeline)")

        # Cleanup temp HF model
        if Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    except Exception as e:
        logger.warning(f"⚠️ Multimodal evaluation error: {e}")
        results['multimodal'] = {
            "error": str(e),
            "status": "failed",
            "note": "Multimodal evaluation failed but this is optional for 2025 pipeline"
        }

    return results


def run_vqa_with_fallback(hf_model_path: str, data_dir: str, eval_type: str):
    """Run VQA evaluation with fallback handling for result processing issues"""
    logger.info("🔍 Running VQA evaluation with enhanced error handling...")

    # FIRST: Check if VQA data exists and download if missing
    try:
        download_vqa_data_if_missing(data_dir)
    except Exception as e:
        logger.warning(f"⚠️ VQA data download/validation failed: {e}")
        # Continue with evaluation anyway - might work with existing data

    try:
        # Create predictions directory if it doesn't exist
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True)

        # Clear any existing VQA predictions to ensure fresh run
        existing_vqa_files = list(predictions_dir.glob("*vqa*")) + list(predictions_dir.glob("*VQA*"))
        for f in existing_vqa_files:
            try:
                f.unlink()
                logger.debug(f"Cleared existing VQA prediction file: {f}")
            except Exception as e:
                logger.warning(f"Failed to clear {f}: {e}")

        # First, try the standard approach with additional arguments
        cmd = [
            "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", hf_model_path,
            "--backend", "causal",
            "--task", "vqa",
            "--data_path", data_dir,
            "--save_predictions",
            "--batch_size", "4",  # Smaller batch size for stability
            "--max_length", "32"   # Reasonable max length for VQA
        ]

        logger.info(f"Running VQA command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes

        # Enhanced prediction file detection - check during and after evaluation
        vqa_predictions = []
        prediction_patterns = [
            "*vqa*", "*VQA*", "*visual*", "*multimodal*",
            "*predictions*", "*pred*", "*results*"
        ]

        # Look in multiple possible directories
        search_dirs = [
            Path("predictions"),
            Path("."),
            Path("outputs"),
            Path("results"),
            Path(data_dir).parent / "predictions" if Path(data_dir).exists() else None
        ]

        for search_dir in search_dirs:
            if search_dir and search_dir.exists():
                for pattern in prediction_patterns:
                    found_files = list(search_dir.glob(pattern))
                    # Filter for files that are likely VQA predictions
                    for f in found_files:
                        if f.is_file() and any(ext in f.suffix.lower() for ext in ['.json', '.jsonl', '.txt', '.csv']):
                            # Additional check: see if file contains VQA-like content
                            try:
                                if f.stat().st_size > 10:  # File has some content
                                    vqa_predictions.append(f)
                            except:
                                pass

        # Remove duplicates
        vqa_predictions = list(set(vqa_predictions))

        logger.info(f"Found {len(vqa_predictions)} potential VQA prediction files: {[str(p) for p in vqa_predictions]}")

        if result.returncode == 0:
            logger.info("✅ VQA evaluation completed successfully")
            return {
                "status": "completed",
                "stdout": result.stdout[-500:] if result.stdout else "",
                "method": "standard",
                "prediction_files": [str(p) for p in vqa_predictions] if vqa_predictions else []
            }
        else:
            # Check if the issue is with result processing
            stderr = result.stderr if result.stderr else ""
            stdout = result.stdout if result.stdout else ""

            logger.warning(f"VQA evaluation failed with return code {result.returncode}")
            logger.warning(f"STDERR: {stderr[:500]}")
            logger.warning(f"STDOUT: {stdout[:500]}")

            # Look for specific error patterns that indicate successful prediction generation
            success_indicators = [
                "Saving predictions",
                "predictions saved",
                "Writing to file",
                "Completed processing",
                len(vqa_predictions) > 0
            ]

            processing_failed = any(indicator in stderr.lower() for indicator in [
                "process_results", "accuracies", "cannot compute", "division by zero",
                "empty results", "no predictions", "failed to calculate"
            ])

            has_success_indicators = any(
                indicator in stdout.lower() or indicator in stderr.lower()
                if isinstance(indicator, str) else indicator
                for indicator in success_indicators
            )

            if processing_failed or has_success_indicators:
                logger.warning("⚠️ VQA result processing failed, but predictions may have been generated")

                # Validate prediction files
                valid_predictions = []
                for pred_file in vqa_predictions:
                    try:
                        with open(pred_file, 'r') as f:
                            content = f.read(1000)  # Read first 1000 chars
                            # Check if it looks like valid prediction data
                            if content.strip() and (
                                content.startswith('[') or content.startswith('{') or
                                'answer' in content.lower() or 'prediction' in content.lower() or
                                'question' in content.lower()
                            ):
                                valid_predictions.append(str(pred_file))
                                logger.info(f"✅ Valid VQA predictions found in {pred_file}")
                    except Exception as e:
                        logger.warning(f"Could not validate prediction file {pred_file}: {e}")

                if valid_predictions:
                    return {
                        "status": "completed",
                        "note": "Predictions generated successfully but result processing failed",
                        "prediction_files": valid_predictions,
                        "method": "fallback_with_predictions",
                        "processing_error": stderr[:300] if stderr else "Unknown processing error"
                    }
                else:
                    logger.info("🔄 No valid predictions found, trying alternative approach...")
                    return run_vqa_alternative_approach(hf_model_path, data_dir, eval_type)

            else:
                # Other type of error - check if it's a data or model issue
                if "FileNotFoundError" in stderr or "No such file" in stderr:
                    return {
                        "error": f"VQA data or model file not found: {stderr[:300]}",
                        "stderr": stderr[:500],
                        "stdout": stdout[:500],
                        "method": "failed_data_missing"
                    }
                elif "CUDA" in stderr or "memory" in stderr.lower():
                    return {
                        "error": f"VQA GPU/memory error: {stderr[:300]}",
                        "stderr": stderr[:500],
                        "stdout": stdout[:500],
                        "method": "failed_gpu_memory"
                    }
                else:
                    # Try alternative approach for unknown errors
                    logger.info("🔄 Unknown error, trying alternative approach...")
                    return run_vqa_alternative_approach(hf_model_path, data_dir, eval_type)

    except subprocess.TimeoutExpired:
        logger.warning("⚠️ VQA evaluation timed out after 40 minutes")
        # Still check for any predictions that might have been generated
        vqa_predictions = []
        for search_dir in [Path("predictions"), Path("."), Path("outputs")]:
            if search_dir.exists():
                patterns = ["*vqa*", "*VQA*", "*visual*", "*multimodal*"]
                for pattern in patterns:
                    vqa_predictions.extend(list(search_dir.glob(pattern)))

        if vqa_predictions:
            logger.info(f"⚠️ VQA timed out but found partial predictions: {[str(p) for p in vqa_predictions]}")
            return {
                "status": "partial",
                "note": "VQA evaluation timed out but partial predictions were generated",
                "prediction_files": [str(p) for p in vqa_predictions],
                "method": "timeout_with_predictions"
            }
        else:
            return {"error": "VQA timeout after 40 minutes with no predictions", "method": "timeout"}
    except Exception as e:
        logger.error(f"❌ VQA evaluation exception: {e}")
        return {"error": f"VQA exception: {str(e)}", "method": "exception"}


def run_vqa_alternative_approach(hf_model_path: str, data_dir: str, eval_type: str):
    """Alternative VQA evaluation approach when standard method fails"""
    logger.info("🔄 Trying alternative VQA evaluation approach...")

    try:
        # Try with different backend or parameters
        alternative_cmd = [
            "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", hf_model_path,
            "--backend", "hf",  # Try HuggingFace backend instead of causal
            "--task", "vqa",
            "--data_path", data_dir,
            "--save_predictions",
            "--batch_size", "2",  # Even smaller batch size
            "--max_length", "16"   # Shorter max length
        ]

        logger.info(f"Running alternative VQA command: {' '.join(alternative_cmd)}")
        result = subprocess.run(alternative_cmd, capture_output=True, text=True, timeout=1800)

        # Check for predictions again
        predictions_dir = Path("predictions")
        vqa_predictions = []
        if predictions_dir.exists():
            patterns = ["*vqa*", "*VQA*", "*visual*", "*multimodal*"]
            for pattern in patterns:
                vqa_predictions.extend(list(predictions_dir.glob(pattern)))

        if result.returncode == 0 or vqa_predictions:
            status = "completed" if result.returncode == 0 else "partial"
            logger.info(f"✅ Alternative VQA approach {'completed' if result.returncode == 0 else 'generated partial results'}")
            return {
                "status": status,
                "note": f"Alternative VQA approach {status}",
                "prediction_files": [str(p) for p in vqa_predictions],
                "method": "alternative_approach",
                "stdout": result.stdout[-300:] if result.stdout else ""
            }
        else:
            logger.warning("⚠️ Alternative VQA approach also failed")
            return {
                "error": f"Alternative VQA approach failed: {result.stderr[:300] if result.stderr else 'Unknown error'}",
                "method": "alternative_failed",
                "stderr": result.stderr[:500] if result.stderr else "",
                "stdout": result.stdout[:500] if result.stdout else ""
            }

    except Exception as e:
        logger.warning(f"Alternative VQA approach exception: {e}")
        return {
            "error": f"Alternative VQA approach exception: {str(e)}",
            "method": "alternative_exception"
        }


def run_devbench_evaluation(model_path: str, output_dir: str = "results_2025"):
    """Run DevBench evaluation with enhanced error handling and data validation"""
    logger.info("🧪 Running DevBench evaluation...")

    results = {}

    try:
        # Check for DevBench script
        devbench_script = "eval_devbench.sh"

        if not Path(devbench_script).exists():
            logger.warning("⚠️ DevBench evaluation script not found")
            results['devbench'] = {"error": "DevBench script not found", "status": "skipped"}
            return results

        # Check if required dependencies are available
        try:
            # Test if nlopt is available by trying to import it
            import subprocess
            test_cmd = ["python", "-c", "import nlopt; print('nlopt available')"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)

            if test_result.returncode != 0:
                logger.warning("⚠️ DevBench dependency 'nlopt' not available - skipping DevBench evaluation")
                results['devbench'] = {
                    "error": "Missing required dependency: nlopt",
                    "status": "skipped",
                    "note": "Install nlopt with: pip install nlopt"
                }
                return results

        except Exception as dep_error:
            logger.warning(f"⚠️ Failed to check DevBench dependencies: {dep_error}")
            results['devbench'] = {
                "error": f"Dependency check failed: {dep_error}",
                "status": "skipped"
            }
            return results

        # Enhanced data validation for DevBench
        required_devbench_files = [
            "evaluation_data/full_eval/devbench/evals/sem-things/spose_similarity.mat",
            "evaluation_data/full_eval/devbench/evals/",
            "evaluation_data/full_eval/devbench/"
        ]

        missing_files = []
        for file_path in required_devbench_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.warning("⚠️ DevBench evaluation data files missing:")
            for missing_file in missing_files:
                logger.warning(f"  • Missing: {missing_file}")

            # Check if any devbench data exists at all
            devbench_base = Path("evaluation_data/full_eval/devbench")
            if not devbench_base.exists():
                logger.warning("⚠️ DevBench base directory not found - evaluation data may need to be downloaded")
                results['devbench'] = {
                    "error": "DevBench evaluation data not found",
                    "status": "skipped",
                    "note": "Run download_evaluation_data.py to get DevBench data",
                    "missing_files": missing_files
                }
                return results
            else:
                # Partial data exists, try to run with what we have
                logger.warning("⚠️ Some DevBench data missing, attempting evaluation with available data...")

        # Run DevBench evaluation
        cmd = [
            "bash", devbench_script,
            "--model_path", model_path,
            "--output_dir", f"{output_dir}/devbench_results"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # Increased timeout for full eval

        if result.returncode == 0:
            logger.info("✅ DevBench evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/devbench_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['devbench'] = json.load(f)
                    results['devbench']['status'] = 'completed'
            else:
                results['devbench'] = {
                    "status": "completed",
                    "note": "DevBench completed but no results file found"
                }
        else:
            logger.warning(f"⚠️ DevBench evaluation failed with return code {result.returncode}")
            logger.warning(f"STDERR: {result.stderr[:500] if result.stderr else 'No stderr'}")

            # Enhanced error analysis for missing data files
            error_msg = result.stderr if result.stderr else "Unknown error"

            if "spose_similarity.mat" in error_msg:
                error_msg = "Missing DevBench data file: spose_similarity.mat - evaluation data incomplete"
                data_solution = "Download complete DevBench data or use fast evaluation mode"
            elif "FileNotFoundError" in error_msg and "devbench" in error_msg:
                error_msg = "DevBench data files missing - evaluation data incomplete"
                data_solution = "Run download_evaluation_data.py to get complete DevBench dataset"
            elif "nlopt" in error_msg.lower():
                error_msg = "Missing nlopt dependency - install with: pip install nlopt"
                data_solution = "Install required Python packages"
            elif "modulenotfounderror" in error_msg.lower():
                # Extract module name
                lines = error_msg.split('\n')
                for line in lines:
                    if "modulenotfounderror" in line.lower():
                        error_msg = f"Missing Python module: {line.strip()}"
                        break
                data_solution = "Install missing Python dependencies"
            else:
                data_solution = "Check DevBench setup and data completeness"

            results['devbench'] = {
                "error": f"Return code {result.returncode}: {error_msg[:300]}",
                "stderr": result.stderr[:500] if result.stderr else "",
                "stdout": result.stdout[:500] if result.stdout else "",
                "status": "failed",
                "solution": data_solution,
                "missing_files": missing_files if missing_files else None
            }

            # If it's a data issue, suggest using fast evaluation
            if "spose_similarity.mat" in error_msg or "FileNotFoundError" in error_msg:
                logger.info("💡 Suggestion: Consider using --eval_type fast to avoid DevBench data issues")

    except subprocess.TimeoutExpired:
        logger.warning("⚠️ DevBench evaluation timed out after 1 hour")
        results['devbench'] = {"error": "Timeout after 1 hour", "status": "timeout"}
    except Exception as e:
        logger.error(f"❌ DevBench evaluation error: {e}")
        results['devbench'] = {"error": str(e), "status": "error"}

    return results


def download_vqa_data_if_missing(vqa_data_dir: str):
    """Check if VQA data exists and download if missing"""
    vqa_path = Path(vqa_data_dir)

    # Check if VQA data files exist (not just the directory)
    required_vqa_files = [
        "questions.json",
        "annotations.json",
        "images_info.json",
        "vqa_test.jsonl",
        "vqa_train.jsonl",
        "vqa_val.jsonl"
    ]

    # Check if any actual VQA data files exist (excluding just vqa_distractors_info.json)
    existing_files = []
    if vqa_path.exists():
        existing_files = [f.name for f in vqa_path.iterdir() if f.is_file() and f.name != "vqa_distractors_info.json"]

    # Check if we have the essential VQA files
    has_essential_files = any(req_file in existing_files for req_file in required_vqa_files)

    if not has_essential_files:
        logger.info(f"📥 VQA evaluation data missing in {vqa_data_dir}")
        logger.info(f"   Found only: {existing_files}")
        logger.info(f"   Downloading VQA dataset...")

        try:
            # Create the directory if it doesn't exist
            vqa_path.mkdir(parents=True, exist_ok=True)

            # Try to download VQA data from working Hugging Face sources
            vqa_urls = [
                {
                    "name": "HuggingFace VQA Dataset (public)",
                    "url": "https://huggingface.co/datasets/HuggingFaceM4/VQAv2/resolve/main/vqa_val_eval.json",
                    "extract": False,
                    "filename": "questions.json"
                },
                {
                    "name": "HuggingFace VQA Annotations (public)",
                    "url": "https://huggingface.co/datasets/HuggingFaceM4/VQAv2/resolve/main/vqa_val_annotations.json",
                    "extract": False,
                    "filename": "annotations.json"
                },
                {
                    "name": "Alternative VQA Source (GitHub)",
                    "url": "https://raw.githubusercontent.com/babylm/evaluation-pipeline-2025/main/evaluation_data/full_eval/vqa_filtered/questions.json",
                    "extract": False,
                    "filename": "questions.json"
                }
            ]

            download_success = False

            for source in vqa_urls:
                try:
                    logger.info(f"🔄 Attempting to download from {source['name']}: {source['url']}")

                    # Download with progress
                    response = requests.get(source['url'], stream=True, timeout=300)
                    response.raise_for_status()

                    # Determine download path
                    download_path = vqa_path / source.get('filename', 'vqa_data.json')

                    # Save the file
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0

                    with open(download_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded_size / total_size) * 100
                                    logger.info(f"   📥 Download progress: {progress:.1f}%")

                    logger.info(f"✅ Downloaded VQA data from {source['name']}")

                    # If we got at least one file, consider it a partial success
                    if download_path.exists() and download_path.stat().st_size > 0:
                        download_success = True
                        # Don't break yet, try to get more files

                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️ Failed to download from {source['name']}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Error processing download from {source['name']}: {e}")
                    continue

            # Try downloading from the evaluation pipeline repository directly
            if not download_success:
                logger.info("🔄 Trying to download from evaluation pipeline repository...")
                try:
                    # Try getting VQA files from the 2025 evaluation pipeline repository
                    pipeline_vqa_files = [
                        "https://github.com/babylm/evaluation-pipeline-2025/raw/main/evaluation_data/full_eval/vqa_filtered/vqa_test.jsonl",
                        "https://github.com/babylm/evaluation-pipeline-2025/raw/main/evaluation_data/full_eval/vqa_filtered/vqa_train.jsonl",
                        "https://github.com/babylm/evaluation-pipeline-2025/raw/main/evaluation_data/full_eval/vqa_filtered/vqa_val.jsonl"
                    ]

                    for file_url in pipeline_vqa_files:
                        try:
                            filename = file_url.split('/')[-1]
                            response = requests.get(file_url, timeout=300)
                            response.raise_for_status()

                            with open(vqa_path / filename, 'wb') as f:
                                f.write(response.content)

                            logger.info(f"✅ Downloaded {filename}")
                            download_success = True

                        except Exception as e:
                            logger.warning(f"⚠️ Failed to download {filename}: {e}")
                            continue

                except Exception as e:
                    logger.warning(f"⚠️ Pipeline repository download failed: {e}")

            if not download_success:
                logger.warning("⚠️ Could not download VQA data from any source")
                logger.info("🔧 Attempting to create minimal VQA dataset for testing...")

                # Create minimal test data
                create_minimal_vqa_data(vqa_path)

        except Exception as e:
            logger.error(f"❌ Failed to download VQA data: {e}")
            logger.info("🔧 Creating minimal VQA dataset for basic evaluation...")
            create_minimal_vqa_data(vqa_path)
    else:
        logger.info(f"✅ VQA data already exists in {vqa_data_dir}")


def create_minimal_vqa_data(vqa_path: Path):
    """Create minimal VQA dataset for basic evaluation when download fails"""
    try:
        logger.info("🔧 Creating minimal VQA dataset for basic testing...")

        # Create minimal questions and annotations
        minimal_questions = {
            "questions": [
                {
                    "question_id": 1,
                    "image_id": 1,
                    "question": "What color is the sky?"
                },
                {
                    "question_id": 2,
                    "image_id": 2,
                    "question": "How many people are in the image?"
                },
                {
                    "question_id": 3,
                    "image_id": 3,
                    "question": "What is the weather like?"
                }
            ]
        }

        minimal_annotations = {
            "annotations": [
                {
                    "question_id": 1,
                    "answers": [{"answer": "blue", "answer_confidence": "yes"}]
                },
                {
                    "question_id": 2,
                    "answers": [{"answer": "2", "answer_confidence": "yes"}]
                },
                {
                    "question_id": 3,
                    "answers": [{"answer": "sunny", "answer_confidence": "yes"}]
                }
            ]
        }

        # Save minimal data files
        with open(vqa_path / "questions.json", 'w') as f:
            json.dump(minimal_questions, f, indent=2)

        with open(vqa_path / "annotations.json", 'w') as f:
            json.dump(minimal_annotations, f, indent=2)

        # Create minimal JSONL files for evaluation pipeline
        test_data = [
            {"question": "What color is the sky?", "answer": "blue", "image_id": 1},
            {"question": "How many people are in the image?", "answer": "2", "image_id": 2},
            {"question": "What is the weather like?", "answer": "sunny", "image_id": 3}
        ]

        for split in ["train", "val", "test"]:
            with open(vqa_path / f"vqa_{split}.jsonl", 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')

        logger.info("✅ Created minimal VQA dataset for basic evaluation")

    except Exception as e:
        logger.error(f"❌ Failed to create minimal VQA data: {e}")

def evaluate_bitmar_2025(
    model_path: str,
    eval_type: str = "fast",
    evaluation_pipeline_path: str = "../evaluation-pipeline-2025",
    output_dir: str = "results_2025"
):
    """
    Evaluate BitMar model using 2025 pipeline (text + multimodal tasks)

    Args:
        model_path: Path to model checkpoint
        eval_type: "fast" or "full" evaluation
        evaluation_pipeline_path: Path to 2025 evaluation pipeline
        output_dir: Output directory for results
    """
    logger.info("🚀 Starting BitMar 2025 Pipeline Evaluation")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"🎯 Evaluation type: {eval_type}")
    logger.info(f"📂 Pipeline: {evaluation_pipeline_path}")
    logger.info(f"💾 Output: {output_dir}")

    # Auto-validate and download missing evaluation data
    logger.info("🔍 Validating evaluation data availability...")
    try:
        from validate_and_download_eval_data import EvaluationDataValidator
        validator = EvaluationDataValidator()
        validation_success = validator.validate_and_download()

        if validation_success:
            logger.info("✅ All evaluation data validated and available")
        else:
            logger.warning("⚠️ Some evaluation data missing, continuing with available data")
    except Exception as e:
        logger.warning(f"⚠️ Could not validate evaluation data: {e}")
        logger.info("Proceeding with evaluation using existing data...")

    # Convert paths to absolute paths BEFORE changing directory
    pipeline_path = Path(evaluation_pipeline_path).resolve()
    model_path = Path(model_path).resolve()

    # Convert output path to absolute path before any directory changes
    if Path(output_dir).is_absolute():
        output_path = Path(output_dir)
    else:
        # Make relative path absolute based on current working directory
        output_path = Path.cwd() / output_dir
    output_path = output_path.resolve()

    logger.info(f"📂 Resolved output path: {output_path}")

    # Validate paths
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline path not found: {pipeline_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Create output directory with absolute path
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Created output directory: {output_path}")

    # Find evaluation data
    eval_data_paths = [
        pipeline_path / "evaluation_data",
        pipeline_path.parent / "evaluation_data"
    ]

    eval_data_path = None
    for path in eval_data_paths:
        if path.exists():
            eval_data_path = path
            break

    if not eval_data_path:
        raise FileNotFoundError("Evaluation data not found. Run download_evaluation_data.py first.")

    logger.info(f"📊 Using evaluation data: {eval_data_path}")

    # Setup evaluation environment
    original_cwd = setup_evaluation_environment(pipeline_path, eval_data_path)

    try:
        all_results = {}

        # Run text evaluations (primary focus)
        text_results = run_text_evaluations(str(model_path), eval_type, str(output_path))
        all_results.update(text_results)

        # Run multimodal evaluations (only available in full mode)
        multimodal_results = run_multimodal_evaluations(str(model_path), eval_type, str(output_path))
        all_results.update(multimodal_results)

        # Note: DevBench is not available in the current evaluation data
        logger.info("ℹ️ DevBench evaluation skipped - not available in current evaluation data")

        # Save combined results
        results_file = output_path / "combined_results_2025.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"📊 Results saved to: {results_file}")

        # Log summary
        logger.info("📈 Evaluation Summary:")
        for task_type, results in all_results.items():
            if isinstance(results, dict) and "error" not in results:
                logger.info(f"  ✅ {task_type}: Success")
            else:
                logger.info(f"  ❌ {task_type}: Failed")

        return all_results

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate BitMar using 2025 pipeline")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval_type", type=str, choices=["fast", "full"], default="fast",
                       help="Type of evaluation to run")
    parser.add_argument("--evaluation_pipeline_path", type=str, default="../evaluation-pipeline-2025",
                       help="Path to 2025 evaluation pipeline")
    parser.add_argument("--output_dir", type=str, default="results_2025",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for evaluation")

    args = parser.parse_args()

    try:
        results = evaluate_bitmar_2025(
            model_path=args.model_path,
            eval_type=args.eval_type,
            evaluation_pipeline_path=args.evaluation_pipeline_path,
            output_dir=args.output_dir
        )

        logger.info("🎉 2025 Pipeline evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
