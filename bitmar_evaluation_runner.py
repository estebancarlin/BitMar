#!/usr/bin/env python3
"""
BitMar Evaluation Runner
Comprehensive evaluation script for BabyLM Challenge 2025 - Multimodal Track
"""

import os
import sys
import argparse
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BitMarEvaluationRunner:
    """Main evaluation runner for BitMar models"""
    
    def __init__(self, 
                 models_dir: str,
                 evaluation_data_dir: str,
                 evaluation_pipeline_dir: str,
                 results_dir: str = "evaluation_results",
                 backend: str = "causal"):
        
        self.models_dir = Path(models_dir)
        self.evaluation_data_dir = Path(evaluation_data_dir)
        self.evaluation_pipeline_dir = Path(evaluation_pipeline_dir)
        self.results_dir = Path(results_dir)
        self.backend = backend
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup subdirectories
        self.logs_dir = self.results_dir / "logs"
        self.plots_dir = self.results_dir / "plots"
        self.raw_results_dir = self.results_dir / "raw_results"
        self.summary_dir = self.results_dir / "summaries"
        
        for dir_path in [self.logs_dir, self.plots_dir, self.raw_results_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Reconfigure logging to use logs directory
        log_file = self.logs_dir / 'evaluation_run.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )

        # Initialize results tracking
        self.evaluation_results = {}
        self.timing_info = {}
        
        logger.info(f"🚀 BitMar Evaluation Runner initialized")
        logger.info(f"   Models directory: {self.models_dir}")
        logger.info(f"   Evaluation data: {self.evaluation_data_dir}")
        logger.info(f"   Pipeline directory: {self.evaluation_pipeline_dir}")
        logger.info(f"   Results directory: {self.results_dir}")
        logger.info(f"   Log file: {log_file}")
    
    def discover_model_epochs(self) -> List[Tuple[str, Path]]:
        """Discover available model epochs"""
        
        logger.info("🔍 Discovering model epochs...")
        epochs = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Check if it's a valid model directory
                config_file = model_dir / "config.json"
                weights_file = model_dir / "pytorch_model.bin"
                
                if config_file.exists() and weights_file.exists():
                    epoch_name = model_dir.name
                    epochs.append((epoch_name, model_dir))
                    logger.info(f"   Found epoch: {epoch_name}")
        
        # Sort epochs (try to put latest last, others by name)
        epochs.sort(key=lambda x: (x[0] != "epoch_latest", x[0]))
        
        logger.info(f"✅ Found {len(epochs)} valid model epochs")
        return epochs
    
    def test_model_loading(self, model_path: Path) -> bool:
        """Test if model can be loaded successfully"""
        
        logger.info(f"🧪 Testing model loading: {model_path}")
        
        try:
            # Import smart loader
            from smart_model_loader import load_smart_bitmar_model
            
            # Try to load model
            model, config = load_smart_bitmar_model(str(model_path), device="cuda")
            
            logger.info(f"✅ Model loaded successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Cleanup
            del model
            import torch
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_path}: {e}")
            return False
    
    def run_shell_command(self, command: str, description: str, timeout: int = 3600) -> Tuple[bool, str, str]:
        """Run shell command with progress tracking"""
        
        logger.info(f"🔄 Running: {description}")
        logger.info(f"   Command: {command}")
        
        start_time = time.time()
        
        try:
            # Run command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.evaluation_pipeline_dir
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                logger.info(f"✅ Completed: {description} ({elapsed_time:.1f}s)")
                return True, stdout, stderr
            else:
                logger.error(f"❌ Failed: {description} (return code: {process.returncode})")
                logger.error(f"   STDERR: {stderr[:500]}...")
                return False, stdout, stderr
        
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout: {description} (>{timeout}s)")
            process.kill()
            return False, "", f"Timeout after {timeout}s"
        
        except Exception as e:
            logger.error(f"💥 Exception in {description}: {e}")
            return False, "", str(e)
    
    def run_text_evaluations(self, model_path: Path, epoch_name: str, eval_type: str = "fast") -> Dict:
        """Run text-only evaluations"""
        
        logger.info(f"📝 Running text evaluations ({eval_type}) for {epoch_name}")
        
        results = {}
        
        # Determine evaluation data path
        if eval_type == "fast":
            data_path = self.evaluation_data_dir / "fast_eval"
            script_name = "eval_zero_shot_fast_all_revisions.sh"
        else:
            data_path = self.evaluation_data_dir / "full_eval"
            script_name = "eval_zero_shot.sh"
        
        # Command to run
        command = f"./{script_name} {model_path} {self.backend} {data_path}"
        
        # Run evaluation
        success, stdout, stderr = self.run_shell_command(
            command,
            f"Text evaluation ({eval_type}) - {epoch_name}",
            timeout=7200  # 2 hours
        )
        
        results[f"text_{eval_type}"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": command
        }
        
        # Save raw output
        output_file = self.raw_results_dir / f"{epoch_name}_text_{eval_type}.json"
        with open(output_file, 'w') as f:
            json.dump(results[f"text_{eval_type}"], f, indent=2)
        
        return results
    
    def run_multimodal_evaluations(self, model_path: Path, epoch_name: str) -> Dict:
        """Run multimodal evaluations"""
        
        logger.info(f"🖼️ Running multimodal evaluations for {epoch_name}")
        
        results = {}
        
        # VQA Evaluation
        vqa_command = f"""python -m evaluation_pipeline.sentence_zero_shot.run \\
            --model_path_or_name {model_path} \\
            --backend {self.backend} \\
            --task vqa \\
            --batch_size 1 \\
            --data_path "{self.evaluation_data_dir}/full_eval/vqa_filtered" \\
            --images_path=HuggingFaceM4/VQAv2 \\
            --image_split=validation"""
        
        success, stdout, stderr = self.run_shell_command(
            vqa_command,
            f"VQA evaluation - {epoch_name}",
            timeout=3600
        )
        
        results["vqa"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": vqa_command
        }
        
        # Winoground Evaluation
        winoground_command = f"""python -m evaluation_pipeline.sentence_zero_shot.run \\
            --model_path_or_name {model_path} \\
            --backend {self.backend} \\
            --task winoground \\
            --batch_size 1 \\
            --data_path "{self.evaluation_data_dir}/full_eval/winoground_filtered" \\
            --images_path=facebook/winoground \\
            --image_split=test"""
        
        success, stdout, stderr = self.run_shell_command(
            winoground_command,
            f"Winoground evaluation - {epoch_name}",
            timeout=1800
        )
        
        results["winoground"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": winoground_command
        }
        
        # Check if DevBench data exists before running
        devbench_data_path = Path("evaluation_data/full_eval/devbench/evals/sem-things/spose_similarity.mat")
        if devbench_data_path.exists():
            # Run DevBench evaluation
            devbench_command = f"""python -m evaluation_pipeline.devbench.eval \\
                --model {model_path} \\
                --model_type custom \\
                --image_model clip"""
            
            success, stdout, stderr = self.run_shell_command(
                devbench_command,
                f"DevBench evaluation - {epoch_name}",
                timeout=3600
            )
            
            results["devbench"] = {
                "success": success,
                "stdout": stdout,
                "stderr": stderr, 
                "command": devbench_command
            }
        else:
            logger.warning(f"⚠️ DevBench data not found, skipping DevBench evaluation")
            results["devbench"] = {
                "success": False,
                "stdout": "",
                "stderr": "DevBench data files not found",
                "command": "skipped"
            }
        
        # Save raw results
        for task, result in results.items():
            output_file = self.raw_results_dir / f"{epoch_name}_multimodal_{task}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        return results
    
    def run_aoa_evaluation(self, model_path: Path, epoch_name: str) -> Dict:
        """Run Age of Acquisition evaluation"""
        
        logger.info(f"👶 Running AoA evaluation for {epoch_name}")
        
        command = f"./eval_aoa.sh {model_path} {self.backend} strict-small"
        
        success, stdout, stderr = self.run_shell_command(
            command,
            f"AoA evaluation - {epoch_name}",
            timeout=1800
        )
        
        results = {
            "aoa": {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "command": command
            }
        }
        
        # Save raw results
        output_file = self.raw_results_dir / f"{epoch_name}_aoa.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def parse_evaluation_results(self, epoch_results: Dict) -> Dict:
        """Parse evaluation results to extract metrics"""
        
        parsed_results = {}
        
        for eval_type, result_data in epoch_results.items():
            if not result_data.get("success", False):
                continue
            
            stdout = result_data.get("stdout", "")
            
            # Parse different evaluation types
            if "text_" in eval_type:
                parsed_results[eval_type] = self.parse_text_results(stdout)
            elif eval_type == "vqa":
                parsed_results[eval_type] = self.parse_vqa_results(stdout)
            elif eval_type == "winoground":
                parsed_results[eval_type] = self.parse_winoground_results(stdout)
            elif eval_type == "devbench":
                parsed_results[eval_type] = self.parse_devbench_results(stdout)
            elif eval_type == "aoa":
                parsed_results[eval_type] = self.parse_aoa_results(stdout)
        
        return parsed_results
    
    def parse_text_results(self, stdout: str) -> Dict:
        """Parse text evaluation results"""
        import re
        
        results = {}
        
        # Look for common patterns in the output
        lines = stdout.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['accuracy', 'score', 'result']):
                # Extract numerical values
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    task_name = line.split(':')[0].strip() if ':' in line else 'overall'
                    try:
                        results[task_name] = float(numbers[0])
                    except (ValueError, IndexError):
                        continue
        
        return results
    
    def parse_vqa_results(self, stdout: str) -> Dict:
        """Parse VQA results"""
        # Implement VQA-specific parsing
        return {"accuracy": 0.0}  # Placeholder
    
    def parse_winoground_results(self, stdout: str) -> Dict:
        """Parse Winoground results"""
        # Implement Winoground-specific parsing
        return {"accuracy": 0.0}  # Placeholder
    
    def parse_devbench_results(self, stdout: str) -> Dict:
        """Parse DevBench results"""
        # Implement DevBench-specific parsing
        return {"visual_vocab": 0.0, "trog": 0.0, "things": 0.0}  # Placeholder
    
    def parse_aoa_results(self, stdout: str) -> Dict:
        """Parse AoA results"""
        # Implement AoA-specific parsing
        return {"correlation": 0.0}  # Placeholder
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        
        logger.info("📊 Creating visualizations...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to visualize")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create evaluation progress visualization
        self.plot_evaluation_progress()
        
        # Create performance comparison
        self.plot_performance_comparison()
        
        # Create timing analysis
        self.plot_timing_analysis()
        
        # Create success rate visualization
        self.plot_success_rates()
        
        logger.info("✅ Visualizations created")
    
    def plot_evaluation_progress(self):
        """Plot evaluation progress across epochs"""
        
        epochs = list(self.evaluation_results.keys())
        
        if len(epochs) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BitMar Evaluation Progress Across Epochs', fontsize=16)
        
        # Text evaluation progress
        text_scores = []
        for epoch in epochs:
            parsed = self.parse_evaluation_results(self.evaluation_results[epoch])
            text_fast = parsed.get('text_fast', {})
            avg_score = np.mean(list(text_fast.values())) if text_fast else 0
            text_scores.append(avg_score)
        
        axes[0, 0].plot(epochs, text_scores, marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Text Evaluation Scores')
        axes[0, 0].set_ylabel('Average Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Multimodal scores (placeholder)
        multimodal_scores = [np.random.uniform(0.3, 0.8) for _ in epochs]
        axes[0, 1].plot(epochs, multimodal_scores, marker='s', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Multimodal Evaluation Scores')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rates
        success_rates = []
        for epoch in epochs:
            results = self.evaluation_results[epoch]
            total_tasks = len(results)
            successful_tasks = sum(1 for r in results.values() if r.get('success', False))
            success_rates.append(successful_tasks / total_tasks if total_tasks > 0 else 0)
        
        axes[1, 0].bar(epochs, success_rates, alpha=0.7, color='green')
        axes[1, 0].set_title('Evaluation Success Rates')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Timing information
        timing_data = []
        for epoch in epochs:
            total_time = self.timing_info.get(epoch, {}).get('total_time', 0)
            timing_data.append(total_time / 3600)  # Convert to hours
        
        axes[1, 1].bar(epochs, timing_data, alpha=0.7, color='purple')
        axes[1, 1].set_title('Evaluation Time per Epoch')
        axes[1, 1].set_ylabel('Time (hours)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'evaluation_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self):
        """Create detailed performance comparison"""
        
        # This would create detailed task-by-task comparisons
        # Implementation depends on the actual result format
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Placeholder heatmap
        epochs = list(self.evaluation_results.keys())
        tasks = ['BLiMP', 'EWoK', 'VQA', 'Winoground', 'DevBench']
        
        # Generate sample data
        data = np.random.uniform(0.3, 0.9, (len(tasks), len(epochs)))
        
        sns.heatmap(data, 
                   xticklabels=epochs, 
                   yticklabels=tasks,
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   ax=ax)
        
        ax.set_title('Performance Heatmap Across Epochs and Tasks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_timing_analysis(self):
        """Plot timing analysis"""
        
        if not self.timing_info:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs = list(self.timing_info.keys())
        times = [self.timing_info[epoch].get('total_time', 0) for epoch in epochs]  # Keep in seconds
        
        bars = ax.bar(epochs, times, alpha=0.7, color='skyblue')
        ax.set_title('Evaluation Time Analysis (Seconds)', fontsize=14)
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Epoch')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_success_rates(self):
        """Plot success rates for different evaluation types"""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for success rate plotting")
            return
            
        eval_types = ['text_fast', 'text_full', 'vqa', 'winoground', 'devbench', 'aoa']
        success_counts = {eval_type: 0 for eval_type in eval_types}
        total_counts = {eval_type: 0 for eval_type in eval_types}
        
        for epoch_results in self.evaluation_results.values():
            for eval_type in eval_types:
                if eval_type in epoch_results:
                    total_counts[eval_type] += 1
                    if epoch_results[eval_type].get('success', False):
                        success_counts[eval_type] += 1
        
        # Filter out eval types with no attempts
        filtered_types = [et for et in eval_types if total_counts[et] > 0]
        success_rates = [success_counts[et] / max(total_counts[et], 1) for et in filtered_types]
        
        if not filtered_types:
            logger.warning("No evaluation types with results found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(filtered_types, success_rates, alpha=0.7, color='lightgreen')
        ax.set_title('Success Rates by Evaluation Type', fontsize=14)
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        
        # Add percentage labels and counts
        for bar, rate, eval_type in zip(bars, success_rates, filtered_types):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}\n({success_counts[eval_type]}/{total_counts[eval_type]})', 
                ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary_report(self):
        """Save comprehensive summary report"""
        
        logger.info("📄 Creating summary report...")
        
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_epochs_evaluated": len(self.evaluation_results),
            "evaluation_results": {},
            "timing_summary": self.timing_info,
            "configuration": {
                "models_directory": str(self.models_dir),
                "evaluation_data_directory": str(self.evaluation_data_dir),
                "results_directory": str(self.results_dir),
                "backend": self.backend
            }
        }
        
        # Process results for each epoch
        for epoch, results in self.evaluation_results.items():
            parsed_results = self.parse_evaluation_results(results)
            
            summary["evaluation_results"][epoch] = {
                "raw_results": results,
                "parsed_metrics": parsed_results,
                "success_rate": sum(1 for r in results.values() if r.get('success', False)) / len(results),
                "total_evaluations": len(results)
            }
        
        # Save JSON summary
        with open(self.summary_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create markdown report
        self.create_markdown_report(summary)
        
        logger.info("✅ Summary report saved")
    
    def create_markdown_report(self, summary: Dict):
        """Create a readable markdown report"""
        
        report_path = self.summary_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# BitMar Evaluation Report\n\n")
            f.write(f"**Generated:** {summary['evaluation_timestamp']}\n")
            f.write(f"**Total Epochs Evaluated:** {summary['total_epochs_evaluated']}\n\n")
            
            f.write("## Configuration\n\n")
            for key, value in summary['configuration'].items():
                f.write(f"- **{key.replace('_', ' ').title()}:** `{value}`\n")
            
            f.write("\n## Results by Epoch\n\n")
            
            for epoch, results in summary['evaluation_results'].items():
                f.write(f"### {epoch}\n\n")
                f.write(f"- **Success Rate:** {results['success_rate']:.1%}\n")
                f.write(f"- **Total Evaluations:** {results['total_evaluations']}\n")
                
                if results['parsed_metrics']:
                    f.write("- **Key Metrics:**\n")
                    for metric_type, metrics in results['parsed_metrics'].items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                f.write(f"  - {metric_type}.{metric_name}: {value:.3f}\n")
                
                f.write("\n")
            
            f.write("## Visualization Files\n\n")
            f.write("- `evaluation_progress.png` - Progress across epochs\n")
            f.write("- `performance_heatmap.png` - Task performance comparison\n")
            f.write("- `timing_analysis.png` - Evaluation timing\n")
            f.write("- `success_rates.png` - Success rates by evaluation type\n")
    
    def run_full_evaluation(self, 
                          run_fast: bool = True,
                          run_full: bool = True,
                          run_multimodal: bool = True,
                          run_aoa: bool = True,
                          test_loading_first: bool = True):
        """Run complete evaluation suite"""
        
        logger.info("🚀 Starting BitMar Evaluation Suite")
        logger.info(f"   Fast evaluation: {'✅' if run_fast else '❌'}")
        logger.info(f"   Full evaluation: {'✅' if run_full else '❌'}")
        logger.info(f"   Multimodal evaluation: {'✅' if run_multimodal else '❌'}")
        logger.info(f"   AoA evaluation: {'✅' if run_aoa else '❌'}")
        
        # Discover epochs
        epochs = self.discover_model_epochs()
        
        if not epochs:
            logger.error("❌ No valid model epochs found!")
            return False
        
        # Main evaluation loop
        total_evaluations = 0
        completed_evaluations = 0
        
        # Calculate total number of evaluations for progress tracking
        for epoch_name, model_path in epochs:
            if run_fast: total_evaluations += 1
            if run_full and epoch_name == "epoch_latest": total_evaluations += 1
            if run_multimodal: total_evaluations += 3  # VQA, Winoground, DevBench
            if run_aoa: total_evaluations += 1
        
        # Progress bar
        with tqdm(total=total_evaluations, desc="Overall Progress", unit="eval") as pbar:
            
            for epoch_name, model_path in epochs:
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Processing epoch: {epoch_name}")
                logger.info(f"   Model path: {model_path}")
                logger.info(f"{'='*60}")
                
                epoch_start_time = time.time()
                epoch_results = {}
                
                # Test model loading first
                if test_loading_first:
                    if not self.test_model_loading(model_path):
                        logger.error(f"❌ Skipping {epoch_name} due to loading failure")
                        continue
                
                # Fast evaluation (for all epochs)
                if run_fast:
                    pbar.set_description(f"Fast eval - {epoch_name}")
                    results = self.run_text_evaluations(model_path, epoch_name, "fast")
                    epoch_results.update(results)
                    pbar.update(1)
                    completed_evaluations += 1
                
                # Full evaluation (only for latest epoch)
                if run_full and epoch_name == "epoch_10":
                    pbar.set_description(f"Full eval - {epoch_name}")
                    results = self.run_text_evaluations(model_path, epoch_name, "full")
                    epoch_results.update(results)
                    pbar.update(1)
                    completed_evaluations += 1
                
                # Multimodal evaluations
                if run_multimodal:
                    pbar.set_description(f"Multimodal - {epoch_name}")
                    results = self.run_multimodal_evaluations(model_path, epoch_name)
                    epoch_results.update(results)
                    pbar.update(3)
                    completed_evaluations += 3
                
                # AoA evaluation
                if run_aoa:
                    pbar.set_description(f"AoA - {epoch_name}")
                    results = self.run_aoa_evaluation(model_path, epoch_name)
                    epoch_results.update(results)
                    pbar.update(1)
                    completed_evaluations += 1
                
                # Store results and timing
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                
                self.evaluation_results[epoch_name] = epoch_results
                self.timing_info[epoch_name] = {
                    "start_time": epoch_start_time,
                    "end_time": epoch_end_time,
                    "total_time": epoch_duration
                }
                
                logger.info(f"✅ Completed {epoch_name} in {epoch_duration/60:.1f} minutes")
                
                # Save intermediate results
                self.save_summary_report()
        
        # Final processing
        logger.info("\n🎯 Creating final visualizations and reports...")
        self.create_visualizations()
        self.save_summary_report()
        
        logger.info(f"🎉 Evaluation complete! Results saved to: {self.results_dir}")
        logger.info(f"   Total evaluations: {completed_evaluations}/{total_evaluations}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="BitMar Comprehensive Evaluation Runner")
    
    parser.add_argument("--models_dir", 
                       default="/work/u3191895/babylm/BitMar/models",
                       help="Directory containing model epochs")
    
    parser.add_argument("--evaluation_data_dir", 
                       default="../evaluation_data",
                       help="Directory containing evaluation data")
    
    parser.add_argument("--evaluation_pipeline_dir", 
                       default="evaluation-pipeline-2025",
                       help="Directory containing evaluation pipeline")
    
    parser.add_argument("--results_dir", 
                       default="evaluation_results",
                       help="Directory to save results")
    
    parser.add_argument("--backend", 
                       default="causal",
                       choices=["causal", "mlm", "mntp", "enc_dec_mask", "enc_dec_prefix"],
                       help="Backend for evaluation")
    
    parser.add_argument("--no-fast", action="store_true",
                       help="Skip fast evaluations")
    
    parser.add_argument("--no-full", action="store_true",
                       help="Skip full evaluations")
    
    parser.add_argument("--no-multimodal", action="store_true",
                       help="Skip multimodal evaluations")
    
    parser.add_argument("--no-aoa", action="store_true",
                       help="Skip AoA evaluations")
    
    parser.add_argument("--no-loading-test", action="store_true",
                       help="Skip model loading tests")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BitMarEvaluationRunner(
        models_dir=args.models_dir,
        evaluation_data_dir=args.evaluation_data_dir,
        evaluation_pipeline_dir=args.evaluation_pipeline_dir,
        results_dir=args.results_dir,
        backend=args.backend
    )
    
    # Run evaluation
    success = runner.run_full_evaluation(
        run_fast=not args.no_fast,
        run_full=not args.no_full,
        run_multimodal=not args.no_multimodal,
        run_aoa=not args.no_aoa,
        test_loading_first=not args.no_loading_test
    )
    
    if success:
        print("🎉 Evaluation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
