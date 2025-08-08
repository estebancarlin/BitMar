# BabyLM 2025 Evaluation Scripts

This folder contains scripts to evaluate models for the BabyLM 2025 Challenge, ensuring compatibility with the official evaluation pipeline and generating submission-ready results.

## Quick Start

1. **Setup environment:**
   ```bash
   cd setup/
   ./setup_environment.sh
   ./download_data.sh
   python hf_authenticate.py
   ```

2. **Validate your model:**
   ```bash
   python model_management/validate_model_compatibility.py --model_path your-model-path --architecture causal
   ```

3. **Run complete evaluation:**
   ```bash
   ./run_complete_evaluation.sh your-model-path causal hf
   ```


## Directory Structure

- setup/ - Environment and data setup scripts
- model_management/ - Model validation and HuggingFace repository management
- evaluation_runners/ - Scripts to run evaluations using the official pipeline
- results_processing/ - Result collation and submission file generation
- analysis/ - Performance analysis and paper table generation


## Usage Examples

Fast Evaluation (All Checkpoints)
   ```bash
   python evaluation_runners/run_all_fast_evals.py huggingface.co/username/model-name
   ```

Full Evaluation (Final Model)
   ```bash
   python evaluation_runners/run_final_full_eval.py huggingface.co/username/model-name
   ```

Generate Submission Files
   ```bash
   python results_processing/generate_submission_files.py your-model-name hf
   ```

Create Analysis Report
   ```bash
   python results_processing/collate_all_results.py
   python analysis/performance_analysis.py your-model-name
   python analysis/create_paper_tables.py --model_name "Your Model Name"
   ```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- HuggingFace Hub access
- Access to BabyLM evaluation data

## Notes
- Ensure your model follows BabyLM checkpoint naming: chck_1M, chck_2M, etc.
- Fast evaluations are required for intermediate checkpoints
- Full evaluation (including fine-tuning) only needed for final model
- Results are automatically formatted for BabyLM submission