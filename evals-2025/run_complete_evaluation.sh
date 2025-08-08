# run_complete_evaluation.sh
#!/bin/bash
# filepath: evals-2025/run_complete_evaluation.sh

MODEL_PATH=$1
ARCHITECTURE=${2:-"causal"}
BACKEND=${3:-"hf"}

echo "Starting complete BabyLM 2025 evaluation for: $MODEL_PATH"

# 1. Run fast evaluations on all checkpoints
echo "=== Running fast evaluations on all checkpoints ==="
python evaluation_runners/run_all_fast_evals.py $MODEL_PATH $ARCHITECTURE

# 2. Run full evaluation on final model
echo "=== Running full evaluation on final model ==="  
python evaluation_runners/run_final_full_eval.py $MODEL_PATH $ARCHITECTURE

# 3. Generate submission files
echo "=== Generating submission files ==="
python results_processing/generate_submission_files.py $MODEL_PATH $BACKEND

# 4. Create analysis reports
echo "=== Creating analysis reports ==="
python analysis/performance_analysis.py $MODEL_PATH

echo "Complete evaluation finished!"