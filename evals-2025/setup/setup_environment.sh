# setup/setup_environment.sh
#!/bin/bash
# filepath: evals-2025/setup/setup_environment.sh
cd ../evaluation-pipeline-2025
pip install -r requirements.txt
echo "Environment setup complete. Use python -m evaluation_pipeline.* commands from evaluation-pipeline-2025 root."