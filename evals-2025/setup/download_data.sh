# setup/download_data.sh  
#!/bin/bash
# filepath: evals-2025/setup/download_data.sh
cd ../evaluation-pipeline-2025
# Download evaluation_data from OSF
echo "Download evaluation_data from: https://osf.io/ryjfm/"
echo "Place in evaluation-pipeline-2025/evaluation_data/"
echo "Run: python -m evaluation_pipeline.ewok.dl_and_filter"