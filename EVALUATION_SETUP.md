# BabyLM 2025 Evaluation Pipeline Setup Guide

This guide walks you through setting up the BabyLM Challenge 2025 evaluation pipelines for automatic evaluation during BitMar training.

## 📋 Overview

The evaluation integration uses two pipelines:
- **evaluation-pipeline-2025**: Text-only and fine-tuning evaluations
- **evaluation-pipeline-2024**: Multimodal evaluations

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

**For Windows (PowerShell):**
```powershell
# Run from your desired directory (e.g., d:\)
.\BitMar\setup_evaluation_pipelines.ps1
```

**For Linux/macOS (Bash):**
```bash
# Run from your desired directory (e.g., ~/projects/)
bash BitMar/setup_evaluation_pipelines.sh
```

### Option 2: Python Download Script

```bash
cd BitMar
python download_evaluation_data.py --pipeline_path "d:/BabyLM/evaluation-pipeline-2025"
```

## 📖 Manual Setup

### Step 1: Clone Repositories

```bash
mkdir BabyLM
cd BabyLM

# Clone evaluation pipelines
git clone https://github.com/babylm/evaluation-pipeline-2025.git
git clone https://github.com/babylm/evaluation-pipeline-2024.git
```

### Step 2: Install Dependencies

**Pipeline 2025:**
```bash
cd evaluation-pipeline-2025
pip install -r requirements.txt
pip install transformers torch scikit-learn numpy pandas statsmodels datasets wandb nltk
cd ..
```

**Pipeline 2024:**
```bash
cd evaluation-pipeline-2024
pip install -e .
pip install minicons
pip install --upgrade accelerate
cd ..
```

### Step 3: Download Evaluation Data

**Pipeline 2025 (Full Evaluation):**
```bash
cd evaluation-pipeline-2025
curl -L "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip=" -o full_eval.zip

# Extract using Python (compatible with all systems)
python3 -c "
import zipfile
import os
os.makedirs('evaluation_data', exist_ok=True)
with zipfile.ZipFile('full_eval.zip', 'r') as zip_ref:
    zip_ref.extractall('evaluation_data/')
print('✅ Evaluation data extracted!')
"
rm full_eval.zip

# Check what was extracted
ls -la evaluation_data/full_eval/
cd ..
```

**Pipeline 2024:**
- Download data from: https://osf.io/ad7qg/
- Extract to: `evaluation-pipeline-2024/evaluation_data/`

### Step 4: Set up HuggingFace Access

```bash
huggingface-cli login
```

Request access to these datasets:
- [Winoground](https://huggingface.co/datasets/facebook/winoground)
- [EWoK](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)

### Step 5: Download Additional Data

**EWoK Data:**
```bash
cd evaluation-pipeline-2025
# Download EWoK data (this will handle both full and fast versions)
python -m evaluation_pipeline.ewok.dl_and_filter

# If the above fails, you can skip this step - EWoK fast evaluation will be handled automatically
cd ..
```

**DevBench Data:**
```bash
cd evaluation-pipeline-2024
bash devbench/download_data.sh
cd ..
```

## 📁 Expected Directory Structure

After setup, your directory structure should look like:

```
d:\BabyLM\
├── BitMar/                           # Your BitMar repository
│   ├── src/
│   │   ├── evaluation_integration.py
│   │   └── training_evaluation_integration.py
│   ├── configs/
│   │   └── bitmar_100M_tokens.yaml
│   ├── train_100M_tokens.py
│   ├── validate_evaluation_setup.py
│   └── download_evaluation_data.py
├── evaluation-pipeline-2024/         # Multimodal evaluations
│   ├── eval_multimodal.sh            # VQA + Winoground (uses HF datasets)
│   ├── eval_blimp.sh
│   ├── eval_ewok.sh
│   ├── devbench/
│   ├── ewok/
│   └── requirements.txt
├── evaluation-pipeline-2025/         # Text & fine-tuning evaluations
│   ├── eval_zero_shot.sh
│   ├── eval_zero_shot_fast.sh
│   ├── eval_finetuning.sh
│   ├── evaluation_pipeline/
│   ├── evaluation_data/              # Downloaded from OSF
│   │   └── full_eval/                # Main evaluation data
│   │       ├── blimp_filtered/       # BLiMP tasks
│   │       ├── supplement_filtered/   # BLiMP supplement
│   │       ├── ewok_filtered/        # EWoK tasks
│   │       ├── entity_tracking/      # Entity tracking tasks
│   │       ├── glue_filtered/        # GLUE tasks for fine-tuning
│   │       ├── wug_adj_nominalization/ # Morphology tasks
│   │       ├── wug_past_tense/       # Morphology tasks
│   │       ├── comps/                # Property knowledge tasks
│   │       ├── reading/              # Reading time tasks
│   │       └── cdi_childes/          # CDI-CHILDES tasks
```

## 🎯 Multimodal Data Notes

**Important**: Multimodal evaluations (VQA and Winoground) do **NOT** use local data files. Instead:

- **VQA**: Downloads `HuggingFaceM4/VQAv2` dataset automatically during evaluation
- **Winoground**: Downloads `facebook/winoground` dataset automatically during evaluation
- **Pipeline**: Uses evaluation-pipeline-2024's `eval_multimodal.sh` with `lm_eval`
- **No setup required**: The datasets are downloaded from HuggingFace automatically

### Multimodal Pipeline Status
- ✅ **2024 Pipeline**: Fully functional multimodal evaluation via `lm_eval`
- ⏳ **2025 Pipeline**: Multimodal evaluation commands are "under construction"
│   └── requirements.txt
└── babylm_dataset/                   # Your training data
    ├── train_50M.zip
    ├── cc_3M_captions.json
    └── ...
```

## ⚙️ Configuration

The evaluation integration is configured in `configs/bitmar_100M_tokens.yaml`:

```yaml
evaluation:
  enabled: true
  pipeline_2024_path: "d:/BabyLM/evaluation-pipeline-2024"
  pipeline_2025_path: "d:/BabyLM/evaluation-pipeline-2025"
  eval_frequency: 1  # Evaluate every epoch
  fast_eval_epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Fast evaluation
  full_eval_epochs: [10]  # Full evaluation
```

## 🧪 Validation

Run the validation script to check your setup:

```bash
cd BitMar
python validate_evaluation_setup.py
```

Expected output:
```
🧪 BabyLM Evaluation Pipeline Setup Validation
==================================================
⚙️ Checking BitMar configuration...
✅ BitMar configuration valid

📁 Checking Pipeline 2025 structure...
✅ Found: eval_zero_shot.sh
✅ Found: eval_finetuning.sh
✅ Found: evaluation_pipeline
✅ Pipeline 2025 structure validation passed

📁 Checking Pipeline 2024 structure...
✅ Found: eval_multimodal.sh
✅ Pipeline 2024 structure validation passed

📦 Checking Python packages...
✅ transformers
✅ torch
✅ numpy
✅ pandas
✅ wandb
✅ datasets
✅ sklearn
✅ statsmodels
✅ All required packages installed

🤗 Checking HuggingFace access...
✅ HuggingFace CLI authenticated

📊 Checking evaluation data...
✅ Pipeline 2025 evaluation data found
  ✅ Found: blimp_filtered
  ✅ Found: supplement_filtered
  ✅ Found: ewok_filtered
  ✅ Found: entity_tracking
  ✅ Found: glue_filtered
  ✅ Found: winoground_filtered
  ✅ Found: vqa_filtered
  ✅ Found: wug_adj_nominalization
  ✅ Found: wug_past_tense
  ✅ Found: comps
  ✅ Found: reading
  ✅ Found: cdi_childes
✅ Pipeline 2024 evaluation data directory found

==================================================
✅ Setup validation PASSED!
```

## 🏃‍♂️ Running Training with Evaluation

Once setup is complete, start training with automatic evaluation:

```bash
cd BitMar
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml
```

The training will automatically:
- Run fast evaluations after epochs 1-9
- Run full evaluation after epoch 10
- Log results to WandB
- Save evaluation results to files

## 📊 Evaluation Components

### Text-Only Evaluations (Pipeline 2025)
- **BLiMP**: Linguistic phenomena understanding
- **BLiMP Supplement**: Additional linguistic tasks
- **EWoK**: Commonsense reasoning
- **Entity Tracking**: State tracking in narratives
- **Morphology Tasks**: Word formation understanding
- **COMPS**: Property knowledge inheritance
- **Reading Time**: Human processing alignment

### Fine-tuning Evaluations (Pipeline 2025)
- **GLUE Tasks**: General language understanding
  - BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, WSC

### Multimodal Evaluations (Pipeline 2024)
- **Winoground**: Visual-textual reasoning
- **VQA**: Visual question answering

## 🔧 Troubleshooting

### Common Issues

**1. Download Fails**
```bash
# Manual download
curl -L -o full_eval.zip "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip="
```

**2. HuggingFace Access Denied**
```bash
huggingface-cli login
# Then visit the dataset pages and request access
```

**3. Missing Dependencies**
```bash
pip install transformers torch scikit-learn numpy pandas statsmodels datasets wandb nltk minicons
```

**4. Path Issues**
- Update paths in `configs/bitmar_100M_tokens.yaml`
- Use absolute paths for reliability

### Getting Help

If you encounter issues:
1. Run `python validate_evaluation_setup.py` to identify problems
2. Check the error messages carefully
3. Ensure all required files are downloaded and in the correct locations
4. Verify HuggingFace CLI authentication and dataset access

## 📈 Results

Evaluation results are saved in:
- `evaluation_results/epoch_N_results.json`: Detailed results per epoch
- WandB logs: Real-time metrics and visualizations
- Pipeline-specific result directories within each evaluation pipeline

The integration provides comprehensive assessment of your BitMar model's performance on the BabyLM 2025 challenge tasks! 🎉
