#!/usr/bin/env python3
"""
Validate model compatibility with BabyLM evaluation requirements
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import argparse
import traceback
import sys

def test_model_loading(model_path, architecture="causal"):
    """Test if model can be loaded with transformers"""
    print(f"Testing model loading for: {model_path}")
    
    try:
        # Test tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
        
        # Test model based on architecture
        if architecture == "causal":
            print("Loading causal language model...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            except OSError as e:
                if "from_tf=True" in str(e):
                    print("Trying with from_tf=True...")
                    model = AutoModelForCausalLM.from_pretrained(model_path, from_tf=True)
                else:
                    raise e
            print(f"✓ Causal model loaded successfully. Parameters: {model.num_parameters():,}")
            
        elif architecture == "mlm":
            print("Loading masked language model...")
            from transformers import AutoModelForMaskedLM
            try:
                model = AutoModelForMaskedLM.from_pretrained(model_path)
            except OSError as e:
                if "from_tf=True" in str(e):
                    print("Trying with from_tf=True...")
                    model = AutoModelForMaskedLM.from_pretrained(model_path, from_tf=True)
                else:
                    raise e
            print(f"✓ MLM model loaded successfully. Parameters: {model.num_parameters():,}")
            
        elif architecture == "encoder":
            print("Loading encoder model...")
            from transformers import AutoModel
            try:
                model = AutoModel.from_pretrained(model_path)
            except OSError as e:
                if "from_tf=True" in str(e):
                    print("Trying with from_tf=True...")
                    model = AutoModel.from_pretrained(model_path, from_tf=True)
                else:
                    raise e
            print(f"✓ Encoder model loaded successfully. Parameters: {model.num_parameters():,}")
            
        elif architecture == "encoder-decoder":
            print("Loading encoder-decoder model...")
            from transformers import AutoModelForSeq2SeqLM
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            except OSError as e:
                if "from_tf=True" in str(e):
                    print("Trying with from_tf=True...")
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, from_tf=True)
                else:
                    raise e
            print(f"✓ Encoder-decoder model loaded successfully. Parameters: {model.num_parameters():,}")
        
        # Test sequence classification compatibility (for GLUE) - skip for encoder-decoder
        if architecture != "encoder-decoder":
            print("Testing sequence classification compatibility...")
            try:
                seq_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=2,
                    ignore_mismatched_sizes=True
                )
                print("✓ Sequence classification compatible")
            except Exception as e:
                print(f"⚠ Sequence classification warning: {e}")
        
        # Test inference
        print("Testing inference...")
        test_input = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            if architecture in ["causal", "mlm", "encoder"]:
                outputs = model(**test_input)
                print("✓ Inference test passed")
            elif architecture == "encoder-decoder":
                # For T5, we need both input_ids and decoder_input_ids
                decoder_input_ids = test_input["input_ids"]
                outputs = model(input_ids=test_input["input_ids"], decoder_input_ids=decoder_input_ids)
                print("✓ Inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False

def validate_checkpoint_structure(model_path):
    """Validate checkpoint has required files"""
    from pathlib import Path
    
    # For HuggingFace model names, skip file validation
    if "/" in model_path and not Path(model_path).exists():
        print(f"Assuming HuggingFace model: {model_path}")
        return True
    
    model_path = Path(model_path)
    required_files = ["config.json"]
    
    print(f"Validating checkpoint structure: {model_path}")
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    else:
        print("✓ Checkpoint structure valid")
        return True

def main():
    parser = argparse.ArgumentParser(description="Validate BabyLM model compatibility")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--architecture", default="causal", 
                       choices=["causal", "mlm", "encoder", "encoder-decoder"],
                       help="Model architecture type")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("BabyLM Model Compatibility Validator")
    print("=" * 50)
    
    # Validate checkpoint structure
    structure_valid = validate_checkpoint_structure(args.model_path)
    
    # Test model loading
    loading_valid = test_model_loading(args.model_path, args.architecture)
    
    print("=" * 50)
    if structure_valid and loading_valid:
        print("✓ Model is compatible with BabyLM evaluation pipeline")
    else:
        print("✗ Model has compatibility issues")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()