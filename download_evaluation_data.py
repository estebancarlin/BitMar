"""
Download script for BabyLM 2025 evaluation data
Downloads and extracts the full_eval dataset from OSF
"""

import requests
import zipfile
import os
from pathlib import Path
import argparse

def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download a file from URL with progress indication"""
    print(f"📥 Downloading from: {url}")
    print(f"📁 Saving to: {output_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\r⏳ Progress: {progress:.1f}%", end='', flush=True)
    
    print(f"\n✅ Download completed: {output_path}")
    return output_path

def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file"""
    print(f"📦 Extracting: {zip_path}")
    print(f"📁 Extract to: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extraction completed")

def download_evaluation_data_2025(pipeline_path: str):
    """Download BabyLM 2025 evaluation data"""
    pipeline_dir = Path(pipeline_path)
    if not pipeline_dir.exists():
        print(f"❌ Pipeline directory not found: {pipeline_path}")
        return False
    
    # Create evaluation_data directory
    eval_data_dir = pipeline_dir / "evaluation_data"
    eval_data_dir.mkdir(exist_ok=True)
    
    # Download URLs
    full_eval_url = "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip="
    
    try:
        # Download full_eval.zip
        zip_path = eval_data_dir / "full_eval.zip"
        download_file(full_eval_url, str(zip_path))
        
        # Extract the zip file
        extract_zip(str(zip_path), str(eval_data_dir))
        
        # Check if extraction was successful
        full_eval_dir = eval_data_dir / "full_eval"
        if full_eval_dir.exists():
            print(f"✅ Successfully set up evaluation data at: {full_eval_dir}")
            
            # List the contents
            print(f"📋 Evaluation data structure:")
            for item in sorted(full_eval_dir.iterdir()):
                if item.is_dir():
                    file_count = len(list(item.iterdir()))
                    print(f"  📁 {item.name}/ ({file_count} files)")
                else:
                    print(f"  📄 {item.name}")
            
            # Clean up zip file
            zip_path.unlink()
            print(f"🧹 Cleaned up: {zip_path}")
            
            return True
        else:
            print(f"❌ Extraction failed - full_eval directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading evaluation data: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download BabyLM 2025 evaluation data")
    parser.add_argument("--pipeline_path", type=str, 
                       default="d:/BabyLM/evaluation-pipeline-2025",
                       help="Path to evaluation-pipeline-2025 directory")
    
    args = parser.parse_args()
    
    print("🚀 BabyLM 2025 Evaluation Data Download")
    print("=" * 50)
    
    success = download_evaluation_data_2025(args.pipeline_path)
    
    if success:
        print("\n✅ Download completed successfully!")
        print("\n📋 Next steps:")
        print("1. Set up HuggingFace CLI: huggingface-cli login")
        print("2. Request access to restricted datasets:")
        print("   - https://huggingface.co/datasets/facebook/winoground")
        print("   - https://huggingface.co/datasets/ewok-core/ewok-core-1.0")
        print("3. Download EWoK data:")
        print(f"   cd {args.pipeline_path}")
        print("   python -m evaluation_pipeline.ewok.dl_and_filter")
        print("4. Validate setup:")
        print("   python validate_evaluation_setup.py")
    else:
        print("\n❌ Download failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
