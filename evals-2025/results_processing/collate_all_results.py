#!/usr/bin/env python3
"""
Collate all evaluation results into organized summaries
"""
import json
import pandas as pd
from pathlib import Path
import argparse
import glob

def load_results_from_directory(results_dir):
    """Load all JSON result files from directory"""
    results_dir = Path(results_dir)
    all_results = {}
    
    # Find all result JSON files
    json_files = list(results_dir.glob("**/*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Use relative path as key
                key = str(json_file.relative_to(results_dir))
                all_results[key] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_results

def create_checkpoint_summary(all_results):
    """Create summary table for checkpoint evaluations"""
    checkpoint_data = []
    
    for file_path, results in all_results.items():
        if "chck_" in file_path:
            # Extract checkpoint name
            checkpoint = None
            for part in file_path.split('/'):
                if 'chck_' in part:
                    checkpoint = part.replace('chck_', '').replace('M', '')
                    break
            
            if checkpoint:
                row_data = {
                    'checkpoint': checkpoint,
                    'file': file_path
                }
                
                # Extract scores from results
                if isinstance(results, dict):
                    for task, score in results.items():
                        if isinstance(score, (int, float)):
                            row_data[task] = score
                        elif isinstance(score, dict) and 'accuracy' in score:
                            row_data[task] = score['accuracy']
                
                checkpoint_data.append(row_data)
    
    if checkpoint_data:
        df = pd.DataFrame(checkpoint_data)
        # Sort by checkpoint number
        df['checkpoint_num'] = pd.to_numeric(df['checkpoint'], errors='coerce')
        df = df.sort_values('checkpoint_num').drop('checkpoint_num', axis=1)
        return df
    else:
        return pd.DataFrame()

def create_task_summary(all_results):
    """Create summary by evaluation task"""
    task_results = {}
    
    for file_path, results in all_results.items():
        if isinstance(results, dict):
            for task, score in results.items():
                if task not in task_results:
                    task_results[task] = []
                
                task_info = {
                    'source_file': file_path,
                    'score': score
                }
                
                # Extract checkpoint info if available
                if "chck_" in file_path:
                    for part in file_path.split('/'):
                        if 'chck_' in part:
                            task_info['checkpoint'] = part
                            break
                
                task_results[task].append(task_info)
    
    return task_results

def save_summaries(checkpoint_df, task_results, output_dir):
    """Save summary files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save checkpoint summary
    if not checkpoint_df.empty:
        checkpoint_file = output_dir / "checkpoint_summary.csv"
        checkpoint_df.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint summary saved to: {checkpoint_file}")
    
    # Save task summaries
    task_summary_file = output_dir / "task_summaries.json"
    with open(task_summary_file, 'w') as f:
        json.dump(task_results, f, indent=2)
    print(f"Task summaries saved to: {task_summary_file}")
    
    # Save combined summary
    combined_file = output_dir / "evaluation_summary.json"
    combined_data = {
        'checkpoint_count': len(checkpoint_df) if not checkpoint_df.empty else 0,
        'tasks_evaluated': list(task_results.keys()),
        'summary_generated': pd.Timestamp.now().isoformat()
    }
    
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Combined summary saved to: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Collate BabyLM evaluation results")
    parser.add_argument("--results_dir", default="../evaluation-pipeline-2025/results", 
                       help="Directory containing evaluation results")
    parser.add_argument("--output_dir", default="./summaries", 
                       help="Directory to save summary files")
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    all_results = load_results_from_directory(args.results_dir)
    print(f"Loaded {len(all_results)} result files")
    
    # Create summaries
    checkpoint_df = create_checkpoint_summary(all_results)
    task_results = create_task_summary(all_results)
    
    print(f"Found {len(checkpoint_df)} checkpoint results")
    print(f"Found {len(task_results)} different tasks")
    
    # Save summaries
    save_summaries(checkpoint_df, task_results, args.output_dir)
    
    print("Result collation complete!")

if __name__ == "__main__":
    main()