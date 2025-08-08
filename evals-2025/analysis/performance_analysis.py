#!/usr/bin/env python3
"""
Analyze model performance across checkpoints and tasks
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import numpy as np

def load_checkpoint_summary(summary_file):
    """Load checkpoint summary CSV"""
    if Path(summary_file).exists():
        return pd.read_csv(summary_file)
    else:
        print(f"Summary file not found: {summary_file}")
        return pd.DataFrame()

def plot_learning_curves(df, output_dir):
    """Plot learning curves for each task"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Get numeric columns (tasks)
    task_columns = df.select_dtypes(include=[np.number]).columns
    task_columns = [col for col in task_columns if col != 'checkpoint']
    
    if not task_columns:
        print("No task score columns found")
        return
    
    # Convert checkpoint to numeric for plotting
    df['checkpoint_num'] = pd.to_numeric(df['checkpoint'], errors='coerce')
    df = df.sort_values('checkpoint_num')
    
    # Plot individual task curves
    for task in task_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['checkpoint_num'], df[task], marker='o')
        plt.xlabel('Checkpoint (M tokens)')
        plt.ylabel('Score')
        plt.title(f'Learning Curve: {task}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"learning_curve_{task}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot all tasks together
    plt.figure(figsize=(12, 8))
    for task in task_columns:
        plt.plot(df['checkpoint_num'], df[task], marker='o', label=task, alpha=0.7)
    
    plt.xlabel('Checkpoint (M tokens)')
    plt.ylabel('Score')
    plt.title('All Tasks Learning Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_file = output_dir / "all_tasks_learning_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curves saved to: {output_dir}")

def generate_performance_report(df, output_dir):
    """Generate text performance report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "performance_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("BabyLM Model Performance Report\n")
        f.write("=" * 40 + "\n\n")
        
        if df.empty:
            f.write("No performance data available.\n")
            return
        
        # Basic stats
        f.write(f"Total checkpoints evaluated: {len(df)}\n")
        f.write(f"Checkpoint range: {df['checkpoint'].min()} - {df['checkpoint'].max()}\n\n")
        
        # Task performance summary
        task_columns = df.select_dtypes(include=[np.number]).columns
        task_columns = [col for col in task_columns if col not in ['checkpoint', 'checkpoint_num']]
        
        f.write("Task Performance Summary:\n")
        f.write("-" * 25 + "\n")
        
        for task in task_columns:
            values = df[task].dropna()
            if len(values) > 0:
                f.write(f"{task}:\n")
                f.write(f"  Best score: {values.max():.4f}\n")
                f.write(f"  Final score: {values.iloc[-1]:.4f}\n")
                f.write(f"  Improvement: {values.iloc[-1] - values.iloc[0]:.4f}\n\n")
        
        # Best checkpoint analysis
        if len(task_columns) > 0:
            f.write("Best Checkpoints by Task:\n")
            f.write("-" * 25 + "\n")
            
            for task in task_columns:
                best_idx = df[task].idxmax()
                if not pd.isna(best_idx):
                    best_checkpoint = df.loc[best_idx, 'checkpoint']
                    best_score = df.loc[best_idx, task]
                    f.write(f"{task}: {best_checkpoint} ({best_score:.4f})\n")
    
    print(f"Performance report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze BabyLM model performance")
    parser.add_argument("model_name", help="Model name for reports")
    parser.add_argument("--summary_file", default="./summaries/checkpoint_summary.csv",
                       help="Path to checkpoint summary CSV")
    parser.add_argument("--output_dir", default="./analysis_output",
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    print(f"Analyzing performance for: {args.model_name}")
    
    # Load data
    df = load_checkpoint_summary(args.summary_file)
    
    if df.empty:
        print("No data found for analysis")
        return
    
    # Generate analysis
    plot_learning_curves(df, args.output_dir)
    generate_performance_report(df, args.output_dir)
    
    print(f"Performance analysis complete! Results in: {args.output_dir}")

if __name__ == "__main__":
    main()