#!/usr/bin/env python3
"""
Create LaTeX tables for paper submission
"""
import pandas as pd
import json
from pathlib import Path
import argparse

def create_latex_table(df, caption, label):
    """Create LaTeX table from DataFrame"""
    latex_str = "\\begin{table}[h]\n"
    latex_str += "\\centering\n"
    latex_str += "\\small\n"
    
    # Create tabular environment
    num_cols = len(df.columns)
    col_spec = "l" + "c" * (num_cols - 1)
    latex_str += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_str += "\\toprule\n"
    
    # Header
    header = " & ".join([col.replace("_", "\\_") for col in df.columns])
    latex_str += header + " \\\\\n"
    latex_str += "\\midrule\n"
    
    # Data rows
    for _, row in df.iterrows():
        row_str = " & ".join([
            str(val) if pd.notna(val) else "-" 
            for val in row.values
        ])
        latex_str += row_str + " \\\\\n"
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    latex_str += "\\end{table}\n"
    
    return latex_str

def format_scores_for_paper(df):
    """Format scores with appropriate precision for paper"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        if col != 'checkpoint':
            # Format to 3 decimal places
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    
    return df

def create_checkpoint_table(summary_file, output_dir):
    """Create table showing performance across checkpoints"""
    df = pd.read_csv(summary_file)
    
    # Select key checkpoints for paper (not all)
    key_checkpoints = ['1', '5', '10', '50', '100']  # Adjust as needed
    df_subset = df[df['checkpoint'].astype(str).isin(key_checkpoints)].copy()
    
    # Format for paper
    df_subset = format_scores_for_paper(df_subset)
    
    # Rename checkpoint column
    df_subset.rename(columns={'checkpoint': 'Checkpoint (M tokens)'}, inplace=True)
    
    latex_table = create_latex_table(
        df_subset,
        caption="Model performance across training checkpoints on BabyLM evaluation tasks.",
        label="tab:checkpoint_performance"
    )
    
    output_file = Path(output_dir) / "checkpoint_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"Checkpoint table saved to: {output_file}")

def create_final_results_table(summary_file, output_dir, model_name="Our Model"):
    """Create table with final results for comparison"""
    df = pd.read_csv(summary_file)
    
    # Get final checkpoint results
    final_row = df.iloc[-1].copy()
    
    # Create comparison table structure
    comparison_data = {
        'Model': [model_name],
        'Parameters': ['10M'],  # Adjust based on your model
        'Training Data': ['BabyLM Corpus']
    }
    
    # Add task scores
    task_cols = df.select_dtypes(include=['float64', 'int64']).columns
    task_cols = [col for col in task_cols if col != 'checkpoint']
    
    for task in task_cols:
        if pd.notna(final_row[task]):
            comparison_data[task] = [f"{final_row[task]:.3f}"]
        else:
            comparison_data[task] = ["-"]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    latex_table = create_latex_table(
        comparison_df,
        caption="Final model performance on BabyLM 2025 evaluation tasks.",
        label="tab:final_results"
    )
    
    output_file = Path(output_dir) / "final_results_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"Final results table saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create LaTeX tables for paper")
    parser.add_argument("--summary_file", default="./summaries/checkpoint_summary.csv",
                       help="Path to checkpoint summary CSV")
    parser.add_argument("--output_dir", default="./paper_tables",
                       help="Directory to save LaTeX tables")
    parser.add_argument("--model_name", default="Our Model",
                       help="Model name for tables")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if Path(args.summary_file).exists():
        create_checkpoint_table(args.summary_file, args.output_dir)
        create_final_results_table(args.summary_file, args.output_dir, args.model_name)
        print("Paper tables created successfully!")
    else:
        print(f"Summary file not found: {args.summary_file}")
        print("Run collate_all_results.py first to generate summaries.")

if __name__ == "__main__":
    main()