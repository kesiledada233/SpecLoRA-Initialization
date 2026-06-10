#!/usr/bin/env python3
"""

"""

import json
import os
from pathlib import Path
import pandas as pd

# 
EXPERIMENT_DIRS = [
    # GSM8K  - α 
    "outputs_gsm8k_ablation_alpha0.6_r16",
    "outputs_gsm8k_ablation_alpha0.8_r16",
    "outputs_gsm8k_ablation_baseline_r16",  # 
    "outputs_gsm8k_ablation_alpha0.9_r16",   # 
    "outputs_gsm8k_ablation_alpha1.1_r16",   # 
    "outputs_gsm8k_ablation_alpha1.5_r16",
    
    # GSM8K  - LoRA 
    "outputs_gsm8k_ablation_r8_baseline",
    "outputs_gsm8k_ablation_r8_alpha1.1",
    "outputs_gsm8k_ablation_r32_baseline",
    "outputs_gsm8k_ablation_r32_alpha1.1",
    
    # 
    "outputs_cmmlu_baseline",
    "outputs_cmmlu_alpha1.1",
    "outputs_sharegpt_baseline",
    "outputs_sharegpt_alpha1.1",
    "outputs_mbpp_baseline",
    "outputs_mbpp_alpha1.1",
    
    # DeepSeek 
    "outputs_deepseek_gsm8k_baseline_r16",
    "outputs_deepseek_gsm8k_alpha1.1_r16",
]

def collect_results():
    """"""
    results_table = []
    
    for exp_dir in EXPERIMENT_DIRS:
        results_file = os.path.join(exp_dir, "results.json")
        
        if not os.path.exists(results_file):
            print(f"  : {results_file}")
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # 
            model_name = data['model_path'].split('/')[-1]
            if 'deepseek' in model_name.lower():
                model_name = 'DeepSeek-7B'
            elif 'pangu' in model_name.lower():
                model_name = 'OpenPangu-7B'
            
            # 
            row = {
                '': exp_dir,
                '': data['dataset'].upper(),
                '': model_name,
                'LoRA': data['lora_r'],
                '': data['init_method'],
                'α': f"{data['init_alpha']:.1f}" if data['init_alpha'] else '-',
                '(min)': f"{data['wall_time_minutes']:.2f}",
                '(GB)': f"{data['peak_memory_gb']:.2f}",
                '(/s)': f"{data['throughput_samples_per_sec']:.2f}",
                'AUC(0-500)': f"{data['auc_500']:.2f}" if data['auc_500'] else 'N/A',
                '': f"{data['best_train_loss']:.4f}",
                '': f"{data['final_test_loss']:.4f}",
                '': f"{data['final_test_loss_std']:.4f}",
            }
            
            results_table.append(row)
            print(f" : {exp_dir}")
            
        except Exception as e:
            print(f"  {results_file}: {e}")
            continue
    
    return pd.DataFrame(results_table)

def main():
    print("="*70)
    print(" ")
    print("="*70)
    print()
    
    # 
    df = collect_results()
    
    if df.empty:
        print("\n ")
        return
    
    print()
    print("="*70)
    print(" ")
    print("="*70)
    print()
    
    #  Markdown 
    print(df.to_markdown(index=False))
    
    #  CSV
    csv_file = "all_experiments_results.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print()
    print(f" : {csv_file}")
    
    #  Excel openpyxl 
    try:
        excel_file = "all_experiments_results.xlsx"
        df.to_excel(excel_file, index=False)
        print(f" : {excel_file}")
    except ImportError:
        print("   openpyxl Excel ")
    
    # 
    print()
    print("="*70)
    print(" ")
    print("="*70)
    
    # 
    print("\n :")
    grouped = df.groupby('').agg({
        'AUC(0-500)': lambda x: x.replace('N/A', '0').astype(float).mean(),
        '': lambda x: x.astype(float).mean(),
        '(min)': lambda x: x.astype(float).mean(),
    })
    print(grouped)
    
    # 
    print("\n :")
    grouped = df.groupby('').agg({
        'AUC(0-500)': lambda x: x.replace('N/A', '0').astype(float).mean(),
        '': lambda x: x.astype(float).mean(),
        '(min)': lambda x: x.astype(float).mean(),
    })
    print(grouped)
    
    print()
    print("="*70)
    print(" !")
    print("="*70)

if __name__ == "__main__":
    main()