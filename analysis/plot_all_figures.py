#!/usr/bin/env python3
"""
4 
- GSM8K: 
- CMMLU: 
- ShareGPT: 
- MBPP: 
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4  × 2  + GSM8K 
EXPERIMENTS = {
    # GSM8K
    'gsm8k': {
        'baseline': 'outputs_gsm8k/baseline',
        'fdt': 'outputs_gsm8k/alpha0.6',
        # 
        # 'alpha06': 'outputs_gsm8k_ablation_alpha0.6_r16',
        # 'alpha08': 'outputs_gsm8k_ablation_alpha0.8_r16',
        # 'alpha09': 'outputs_gsm8k_ablation_alpha0.9_r16',
        # 'alpha15': 'outputs_gsm8k_ablation_alpha1.5_r16',
        # 'r8_baseline': 'outputs_gsm8k_ablation_r8_baseline',
        # 'r8_fdt': 'outputs_gsm8k_ablation_r8_alpha1.1',
        # 'r32_baseline': 'outputs_gsm8k_ablation_r32_baseline',
        # 'r32_fdt': 'outputs_gsm8k_ablation_r32_alpha1.1',
    },
    
    # CMMLU
    'cmmlu': {
        'baseline': 'outputs_cmmlu/baseline',
        'fdt': 'outputs_cmmlu/alpha0.6',
    },
    
    # ShareGPT
    'sharegpt': {
        'baseline': 'outputs_sharegpt/baseline',
        'fdt': 'outputs_sharegpt/alpha0.6',
    },
    
    # MBPP
    'mbpp': {
        'baseline': 'outputs_mbpp/baseline',
        'fdt': 'outputs_mbpp/alpha0.6',
    },
}


#  1: 4 1×4 
def plot_4dataset_training_curves():
    """
    4 
    Baseline vs FDT
    1 1×4 
    """

    # 1×4 
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

    datasets = ['gsm8k', 'cmmlu', 'sharegpt', 'mbpp']
    dataset_names = {
        'gsm8k': 'GSM8K (Math)',
        'cmmlu': 'CMMLU (Chinese)',
        'sharegpt': 'ShareGPT (Dialogue)',
        'mbpp': 'MBPP (Code)',
    }

    colors = {'Baseline': '#E74C3C', 'FDT (α=0.6)': '#3498DB'}

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for method_name, method_key in [('Baseline', 'baseline'), ('FDT (α=0.6)', 'fdt')]:
            exp_dir = EXPERIMENTS[dataset][method_key]
            losses_file = os.path.join(exp_dir, "training_losses.npy")

            if not os.path.exists(losses_file):
                print(f"  : {losses_file}")
                continue

            losses = np.load(losses_file)
            steps = np.arange(1, len(losses) + 1)

            window = 20
            losses_smooth = np.convolve(losses, np.ones(window) / window, mode='valid')
            steps_smooth = steps[:len(losses_smooth)]

            mask = steps_smooth <= 500

            ax.plot(
                steps_smooth[mask],
                losses_smooth[mask],
                label=method_name,
                color=colors[method_name],
                linewidth=1.5,
                alpha=0.9
            )

        # 
        ax.set_xlabel('Training Steps', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Training Loss', fontsize=9)

        #  ax.set_title
        ax.text(
            0.5, -0.28,
            f'({chr(97 + idx)}) {dataset_names[dataset]}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10,
            fontweight='bold'
        )

        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 500)

    # “”
    fig.subplots_adjust(bottom=0.28, wspace=0.25)

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig1_4dataset_curves.{fmt}", format=fmt)

    print(f"  1: 4  → {OUTPUT_DIR}/fig1_4dataset_curves.pdf")
    plt.close()

#  B: Step-AUC 
def plot_step_auc_curves():
    """Step-AUC 

     AUC(0–t)  t 
     loss  500 
    """

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    datasets = ['gsm8k', 'cmmlu', 'sharegpt', 'mbpp']
    dataset_names = {
        'gsm8k': 'GSM8K (Math)',
        'cmmlu': 'CMMLU (Chinese)',
        'sharegpt': 'ShareGPT (Dialogue)',
        'mbpp': 'MBPP (Code)',
    }

    colors = {'Baseline': '#E74C3C', 'FDT (α=1.1)': '#3498DB'}

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for method_name, method_key in [('Baseline', 'baseline'), ('FDT (α=1.1)', 'fdt')]:
            exp_dir = EXPERIMENTS[dataset][method_key]
            losses_file = os.path.join(exp_dir, "training_losses.npy")

            if not os.path.exists(losses_file):
                print(f"  : {losses_file}")
                continue

            losses = np.load(losses_file)
            steps = np.arange(1, len(losses) + 1)

            # 
            window = 20
            if len(losses) < window:
                losses_smooth = losses
                steps_smooth = steps
            else:
                losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
                steps_smooth = steps[:len(losses_smooth)]

            #  500 
            mask = steps_smooth <= 500
            steps_smooth = steps_smooth[mask]
            losses_smooth = losses_smooth[mask]

            # Step-AUC: 
            auc_curve = np.cumsum(losses_smooth)

            ax.plot(
                steps_smooth,
                auc_curve,
                label=method_name,
                color=colors[method_name],
                linewidth=1.5,
                alpha=0.9
            )

        ax.set_xlabel('Training Steps', fontsize=9)
        ax.set_ylabel('Cumulative AUC (0–t)', fontsize=9)
        ax.set_title(f'({chr(97+idx)}) {dataset_names[dataset]}', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 500)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig1b_step_auc_curves.{fmt}", format=fmt)

    print(f"  B: Step-AUC  → {OUTPUT_DIR}/fig1b_step_auc_curves.pdf")
    plt.close()


#  2:  AUC 
def plot_cross_dataset_auc():
    """
    AUC(0-500) 
    - 4 
    - Baseline vs FDT
    """
    
    datasets = ['gsm8k', 'cmmlu', 'sharegpt', 'mbpp']
    dataset_labels = ['GSM8K\n(Math)', 'CMMLU\n(Chinese)', 'ShareGPT\n(Dialogue)', 'MBPP\n(Code)']
    
    baseline_values = []
    fdt_values = []
    
    for dataset in datasets:
        # Baseline
        baseline_dir = EXPERIMENTS[dataset]['baseline']
        baseline_file = os.path.join(baseline_dir, "results.json")
        
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                data = json.load(f)
            baseline_values.append(data.get('auc_500', 0))
        else:
            baseline_values.append(0)
            print(f"  : {baseline_file}")
        
        # FDT
        fdt_dir = EXPERIMENTS[dataset]['fdt']
        fdt_file = os.path.join(fdt_dir, "results.json")
        
        if os.path.exists(fdt_file):
            with open(fdt_file, 'r') as f:
                data = json.load(f)
            fdt_values.append(data.get('auc_500', 0))
        else:
            fdt_values.append(0)
            print(f"  : {fdt_file}")
    
    # 
    improvements = [(b - f) / b * 100 if b > 0 else 0 for b, f in zip(baseline_values, fdt_values)]
    
    # 
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    x = np.arange(len(datasets))
    width = 0.36

    # 
    bars1 = ax.bar(
        x - width/2,
        baseline_values,
        width,
        label='PEFT Default',
        color='#BDC3C7',
        edgecolor='#7F8C8D',
        linewidth=0.8,
    )

    bars2 = ax.bar(
        x + width/2,
        fdt_values,
        width,
        label='FDT (α=1.1)',
        color='#2980B9',
        edgecolor='#1F618D',
        linewidth=0.8,
    )

    #  y 
    max_height = max([b.get_height() for b in list(bars1) + list(bars2)] + [1])
    ylim_top = max_height * 1.3
    ax.set_ylim(0, ylim_top)

    # 
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        text_y = height + max_height * 0.06
        text_y = min(text_y, ylim_top * 0.95)
        ax.text(
            i,
            text_y,
            f'↓{imp:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#27AE60',
        )

    # 
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('AUC(0-500) ↓', fontweight='bold')
    ax.set_title('Convergence Speed Across Datasets (Lower is Better)', fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=9)

    # 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=9)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig2_cross_dataset_auc.{fmt}", format=fmt)
    
    print(f"  2:  AUC → {OUTPUT_DIR}/fig2_cross_dataset_auc.pdf")
    plt.close()


#  3: GSM8K α vs 
def plot_gsm8k_ablation_heatmap():
    """
    GSM8K 
    - : α  [0.6, 0.8, 0.9, 1.1, 1.5]
    - : LoRA  [8, 16, 32]
    """
    
    alphas = [0.6, 0.8, 0.9, 1.1, 1.5]
    ranks = [8, 16, 32]
    
    exp_map = {
        # EXPERIMENTS  alpha06/08/09/15
        (0.6, 16): EXPERIMENTS['gsm8k']['alpha06'],
        (0.8, 16): EXPERIMENTS['gsm8k']['alpha08'],
        (0.9, 16): EXPERIMENTS['gsm8k']['alpha09'],
        (1.1, 16): EXPERIMENTS['gsm8k']['fdt'],
        (1.5, 16): EXPERIMENTS['gsm8k']['alpha15'], 
        (1.1, 8): EXPERIMENTS['gsm8k']['r8_fdt'],
        (1.1, 32): EXPERIMENTS['gsm8k']['r32_fdt'],
    }
    
    matrix = np.full((len(alphas), len(ranks)), np.nan)

    for (alpha, rank), exp_dir in exp_map.items():
        results_file = os.path.join(exp_dir, "results.json")

        if not os.path.exists(results_file):
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        auc = results.get('auc_500')
        if auc is not None:
            i = alphas.index(alpha)
            j = ranks.index(rank)
            matrix[i, j] = auc

    if np.all(np.isnan(matrix)):
        print("  GSM8K :  AUC ")
        return

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    #  DataFrame  seaborn 
    df = pd.DataFrame(
        matrix,
        index=[f"α={a}" for a in alphas],
        columns=[f"r={r}" for r in ranks],
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    #  NaN 
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "AUC(0-500) ↓"},
    )

    ax.set_xlabel('LoRA Rank', fontweight='bold')
    ax.set_ylabel('FDT Alpha (α)', fontweight='bold')
    ax.set_title('GSM8K Ablation Study: AUC(0-500)', fontsize=11, pad=8)

    # 
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(10)
    cbar.ax.yaxis.label.set_fontweight('bold')

    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig3_gsm8k_ablation.{fmt}", format=fmt)
    
    print(f"  3: GSM8K  → {OUTPUT_DIR}/fig3_gsm8k_ablation.pdf")
    plt.close()


#  4: SOC
def plot_frequency_validation():
    """
    FDT 
    """
    
    exp_dir = EXPERIMENTS['gsm8k']['fdt']
    spectra_dir = os.path.join(exp_dir, "init_spectra")
    
    if not os.path.exists(spectra_dir):
        print(f"  : {spectra_dir}")
        return
    
    npz_files = list(Path(spectra_dir).glob("*.npz"))
    
    if not npz_files:
        print(f"  {spectra_dir}  .npz ")
        return
    
    data = np.load(npz_files[0])
    freqs = data['freqs']
    psd = data['psd']
    alpha_fit = float(data['alpha_fit'])
    
    # 
    freq_theory = freqs[freqs > 0]
    psd_theory = freq_theory ** (-1.1)
    psd_theory *= psd[len(psd)//4] / psd_theory[len(psd_theory)//4]
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    ax.loglog(freqs[freqs > 0], psd[freqs > 0], 
              'o', markersize=4, alpha=0.6, label='Measured PSD', 
              color='#3498DB')
    
    ax.loglog(freq_theory, psd_theory, 
              '--', linewidth=2.5, label=r'Theoretical $1/f^{1.1}$', 
              color='#E74C3C')
    
    ax.set_xlabel('Frequency (log scale)', fontweight='bold')
    ax.set_ylabel('Power Spectral Density (log scale)', fontweight='bold')
    ax.set_title(f'FDT Initialization Validation', fontsize=11)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    ax.text(0.05, 0.05, 
            f'Measured α = {alpha_fit:.2f}\nTarget α = 1.1\nError = {abs(alpha_fit - 1.1):.2f}', 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig4_frequency_validation.{fmt}", format=fmt)
    
    print(f"  4:  → {OUTPUT_DIR}/fig4_frequency_validation.pdf")
    plt.close()


#  5: 4 
def plot_efficiency_radar():
    """
     4 
    - 
    - 
    - 
    - AUC 
    """
    
    datasets = ['gsm8k', 'cmmlu', 'sharegpt', 'mbpp']
    dataset_labels = ['GSM8K', 'CMMLU', 'ShareGPT', 'MBPP']
    
    # 
    time_ratios = []
    memory_ratios = []
    throughput_ratios = []
    auc_improvements = []
    
    for dataset in datasets:
        baseline_file = os.path.join(EXPERIMENTS[dataset]['baseline'], "results.json")
        fdt_file = os.path.join(EXPERIMENTS[dataset]['fdt'], "results.json")
        
        if not (os.path.exists(baseline_file) and os.path.exists(fdt_file)):
            continue
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        with open(fdt_file, 'r') as f:
            fdt = json.load(f)
        
        # FDT / Baseline 1 
        time_ratio = fdt.get('wall_time_minutes', 0) / baseline.get('wall_time_minutes', 1)
        time_ratios.append(time_ratio)
        
        # 
        memory_ratio = fdt.get('peak_memory_gb', 0) / baseline.get('peak_memory_gb', 1)
        memory_ratios.append(memory_ratio)
        
        # FDT / Baseline
        throughput_ratio = fdt.get('throughput_samples_per_sec', 0) / baseline.get('throughput_samples_per_sec', 1)
        throughput_ratios.append(throughput_ratio)
        
        # AUC 
        baseline_auc = baseline.get('auc_500', 0)
        fdt_auc = fdt.get('auc_500', 0)
        improvement = (baseline_auc - fdt_auc) / baseline_auc * 100 if baseline_auc > 0 else 0
        auc_improvements.append(improvement)
    
    #  0-100
    def normalize(values):
        min_val, max_val = min(values), max(values)
        return [(v - min_val) / (max_val - min_val) * 100 if max_val > min_val else 50 for v in values]
    
    # 4 
    categories = ['Training Time\n(Lower Better)', 
                  'Peak Memory\n(Lower Better)', 
                  'Throughput\n(Higher Better)',
                  'AUC Improvement\n(Higher Better)']
    
    #  GSM8K
    gsm8k_idx = 0
    values = [
        100 - normalize(time_ratios)[gsm8k_idx],  # 
        100 - normalize(memory_ratios)[gsm8k_idx],  # 
        normalize(throughput_ratios)[gsm8k_idx],  # 
        normalize(auc_improvements)[gsm8k_idx],  # 
    ]
    
    # 
    values += values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2, label='FDT (α=1.1)', color='#3498DB')
    ax.fill(angles, values, alpha=0.25, color='#3498DB')
    
    # Baseline 50 
    baseline_values = [50] * len(angles)
    ax.plot(angles, baseline_values, '--', linewidth=1.5, label='PEFT Default', color='#95A5A6')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=8)
    ax.set_title('Efficiency Profile (GSM8K)', fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig5_efficiency_radar.{fmt}", format=fmt)
    
    print(f"  5:  → {OUTPUT_DIR}/fig5_efficiency_radar.pdf")
    plt.close()


#  6: 4 
def plot_detailed_efficiency():
    """
    3 
     Baseline vs FDT
    """
    
    datasets = ['gsm8k', 'cmmlu', 'sharegpt', 'mbpp']
    dataset_labels = ['GSM8K', 'CMMLU', 'ShareGPT', 'MBPP']
    
    metrics = {
        'Training Time (min)': 'wall_time_minutes',
        'Peak Memory (GB)': 'peak_memory_gb',
        'Throughput (samples/s)': 'throughput_samples_per_sec',
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    for ax, (metric_name, json_key) in zip(axes, metrics.items()):
        baseline_values = []
        fdt_values = []
        
        for dataset in datasets:
            baseline_file = os.path.join(EXPERIMENTS[dataset]['baseline'], "results.json")
            fdt_file = os.path.join(EXPERIMENTS[dataset]['fdt'], "results.json")
            
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                baseline_values.append(data.get(json_key, 0))
            else:
                baseline_values.append(0)
            
            if os.path.exists(fdt_file):
                with open(fdt_file, 'r') as f:
                    data = json.load(f)
                fdt_values.append(data.get(json_key, 0))
            else:
                fdt_values.append(0)
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, label='PEFT Default', 
               color='#95A5A6', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, fdt_values, width, label='FDT (α=1.1)', 
               color='#3498DB', edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(metric_name, fontweight='bold', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=8, rotation=15, ha='right')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        plt.savefig(f"{OUTPUT_DIR}/fig6_detailed_efficiency.{fmt}", format=fmt)
    
    print(f"  6:  → {OUTPUT_DIR}/fig6_detailed_efficiency.pdf")
    plt.close()


# 
def main():
    print("="*70)
    print(" 4 ")
    print("="*70)
    print()
    print(":")
    print("  • GSM8K: ")
    print("  • CMMLU: ")
    print("  • ShareGPT: ")
    print("  • MBPP: ")
    print()
    print(":", OUTPUT_DIR)
    print()
    
    # 
    plot_4dataset_training_curves()
    plot_step_auc_curves()
    plot_cross_dataset_auc()
    plot_gsm8k_ablation_heatmap()
    plot_frequency_validation()
    plot_efficiency_radar()
    plot_detailed_efficiency()
    
    print()
    print("="*70)
    print(" !")
    print("="*70)
    print()
    print(":")
    for file in sorted(Path(OUTPUT_DIR).glob("*.pdf")):
        print(f"  • {file}")
    print()


if __name__ == "__main__":
    main()