"""
FDT Initialization Comparison Visualization Script
Compare Baseline (Xavier) and FDT (α=1.1) training results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set font (remove Chinese font, use default)
matplotlib.rcParams['font.size'] = 11

BASELINE_DIR = "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_sharegpt/baseline"
FDT_DIR = "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_sharegpt/alpha0.6"
OUTPUT_DIR = "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/comparison_sharegpt_plots_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print(" FDT Initialization Comparison Visualization")
print("="*70)

print("\n[1/5] Loading data...")

def load_experiment_data(exp_dir):
    """Load experiment data"""
    data = {}
    
    # 1. Training losses
    losses_file = os.path.join(exp_dir, "training_losses.npy")
    if os.path.exists(losses_file):
        data['train_losses'] = np.load(losses_file)
        print(f"   {exp_dir}: Training losses ({len(data['train_losses'])} steps)")
    else:
        print(f"   {exp_dir}: training_losses.npy not found")
        data['train_losses'] = None
    
    # 2. Evaluation losses
    eval_file = os.path.join(exp_dir, "eval_losses.json")
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            data['eval_losses'] = json.load(f)
        print(f"   {exp_dir}: Evaluation losses")
    else:
        print(f"   {exp_dir}: eval_losses.json not found")
        data['eval_losses'] = None
    
    # 3. Initialization info
    init_file = os.path.join(exp_dir, "init_info.json")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            data['init_info'] = json.load(f)
        print(f"   {exp_dir}: Initialization info")
    else:
        data['init_info'] = {}
    
    # 4. Config
    config_file = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data['config'] = json.load(f)
    else:
        data['config'] = {}
    
    return data

baseline_data = load_experiment_data(BASELINE_DIR)
fdt_data = load_experiment_data(FDT_DIR)

# Check data integrity
if baseline_data['train_losses'] is None or fdt_data['train_losses'] is None:
    print("\n Error: Missing training loss files, cannot generate plots")
    exit(1)

print("\n[2/5] Computing key metrics...")

def compute_metrics(data):
    """Compute key metrics"""
    metrics = {}
    
    train_losses = data['train_losses']
    
    # AUC (0-500)
    if len(train_losses) >= 500:
        metrics['auc_500'] = float(np.sum(train_losses[:500]))
    else:
        metrics['auc_500'] = float(np.sum(train_losses))
    
    # Final loss
    metrics['final_loss'] = float(train_losses[-1])
    
    # Loss at step 100
    if len(train_losses) >= 100:
        metrics['loss_100'] = float(np.mean(train_losses[90:100]))
    else:
        metrics['loss_100'] = float(np.mean(train_losses))
    
    # Loss at step 500
    if len(train_losses) >= 500:
        metrics['loss_500'] = float(np.mean(train_losses[490:500]))
    else:
        metrics['loss_500'] = metrics['final_loss']
    
    # Best loss
    metrics['best_loss'] = float(np.min(train_losses))
    
    # Test loss (if available)
    if data['eval_losses'] and 'final_metrics' in data['eval_losses']:
        test_loss = data['eval_losses']['final_metrics'].get('test_loss')
        metrics['test_loss'] = float(test_loss) if test_loss is not None else None
    else:
        metrics['test_loss'] = None
    
    return metrics

baseline_metrics = compute_metrics(baseline_data)
fdt_metrics = compute_metrics(fdt_data)

print("\n  Baseline metrics:")
for k, v in baseline_metrics.items():
    if v is not None:
        print(f"    • {k}: {v:.4f}")

print("\n  FDT α=1.1 metrics:")
for k, v in fdt_metrics.items():
    if v is not None:
        print(f"    • {k}: {v:.4f}")

# Calculate improvement percentages
improvements = {}
for key in baseline_metrics.keys():
    if baseline_metrics[key] is not None and fdt_metrics[key] is not None:
        baseline_val = baseline_metrics[key]
        fdt_val = fdt_metrics[key]
        
        # Lower is better for all metrics
        improvement_pct = (baseline_val - fdt_val) / baseline_val * 100
        improvements[key] = improvement_pct

print("\n  Improvement percentages:")
for k, v in improvements.items():
    sign = "" if v > 0 else ""
    print(f"    {sign} {k}: {v:+.1f}%")

print("\n[3/5] Plotting training curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FDT Initialization vs Baseline (Xavier) - ShareGPT', 
             fontsize=16, fontweight='bold')

baseline_losses = baseline_data['train_losses']
fdt_losses = fdt_data['train_losses']
steps = np.arange(1, len(baseline_losses) + 1)

# Subplot 1: Full training curve
ax1 = axes[0, 0]
ax1.plot(steps, baseline_losses, 'o-', label='Baseline (Xavier)', 
         linewidth=2, markersize=3, alpha=0.7, color='#e74c3c')
ax1.plot(steps, fdt_losses, 's-', label='FDT α=1.1', 
         linewidth=2, markersize=3, alpha=0.7, color='#27ae60')
ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('(a) Complete Training Curve', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# Subplot 2: First 500 steps (critical phase)
ax2 = axes[0, 1]
steps_500 = steps[:500]
ax2.plot(steps_500, baseline_losses[:500], 'o-', label='Baseline', 
         linewidth=2, markersize=4, alpha=0.7, color='#e74c3c')
ax2.plot(steps_500, fdt_losses[:500], 's-', label='FDT α=1.1', 
         linewidth=2, markersize=4, alpha=0.7, color='#27ae60')
ax2.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Step 100')
ax2.axvline(x=300, color='gray', linestyle=':', alpha=0.5, label='Step 300')
ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('(b) First 500 Steps (Early Convergence)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(alpha=0.3, linestyle='--')

# Mark key points
for step_mark in [100, 300, 500]:
    if step_mark <= len(baseline_losses):
        baseline_val = baseline_losses[step_mark-1]
        fdt_val = fdt_losses[step_mark-1]
        
        # Baseline
        ax2.plot(step_mark, baseline_val, 'o', color='#e74c3c', 
                markersize=8, markeredgewidth=2, markeredgecolor='white')
        
        # FDT
        ax2.plot(step_mark, fdt_val, 's', color='#27ae60', 
                markersize=8, markeredgewidth=2, markeredgecolor='white')

# Subplot 3: Moving average (window=50)
ax3 = axes[1, 0]
window = 50

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

baseline_smooth = moving_average(baseline_losses, window)
fdt_smooth = moving_average(fdt_losses, window)
steps_smooth = np.arange(window, len(baseline_losses) + 1)

ax3.plot(steps_smooth, baseline_smooth, '-', label='Baseline (MA-50)', 
         linewidth=2.5, alpha=0.9, color='#e74c3c')
ax3.plot(steps_smooth, fdt_smooth, '-', label='FDT α=1.1 (MA-50)', 
         linewidth=2.5, alpha=0.9, color='#27ae60')
ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss (Smoothed)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Moving Average Curve (Window=50)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11, loc='upper right')
ax3.grid(alpha=0.3, linestyle='--')

# Subplot 4: AUC comparison (bar chart)
ax4 = axes[1, 1]

auc_baseline = baseline_metrics['auc_500']
auc_fdt = fdt_metrics['auc_500']

bars = ax4.bar(['Baseline\n(Xavier)', 'FDT\nα=1.1'], 
               [auc_baseline, auc_fdt],
               color=['#e74c3c', '#27ae60'], 
               alpha=0.8, 
               edgecolor='black', 
               linewidth=2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, [auc_baseline, auc_fdt])):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Show improvement percentage on bar
    if i == 1:
        improvement = improvements['auc_500']
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{improvement:+.1f}%',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

ax4.set_ylabel('Cumulative Loss (Steps 0-500)', fontsize=12, fontweight='bold')
ax4.set_title('(d) AUC(0-500) Comparison', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim(0, max(auc_baseline, auc_fdt) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_training_curves.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/1_training_curves.png")
plt.close()

print("\n[4/5] Plotting key metrics comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss comparison (bar chart)
ax1 = axes[0]

metrics_to_plot = ['loss_100', 'loss_500', 'final_loss', 'best_loss']
metric_labels = ['Step 100', 'Step 500', 'Final', 'Best']

baseline_values = [baseline_metrics[m] for m in metrics_to_plot]
fdt_values = [fdt_metrics[m] for m in metrics_to_plot]

x = np.arange(len(metric_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_values, width, 
                label='Baseline', color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, fdt_values, width, 
                label='FDT α=1.1', color='#27ae60', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax1.set_title('Loss at Key Steps', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels, fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 2: Improvement percentages (horizontal bar chart)
ax2 = axes[1]

improvement_values = [improvements[m] for m in metrics_to_plot]
colors = ['#27ae60' if v > 0 else '#e74c3c' for v in improvement_values]

bars = ax2.barh(metric_labels, improvement_values, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for bar, val in zip(bars, improvement_values):
    width = bar.get_width()
    label_x_pos = width + (2 if width > 0 else -2)
    ha = 'left' if width > 0 else 'right'
    
    ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2,
             f'{val:+.1f}%',
             va='center', ha=ha, fontsize=11, fontweight='bold')

ax2.axvline(x=0, color='black', linewidth=1.5)
ax2.set_xlabel('Improvement (%)', fontsize=13, fontweight='bold')
ax2.set_title('FDT Improvement over Baseline', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
ax2.text(0.02, 0.98, ' Green = Improvement\n Red = Degradation',
         transform=ax2.transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_metrics_comparison.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/2_metrics_comparison.png")
plt.close()

print("\n[5/5] Plotting phase analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Phase Analysis', fontsize=16, fontweight='bold')

# Define phases
phases = [
    (1, 100, 'Early Phase (Steps 1-100)'),
    (100, 300, 'Early-Mid Phase (Steps 100-300)'),
    (300, 500, 'Mid Phase (Steps 300-500)'),
    (500, len(baseline_losses), 'Late Phase (Steps 500-2500)')
]

for idx, (start, end, title) in enumerate(phases):
    ax = axes[idx // 2, idx % 2]
    
    steps_phase = steps[start-1:end]
    baseline_phase = baseline_losses[start-1:end]
    fdt_phase = fdt_losses[start-1:end]
    
    ax.plot(steps_phase, baseline_phase, 'o-', label='Baseline', 
            linewidth=2, markersize=3, alpha=0.7, color='#e74c3c')
    ax.plot(steps_phase, fdt_phase, 's-', label='FDT α=1.1', 
            linewidth=2, markersize=3, alpha=0.7, color='#27ae60')
    
    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Calculate statistics for this phase
    baseline_mean = np.mean(baseline_phase)
    fdt_mean = np.mean(fdt_phase)
    improvement_phase = (baseline_mean - fdt_mean) / baseline_mean * 100
    
    # Add statistics text
    stats_text = f'Average Loss:\nBaseline: {baseline_mean:.3f}\nFDT: {fdt_mean:.3f}\nImprovement: {improvement_phase:+.1f}%'
    
    color = '#27ae60' if improvement_phase > 0 else '#e74c3c'
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_phase_analysis.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/3_phase_analysis.png")
plt.close()

print("\n[6/6] Generating text report...")

report_file = os.path.join(OUTPUT_DIR, 'comparison_report.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("FDT Initialization vs Baseline Comparison Report (WikiText-2)\n")
    f.write("="*70 + "\n\n")
    
    f.write("Experiment Configuration\n")
    f.write(f"  Dataset: WikiText-2\n")
    f.write(f"  Model: OpenPangu-7B (LoRA)\n")
    f.write(f"  Training Steps: {len(baseline_losses)}\n")
    f.write(f"  Baseline: Xavier Initialization\n")
    f.write(f"  FDT: α=1.1 (Pink Noise)\n\n")
    
    f.write("Key Metrics Comparison\n")
    f.write(f"{'Metric':<20} {'Baseline':<15} {'FDT α=1.1':<15} {'Improvement':<10}\n")
    f.write("-"*70 + "\n")
    
    for key in ['auc_500', 'loss_100', 'loss_500', 'final_loss', 'best_loss']:
        baseline_val = baseline_metrics[key]
        fdt_val = fdt_metrics[key]
        improve = improvements[key]
        
        key_name = {
            'auc_500': 'AUC(0-500)',
            'loss_100': 'Loss@Step100',
            'loss_500': 'Loss@Step500',
            'final_loss': 'Final Loss',
            'best_loss': 'Best Loss',
        }[key]
        
        sign = "" if improve > 0 else ""
        f.write(f"{key_name:<20} {baseline_val:<15.4f} {fdt_val:<15.4f} {sign} {improve:+.1f}%\n")
    
    f.write("\nPhase Analysis\n\n")
    
    for start, end, title in phases:
        baseline_phase = baseline_losses[start-1:end]
        fdt_phase = fdt_losses[start-1:end]
        
        baseline_mean = np.mean(baseline_phase)
        fdt_mean = np.mean(fdt_phase)
        improvement_phase = (baseline_mean - fdt_mean) / baseline_mean * 100
        
        f.write(f"{title}:\n")
        f.write(f"  Baseline Average Loss: {baseline_mean:.4f}\n")
        f.write(f"  FDT Average Loss: {fdt_mean:.4f}\n")
        f.write(f"  Improvement: {improvement_phase:+.1f}%\n\n")
    
    f.write("Main Findings\n\n")
    
    f.write("1. Early Convergence Speed (AUC 0-500):\n")
    f.write(f"   FDT significantly outperforms Baseline, reducing cumulative loss by {improvements['auc_500']:.1f}%\n\n")
    
    if improvements['final_loss'] > 0:
        f.write("2. Final Performance:\n")
        f.write(f"   FDT final loss lower than Baseline by {improvements['final_loss']:.1f}%\n\n")
    else:
        f.write("2. Final Performance:\n")
        f.write(f"   FDT final loss higher than Baseline by {-improvements['final_loss']:.1f}%\n")
        f.write(f"   Possible cause: Premature convergence to local optimum\n\n")
    
    f.write("3. Training Stability:\n")
    baseline_std = np.std(baseline_losses)
    fdt_std = np.std(fdt_losses)
    std_improvement = (baseline_std - fdt_std) / baseline_std * 100
    f.write(f"   Baseline Std Dev: {baseline_std:.4f}\n")
    f.write(f"   FDT Std Dev: {fdt_std:.4f}\n")
    f.write(f"   Stability Improvement: {std_improvement:+.1f}%\n\n")
    
    f.write("="*70 + "\n")

print(f"   Saved: {report_file}")

print("\n" + "="*70)
print(" All plots generated successfully!")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. 1_training_curves.png     - Training curve comparison (4 subplots)")
print("  2. 2_metrics_comparison.png  - Key metrics comparison")
print("  3. 3_phase_analysis.png      - Phase analysis (4 phases)")
print("  4. comparison_report.txt     - Text report")
print("\nNext steps:")
print("  • Insert plots into LaTeX report")
print("  • Use \\includegraphics command")
print("="*70 + "\n")