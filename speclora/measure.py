"""
功率律指数测量工具

功能：
1. 测量参数的功率律指数 α
2. 绘制功率谱（验证初始化质量）
3. 批量分析模型所有 LoRA 层
4. 验证 FDA 初始化是否成功（α 是否接近目标值）

"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def measure_alpha(
    tensor: torch.Tensor,
    method: str = 'fft',
    fit_range: Optional[Tuple[float, float]] = None,
    return_full_spectrum: bool = False
) -> float:
    """
    
    Args:
        tensor: 参数张量（任意形状）
        method: 'fft'（频域）或 'dfa'（去趋势波动分析，未实现）
        fit_range: 拟合的频率范围（整数索引，如 (10, 1000)），None 表示自动
        return_full_spectrum: 是否返回完整功率谱
    
    Returns:
        α: 功率律指数（如果 return_full_spectrum=True，返回 (α, freqs, power)）
    """
    with torch.no_grad():
        # 展平并转到 CPU
        data = tensor.detach().flatten().cpu().numpy()
        
        if len(data) < 10:
            return np.nan
        
        # 中心化（去除均值）
        data = data - data.mean()
        
        if method == 'fft':
            # FFT 功率谱
            fft_result = np.fft.rfft(data)
            power_spectrum = np.abs(fft_result) ** 2
            
            # 使用整数频率索引
            n_freqs = len(power_spectrum)
            freqs = np.arange(1, n_freqs)  # [1, 2, 3, ..., N/2]（跳过 DC）
            power = power_spectrum[1:]      # 去掉 DC 分量
            
            if len(freqs) < 3:
                return np.nan
            
            # 确定拟合范围
            if fit_range is not None:
                # 用户指定范围（整数索引）
                low_idx = max(0, fit_range[0] - 1)  # freqs[0] = 1
                high_idx = min(len(freqs), fit_range[1] - 1)
            else:
                # 默认：跳过极低频（前 5%）和极高频（后 10%）
                low_idx = max(0, int(len(freqs) * 0.05))
                high_idx = int(len(freqs) * 0.90)
            
            if high_idx - low_idx < 3:
                return np.nan
            
            # 对数回归（使用整数频率）
            log_freq = np.log(freqs[low_idx:high_idx].astype(np.float64))
            log_power = np.log(power[low_idx:high_idx] + 1e-12)
            
            # 最小二乘拟合 log(P) = a + b*log(f)
            coeffs = np.polyfit(log_freq, log_power, 1)
            alpha = -coeffs[0]  # 斜率取负
            
            if return_full_spectrum:
                return alpha, freqs, power
            else:
                return alpha
        
        else:
            raise ValueError(f"Unknown method: {method}")


def plot_power_spectrum(
    tensor: torch.Tensor,
    title: str = "Power Spectrum",
    save_path: Optional[str] = None,
    show_fit: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制功率谱（对数-对数坐标）
    
    Args:
        tensor: 参数张量
        title: 图表标题
        save_path: 保存路径（None 表示只显示）
        show_fit: 是否显示功率律拟合线
        figsize: 图表大小
    
    Example:
        >>> plot_power_spectrum(lora_A, title="LoRA A Initialization")
    """
    alpha, freqs, power = measure_alpha(tensor, return_full_spectrum=True)
    
    plt.figure(figsize=figsize)
    
    # 功率谱（对数坐标）
    plt.loglog(freqs, power, 'o-', markersize=3, alpha=0.6, label='Power Spectrum')
    
    # 功率律拟合线
    if show_fit and not np.isnan(alpha):
        fit_power = freqs[0] ** (-alpha) * power[0] / (freqs[0] ** (-alpha))
        fit_line = fit_power * (freqs ** (-alpha))
        plt.loglog(freqs, fit_line, 'r--', linewidth=2, 
                   label=f'Power Law Fit: P(f) ~ f^(-{alpha:.2f})')
    
    plt.xlabel('Frequency (normalized)', fontsize=12)
    plt.ylabel('Power Spectral Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_lora_spectra(
    model: torch.nn.Module,
    save_dir: Optional[str] = None,
    plot_top_n: int = 3,
    verbose: bool = True
) -> Dict[str, float]:
    """
    分析模型中所有 LoRA 层的功率律指数
    
    Args:
        model: 带 LoRA 的模型
        save_dir: 保存图表的目录（None 表示不保存）
        plot_top_n: 绘制前 N 个层的功率谱
        verbose: 是否打印详细信息
    
    Returns:
        {层名称: α 值} 的字典
    
    Example:
        >>> alphas = analyze_lora_spectra(model, save_dir='./spectra', plot_top_n=5)
        >>> print(f"Average alpha: {np.mean(list(alphas.values())):.3f}")
    """
    results = {}
    plotted_count = 0
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print("LoRA Layer Spectrum Analysis".center(70))
        print("="*70)
        print(f"{'Layer Name':50s} {'Alpha':>10s} {'Shape':>15s}")
        print("-"*70)
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            alpha = measure_alpha(param)
            results[name] = alpha
            
            if verbose:
                shape_str = str(list(param.shape))
                status = "✓" if 0.8 <= alpha <= 1.5 else "!"
                print(f"{status} {name:48s} {alpha:10.3f} {shape_str:>15s}")
            
            # 绘制前 N 个层
            if save_dir and plotted_count < plot_top_n:
                safe_name = name.replace('.', '_').replace('/', '_')
                plot_path = Path(save_dir) / f"{safe_name}_spectrum.png"
                
                plot_power_spectrum(
                    param,
                    title=f"{name} (alpha={alpha:.3f})",
                    save_path=str(plot_path),
                    show_fit=True
                )
                plotted_count += 1
    
    if verbose:
        print("-"*70)
        valid_alphas = [a for a in results.values() if not np.isnan(a)]
        if valid_alphas:
            print(f"Statistics: Mean alpha = {np.mean(valid_alphas):.3f} ± {np.std(valid_alphas):.3f}")
            print(f"            Range [{np.min(valid_alphas):.3f}, {np.max(valid_alphas):.3f}]")
        print("="*70 + "\n")
    
    return results


def verify_fda_initialization(
    model: torch.nn.Module,
    target_alpha: float,
    tolerance: float = 0.1,
    verbose: bool = True
) -> bool:
    """
    验证 FDA 初始化是否成功
    
    Args:
        model: 初始化后的模型
        target_alpha: 目标 α 值
        tolerance: 允许误差
        verbose: 是否打印详细信息
    
    Returns:
        True 如果所有 LoRA 层都在容差范围内
    
    Example:
        >>> apply_fda_to_lora(model, alpha=1.2)
        >>> success = verify_fda_initialization(model, target_alpha=1.2, tolerance=0.1)
    """
    alphas = analyze_lora_spectra(model, verbose=False)
    
    valid_alphas = [a for a in alphas.values() if not np.isnan(a)]
    
    if len(valid_alphas) == 0:
        if verbose:
            print("Verification Failed: Cannot measure any alpha values")
        return False
    
    errors = [abs(a - target_alpha) for a in valid_alphas]
    max_error = max(errors)
    mean_error = np.mean(errors)
    
    success = max_error < tolerance
    
    if verbose:
        print("\n" + "="*70)
        print("FDA Initialization Verification".center(70))
        print("="*70)
        print(f"Target alpha: {target_alpha:.3f}")
        print(f"Measured alpha: {np.mean(valid_alphas):.3f} ± {np.std(valid_alphas):.3f}")
        print(f"Max error: {max_error:.3f} (tolerance: {tolerance:.3f})")
        print(f"Mean error: {mean_error:.3f}")
        print("-"*70)
        
        if success:
            print("Verification PASSED! All layers within tolerance.")
        else:
            print(f"Verification FAILED! Max error {max_error:.3f} exceeds tolerance {tolerance:.3f}")
            print("\nLayers exceeding tolerance:")
            for name, alpha in alphas.items():
                if abs(alpha - target_alpha) >= tolerance:
                    print(f"  • {name}: alpha={alpha:.3f} (error {abs(alpha - target_alpha):.3f})")
        
        print("="*70 + "\n")
    
    return success


def compare_initializations(
    model_baseline: torch.nn.Module,
    model_fda: torch.nn.Module,
    layer_name: str,
    save_path: Optional[str] = None
):
    """
    对比两种初始化的功率谱
    
    Args:
        model_baseline: 标准初始化的模型
        model_fda: FDA 初始化的模型
        layer_name: 要对比的层名称（如 'model.layers.0.self_attn.q_proj.lora_A')
        save_path: 保存路径
    
    Example:
        >>> compare_initializations(
        ...     model_xavier,
        ...     model_fda,
        ...     'model.layers.0.self_attn.q_proj.lora_A',
        ...     save_path='comparison.png'
        ... )
    """
    param_baseline = dict(model_baseline.named_parameters())[layer_name]
    param_fda = dict(model_fda.named_parameters())[layer_name]
    
    alpha_baseline, freqs_b, power_b = measure_alpha(param_baseline, return_full_spectrum=True)
    alpha_fda, freqs_f, power_f = measure_alpha(param_fda, return_full_spectrum=True)
    
    plt.figure(figsize=(12, 6))
    
    # Baseline
    plt.loglog(freqs_b, power_b, 'o-', markersize=3, alpha=0.6, 
               label=f'Xavier (alpha={alpha_baseline:.3f})', color='blue')
    
    # FDA
    plt.loglog(freqs_f, power_f, 's-', markersize=3, alpha=0.6,
               label=f'FDA (alpha={alpha_fda:.3f})', color='red')
    
    # 理论线
    f_ref = freqs_f[len(freqs_f)//4]
    p_ref_fda = power_f[len(power_f)//4]
    
    theory_line = p_ref_fda * (freqs_f / f_ref) ** (-alpha_fda)
    plt.loglog(freqs_f, theory_line, 'r--', linewidth=2, alpha=0.5,
               label=f'Theory: f^(-{alpha_fda:.2f})')
    
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power Spectral Density', fontsize=12)
    plt.title(f'Initialization Comparison: {layer_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()