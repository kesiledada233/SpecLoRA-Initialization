"""
频域感知初始化器（Frequency-Domain-Aware Initialization）

核心思想：
- 传统初始化（Xavier/He）→ 白噪声（α ≈ 0）
- FDT 初始化 → 粉红噪声（α ≈ 1.2），本文针对 LoRA 推荐 α = 0.6
- 匹配收敛模型的频域特性，加速早期收敛

实现方法：
1. FFT 合成法：精确生成 P(f) ∝ f^(-α)
2. 自回归法：快速近似，适合大规模部署

适用场景：
- LoRA 微调（主要用途）
- 全参数微调
- 从头预训练
"""

import torch
import numpy as np
from typing import Optional, Tuple
import warnings
import math

def fdt_initialize_(
    tensor: torch.Tensor,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    unroll_order: str = 'row',  # ⬅️ 新增：'row' (按行展开) 或 'col' (按列展开)
    verbose: bool = False
) -> torch.Tensor:
    """
    频域感知初始化（只生成功率律噪声，不做方差归一化）
    
    核心原理：
    1. 生成功率谱 P(f) ∝ f^(-α) 的随机信号
    2. 返回**未归一化**的噪声（让调用者决定方差）
    
    Args:
        tensor: 要初始化的参数张量（任意形状）
        alpha: 功率律指数（推荐 0.6 或 1.2）
        temp_ratio: 高频/低频能量比（可选）
        method: 'fft'（精确）或 'ar'（快速）
        unroll_order: 'row' (C-contiguous) 或 'col' (Fortran-contiguous)
        verbose: 是否打印详细信息
    
    Returns:
        初始化后的张量（原地修改，**未归一化**）
    """
    with torch.no_grad():
        original_shape = tensor.shape
        original_device = tensor.device
        original_dtype = tensor.dtype
        
        # 展平为 1D
        numel = tensor.numel()
        
        if verbose:
            print(f"🔧 FDT 初始化: shape={original_shape}, α={alpha:.2f}, method={method}, order={unroll_order}")
        
        # === 方法 1: FFT 频域合成（精确）===
        if method == 'fft':
            n_freqs = numel // 2 + 1
            freqs = np.arange(1, n_freqs)
            
            if len(freqs) == 0:
                warnings.warn(f"张量太小 (numel={numel})，使用标准正态分布")
                tensor.normal_(0, 1)
                return tensor
            
            # 功率谱 P(f) ∝ f^(-α)
            power_spectrum = freqs.astype(np.float64) ** (-alpha)
            
            # 可选：温度比调整
            if temp_ratio is not None and temp_ratio != 1.0:
                cutoff_idx = max(1, len(freqs) // 16)
                
                energy_low = power_spectrum[:cutoff_idx].sum()
                energy_high = power_spectrum[cutoff_idx:].sum()
                
                if energy_low > 0 and energy_high > 0:
                    current_ratio = energy_high / energy_low
                    scale = np.sqrt(temp_ratio / current_ratio)
                    power_spectrum[cutoff_idx:] *= scale
                    
                    if verbose:
                        print(f"  温度比调整: {current_ratio:.3f} → {temp_ratio:.3f}")
            
            # 振幅 A(f) = √P(f)
            amplitudes = np.sqrt(power_spectrum)
            
            # 随机相位
            phases = np.random.uniform(0, 2 * np.pi, len(freqs))
            
            # 构造复数 FFT 系数
            fft_coeffs = amplitudes * np.exp(1j * phases)
            fft_coeffs_full = np.concatenate([[0.0 + 0.0j], fft_coeffs])
            
            # 逆 FFT
            time_series = np.fft.irfft(fft_coeffs_full, n=numel)
            
            # 只做零均值
            time_series = time_series - time_series.mean()
        
        # === 方法 2: 自回归近似 ===
        elif method == 'ar':
            beta = 0.5 + 0.3 * (alpha - 1.0)
            beta = np.clip(beta, 0.1, 0.9)
            
            time_series = np.zeros(numel)
            time_series[0] = np.random.randn()
            
            noise_scale = np.sqrt(1 - beta**2)
            
            for i in range(1, numel):
                time_series[i] = beta * time_series[i-1] + noise_scale * np.random.randn()
            
            time_series = time_series - time_series.mean()
            
            if verbose:
                print(f"  AR 参数: β={beta:.3f}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # === 将 1D 序列映射回多维 Tensor ===
        tensor_flat = torch.from_numpy(time_series).to(dtype=original_dtype, device=original_device)
        
        if unroll_order == 'row':
            # 默认按行展开 (C-contiguous)
            tensor.copy_(tensor_flat.reshape(original_shape))
        elif unroll_order == 'col':
            # 按列展开 (Fortran-contiguous)
            # 做法：先按反向的 shape 填充，然后再转置维度
            reversed_shape = tuple(reversed(original_shape))
            dims_reversed = tuple(range(len(original_shape) - 1, -1, -1))
            tensor.copy_(tensor_flat.reshape(reversed_shape).permute(dims_reversed))
        else:
            raise ValueError(f"Unknown unroll_order: {unroll_order}. Use 'row' or 'col'.")
        
        if verbose:
            actual_std = tensor.std().item()
            print(f"  ✓ 完成: 原始 std={actual_std:.6f}（未归一化）")

    return tensor

def apply_fdt_to_lora(
    model: torch.nn.Module,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    unroll_order: str = 'row',  # ⬅️ 暴露给上层接口
    verbose: bool = True
) -> torch.nn.Module:
    """对模型中的 LoRA 层应用 FDT 初始化（带方差归一化）"""
    
    initialized_count = 0
    skipped_count = 0
    total_trainable = 0
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 应用 FDT 初始化到 LoRA 层（带方差归一化）")
        print("="*70)
        print(f"配置: α={alpha:.2f}", end='')
        if temp_ratio:
            print(f", τ={temp_ratio:.2f}", end='')
        print(f", method={method}, order={unroll_order}")
        print("-"*70)
    
    lora_patterns = ['lora', 'adapter', 'delta', 'ia3']
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        total_trainable += 1
        is_lora = any(pattern in name.lower() for pattern in lora_patterns)
        
        if is_lora:
            try:
                if verbose:
                    print(f"[{initialized_count+1}] {name}")
                    print(f"    形状: {list(param.shape)}")
                
                # 步骤 1: 计算 Xavier 目标标准差
                dimensions = param.dim()
                if dimensions >= 2:
                    fan_in = param.size(1)
                    fan_out = param.size(0)
                    
                    if dimensions > 2:
                        receptive_field_size = param[0][0].numel()
                        fan_in *= receptive_field_size
                        fan_out *= receptive_field_size
                    
                    # Xavier/Glorot 标准差
                    target_std = math.sqrt(2.0 / (fan_in + fan_out))
                else:
                    # 1D 张量（很少见）
                    target_std = 1.0 / math.sqrt(param.numel())
                
                if verbose:
                    print(f"    Xavier std: {target_std:.6f} (fan_in={fan_in}, fan_out={fan_out})")
                
                # 步骤 2: 生成 FDT 噪声（未归一化，传入 unroll_order）
                fdt_initialize_(
                    param.data,
                    alpha=alpha,
                    temp_ratio=temp_ratio,
                    method=method,
                    unroll_order=unroll_order,  # ⬅️ 传递参数
                    verbose=False
                )
                
                # 步骤 3: 归一化到 Xavier 方差
                with torch.no_grad():
                    current_std = param.data.std().item()
                    
                    if current_std > 1e-8:
                        scale_factor = target_std / current_std
                        param.data.mul_(scale_factor)
                        
                        if verbose:
                            actual_std = param.data.std().item()
                            print(f"    归一化: {current_std:.6f} × {scale_factor:.4f} = {actual_std:.6f}")
                    else:
                        if verbose:
                            print(f"    ⚠️  标准差过小 ({current_std:.2e})，使用随机初始化")
                        param.data.normal_(0, target_std)
                
                # 步骤 4: 验证功率律指数
                try:
                    # 如果你有 measure_alpha 函数可以在这里调用
                    # from measure_alpha import measure_alpha
                    # measured_alpha = measure_alpha(param.data)
                    # error = abs(measured_alpha - alpha)
                    # status = "✓" if error < 0.15 else "⚠️"
                    # if verbose:
                    #     print(f"    测量 α: {measured_alpha:.3f} {status} (误差={error:.3f})")
                    pass
                except Exception as e:
                    if verbose:
                        print(f"    ⚠️  无法测量 α: {e}")
                
                if verbose:
                    print()
                
                initialized_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"    ✗ 失败: {e}\n")
                import traceback
                traceback.print_exc()
                skipped_count += 1
        else:
            skipped_count += 1
    
    if verbose:
        print("-"*70)
        print(f"统计:")
        print(f"  • 总可训练参数: {total_trainable}")
        print(f"  • 已初始化: {initialized_count}")
        print(f"  • 跳过: {skipped_count}")
        
        if initialized_count == 0:
            print("\n❌ 警告：没有参数被初始化！")
        elif initialized_count < total_trainable * 0.5:
            print(f"\n⚠️  警告：只初始化了 {initialized_count}/{total_trainable} 个参数")
        else:
            print(f"\n✅ 成功初始化 {initialized_count} 个 LoRA 层")
        print("="*70)
    
    return model


# ============ 便捷函数 ============

def init_lora_with_pink_noise(
    model: torch.nn.Module,
    alpha: float = 0.6,
    unroll_order: str = 'row',
    verbose: bool = True
) -> torch.nn.Module:
    """
    便捷函数：用粉红噪声初始化 LoRA（推荐配置）
    """
    return apply_fdt_to_lora(model, alpha=alpha, method='fft', unroll_order=unroll_order, verbose=verbose)