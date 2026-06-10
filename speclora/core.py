"""
Frequency-domain-aware initialization for LoRA fine-tuning.

Generates power-law noise (pink noise, alpha ~ 0.6-1.2) to match
the spectral properties of converged models and accelerate early convergence.

Methods:
    fft: Exact frequency-domain synthesis via inverse FFT.
    ar:  Fast autoregressive approximation for large-scale deployment.
"""

import torch
import numpy as np
from typing import Optional
import warnings
import math


def fdt_initialize_(
    tensor: torch.Tensor,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    unroll_order: str = 'row',
    verbose: bool = False
) -> torch.Tensor:
    """
    In-place power-law initialization. Returns unnormalized noise;
    the caller decides the variance scaling.

    Args:
        tensor: Parameter tensor of arbitrary shape.
        alpha: Power-law exponent (recommended 0.6 for LoRA, or 1.2).
        temp_ratio: Optional high/low frequency energy ratio.
        method: 'fft' (exact) or 'ar' (fast approximate).
        unroll_order: 'row' (C-contiguous) or 'col' (Fortran-contiguous).
        verbose: Print detailed info.

    Returns:
        Initialized tensor (modified in-place, unnormalized).
    """
    with torch.no_grad():
        original_shape = tensor.shape
        original_device = tensor.device
        original_dtype = tensor.dtype

        numel = tensor.numel()

        if verbose:
            print(f"FDT init: shape={original_shape}, alpha={alpha:.2f}, method={method}, order={unroll_order}")

        if method == 'fft':
            n_freqs = numel // 2 + 1
            freqs = np.arange(1, n_freqs)

            if len(freqs) == 0:
                warnings.warn(f"Tensor too small (numel={numel}), falling back to standard normal")
                tensor.normal_(0, 1)
                return tensor

            power_spectrum = freqs.astype(np.float64) ** (-alpha)

            if temp_ratio is not None and temp_ratio != 1.0:
                cutoff_idx = max(1, len(freqs) // 16)

                energy_low = power_spectrum[:cutoff_idx].sum()
                energy_high = power_spectrum[cutoff_idx:].sum()

                if energy_low > 0 and energy_high > 0:
                    current_ratio = energy_high / energy_low
                    scale = np.sqrt(temp_ratio / current_ratio)
                    power_spectrum[cutoff_idx:] *= scale

                    if verbose:
                        print(f"  temp_ratio: {current_ratio:.3f} -> {temp_ratio:.3f}")

            amplitudes = np.sqrt(power_spectrum)
            phases = np.random.uniform(0, 2 * np.pi, len(freqs))

            fft_coeffs = amplitudes * np.exp(1j * phases)
            fft_coeffs_full = np.concatenate([[0.0 + 0.0j], fft_coeffs])

            time_series = np.fft.irfft(fft_coeffs_full, n=numel)
            time_series = time_series - time_series.mean()

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
                print(f"  AR param: beta={beta:.3f}")

        else:
            raise ValueError(f"Unknown method: {method}")

        tensor_flat = torch.from_numpy(time_series).to(dtype=original_dtype, device=original_device)

        if unroll_order == 'row':
            tensor.copy_(tensor_flat.reshape(original_shape))
        elif unroll_order == 'col':
            reversed_shape = tuple(reversed(original_shape))
            dims_reversed = tuple(range(len(original_shape) - 1, -1, -1))
            tensor.copy_(tensor_flat.reshape(reversed_shape).permute(dims_reversed))
        else:
            raise ValueError(f"Unknown unroll_order: {unroll_order}. Use 'row' or 'col'.")

        if verbose:
            actual_std = tensor.std().item()
            print(f"  done: raw std={actual_std:.6f} (unnormalized)")

    return tensor


def apply_fdt_to_lora(
    model: torch.nn.Module,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    unroll_order: str = 'row',
    verbose: bool = True
) -> torch.nn.Module:
    """Apply FDT initialization to LoRA layers with Xavier variance normalization."""

    initialized_count = 0
    skipped_count = 0
    total_trainable = 0

    if verbose:
        print("\n" + "="*70)
        print("Apply FDT initialization to LoRA layers (with Xavier normalization)")
        print("="*70)
        print(f"Config: alpha={alpha:.2f}", end='')
        if temp_ratio:
            print(f", temp_ratio={temp_ratio:.2f}", end='')
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
                    print(f"    shape: {list(param.shape)}")

                dimensions = param.dim()
                if dimensions >= 2:
                    fan_in = param.size(1)
                    fan_out = param.size(0)

                    if dimensions > 2:
                        receptive_field_size = param[0][0].numel()
                        fan_in *= receptive_field_size
                        fan_out *= receptive_field_size

                    target_std = math.sqrt(2.0 / (fan_in + fan_out))
                else:
                    target_std = 1.0 / math.sqrt(param.numel())

                if verbose:
                    print(f"    Xavier std: {target_std:.6f} (fan_in={fan_in}, fan_out={fan_out})")

                fdt_initialize_(
                    param.data,
                    alpha=alpha,
                    temp_ratio=temp_ratio,
                    method=method,
                    unroll_order=unroll_order,
                    verbose=False
                )

                with torch.no_grad():
                    current_std = param.data.std().item()

                    if current_std > 1e-8:
                        scale_factor = target_std / current_std
                        param.data.mul_(scale_factor)

                        if verbose:
                            actual_std = param.data.std().item()
                            print(f"    normalize: {current_std:.6f} * {scale_factor:.4f} = {actual_std:.6f}")
                    else:
                        if verbose:
                            print(f"    std too small ({current_std:.2e}), using random init")
                        param.data.normal_(0, target_std)

                if verbose:
                    print()

                initialized_count += 1

            except Exception as e:
                if verbose:
                    print(f"    failed: {e}\n")
                import traceback
                traceback.print_exc()
                skipped_count += 1
        else:
            skipped_count += 1

    if verbose:
        print("-"*70)
        print(f"Stats:")
        print(f"  total trainable: {total_trainable}")
        print(f"  initialized: {initialized_count}")
        print(f"  skipped: {skipped_count}")

        if initialized_count == 0:
            print("\nWarning: no parameters were initialized!")
            print("Possible reasons:")
            print("  1. Parameter names do not contain 'lora' etc.")
            print("  2. All parameters have requires_grad=False")
        elif initialized_count < total_trainable * 0.5:
            print(f"\nWarning: only initialized {initialized_count}/{total_trainable} parameters")
        else:
            print(f"\nSuccessfully initialized {initialized_count} LoRA layers")
        print("="*70)

    return model


def apply_fdt_to_all_params(
    model: torch.nn.Module,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    skip_frozen: bool = True,
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply FDT initialization to all parameters (for full fine-tuning or pretraining).

    Args:
        model: Model to initialize.
        alpha: Power-law exponent.
        temp_ratio: Optional temperature ratio.
        method: Initialization method.
        skip_frozen: Skip frozen parameters.
        verbose: Print logs.

    Returns:
        Initialized model (modified in-place).
    """
    initialized_count = 0
    skipped_count = 0
    total_params = 0

    if verbose:
        print("\n" + "="*70)
        print("FDT initialization - all parameters".center(70))
        print("="*70)

    for name, param in model.named_parameters():
        if skip_frozen and not param.requires_grad:
            skipped_count += 1
            continue

        if len(param.shape) >= 2:
            if verbose and initialized_count < 5:
                print(f"{name:60s} {list(param.shape)}")

            fdt_initialize_(
                param,
                alpha=alpha,
                temp_ratio=temp_ratio,
                method=method,
                verbose=False
            )

            initialized_count += 1
            total_params += param.numel()

    if verbose:
        print("-"*70)
        print(f"Done: {initialized_count} params, {total_params:,} elements")
        if skipped_count > 0:
            print(f"  skipped: {skipped_count} frozen/1D params")
        print("="*70 + "\n")

    return model


def init_lora_with_pink_noise(
    model: torch.nn.Module,
    alpha: float = 0.6,
    unroll_order: str = 'row',
    verbose: bool = True
) -> torch.nn.Module:
    """Convenience function: initialize LoRA with pink noise (recommended config)."""
    return apply_fdt_to_lora(model, alpha=alpha, method='fft', unroll_order=unroll_order, verbose=verbose)


def init_lora_with_custom_spectrum(
    model: torch.nn.Module,
    alpha: float,
    temp_ratio: float,
    verbose: bool = True
) -> torch.nn.Module:
    """Convenience function: initialize LoRA with custom spectrum (advanced)."""
    return apply_fdt_to_lora(
        model,
        alpha=alpha,
        temp_ratio=temp_ratio,
        method='fft',
        verbose=verbose
    )
