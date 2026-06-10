import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    Measure the power-law exponent alpha of a tensor.

    Args:
        tensor: Parameter tensor of arbitrary shape.
        method: 'fft' (frequency domain) or 'dfa' (not implemented).
        fit_range: Frequency range for fitting (integer indices), None for auto.
        return_full_spectrum: If True, returns (alpha, freqs, power).

    Returns:
        alpha: Power-law exponent (or tuple if return_full_spectrum=True).
    """
    with torch.no_grad():
        data = tensor.detach().flatten().cpu().numpy()

        if len(data) < 10:
            return np.nan

        data = data - data.mean()

        if method == 'fft':
            fft_result = np.fft.rfft(data)
            power_spectrum = np.abs(fft_result) ** 2

            n_freqs = len(power_spectrum)
            freqs = np.arange(1, n_freqs)
            power = power_spectrum[1:]

            if len(freqs) < 3:
                return np.nan

            if fit_range is not None:
                low_idx = max(0, fit_range[0] - 1)
                high_idx = min(len(freqs), fit_range[1] - 1)
            else:
                low_idx = max(0, int(len(freqs) * 0.05))
                high_idx = int(len(freqs) * 0.90)

            if high_idx - low_idx < 3:
                return np.nan

            log_freq = np.log(freqs[low_idx:high_idx].astype(np.float64))
            log_power = np.log(power[low_idx:high_idx] + 1e-12)

            coeffs = np.polyfit(log_freq, log_power, 1)
            alpha = -coeffs[0]

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
    """Plot power spectrum in log-log scale."""
    alpha, freqs, power = measure_alpha(tensor, return_full_spectrum=True)

    plt.figure(figsize=figsize)
    plt.loglog(freqs, power, 'o-', markersize=3, alpha=0.6, label='Power Spectrum')

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
    Analyze power-law exponents of all LoRA layers in a model.

    Args:
        model: Model with LoRA layers.
        save_dir: Directory to save plots (None to skip).
        plot_top_n: Plot spectra for top N layers.
        verbose: Print info.

    Returns:
        Dict mapping layer name to alpha value.
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
                status = " " if 0.8 <= alpha <= 1.5 else "!"
                print(f"{status} {name:48s} {alpha:10.3f} {shape_str:>15s}")

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
            print(f"Statistics: Mean alpha = {np.mean(valid_alphas):.3f} +/- {np.std(valid_alphas):.3f}")
            print(f"            Range [{np.min(valid_alphas):.3f}, {np.max(valid_alphas):.3f}]")
        print("="*70 + "\n")

    return results


def verify_speclora_initialization(
    model: torch.nn.Module,
    target_alpha: float,
    tolerance: float = 0.1,
    verbose: bool = True
) -> bool:
    """
    Verify whether SpeLoRA initialization succeeded.

    Args:
        model: Initialized model.
        target_alpha: Target alpha value.
        tolerance: Allowed error.
        verbose: Print info.

    Returns:
        True if all LoRA layers are within tolerance.
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
        print("SpeLoRA Initialization Verification".center(70))
        print("="*70)
        print(f"Target alpha: {target_alpha:.3f}")
        print(f"Measured alpha: {np.mean(valid_alphas):.3f} +/- {np.std(valid_alphas):.3f}")
        print(f"Max error: {max_error:.3f} (tolerance: {tolerance:.3f})")
        print(f"Mean error: {mean_error:.3f}")
        print("-"*70)

        if success:
            print("Verification PASSED. All layers within tolerance.")
        else:
            print(f"Verification FAILED. Max error {max_error:.3f} exceeds tolerance {tolerance:.3f}")
            print("\nLayers exceeding tolerance:")
            for name, alpha in alphas.items():
                if abs(alpha - target_alpha) >= tolerance:
                    print(f"  - {name}: alpha={alpha:.3f} (error {abs(alpha - target_alpha):.3f})")

        print("="*70 + "\n")

    return success


def compare_initializations(
    model_baseline: torch.nn.Module,
    model_speclora: torch.nn.Module,
    layer_name: str,
    save_path: Optional[str] = None
):
    """
    Compare power spectra of two initializations for a given layer.

    Args:
        model_baseline: Baseline-initialized model.
        model_speclora: SpeLoRA-initialized model.
        layer_name: Layer name to compare.
        save_path: Save path.
    """
    param_baseline = dict(model_baseline.named_parameters())[layer_name]
    param_speclora = dict(model_speclora.named_parameters())[layer_name]

    alpha_baseline, freqs_b, power_b = measure_alpha(param_baseline, return_full_spectrum=True)
    alpha_speclora, freqs_f, power_f = measure_alpha(param_speclora, return_full_spectrum=True)

    plt.figure(figsize=(12, 6))

    plt.loglog(freqs_b, power_b, 'o-', markersize=3, alpha=0.6,
               label=f'Xavier (alpha={alpha_baseline:.3f})', color='blue')
    plt.loglog(freqs_f, power_f, 's-', markersize=3, alpha=0.6,
               label=f'SpeLoRA (alpha={alpha_speclora:.3f})', color='red')

    f_ref = freqs_f[len(freqs_f)//4]
    p_ref_speclora = power_f[len(power_f)//4]

    theory_line = p_ref_speclora * (freqs_f / f_ref) ** (-alpha_speclora)
    plt.loglog(freqs_f, theory_line, 'r--', linewidth=2, alpha=0.5,
               label=f'Theory: f^(-{alpha_speclora:.2f})')

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
