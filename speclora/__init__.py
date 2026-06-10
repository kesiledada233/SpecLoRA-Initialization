"""
SpecLoRA Initialization

Frequency-domain-aware initialization for LoRA fine-tuning.
Generates power-law noise (pink noise, alpha ~ 0.6-1.2) to match
the spectral properties of converged models and accelerate early convergence.
"""

from .core import (
    fdt_initialize_,
    apply_fdt_to_lora,
    apply_fdt_to_all_params,
    init_lora_with_pink_noise,
    init_lora_with_custom_spectrum,
)
from .measure import (
    measure_alpha,
    plot_power_spectrum,
    analyze_lora_spectra,
    verify_fda_initialization,
    compare_initializations,
)

__version__ = "0.1.0"
__all__ = [
    "fdt_initialize_",
    "apply_fdt_to_lora",
    "apply_fdt_to_all_params",
    "init_lora_with_pink_noise",
    "init_lora_with_custom_spectrum",
    "measure_alpha",
    "plot_power_spectrum",
    "analyze_lora_spectra",
    "verify_fda_initialization",
    "compare_initializations",
]
