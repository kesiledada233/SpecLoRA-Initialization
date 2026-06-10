"""
SpecLoRA Initialization

Frequency-domain-aware initialization for LoRA fine-tuning.
Generates power-law noise (pink noise, alpha ~ 0.6-1.2) to match
the spectral properties of converged models and accelerate early convergence.
"""

from .core import (
    speclora_initialize_,
    apply_speclora_to_lora,
)
from .measure import (
    measure_alpha,
    plot_power_spectrum,
    analyze_lora_spectra,
    verify_speclora_initialization,
    compare_initializations,
)

__version__ = "0.1.0"
__all__ = [
    "speclora_initialize_",
    "apply_speclora_to_lora",
    "measure_alpha",
    "plot_power_spectrum",
    "analyze_lora_spectra",
    "verify_speclora_initialization",
    "compare_initializations",
]
