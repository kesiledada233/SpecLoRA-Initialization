# SpecLoRA Initialization

Frequency-domain-aware initialization for LoRA fine-tuning.

SpecLoRA replaces the conventional Xavier/He initialization (white noise, α ≈ 0) with power-law noise (pink noise, α ≈ 0.6–1.2) that matches the spectral properties of converged models, accelerating early-stage convergence.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from speclora import apply_speclora_to_lora

# Apply SpecLoRA initialization before training
model = apply_speclora_to_lora(model, alpha=0.6, method='fft')
```

## Core API

| Function | Description |
|----------|-------------|
| `speclora_initialize_(tensor, alpha=1.2, method='fft')` | In-place power-law initialization of a tensor |
| `apply_speclora_to_lora(model, alpha=0.6)` | Apply to all LoRA layers with Xavier variance normalization |
| `measure_alpha(tensor)` | Measure the power-law exponent of a tensor |
| `analyze_lora_spectra(model)` | Analyze all LoRA layers in a model |
| `verify_speclora_initialization(model, target_alpha)` | Verify initialization quality |
| `compare_initializations(baseline, speclora, layer_name)` | Compare spectra of two initializations |

## Repository Structure

```
speclora-repo/
├── speclora/              # Core library
│   ├── core.py            # Initialization implementations (FFT + AR)
│   └── measure.py         # Spectral analysis tools
├── experiments/
│   └── train_openpangu_speclora.py   # Unified training script (4 datasets)
├── tests/
│   └── test_speclora_initialization.py
├── results/               # Key experimental results (alpha=0.6)
│   ├── openpangu_cmmlu_alpha0.6.json
│   ├── openpangu_gsm8k_alpha0.6.json
│   ├── openpangu_mbpp_alpha0.6.json
│   ├── openpangu_sharegpt_alpha0.6.json
│   ├── qwen2.5_cmmlu_alpha0.6.json
│   ├── qwen2.5_gsm8k_alpha0.6.json
│   ├── qwen2.5_mbpp_alpha0.6.json
│   └── qwen2.5_sharegpt_alpha0.6.json
├── requirements.txt
└── README.md
```

## Datasets

The unified training script supports the following datasets:
- **GSM8K**: Mathematical reasoning
- **CMMLU**: Chinese general knowledge
- **ShareGPT**: Dialogue interaction
- **MBPP**: Code generation

## Reproduce Main Experiments

```bash
python experiments/train_openpangu_speclora.py \
    --dataset gsm8k \
    --init_method speclora \
    --alpha 0.6 \
    --lora_r 16 \
    --out_dir outputs_gsm8k_speclora_r16
```

## License

MIT
