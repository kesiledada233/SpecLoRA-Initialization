# SpecLoRA Initialization

Frequency-domain-aware initialization for LoRA fine-tuning.

SpecLoRA replaces the conventional Xavier/He initialization (white noise, α ≈ 0) with power-law noise (pink noise, α ≈ 0.6–1.2) that matches the spectral properties of converged models, accelerating early-stage convergence.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from speclora import apply_fdt_to_lora

# Apply SpecLoRA initialization before training
model = apply_fdt_to_lora(model, alpha=0.6, method='fft')
```

## Core API

| Function | Description |
|----------|-------------|
| `fdt_initialize_(tensor, alpha=1.2, method='fft')` | In-place power-law initialization of a tensor |
| `apply_fdt_to_lora(model, alpha=0.6)` | Apply to all LoRA layers with Xavier variance normalization |
| `apply_fdt_to_all_params(model, alpha=1.2)` | Apply to all trainable parameters (for full fine-tuning) |
| `measure_alpha(tensor)` | Measure the power-law exponent of a tensor |
| `analyze_lora_spectra(model)` | Analyze all LoRA layers in a model |

## Repository Structure

```
speclora-repo/
├── speclora/              # Core library
│   ├── core.py            # Initialization implementations (FFT + AR)
│   └── measure.py         # Spectral analysis tools
├── experiments/           # Main experiments
│   ├── train_benchmark_fdt_init.py
│   ├── train_deepseek_fdt_lora_final.py
│   ├── train_openpangu_fdt_lora_final.py
│   ├── train_qwen2.5_fdt_lora_final.py
│   ├── train_sharegpt_fdt_init.py
│   ├── train_wikitext_fdt_init.py
│   ├── evaluate_downstream.py
│   └── run_mistral.sh
├── rebuttal/              # Rebuttal experiments
│   ├── train_openpangu_loraone.py
│   ├── train_openpangu_dora.py
│   └── train_openpangu_multi_init.py
├── analysis/              # Plotting and result collection
│   ├── plot_all_figures.py
│   ├── plot_ablation.py
│   ├── plot_comparison.py
│   ├── analyze_time_to_threshold.py
│   └── collect_all_results.py
├── tests/                 # Unit tests
│   ├── test_multi_init.py
│   └── test_fdt_initialization.py
├── results/               # Key experimental results
│   ├── all_experiments_results.csv
│   └── time_to_threshold_results.json
├── data/                  # Data directory (populate via README links)
├── requirements.txt
└── README.md
```

## Reproduce Main Experiments

### Benchmark Models

```bash
# Mistral
bash experiments/run_mistral.sh

# DeepSeek
python experiments/train_deepseek_fdt_lora_final.py

# OpenPangu
python experiments/train_openpangu_fdt_lora_final.py

# Qwen2.5
python experiments/train_qwen2.5_fdt_lora_final.py
```

### Downstream Evaluation

```bash
python experiments/evaluate_downstream.py --model_path <path> --task <task_name>
```

### Generate Figures

```bash
python analysis/plot_all_figures.py
python analysis/plot_ablation.py
python analysis/plot_comparison.py
```

## Rebuttal Experiments

See [`rebuttal/README.md`](rebuttal/README.md) for details on the supplementary experiments comparing against LoRA-One, DoRA, and PiSSA.

```bash
python rebuttal/train_openpangu_loraone.py
python rebuttal/train_openpangu_dora.py
python rebuttal/train_openpangu_multi_init.py
```

## Citation

```bibtex
@article{speclora2025,
  title={SpecLoRA: Frequency-Domain-Aware Initialization for LoRA Fine-Tuning},
  author={...},
  year={2025}
}
```

## License

MIT
