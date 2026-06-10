# Rebuttal Experiments

This directory contains supplementary experiments conducted during the rebuttal phase to address reviewer concerns.

## Experiments

### 1. Comparison with LoRA-One

**Script:** `train_openpangu_loraone.py`

Compares SpecLoRA against LoRA-One initialization on the OpenPangu model.

```bash
python rebuttal/train_openpangu_loraone.py
```

### 2. Comparison with DoRA

**Script:** `train_openpangu_dora.py`

Compares SpecLoRA against DoRA (Weight-Decomposed Low-Rank Adaptation).

```bash
python rebuttal/train_openpangu_dora.py
```

### 3. Multiple Initialization Methods

**Script:** `train_openpangu_multi_init.py`

Systematic comparison of multiple initialization strategies (Xavier, Gaussian, Orthogonal, etc.) against SpecLoRA.

```bash
python rebuttal/train_openpangu_multi_init.py
```

## Note

- All rebuttal experiments use the OpenPangu model as the base model for consistency.
- Results are saved to `outputs_*` directories (gitignored).
