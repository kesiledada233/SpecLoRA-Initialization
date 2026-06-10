import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from speclora import apply_fdt_to_lora, measure_alpha


print("=" * 70)
print("FDT Initialization Test")
print("=" * 70)

# 1. Load base model
print("\n[1/4] Loading base model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Apply LoRA
print("\n[2/4] Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 3. Collect LoRA parameters
print("\n[3/4] Collecting LoRA parameters...")
lora_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  - {name}: {param.shape}")
        lora_params.append((name, param))

print(f"\nTotal LoRA params: {len(lora_params)}")

# 4. Apply FDT initialization
print("\n[4/4] Applying FDT initialization...")
model = apply_fdt_to_lora(model, alpha=1.2, verbose=True)

# 5. Verify alpha values
print("\n" + "=" * 70)
print("Verification")
print("=" * 70)

for name, param in lora_params:
    alpha = measure_alpha(param)
    error = abs(alpha - 1.2)
    status = "PASS" if error < 0.15 else "FAIL"
    print(f"{status} {name}: alpha={alpha:.3f} (error={error:.3f})")

print("=" * 70)
