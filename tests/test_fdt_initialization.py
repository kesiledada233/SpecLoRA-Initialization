import sys
sys.path.insert(0, '/root/nvme0n1/Noneq_Neural_Network/FDT_Init')

import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# 清除缓存
for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]

from fdt_init import apply_fdt_to_lora
from measure_alpha import measure_alpha

print("="*70)
print("快速验证：FDT 初始化是否生效")
print("="*70)

# 1. 加载小模型
print("\n[1/4] 加载模型...")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. 应用 LoRA
print("\n[2/4] 应用 LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 3. 检查参数名
print("\n[3/4] 检查参数名...")
lora_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  • {name}: {param.shape}")
        lora_params.append((name, param))

print(f"\n共 {len(lora_params)} 个可训练参数")

# 4. 应用 FDT 初始化
print("\n[4/4] 应用 FDT 初始化...")
model = apply_fdt_to_lora(model, alpha=1.2, verbose=True)

# 5. 验证
print("\n" + "="*70)
print("验证结果")
print("="*70)

for name, param in lora_params:
    alpha = measure_alpha(param)
    error = abs(alpha - 1.2)
    status = "✓" if error < 0.15 else "✗"
    print(f"{status} {name}: α={alpha:.3f} (误差={error:.3f})")

print("="*70)