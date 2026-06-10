#!/bin/bash
# Mistral-7B 训练命令示例

# ==================== 下载模型 ====================
cd /root/nvme0n1/Noneq_Neural_Network/pretrained_models
bash download_mistral.sh

# ==================== GSM8K 任务 ====================

# Baseline
cd /root/nvme0n1/Noneq_Neural_Network/FDT_Init

python train_openpangu_fda_lora_final.py \
    --dataset gsm8k \
    --model_path /root/nvme0n1/Noneq_Neural_Network/pretrained_models/mistral-7b \
    --init_preset baseline \
    --out_dir outputs_mistral_gsm8k_baseline \
    --device npu:1

# Alpha 0.6
python train_openpangu_fda_lora_final.py \
    --dataset gsm8k \
    --model_path /root/nvme0n1/Noneq_Neural_Network/pretrained_models/mistral-7b \
    --init_preset medium \
    --out_dir outputs_mistral_gsm8k_alpha0.6 \
    --device npu:1

# Alpha 0.2, 0.4, 0.8 (消融实验)
for alpha in 0.2 0.4 0.8; do
    python train_openpangu_fda_lora_final.py \
        --dataset gsm8k \
        --model_path /root/nvme0n1/Noneq_Neural_Network/pretrained_models/mistral-7b \
        --init_preset soft \
        --out_dir outputs_mistral_gsm8k_alpha${alpha} \
        --device npu:1
done

# ==================== ShareGPT 任务 ====================

# Baseline
python train_openpangu_fda_lora_final.py \
    --dataset sharegpt \
    --model_path /root/nvme0n1/Noneq_Neural_Network/pretrained_models/mistral-7b \
    --init_preset baseline \
    --out_dir outputs_mistral_sharegpt_baseline \
    --device npu:1

# Alpha 0.6
python train_openpangu_fda_lora_final.py \
    --dataset sharegpt \
    --model_path /root/nvme0n1/Noneq_Neural_Network/pretrained_models/mistral-7b \
    --init_preset medium \
    --out_dir outputs_mistral_sharegpt_alpha0.6 \
    --device npu:1
