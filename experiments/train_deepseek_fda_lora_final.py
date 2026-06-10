"""
DeepSeek FDT 初始化训练脚本 - 完整修复版

修复内容:
  1. ✅ 索引越界问题（step < 500 时的安全检查）
  2. ✅ NPU环境下的模型保存（使用bin格式）
  3. ✅ NaN检测和早停机制
  4. ✅ 降低学习率避免训练不稳定
  5. ✅ 数据集格式兼容性增强

支持数据集:
  - gsm8k: 数学推理
  - cmmlu: 中文知识
  - sharegpt: 对话交互
  - mbpp: 代码生成

运行示例:
  # Baseline
  python train_deepseek_fdt_final.py \
      --dataset gsm8k \
      --model_path /path/to/deepseek-7b \
      --lora_r 16 \
      --out_dir outputs/deepseek_baseline_r16 \
      --device npu:1

  # FDT (alpha=0.6)
  python train_deepseek_fdt_final.py \
      --dataset gsm8k \
      --model_path /path/to/deepseek-7b \
      --use_fdt_init \
      --fdt_alpha 0.6 \
      --lora_r 16 \
      --out_dir outputs/deepseek_fdt0.6_r16 \
      --device npu:1
"""

import os
os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'
os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'

import sys
import time
import argparse
import random
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# 检查 PEFT
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError as e:
    print("错误: 未找到peft库，请先安装: pip install peft")
    print(f"详细错误: {e}")
    PEFT_AVAILABLE = False

# 检查 datasets
try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    print("警告: 未找到datasets库")
    DATASETS_AVAILABLE = False

# ==================== 导入 FDT 模块 ====================
print("\n" + "="*70)
print("🔧 加载 FDT 初始化模块")
print("="*70)

for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]

FDT_INIT_PATH = '/root/nvme0n1/Noneq_Neural_Network/FDT_Init'

for path in sys.path[:]:
    if 'FDT_Init' in path and path != FDT_INIT_PATH:
        sys.path.remove(path)

if FDT_INIT_PATH in sys.path:
    sys.path.remove(FDT_INIT_PATH)
sys.path.insert(0, FDT_INIT_PATH)

try:
    from fdt_init import apply_fdt_to_lora
    from measure_alpha import analyze_lora_spectra, verify_fdt_initialization
    
    FDT_INIT_AVAILABLE = True
    print("[导入] ✓ FDT 模块加载成功")
    
except ImportError as e:
    print(f"[导入] ✗ 失败: {e}")
    raise

print("="*70 + "\n")

# ==================== 数据集路径配置 ====================
DATASET_PATHS = {
    'gsm8k': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/gsm8k',
    'cmmlu': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/cmmlu/processed',
    'sharegpt': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/sharegpt',
    'mbpp': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/mbpp/processed',
}

# ==================== 数据集类 ====================
class BenchmarkDataset(Dataset):
    """评测数据集类"""
    
    def __init__(self, tokenizer, examples, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"[Dataset] Tokenization {len(examples)} 个样本...")
        
        for idx, text in enumerate(examples):
            if len(text.strip()) < 10:
                continue
            
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
            })
            
            if (idx + 1) % 500 == 0:
                print(f"  进度: {idx+1}/{len(examples)}")
        
        print(f"[Dataset] ✓ 完成: {len(self.examples)} 个有效样本\n")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['input_ids'].clone(),
        }


# ==================== 工具函数 ====================
def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def convert_to_json_serializable(obj):
    """转换 numpy 类型"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)


def compute_gradient_norm(model):
    """计算模型梯度的L2范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_auc_intervals(losses):
    """计算不同阶段的AUC（安全版本）"""
    intervals = {
        'auc_0_100': (0, 100),
        'auc_100_500': (100, 500),
        'auc_500_1000': (500, 1000),
        'auc_1000_2500': (1000, 2500),
        'auc_0_500': (0, 500),
        'auc_0_2500': (0, 2500),
    }
    
    results = {}
    for name, (start, end) in intervals.items():
        if len(losses) >= end:
            results[name] = float(sum(losses[start:end]))
        else:
            results[name] = None
    
    return results


# ==================== 参数配置 ====================
def get_args():
    ap = argparse.ArgumentParser(description="DeepSeek FDT 训练脚本")
    
    # 数据集配置
    ap.add_argument("--dataset", type=str, required=True,
                   choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'],
                   help="数据集名称")
    ap.add_argument("--num_samples", type=int, default=0,
                   help="训练样本数（0=全部）")
    
    # 模型配置
    ap.add_argument("--model_path", type=str,
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="预训练模型路径")
    
    # LoRA 配置
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "v_proj"])
    
    # 训练配置（⚡ DeepSeek 优化）
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=2500)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    
    # 优化器配置（⚡ 降低学习率，增强梯度裁剪）
    ap.add_argument("--lr", type=float, default=3e-5,
                   help="学习率（DeepSeek推荐3e-5）")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=0.5,
                   help="梯度裁剪（DeepSeek推荐0.5）")
    
    # FDT 初始化
    ap.add_argument("--use_fdt_init", action="store_true")
    ap.add_argument("--fdt_alpha", type=float, default=1.1)
    ap.add_argument("--fdt_method", type=str, default='fft',
                   choices=['fft', 'ar'])
    ap.add_argument("--verify_fdt", action="store_true")
    ap.add_argument("--plot_spectra", action="store_true")
    ap.add_argument("--init_preset", type=str, default=None,
                   choices=['baseline', 'soft', 'medium', 'strong'])
    
    # 记录选项
    ap.add_argument("--record_gradnorm", action="store_true",
                   help="记录每步的梯度范数")
    ap.add_argument("--full_test_eval", action="store_true",
                   help="完整测试集评估（不限制batch数）")
    ap.add_argument("--measure_final_spectrum", action="store_true",
                   help="训练结束后测量功率谱")
    
    # 输出配置
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--device", type=str, default="npu:1")
    
    return ap.parse_args()


# ==================== 主函数 ====================
def main():
    if not PEFT_AVAILABLE:
        print("错误: PEFT库不可用")
        return
    
    args = get_args()
    
    # 获取数据集路径
    dataset_path = DATASET_PATHS.get(args.dataset)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(args.out_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ 配置保存: {config_file}\n")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设备设置
    device = torch.device(args.device)
    device_type = args.device.split(':')[0]
    
    if device_type == 'npu':
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            torch_npu.npu.manual_seed_all(args.seed)
            print(f"[设备] ✓ NPU 初始化成功: {device}\n")
        except Exception as e:
            print(f"[设备] ❌ NPU 初始化失败: {e}")
            return
    else:
        print(f"[设备] 使用: {device}\n")
    
    # ==================== 步骤 1: 加载模型 ====================
    print("="*70)
    print("📦 步骤 1: 加载模型")
    print("="*70)
    
    print(f"[模型] 路径: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[模型] ✓ Tokenizer: vocab_size={tokenizer.vocab_size}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    total_params, _ = count_parameters(model)
    print(f"[模型] ✓ 加载成功: {total_params/1e9:.2f}B 参数\n")
    
    # ==================== 步骤 2: 应用 LoRA ====================
    print("="*70)
    print("🔧 步骤 2: 应用 LoRA")
    print("="*70)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"[LoRA] r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[LoRA] target_modules={args.lora_target_modules}")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"[LoRA] ✓ 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)\n")
    
    # ==================== 步骤 3: FDT 初始化 ====================
    print("="*70)
    print("🎯 步骤 3: FDT 初始化")
    print("="*70)
    
    init_start_time = time.time()
    
    # 解析预设
    if args.init_preset:
        preset_configs = {
            'baseline': {'use_fdt': False, 'alpha': None, 'name': 'PEFT Default (Kaiming+Zero)'},
            'soft': {'use_fdt': True, 'alpha': 0.8, 'name': 'FDT-Soft (α=0.8)'},
            'medium': {'use_fdt': True, 'alpha': 1.1, 'name': 'FDT-Medium (α=1.1)'},
            'strong': {'use_fdt': True, 'alpha': 1.5, 'name': 'FDT-Strong (α=1.5)'},
        }
        
        config = preset_configs[args.init_preset]
        print(f"[预设] {config['name']}\n")
        
        if config['use_fdt']:
            args.use_fdt_init = True
            args.fdt_alpha = config['alpha']
    
    # 初始化信息
    init_info = {
        'use_fdt': args.use_fdt_init,
        'preset': args.init_preset,
        'alpha': None,
        'method': 'peft_default',
        'lora_a_init': 'kaiming_uniform',
        'lora_b_init': 'zero',
        'init_time_seconds': None,
        'measured_alphas_init': {},
        'measured_alphas_final': {},
        'verification_passed': None,
    }
    
    if args.use_fdt_init:
        print(f"[FDT] 应用初始化: α={args.fdt_alpha:.2f}, 方法={args.fdt_method}")
        
        apply_fdt_to_lora(
            model,
            alpha=args.fdt_alpha,
            method=args.fdt_method,
            verbose=args.verbose
        )
        
        init_info['alpha'] = args.fdt_alpha
        init_info['method'] = args.fdt_method
        
        print("[FDT] ✓ 初始化完成")
        
        if args.verify_fdt:
            print("\n[FDT] 验证初始化质量...")
            verify_success = verify_fdt_initialization(
                model,
                target_alpha=args.fdt_alpha,
                tolerance=0.15,
                verbose=True
            )
            init_info['verification_passed'] = verify_success
        
        if args.plot_spectra:
            print("\n[FDT] 分析初始功率谱...")
            spectra_dir = os.path.join(args.out_dir, 'init_spectra')
            os.makedirs(spectra_dir, exist_ok=True)
            
            alphas = analyze_lora_spectra(
                model,
                save_dir=spectra_dir,
                plot_top_n=3,
                verbose=args.verbose
            )
            
            init_info['measured_alphas_init'] = {k: float(v) for k, v in alphas.items()}
            print(f"[FDT] ✓ 功率谱保存到: {spectra_dir}")
    
    else:
        print("[FDT] 使用 PEFT 默认初始化 (Kaiming Uniform + Zero) (Baseline)")
    
    init_time = time.time() - init_start_time
    init_info['init_time_seconds'] = float(init_time)
    print(f"[FDT] 初始化耗时: {init_time*1000:.2f} ms\n")
    
    # ==================== 步骤 4: 加载数据集 ====================
    print("="*70)
    print(f"📊 步骤 4: 加载数据集 ({args.dataset.upper()})")
    print("="*70)
    
    print(f"[数据] 路径: {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        
        print(f"[数据] ✓ 加载成功")
        print(f"  • 训练集: {len(dataset['train'])} 样本")
        print(f"  • 测试集: {len(dataset['test'])} 样本")
        
        # 限制训练样本
        train_raw = dataset['train']
        if args.num_samples > 0 and args.num_samples < len(train_raw):
            train_raw = train_raw.select(range(args.num_samples))
            print(f"[数据] 限制训练集: {len(train_raw)} 样本")
        
        train_dataset_raw = train_raw
        test_dataset_raw = dataset['test']
        
        if not args.full_test_eval:
            max_test_samples = 1000
            if len(test_dataset_raw) > max_test_samples:
                test_dataset_raw = test_dataset_raw.select(range(max_test_samples))
                print(f"[数据] 限制测试集: {max_test_samples} 样本（快速评估）")
        else:
            print(f"[数据] 使用完整测试集: {len(test_dataset_raw)} 样本")
        
        print(f"\n[数据] 数据划分:")
        print(f"  • 训练: {len(train_dataset_raw)} 样本")
        print(f"  • 测试: {len(test_dataset_raw)} 样本")
        
        # ⚡⚡⚡ 增强的数据格式化函数 ⚡⚡⚡
        def format_example(example):
            """格式化样本文本（增强版）"""
            
            if args.dataset == 'gsm8k':
                question = example.get('question', '')
                answer = example.get('answer', '')
                return f"问题：{question}\n解答：{answer}"
            
            elif args.dataset == 'cmmlu':
                question = example.get('Question', '')
                a = example.get('A', '')
                b = example.get('B', '')
                c = example.get('C', '')
                d = example.get('D', '')
                answer = example.get('Answer', '')
                choices = f"A. {a}  B. {b}  C. {c}  D. {d}"
                return f"问题：{question}\n选项：{choices}\n答案：{answer}"
            
            elif args.dataset == 'sharegpt':
                # 兼容多种字段名
                conversations = example.get('conversations') or example.get('messages', [])
                
                if not conversations:
                    return "Empty conversation"
                
                text = ""
                for turn in conversations:
                    # 兼容不同的字段名
                    role = turn.get('from') or turn.get('role', 'unknown')
                    content = turn.get('value') or turn.get('content', '')
                    
                    # 角色标准化
                    if role in ['human', 'user']:
                        role = 'User'
                    elif role in ['gpt', 'assistant']:
                        role = 'Assistant'
                    elif role == 'system':
                        role = 'System'
                    
                    if content.strip():
                        text += f"{role}: {content}\n"
                
                return text.strip() if text.strip() else "Empty conversation"
            
            elif args.dataset == 'mbpp':
                # 兼容多种字段名
                text = example.get('text') or example.get('prompt') or example.get('description', '')
                code = example.get('code') or example.get('solution', '')
                
                # 可选：添加测试用例
                test_list = example.get('test_list', [])
                
                if not text and not code:
                    return "Empty code problem"
                
                formatted = f"# Problem\n{text}\n\n"
                
                if test_list and len(test_list) > 0:
                    tests = "\n".join([f"# Test: {t}" for t in test_list[:2]])
                    formatted += f"{tests}\n\n"
                
                formatted += f"# Solution\n{code}"
                
                return formatted
            
            else:
                raise ValueError(f"未知数据集: {args.dataset}")
        
        print(f"\n[数据] 格式化文本...")
        train_texts = [format_example(item) for item in train_dataset_raw]
        test_texts = [format_example(item) for item in test_dataset_raw]
        
        print(f"\n[数据] 样本预览 (前 150 字符):")
        print(f"  {train_texts[0][:150]}...\n")
        
        # 创建 Dataset
        train_dataset = BenchmarkDataset(tokenizer, train_texts, args.max_length)
        test_dataset = BenchmarkDataset(tokenizer, test_texts, args.max_length)
        
    except Exception as e:
        print(f"[数据] ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[数据] DataLoader:")
    print(f"  • 训练: {len(train_loader)} 批次")
    print(f"  • 测试: {len(test_loader)} 批次\n")
    
    # ==================== 步骤 5: 配置优化器 ====================
    print("="*70)
    print("⚙️ 步骤 5: 配置优化器")
    print("="*70)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )
    
    print(f"[优化器] AdamW (lr={args.lr:.2e}, wd={args.weight_decay})")
    print(f"[调度器] Warmup {args.warmup_steps} 步")
    print(f"[梯度] 裁剪阈值={args.max_grad_norm}")
    print(f"[DeepSeek] 使用优化配置: lr={args.lr:.2e}, grad_clip={args.max_grad_norm}\n")
    
    # 初始化效率监控
    if device_type == 'npu':
        import torch_npu
        start_memory = torch_npu.npu.memory_allocated(device)
        peak_memory = start_memory
    else:
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = start_memory
    
    training_log = []
    
    # ==================== 步骤 6: 训练 ====================
    print("="*70)
    print("🚀 步骤 6: 开始训练")
    print("="*70)
    
    model.train()
    
    training_losses = []
    test_losses_history = []
    best_loss = float('inf')
    
    # ⚡⚡⚡ NaN 检测和早停 ⚡⚡⚡
    nan_count = 0
    max_nan_allowed = 10
    
    data_iter = iter(train_loader)
    start_time = time.time()
    
    print(f"\n[训练] {args.max_iters} 步")
    print(f"[DeepSeek] NaN容忍度: {max_nan_allowed} 次")
    print("-"*70)
    
    for step in range(1, args.max_iters + 1):
        step_start_time = time.time()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # NPU FP16 转换
        if device_type == 'npu':
            batch = {
                k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                for k, v in batch.items()
            }
        
        try:
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            
            # ⚡⚡⚡ 增强的NaN检测 ⚡⚡⚡
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"\n⚠️ 异常损失 (Step {step}): {loss.item()} (累计 {nan_count}/{max_nan_allowed} 次)")
                
                if nan_count > max_nan_allowed:
                    print(f"\n❌ 异常损失过多 ({nan_count} 次)，提前终止训练!")
                    break
                
                optimizer.zero_grad()
                continue
            
            # ⚡⚡⚡ 重置NaN计数 ⚡⚡⚡
            if nan_count > 0 and step % 10 == 0:
                nan_count = max(0, nan_count - 1)
            
            loss.backward()
            
        except Exception as e:
            print(f"\n⚠️ 训练错误 (Step {step}): {e}")
            optimizer.zero_grad()
            continue
        
        # 记录梯度范数
        grad_norm = None
        if args.record_gradnorm:
            grad_norm = compute_gradient_norm(model)
        
        # 梯度累积
        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        
        if device_type == 'npu':
            current_memory = torch_npu.npu.memory_allocated(device)
        else:
            current_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = max(peak_memory, current_memory)
        
        step_time = time.time() - step_start_time
        
        log_entry = {
            'step': step,
            'train_loss': current_loss,
            'learning_rate': current_lr,
            'grad_norm': grad_norm,
            'step_time_ms': step_time * 1000,
            'memory_gb': (current_memory - start_memory) / 1e9,
            'test_loss': None,
        }
        
        # 评估
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train = np.mean(training_losses[-args.eval_interval:])
            
            # 测试集评估
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                max_eval_batches = None if args.full_test_eval else 10
                
                for batch_idx, test_batch in enumerate(test_loader):
                    if max_eval_batches and batch_idx >= max_eval_batches:
                        break
                    
                    test_batch = {k: v.to(device) for k, v in test_batch.items()}
                    
                    if device_type == 'npu':
                        test_batch = {
                            k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                            for k, v in test_batch.items()
                        }
                    
                    try:
                        test_outputs = model(**test_batch)
                        test_losses.append(test_outputs.loss.item())
                    except:
                        break
            
            model.train()
            
            test_loss_avg = np.mean(test_losses) if test_losses else None
            
            if test_loss_avg:
                test_losses_history.append({
                    'step': step,
                    'test_loss': test_loss_avg,
                })
                log_entry['test_loss'] = test_loss_avg
            
            # ⚡⚡⚡ 安全的早期收敛指标计算 ⚡⚡⚡
            metric_str = ""
            current_len = len(training_losses)
            
            # Loss@100
            if current_len >= 100:
                loss_100 = training_losses[99]
                metric_str += f", L@100={loss_100:.4f}"
            
            # Loss@500 和 AUC(0-500)
            if current_len >= 500:
                loss_500 = training_losses[499]
                auc_500 = sum(training_losses[:500])
                metric_str += f", L@500={loss_500:.4f}, AUC(0-500)={auc_500:.2f}"
            
            # 日志输出
            memory_gb = (peak_memory - start_memory) / 1e9
            
            log = f"[{step:5d}/{args.max_iters}] "
            log += f"训练={current_loss:.4f}, 均值={avg_train:.4f}"
            
            if test_loss_avg:
                log += f", 测试={test_loss_avg:.4f}"
            
            log += f", lr={current_lr:.2e}, {elapsed:.1f}s"
            log += f", Mem={memory_gb:.2f}GB"
            
            if args.record_gradnorm and grad_norm:
                log += f", GradNorm={grad_norm:.4f}"
            
            log += metric_str
            
            print(log)
            
            # ⚡⚡⚡ NPU环境安全保存模型 ⚡⚡⚡
            metric = test_loss_avg if test_loss_avg else avg_train
            
            if metric < best_loss:
                best_loss = metric
                best_model_path = os.path.join(args.out_dir, "best_model")
                
                try:
                    if device_type == 'npu':
                        # NPU: 保存为bin格式
                        os.makedirs(best_model_path, exist_ok=True)
                        
                        adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if 'lora' in k}
                        torch.save(adapter_state, os.path.join(best_model_path, "adapter_model.bin"))
                        
                        model.peft_config['default'].save_pretrained(best_model_path)
                        
                        print(f"  → 保存最佳模型 (损失={best_loss:.4f})")
                    else:
                        model.save_pretrained(best_model_path)
                        print(f"  → 保存最佳模型 (损失={best_loss:.4f})")
                except Exception as e:
                    print(f"  ⚠️ 保存最佳模型失败: {e}")
        
        training_log.append(log_entry)
    
    total_time = time.time() - start_time
    
    print("-"*70)
    print(f"[训练] ✓ 完成! 耗时: {total_time/60:.2f} 分钟")
    print(f"[训练] 最佳损失: {best_loss:.4f}")
    print(f"[训练] 峰值内存: {(peak_memory - start_memory) / 1e9:.2f} GB")
    if nan_count > 0:
        print(f"[训练] ⚠️ NaN次数: {nan_count} 次")
    print()
    
    # ==================== 步骤 7: 测试集完整评估 ====================
    print("="*70)
    print("🧪 步骤 7: 测试集完整评估")
    print("="*70)
    
    print(f"[测试] 使用训练完成的模型进行完整评估")
    
    model.eval()
    test_losses = []
    
    print(f"[测试] 在 {len(test_loader)} 个批次上评估...")
    
    with torch.no_grad():
        for idx, test_batch in enumerate(test_loader):
            test_batch = {k: v.to(device) for k, v in test_batch.items()}
            
            if device_type == 'npu':
                test_batch = {
                    k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                    for k, v in test_batch.items()
                }
            
            try:
                test_outputs = model(**test_batch)
                test_losses.append(test_outputs.loss.item())
            except Exception as e:
                print(f"\n  ⚠️ 批次 {idx} 评估失败: {str(e)[:100]}")
                continue
            
            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx+1}/{len(test_loader)}")
    
    test_loss = np.mean(test_losses) if test_losses else float('inf')
    test_std = np.std(test_losses) if test_losses else 0.0
    
    print(f"\n[测试] 结果:")
    print(f"  • 平均损失: {test_loss:.4f}")
    print(f"  • 标准差: {test_std:.4f}")
    print(f"  • 有效批次: {len(test_losses)}/{len(test_loader)}\n")
    
    # ==================== 步骤 7.5: 训练后功率谱测量 ====================
    if args.use_fdt_init and args.measure_final_spectrum:
        print("="*70)
        print("📊 步骤 7.5: 训练后功率谱测量")
        print("="*70)
        
        spectra_dir_final = os.path.join(args.out_dir, 'final_spectra')
        os.makedirs(spectra_dir_final, exist_ok=True)
        
        print("[FDT] 分析训练后的功率谱...")
        alphas_final = analyze_lora_spectra(
            model,
            save_dir=spectra_dir_final,
            plot_top_n=3,
            verbose=args.verbose
        )
        
        init_info['measured_alphas_final'] = {k: float(v) for k, v in alphas_final.items()}
        
        print(f"[FDT] ✓ 训练后功率谱保存到: {spectra_dir_final}")
        
        if init_info['measured_alphas_init']:
            print("\n[FDT] 功率谱演化:")
            for key in init_info['measured_alphas_init']:
                if key in init_info['measured_alphas_final']:
                    alpha_init = init_info['measured_alphas_init'][key]
                    alpha_final = init_info['measured_alphas_final'][key]
                    delta = alpha_final - alpha_init
                    print(f"  {key}: {alpha_init:.3f} → {alpha_final:.3f} (Δ={delta:+.3f})")
        print()
    
    # ==================== 步骤 8: 保存结果 ====================
    print("="*70)
    print("💾 步骤 8: 保存结果")
    print("="*70)
    
    # 1. 训练损失数组
    losses_file = os.path.join(args.out_dir, "training_losses.npy")
    np.save(losses_file, np.array(training_losses))
    print(f"[保存] ✓ 训练损失数组: {losses_file}")
    
    # 2. 训练日志CSV
    csv_file = os.path.join(args.out_dir, "training_log.csv")
    with open(csv_file, 'w', newline='') as f:
        if training_log:
            fieldnames = training_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_log)
    print(f"[保存] ✓ 训练日志CSV: {csv_file}")
    
    # 3. 测试集损失历史
    test_history_file = os.path.join(args.out_dir, "test_loss_history.json")
    with open(test_history_file, 'w') as f:
        json.dump(test_losses_history, f, indent=2)
    print(f"[保存] ✓ 测试集损失历史: {test_history_file}")
    
    # 4. 计算AUC
    auc_metrics = compute_auc_intervals(training_losses)
    
    # 5. 早期收敛指标
    early_convergence = {
        'loss_at_100': float(training_losses[99]) if len(training_losses) >= 100 else None,
        'loss_at_200': float(training_losses[199]) if len(training_losses) >= 200 else None,
        'loss_at_500': float(training_losses[499]) if len(training_losses) >= 500 else None,
        'loss_at_1000': float(training_losses[999]) if len(training_losses) >= 1000 else None,
    }
    
    # 6. 结果汇总
    results = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'init_method': 'FDT' if args.use_fdt_init else 'PEFT_Default',
        'init_alpha': args.fdt_alpha if args.use_fdt_init else None,
        'init_preset': args.init_preset,
        
        'best_train_loss': float(best_loss),
        'final_test_loss': float(test_loss),
        'final_test_loss_std': float(test_std),
        
        'early_convergence': early_convergence,
        'auc_metrics': auc_metrics,
        
        'wall_time_seconds': float(total_time),
        'wall_time_minutes': float(total_time / 60),
        'peak_memory_gb': float((peak_memory - start_memory) / 1e9),
        'throughput_samples_per_sec': float(len(train_dataset) / total_time),
        'avg_time_per_step_ms': float(total_time * 1000 / len(training_losses)),
        
        'init_time_seconds': init_info['init_time_seconds'],
        'init_time_ms': init_info['init_time_seconds'] * 1000 if init_info['init_time_seconds'] else None,
        
        'num_train_samples': len(train_dataset),
        'num_test_samples': len(test_dataset),
        'test_batches_evaluated': len(test_losses),
        
        'nan_count': nan_count,
        
        'measured_alphas_init': init_info.get('measured_alphas_init', {}),
        'measured_alphas_final': init_info.get('measured_alphas_final', {}),
    }
    
    results_file = os.path.join(args.out_dir, "results.json")
    results_serializable = convert_to_json_serializable(results)
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"[保存] ✓ 结果汇总: {results_file}")
    
    # 7. 初始化信息
    init_info_file = os.path.join(args.out_dir, 'init_info.json')
    init_info_serializable = convert_to_json_serializable(init_info)
    with open(init_info_file, 'w') as f:
        json.dump(init_info_serializable, f, indent=2)
    print(f"[保存] ✓ 初始化信息: {init_info_file}")
    
    # 8. 最终模型
    final_path = os.path.join(args.out_dir, "final_model")
    try:
        if device_type == 'npu':
            os.makedirs(final_path, exist_ok=True)
            adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if 'lora' in k}
            torch.save(adapter_state, os.path.join(final_path, "adapter_model.bin"))
            model.peft_config['default'].save_pretrained(final_path)
            print(f"[保存] ✓ 最终模型: {final_path}")
        else:
            model.save_pretrained(final_path)
            print(f"[保存] ✓ 最终模型: {final_path}")
    except Exception as e:
        print(f"[保存] ⚠️ 最终模型保存失败: {e}")
    
    # 9. Tokenizer
    try:
        tokenizer.save_pretrained(args.out_dir)
        print(f"[保存] ✓ Tokenizer: {args.out_dir}\n")
    except Exception as e:
        print(f"[保存] ⚠️ Tokenizer 保存失败: {e}\n")
    
    # ==================== 完成 ====================
    print("="*70)
    print("🎉 训练完成!")
    print("="*70)
    print(f"\n数据集: {args.dataset.upper()}")
    print(f"模型: {args.model_path.split('/')[-1]}")
    print(f"LoRA 秩: r={args.lora_r}")
    print(f"初始化: {'FDT (α='+str(args.fdt_alpha)+')' if args.use_fdt_init else 'PEFT Default (Kaiming+Zero)'}")
    
    print(f"\n📊 训练指标:")
    print(f"  • 最佳训练损失: {best_loss:.4f}")
    print(f"  • 最终测试损失: {test_loss:.4f}")
    
    if early_convergence['loss_at_100']:
        print(f"\n📈 早期收敛:")
        print(f"  • Loss@100: {early_convergence['loss_at_100']:.4f}")
        if early_convergence['loss_at_500']:
            print(f"  • Loss@500: {early_convergence['loss_at_500']:.4f}")
    
    if auc_metrics.get('auc_0_500'):
        print(f"\n📉 AUC 指标:")
        print(f"  • AUC(0-500): {auc_metrics['auc_0_500']:.2f}")
        if auc_metrics.get('auc_0_2500'):
            print(f"  • AUC(0-2500): {auc_metrics['auc_0_2500']:.2f}")
    
    print(f"\n⏱️ 效率指标:")
    print(f"  • 训练时间: {total_time/60:.2f} 分钟")
    print(f"  • 峰值内存: {(peak_memory - start_memory) / 1e9:.2f} GB")
    print(f"  • 吞吐量: {len(train_dataset) / total_time:.2f} samples/s")
    
    if init_info['init_time_seconds']:
        print(f"  • 初始化耗时: {init_info['init_time_seconds']*1000:.2f} ms")
    
    if nan_count > 0:
        print(f"\n⚠️ 训练异常:")
        print(f"  • NaN次数: {nan_count} 次")
    
    print(f"\n💾 输出目录: {args.out_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()