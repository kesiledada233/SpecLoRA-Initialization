"""
OpenPangu FDT 验证脚本 - LoRA版本（多卡优化版）
支持三种优化器:
1. AdamW (标准优化器)
2. FDT-FreqAdamW v2.1 (旧版FDT优化器)
3. FDT-SOC AdamW (新版混合自适应优化器 + 多卡FFT卸载)
"""

import os

os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'  # ← 必须在这里！
os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'

import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from datasets import Dataset
import json
import gzip

# 检查PEFT是否安装
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
    exit(1)

# ==================== ⚡ 强制导入正确版本的 FDT 模块 ====================
print("\n" + "="*70)
print("🔧 加载 FDT 初始化模块")
print("="*70)

# 1. 清除旧缓存
for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"[清理] 移除模块缓存: {mod}")

# 2. 设置正确路径
FDT_INIT_PATH = '/root/nvme0n1/Noneq_Neural_Network/FDT_Init'

# 移除所有可能的旧路径
for path in sys.path[:]:
    if 'FDT_Init' in path and path != FDT_INIT_PATH:
        sys.path.remove(path)
        print(f"[清理] 移除旧路径: {path}")

# 插入正确路径
if FDT_INIT_PATH in sys.path:
    sys.path.remove(FDT_INIT_PATH)
sys.path.insert(0, FDT_INIT_PATH)

print(f"[路径] FDT 路径: {FDT_INIT_PATH}")

# 3. 验证文件存在
fdt_init_file = os.path.join(FDT_INIT_PATH, 'fdt_init.py')
measure_alpha_file = os.path.join(FDT_INIT_PATH, 'measure_alpha.py')

if not os.path.exists(fdt_init_file):
    raise FileNotFoundError(f"找不到: {fdt_init_file}")
if not os.path.exists(measure_alpha_file):
    raise FileNotFoundError(f"找不到: {measure_alpha_file}")

print(f"[验证] ✓ fdt_init.py 存在")
print(f"[验证] ✓ measure_alpha.py 存在")

# 4. 导入模块
try:
    from fdt_init import (
        fdt_initialize_,
        apply_fdt_to_lora,
        init_lora_with_pink_noise,
        init_lora_with_custom_spectrum
    )
    from measure_alpha import (
        measure_alpha,
        analyze_lora_spectra,
        verify_fdt_initialization,
        plot_power_spectrum
    )
    
    # 5. 验证导入路径
    import fdt_init as _fdt_check
    actual_path = _fdt_check.__file__
    print(f"[导入] 实际路径: {actual_path}")
    
    # 6. 检查文件修改时间
    import datetime
    mtime = os.path.getmtime(actual_path)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[导入] 修改时间: {mtime_str}")
    
    # 7. 检查关键修复
    with open(actual_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'np.arange(1, n_freqs)' in content:
            print(f"[导入] ✓ 确认为修复版本（整数频率）")
        else:
            raise ImportError(
                f"❌ 导入的是旧版本！\n"
                f"   路径: {actual_path}\n"
                f"   请检查是否有多个 fdt_init.py 文件。"
            )
    
    FDT_INIT_AVAILABLE = True
    print(f"[导入] ✓ FDT 模块加载成功")
    
except ImportError as e:
    print(f"[导入] ✗ 失败: {e}")
    raise

print("="*70 + "\n")

# ==================== 参数配置 ====================
def get_args():
    ap = argparse.ArgumentParser()
    
    # 基础参数
    ap.add_argument("--model_name", type=str, 
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="openPangu 模型路径")
    
    ap.add_argument("--use_flash_attention", action="store_true",
                   help="使用 Flash Attention")

    ap.add_argument("--block_size", type=int, default=256,
                   help="序列长度")
    ap.add_argument("--batch_size", type=int, default=2,
                   help="批大小")
    ap.add_argument("--max_iters", type=int, default=2500,
                   help="训练步数")
    ap.add_argument("--eval_interval", type=int, default=100,
                   help="评估间隔")
    ap.add_argument("--seed", type=int, default=1107,
                   help="随机种子")
    ap.add_argument("--max_train_samples", type=int, default=10000,
                   help="限制训练样本最大数量（截断前随机）")
    ap.add_argument("--max_val_samples", type=int, default=500,
                   help="限制验证样本最大数量（截断前随机）")
    ap.add_argument("--device", type=str, 
                   default="npu:0" ,
                   help="训练设备 (npu:0, npu:1, cuda:0, etc.)")
    
    ap.add_argument("--save_interval", type=int, default=500,
                   help="保存间隔")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                   help="梯度累积步数")
    
    # LoRA 相关参数
    ap.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    ap.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, nargs="+", 
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA目标模块")
    

    # 优化器配置
    ap.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "fdt_v21", "fdt_soc"],
                   help="优化器类型")
    ap.add_argument("--lr", type=float, default=5e-5,
                   help="学习率")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                   help="权重衰减")
    ap.add_argument("--warmup_steps", type=int, default=100,
                   help="预热步数")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="梯度裁剪阈值")
    
    # 数据配置
    ap.add_argument("--dataset", type=str, default="ShareGPT",
                   help="数据集名 --data_files 使用本地文件")
    ap.add_argument("--dataset_config", type=str, default="computer_en",
                   help="数据集配置名称")
    ap.add_argument("--dataset_split", type=str, default="train",
                   help="数据集 split（对本地文件无实际分割意义，仅作标签）")
    ap.add_argument("--data_files", type=str, nargs="+", default=["computer_en_26k.jsonl"],
                   help="本地数据文件（json/jsonl/csv 等）")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                   help="验证集比例（0-1），对本地 JSONL 生效")
    ap.add_argument("--val_max_steps", type=int, default=100,
                   help="评估时抽样的批次数")
    
    # FDT 初始化配置
    ap.add_argument("--use_fdt_init", action="store_true",
                   help="启用 FDT 初始化（频域感知初始化）")
    ap.add_argument("--fdt_alpha", type=float, default=1.2,
                   help="FDT 初始化的功率律指数 (0.8-1.5)")
    ap.add_argument("--fdt_temp_ratio", type=float, default=None,
                   help="FDT 初始化的温度比 (可选)")
    ap.add_argument("--fdt_method", type=str, default='fft',
                   choices=['fft', 'ar'],
                   help="FDT 初始化方法: fft (精确) 或 ar (快速)")
    ap.add_argument("--verify_fdt", action="store_true",
                   help="初始化后验证功率律指数")
    ap.add_argument("--plot_spectra", action="store_true",
                   help="绘制功率谱图（保存到输出目录）")
    ap.add_argument("--init_preset", type=str, default=None,
                   choices=['baseline', 'soft', 'medium', 'strong', 'temp'],
                   help="初始化预设: baseline/soft/medium/strong/temp")
    
    # FDT Recorder 配置
    ap.add_argument("--max_elems", type=int, default=4096,
                   help="FDT Recorder 最大追踪元素数")
    
    # 输出配置
    ap.add_argument("--out_dir", type=str, default="outputs_fdt_init",
                   help="输出目录")
    ap.add_argument("--verbose", action="store_true",
                   help="详细输出")
    

    args = ap.parse_args()
    # 兼容旧代码：统一两个字段，后续代码两者都可用
    setattr(args, 'model_path', args.model_name)
    return args

# ==================== 数据处理 ====================
class TextDataset:
    def __init__(self, tokenizer, dataset, block_size, device):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device
        
        # 过滤空文本
        dataset = dataset.filter(lambda x: isinstance(x.get('text', ''), str) and len(x['text'].strip()) > 0)
        
        # 分词函数
        def tokenize_fn(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=block_size,
                padding='max_length',
                return_tensors=None
            )
        
        self.data = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"[Dataset] 总样本数: {len(self.data)}")
        
        # 验证数据格式
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"[Dataset] 样本键名: {list(sample.keys())}")
            print(f"[Dataset] 输入ID长度: {len(sample['input_ids'])}")
    
    def get_batch(self, batch_size):
        # 随机采样
        indices = torch.randint(0, len(self.data), (batch_size,))
        batch = [self.data[int(i)] for i in indices]
        
        # 构造输入和标签
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch])
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch])
        
        # 标签：input_ids 右移一位
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        
        return {
            'input_ids': input_ids.to(self.device),
            'labels': labels.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }

@torch.no_grad()
def evaluate(model, dataset, batch_size, steps):
    model.eval()
    losses = []
    for _ in range(steps):
        batch = dataset.get_batch(batch_size)
        out = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

# ==================== 内存优化函数 ====================
def setup_memory_optimizations(model):
    """设置内存优化"""
    if hasattr(model, 'config'):
        model.config.use_cache = False
        print("[Memory] 已禁用KV缓存")
    return model

def get_memory_info(device_str):
    """获取指定设备的显存信息"""
    try:
        if 'cuda' in device_str:
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            return allocated, reserved
        elif 'npu' in device_str:
            import torch_npu
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            allocated = torch_npu.npu.memory_allocated(device_id) / 1024**3
            reserved = torch_npu.npu.memory_reserved(device_id) / 1024**3
            return allocated, reserved
    except:
        pass
    return None, None

# ==================== ShareGPT 线性化 ====================
def linearize_sharegpt(example):
    """
    将 ShareGPT 的 conversations/messages 转为单条 text
    支持条目格式：
    - {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    - {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
    """
    turns = None
    if isinstance(example, dict):
        if 'conversations' in example and example['conversations'] is not None:
            turns = example['conversations']
        elif 'messages' in example and example['messages'] is not None:
            turns = example['messages']
    if not turns:
        # 兜底：已有 text 字段
        if 'text' in example:
            return {'text': example['text']}
        if 'content' in example:
            return {'text': example['content']}
        return {'text': ''}

    parts = []
    # 可选：前置 system 提示
    sys_prompt = None
    if isinstance(turns, list):
        for t in turns:
            role = (t.get('from') or t.get('role') or '').lower()
            content = (t.get('value') or t.get('content') or '') or ''
            if role in ['system'] and content:
                sys_prompt = content.strip() if not sys_prompt else sys_prompt
                break
    if sys_prompt:
        parts.append(f"System: {sys_prompt}")

    for t in turns:
        role = (t.get('from') or t.get('role') or '').lower()
        content = t.get('value') or t.get('content') or ''
        if not content:
            continue
        if role in ['human', 'user']:
            parts.append(f"User: {content}")
        elif role in ['gpt', 'assistant', 'bot']:
            parts.append(f"Assistant: {content}")
        elif role in ['system']:
            # 已在开头加过，跳过
            continue
        else:
            parts.append(str(content))
    text = "\n".join(parts).strip()
    return {'text': text}

# ==================== 主训练循环 ====================
def main():
    if not PEFT_AVAILABLE:
        print("错误: PEFT库不可用，请先安装: pip install peft")
        return
    
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*70)
    print("[1/6] 加载模型和分词器...")
    print("="*70)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 设置特殊token（适配 OpenPangu）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Tokenizer] Pad token 设置为 EOS: {tokenizer.pad_token}")
    else:
        print(f"[Tokenizer] Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # 精度选择
    if 'cuda' in args.device:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif 'npu' in args.device:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"[Model] 使用 {dtype} 加载模型（显存优化）...")
    # 正确的 kwargs 构造
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': dtype,
    }
    
    if args.use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("[加载] 使用 Flash Attention 2")

    # 只调用一次 from_pretrained，且用 kwargs
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,   # 与 get_args 中一致
        **model_kwargs
    )

    # 应用内存优化
    model = setup_memory_optimizations(model)
    
    # 将模型移动到设备
    model = model.to(args.device)

    if 'npu' in args.device:
        model.half()
    
    print(f"[Model] 参数量: {model.num_parameters():,}")
    print(f"[Model] 设备: {next(model.parameters()).device}")
    print(f"[Model] 数据类型: {next(model.parameters()).dtype}")
    
    # 计算模型显存占用（粗略）
    model_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    model_memory_gb = model_params * bytes_per_param / 1e9
    print(f"[Model] 基础模型显存: ~{model_memory_gb:.2f} GB")
    
    # ==================== LoRA 配置 ====================
    print("\n[2/6] 设置LoRA...")
    print("="*70)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ⚡⚡⚡ 显式验证 LoRA 配置 ⚡⚡⚡
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100 * trainable_params / total_params
    
    print(f"\n[LoRA 验证]")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练占比: {trainable_ratio:.3f}%")
    
    if trainable_ratio > 5.0:
        print(f"\n{'='*70}")
        print("⚠️⚠️⚠️  警告: 可训练参数占比过高！这不像是正确的 LoRA 配置（应 < 2%）")
        print(f"{'='*70}")
        non_lora_params = []
        lora_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append((name, param.numel()))
                else:
                    non_lora_params.append((name, param.numel()))
        if non_lora_params:
            print("正在强制冻结所有非 LoRA 参数...")
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' not in name.lower():
                    param.requires_grad = False
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_ratio = 100 * trainable_params / total_params
            print(f"修复后可训练占比: {trainable_ratio:.3f}%")
    

    # ==================== 步骤 3: FDT 初始化 ====================
    print("\n" + "="*70)
    print("🎯 步骤 3: FDT 初始化")
    print("="*70)
    
    # 解析预设配置
    if args.init_preset:
        preset_configs = {
            'baseline': {'use_fdt': False, 'alpha': None, 'temp': None, 
                        'name': 'Xavier Baseline (白噪声 α≈0)'},
            'soft': {'use_fdt': True, 'alpha': 0.8, 'temp': None, 
                    'name': 'FDT-Soft (α=0.8, 弱相关)'},
            'medium': {'use_fdt': True, 'alpha': 1.2, 'temp': None, 
                      'name': 'FDT-Medium (α=1.2, 粉红噪声)'},
            'strong': {'use_fdt': True, 'alpha': 1.5, 'temp': None, 
                      'name': 'FDT-Strong (α=1.5, 强功率律)'},
            'temp': {'use_fdt': True, 'alpha': 1.2, 'temp': 1.5, 
                    'name': 'FDT-Temp (α=1.2, τ=1.5)'},
        }
        
        config = preset_configs[args.init_preset]
        print(f"\n[预设] {config['name']}")
        
        if config['use_fdt']:
            args.use_fdt_init = True
            args.fdt_alpha = config['alpha']
            args.fdt_temp_ratio = config['temp']
    
    # 执行初始化
    init_info = {
        'use_fdt': args.use_fdt_init,
        'preset': args.init_preset,
        'alpha': None,
        'temp_ratio': None,
        'method': 'xavier',
        'measured_alphas': {},
        'verification_passed': None,
    }
    
    if args.use_fdt_init:
        if not FDT_INIT_AVAILABLE:
            print("\n❌ 错误: FDT 初始化模块不可用!")
            print("请确保以下文件存在:")
            print("  - deepseek/initializers/fdt_init.py")
            print("  - deepseek/initializers/measure_alpha.py")
            print("\n将使用默认 Xavier 初始化继续...")
            args.use_fdt_init = False
        else:
            print(f"\n[1/3] 应用 FDT 初始化...")
            print(f"  配置: α={args.fdt_alpha:.2f}", end='')
            if args.fdt_temp_ratio:
                print(f", τ={args.fdt_temp_ratio:.2f}", end='')
            print(f", 方法={args.fdt_method}")
            
            init_start_time = time.time()
            
            apply_fdt_to_lora(
                model,
                alpha=args.fdt_alpha,
                temp_ratio=args.fdt_temp_ratio,
                method=args.fdt_method,
                verbose=args.verbose
            )
            
            init_duration = time.time() - init_start_time
            print(f"  ✓ 初始化完成，耗时 {init_duration:.2f} 秒")
            
            init_info['alpha'] = args.fdt_alpha
            init_info['temp_ratio'] = args.fdt_temp_ratio
            init_info['method'] = args.fdt_method
            
            # 验证初始化
            if args.verify_fdt:
                print(f"\n[2/3] 验证初始化质量...")
                
                verify_success = verify_fdt_initialization(
                    model,
                    target_alpha=args.fdt_alpha,
                    tolerance=0.15,  # 允许 15% 误差
                    verbose=True
                )
                
                init_info['verification_passed'] = verify_success
                
                if not verify_success:
                    print("\n⚠️  警告: 初始化验证未完全通过，但训练将继续")
            
            # 分析并保存功率谱
            if args.plot_spectra:
                print(f"\n[3/3] 分析功率谱...")
                
                spectra_dir = os.path.join(args.out_dir, 'init_spectra')
                os.makedirs(spectra_dir, exist_ok=True)
                
                alphas = analyze_lora_spectra(
                    model,
                    save_dir=spectra_dir,
                    plot_top_n=3,
                    verbose=args.verbose
                )
                
                init_info['measured_alphas'] = {k: float(v) for k, v in alphas.items() 
                                                 if not np.isnan(v)}
                
                # 保存 α 值到文件
                alpha_file = os.path.join(args.out_dir, 'init_alpha_values.txt')
                with open(alpha_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("FDT 初始化报告\n")
                    f.write("="*60 + "\n\n")
                    
                    f.write("配置:\n")
                    f.write(f"  目标 α: {args.fdt_alpha:.3f}\n")
                    if args.fdt_temp_ratio:
                        f.write(f"  目标 τ: {args.fdt_temp_ratio:.3f}\n")
                    f.write(f"  方法: {args.fdt_method}\n\n")
                    
                    f.write("测量结果:\n")
                    for name, alpha in alphas.items():
                        error = abs(alpha - args.fdt_alpha)
                        status = "✓" if error < 0.15 else "⚠️"
                        f.write(f"  {status} {name}: α={alpha:.3f} (误差={error:.3f})\n")
                    
                    valid_alphas = [a for a in alphas.values() if not np.isnan(a)]
                    if valid_alphas:
                        f.write(f"\n统计:\n")
                        f.write(f"  平均: {np.mean(valid_alphas):.3f}\n")
                        f.write(f"  标准差: {np.std(valid_alphas):.3f}\n")
                        f.write(f"  范围: [{np.min(valid_alphas):.3f}, {np.max(valid_alphas):.3f}]\n")
                        
                        avg_error = np.mean([abs(a - args.fdt_alpha) for a in valid_alphas])
                        f.write(f"  平均误差: {avg_error:.3f}\n")
                    
                    f.write("="*60 + "\n")
                
                print(f"  ✓ 报告保存到: {alpha_file}")
    
    else:
        print("\n[跳过] 使用默认 Xavier 初始化（Baseline）")
    
    # ⚡⚡⚡ 保存初始化信息（修复 JSON 序列化 + numpy 兼容性）⚡⚡⚡
    init_info_file = os.path.join(args.out_dir, 'init_info.json')
    
    # 转换函数：处理 numpy 类型（兼容 numpy >= 1.20）
    def convert_to_json_serializable(obj):
        """递归转换 numpy 类型为 Python 原生类型（兼容 numpy 1.20+）"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        # ⚡ 修复：移除 np.bool，只保留 np.bool_
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (int, float, str)):
            return obj
        else:
            # 其他类型尝试转换为字符串
            try:
                return str(obj)
            except:
                return None
    
    # 转换并保存
    try:
        init_info_serializable = convert_to_json_serializable(init_info)
        
        with open(init_info_file, 'w') as f:
            json.dump(init_info_serializable, f, indent=2)
        print(f"[初始化] ✓ 初始化信息保存到: {init_info_file}")
    except Exception as e:
        print(f"[初始化] ⚠️ 初始化信息保存失败: {e}")
        # 保存一个简化版本
        try:
            simple_info = {
                'use_fdt': bool(init_info.get('use_fdt', False)),
                'preset': str(init_info.get('preset', 'unknown')),
                'alpha': float(init_info.get('alpha')) if init_info.get('alpha') is not None else None,
            }
            with open(init_info_file, 'w') as f:
                json.dump(simple_info, f, indent=2)
            print(f"[初始化] ✓ 保存了简化版本")
        except:
            pass
    
    print("="*70)
    # ==================== 优化器 ====================
    print("\n" + "="*70)
    print("⚙️ 步骤 4: 配置优化器")
    print("="*70)
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"[优化器] AdamW (lr={args.lr}, wd={args.weight_decay})")
    
    # elif args.optimizer == "fdt_v21":
    #     if not FDT_V21_AVAILABLE:
    #         print("[优化器] ⚠️ FDT v2.1 不可用，回退到 AdamW")
    #         optimizer = torch.optim.AdamW(
    #             trainable_params_list,
    #             lr=args.lr,
    #             weight_decay=args.weight_decay
    #         )
    #     else:
    #         optimizer = FDTFreqAdamWv21(
    #             trainable_params_list,
    #             lr=args.lr,
    #             weight_decay=args.weight_decay
    #         )
    #         print(f"[优化器] FDT-FreqAdamW v2.1 (lr={args.lr})")
    
    # elif args.optimizer == "fdt_soc":
    #     if not FDT_SOC_AVAILABLE:
    #         print("[优化器] ⚠️ FDT-SOC 不可用，回退到 AdamW")
    #         optimizer = torch.optim.AdamW(
    #             trainable_params_list,
    #             lr=args.lr,
    #             weight_decay=args.weight_decay
    #         )
    #     else:
    #         optimizer = FDTSOCAdamW(
    #             trainable_params_list,
    #             lr=args.lr,
    #             weight_decay=args.weight_decay
    #         )
    #         print(f"[优化器] FDT-SOC AdamW (lr={args.lr})")
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )
    print(f"[调度器] Linear warmup ({args.warmup_steps} steps) + decay")
    
    print("="*70)
    
    # ==================== 数据集 ====================
    print("\n[5/6] 加载数据集 (ShareGPT)...")
    print("="*70)

    def _join_turns(turns, sep="\n"):
        # 将 [{role,text}/{human,assistant}] 线性化为单一字符串
        parts = []
        for t in turns:
            # 兼容字段名
            if "role" in t and "content" in t:
                role = t["role"]
                text = t.get("content") or ""
            else:
                # ShareGPT 常见：human/assistant
                if "human" in t:
                    role = "user"
                    text = t.get("human") or ""
                elif "assistant" in t:
                    role = "assistant"
                    text = t.get("assistant") or ""
                else:
                    role = "unknown"
                    text = ""
            text = str(text).strip()
            if not text:
                continue
            # 带前缀以增强上下文清晰度
            prefix = "User:" if role == "user" else ("Assistant:" if role == "assistant" else "")
            parts.append(f"{prefix} {text}" if prefix else text)
        return sep.join(parts).strip()

    def _linearize_example(ex):
        # 优先使用已有 text，其次 conversations/messages，最后 conversation
        if isinstance(ex.get("text"), str) and ex["text"].strip():
            return ex["text"].strip()

        turns = None
        if isinstance(ex.get("conversations"), list):
            turns = ex["conversations"]
        elif isinstance(ex.get("messages"), list):
            turns = ex["messages"]
        elif isinstance(ex.get("conversation"), list):  # 你的数据用这个键
            turns = ex["conversation"]

        if isinstance(turns, list) and turns:
            return _join_turns(turns)

        # 兜底：instruction/output 结构
        if ("instruction" in ex) and ("output" in ex):
            instr = str(ex.get("instruction") or "").strip()
            inp = str(ex.get("input") or "").strip()
            outp = str(ex.get("output") or "").strip()
            body = "\n".join(s for s in [instr, inp, outp] if s).strip()
            return body

        return ""

    def _load_pair():
        if args.data_files:
            print(f"[Dataset] 本地读取（手工解析，跳过 pyarrow）: {args.data_files}")

            def iter_examples(paths):
                import json, gzip
                for fp in paths:
                    opener = gzip.open if fp.endswith(".gz") else open
                    with opener(fp, "rt", encoding="utf-8") as f:
                        first = f.read(1)
                        if not first:
                            continue
                        f.seek(0)
                        if first == "[":
                            try:
                                arr = json.load(f)
                            except Exception as e:
                                print(f"[警告] 读取失败（跳过）{fp}: {e}")
                                continue
                            for ex in arr:
                                yield ex
                        else:
                            for i, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    yield json.loads(line)
                                except Exception as e:
                                    if i <= 3:
                                        print(f"[警告] 解析失败 {fp}:{i}: {e}（跳过该行）")

            def gen():
                dropped = 0
                kept = 0
                for ex in iter_examples(args.data_files):
                    # 去掉易触发 Arrow 类型冲突字段
                    ex.pop("category", None)

                    txt = _linearize_example(ex)
                    if not txt:
                        dropped += 1
                        continue
                    kept += 1
                    yield {"text": txt}
                print(f"[Dataset] 生成器统计: 保留 {kept} 条，丢弃空样本 {dropped} 条")

            base = Dataset.from_generator(gen)
            split = base.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
            return split["train"], split["test"]
        else:
            name = args.dataset_config if (args.dataset_config not in [None, "", "none", "None"]) else None
            base = load_dataset(args.dataset, name, split=args.dataset_split)
            split = base.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
            return split["train"], split["test"]

    raw_train, raw_val = _load_pair()

    # 将对话结构映射为 text
    def ensure_text(ds):
        cols = ds.column_names
        if 'text' in cols:
            return ds
        if 'conversations' in cols or 'messages' in cols:
            print("[Dataset] 线性化对话为 'text'")
            return ds.map(linearize_sharegpt, remove_columns=cols, desc="Linearizing ShareGPT")
        if 'content' in cols:
            return ds.rename_column('content', 'text')
        if all(c in cols for c in ['instruction', 'output']):
            def _inst_fmt(x):
                inp = x.get('input') or ''
                if inp:
                    return {'text': f"User: {x['instruction']}\nInput: {inp}\nAssistant: {x['output']}"}
                return {'text': f"User: {x['instruction']}\nAssistant: {x['output']}"}
            return ds.map(_inst_fmt, remove_columns=cols)
        raise ValueError(f"无法从数据集中推断文本字段，列: {cols}")

    raw_train = ensure_text(raw_train)
    raw_val = ensure_text(raw_val)

    # 过滤空样本
    raw_train = raw_train.filter(lambda x: isinstance(x.get('text',''), str) and len(x['text'].strip())>0)
    raw_val = raw_val.filter(lambda x: isinstance(x.get('text',''), str) and len(x['text'].strip())>0)

    if args.max_train_samples is not None and len(raw_train) > args.max_train_samples:
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.max_train_samples))
        print(f"[Subset] 训练样本截断到 {len(raw_train)}")
    if args.max_val_samples is not None and len(raw_val) > args.max_val_samples:
        raw_val = raw_val.shuffle(seed=args.seed).select(range(args.max_val_samples))
        print(f"[Subset] 验证样本截断到 {len(raw_val)}")

    print(f"[Dataset] 训练样本: {len(raw_train)} | 验证样本: {len(raw_val)}")

    train_dataset = TextDataset(tokenizer, raw_train, args.block_size, args.device)
    val_dataset = TextDataset(tokenizer, raw_val, args.block_size, args.device)

    
    # ==================== 训练循环 ====================
    print("\n[6/6] 开始训练...")
    print("="*70)
    
    model.train()
    t0 = time.time()
    t1 = time.time()
    # losses = []
    training_losses = []
    val_history = []
    global_step = 0
    
    # 初始内存状态
    allocated, reserved = get_memory_info(args.device)
    if allocated is not None:
        print(f"[Memory] 训练设备 ({args.device}) - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    if args.optimizer == "fdt_soc" and hasattr(optimizer, 'fft_device'):
        fft_device = optimizer.fft_device
        fft_allocated, fft_reserved = get_memory_info(fft_device)
        if fft_allocated is not None:
            print(f"[Memory] FFT 设备 ({fft_device}) - 已分配: {fft_allocated:.2f}GB, 已保留: {fft_reserved:.2f}GB")
    
    for step in range(1, args.max_iters + 1):
        batch = train_dataset.get_batch(args.batch_size)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss / args.grad_accum_steps
        
        loss.backward()

        # 梯度累积
        if step % args.grad_accum_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
            
            # 优化器步骤
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        if step % args.eval_interval == 0 or step == 1:
            dt = time.time() - t1
            recent_losses = training_losses[-args.eval_interval:] if len(training_losses) >= args.eval_interval else training_losses
            avg_loss = sum(recent_losses) / len(recent_losses)
            
            info_str = f"[步骤 {step:5d}/{args.max_iters}] 损失={current_loss:.4f} (均值: {avg_loss:.4f})"
            
            info_str += f" | 时间={dt:.1f}s"

            # 计算 AUC (前 500 步)
            auc_str = ""
            if step >= 500:
                auc_500 = sum(training_losses[:500])
                auc_str = f", AUC(0-500)={auc_500:.2f}"
                info_str += auc_str

            print(info_str)

            if step % (args.eval_interval * 2) == 0:
                if 'npu' in args.device:
                    try:
                        import torch_npu
                        torch_npu.npu.empty_cache()
                    except:
                        pass
            val_loss = evaluate(model, val_dataset, args.batch_size, args.val_max_steps)
            print(f"[验证] step {step} | val_loss={val_loss:.4f}")

            val_history.append((step, float(val_loss)))
            val_csv = os.path.join(args.out_dir, "val_losses.csv")
            write_header = not os.path.exists(val_csv)
            try:
                with open(val_csv, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("step,val_loss\n")
                    f.write(f"{step},{val_loss:.6f}\n")
            except Exception as e:
                print(f"⚠️ [警告] 写入验证损失CSV失败: {e}")

        # 定期保存检查点
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_step{step}")
            model.save_pretrained(checkpoint_path)
            print(f"  → 保存检查点: {checkpoint_path}")

            t1 = time.time()

    total_time = time.time() - t0

    print("-"*70)
    print(f"[训练] ✓ 完成! 总耗时: {total_time/60:.2f} 分钟")
    
    # ==================== 保存数据 ====================
    print("\n[保存] 保存训练数据...")
    print("="*70)
    
    lora_dir = os.path.join(args.out_dir, "lora_adapter")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    
    # ⚡⚡⚡ 修复：保存正确的损失列表 ⚡⚡⚡
    loss_file = os.path.join(args.out_dir, "training_losses.npy")

    # 添加调试信息
    print(f"[保存] 训练损失数量: {len(training_losses)}")

    if len(training_losses) == 0:
        print(f"[保存] ⚠️ 警告: 训练损失列表为空!")
        print(f"[保存] 可能原因:")
        print(f"  1. 训练循环未正常执行")
        print(f"  2. loss.item() 调用失败")
        print(f"  3. training_losses.append() 未执行")
    else:
        np.save(loss_file, np.array(training_losses))  # ✅ 修复：保存正确的列表
        print(f"[保存] ✓ 训练损失 -> {loss_file}")
        print(f"[保存]   前5个损失: {training_losses[:5]}")
        print(f"[保存]   后5个损失: {training_losses[-5:]}")

    # 验证文件是否成功保存
    if os.path.exists(loss_file):
        saved_losses = np.load(loss_file)
        print(f"[保存] ✓ 验证：文件包含 {len(saved_losses)} 个损失值")
    else:
        print(f"[保存] ❌ 错误：文件未成功创建")

    try:
        val_npy = os.path.join(args.out_dir, "val_losses.npy")
        np.save(val_npy, np.array([v for _, v in val_history], dtype=np.float32))
        print(f"[保存] ✓ 验证损失 -> {val_npy}")
        print(f"[保存] ✓ 验证损失CSV -> {os.path.join(args.out_dir, 'val_losses.csv')}")
    except Exception as e:
        print(f"⚠️  [警告] 验证损失保存失败: {e}")

    print(f"[保存] ✓ LoRA适配器 -> {lora_dir}")
    
    print("\n" + "="*70)
    print("🎉 [完成] 训练完成!")
    print("="*70)

if __name__ == "__main__":
    main()