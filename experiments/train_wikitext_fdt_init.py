"""
OpenPangu FDT 初始化验证脚本 - LoRA版本
支持多种优化器 + 五种初始化方式:

优化器:
1. AdamW (标准)
2. FDT-FreqAdamW v2.1
3. FDT-SOC AdamW

初始化:
1. Xavier (Baseline) - 白噪声谱 α≈0
2. FDT-Soft (α=0.8) - 弱长程相关
3. FDT-Medium (α=1.2) - 粉红噪声（推荐）
4. FDT-Strong (α=1.5) - 强功率律
5. FDT-Temp (α=1.2, τ=1.5) - 结合温度比控制

运行示例:
  # Baseline
  python train_openpangu_fdt_init.py --init_preset baseline --out_dir outputs_baseline
  
  # FDT-Medium (推荐)
  python train_openpangu_fdt_init.py --init_preset medium --verify_fdt --out_dir outputs_fdt
  
  # 自定义 α 值
  python train_openpangu_fdt_init.py --use_fdt_init --fdt_alpha 1.3 --out_dir outputs_custom
"""

import os

os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'  # ← 必须在这里！
os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'

import sys
import time
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# 检查 PEFT 是否安装
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

# 检查 datasets 是否安装
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("警告: 未找到datasets库，将使用虚拟数据")
    DATASETS_AVAILABLE = False

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

# ==================== 导入优化器 ====================
optimizer_paths = [
    '/root/nvme0n1/Noneq_Neural_Network',
    os.path.join(os.path.dirname(__file__), 'deepseek', 'optimizers'),
]

for path in optimizer_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)

try:
    from fdt_freq_adamw_v21 import FDTFreqAdamWv21
    FDT_V21_AVAILABLE = True
    print("[Import] ✓ FDT-FreqAdamW v2.1 优化器加载成功")
except ImportError as e:
    print(f"[Import] ✗ 无法导入 FDT v2.1 优化器: {e}")
    FDT_V21_AVAILABLE = False

try:
    from FDTSOCAdamW import FDTSOCAdamW
    FDT_SOC_AVAILABLE = True
    print("[Import] ✓ FDT-SOC AdamW 优化器加载成功")
except ImportError as e:
    print(f"[Import] ✗ 无法导入 FDT-SOC 优化器: {e}")
    FDT_SOC_AVAILABLE = False

# 尝试导入 FDT Recorder
try:
    from fdt import FDTRecorder
    FDT_RECORDER_AVAILABLE = True
except ImportError:
    FDT_RECORDER_AVAILABLE = False
    print("[Import] ✗ FDTRecorder 不可用，将跳过轨迹记录")


# ==================== 数据集类 ====================
class TextDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            if len(text.strip()) > 0:
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
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = self.examples[idx]['input_ids']
        attention_mask = self.examples[idx]['attention_mask']
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,  # Long + -100 忽略 pad
        }


class DummyDataset(Dataset):
    """虚拟数据集（用于测试）"""
    
    def __init__(self, vocab_size, seq_length=512, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
        }


# ==================== 工具函数 ====================
def get_memory_info():
    """获取显存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB"
    return "CUDA 不可用"


def setup_memory_optimizations():
    """设置内存优化"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 启用 TF32（如果支持）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==================== 参数配置 ====================
def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDT 初始化训练脚本")
    
    # 模型配置
    ap.add_argument("--model_path", type=str, 
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="预训练模型路径")
    ap.add_argument("--use_flash_attention", action="store_true",
                   help="使用 Flash Attention")
    
    # LoRA 配置
    ap.add_argument("--lora_r", type=int, default=16,
                   help="LoRA 秩")
    ap.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA 目标模块")
    
    # 训练配置
    ap.add_argument("--batch_size", type=int, default=4,
                   help="批次大小")
    ap.add_argument("--max_length", type=int, default=512,
                   help="最大序列长度")
    ap.add_argument("--max_iters", type=int, default=2000,
                   help="最大训练步数")
    ap.add_argument("--eval_interval", type=int, default=100,
                   help="评估间隔")
    ap.add_argument("--save_interval", type=int, default=500,
                   help="保存间隔")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                   help="梯度累积步数")
    
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
    ap.add_argument("--dataset", type=str, default="wikitext",
                   choices=["wikitext", "dummy"],
                   help="数据集类型")
    ap.add_argument("--dataset_path", type=str, 
                   default="/root/nvme0n1/Noneq_Neural_Network/pretrained_models/wikitext_wikitext-2-raw-v1",
                   help="数据集路径（本地路径或 HuggingFace 名称）")
    ap.add_argument("--num_samples", type=int, default=1000,
                   help="虚拟数据集样本数")
    
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
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    ap.add_argument("--verbose", action="store_true",
                   help="详细输出")
    
    # 设备配置 
    ap.add_argument("--device", type=str, default=None,
                   help="计算设备 (如 'cuda:0', 'npu:2', 'cpu'，None 表示自动检测)")

    return ap.parse_args()


# ==================== 主训练循环 ====================
def main():
    # 检查依赖
    if not PEFT_AVAILABLE:
        print("错误: PEFT库不可用，请先安装: pip install peft")
        return
    
    args = get_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(args.out_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"[配置] 保存到: {config_file}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ==================== ⚡ 设备设置（支持 NPU/CUDA/CPU） ====================
    if args.device:
        # 用户指定设备
        device = torch.device(args.device)
        device_type = args.device.split(':')[0]  # 提取设备类型 (npu/cuda/cpu)
        
        print(f"[设备] 使用指定设备: {device}")
        
        # 如果是 NPU，尝试导入并初始化
        if device_type == 'npu':
            try:
                import torch_npu
                torch_npu.npu.set_device(device)
                print(f"[设备] ✓ NPU 初始化成功")
                
                # 设置随机种子（NPU）
                torch_npu.npu.manual_seed_all(args.seed)
            except ImportError:
                print(f"[设备] ⚠️ 未找到 torch_npu 模块")
                print("请安装: pip install torch-npu (华为昇腾)")
                return
            except Exception as e:
                print(f"[设备] ⚠️ NPU 设置失败: {e}")
                return
        
        elif device_type == 'cuda':
            if not torch.cuda.is_available():
                print(f"[设备] ⚠️ CUDA 不可用，回退到 CPU")
                device = torch.device('cpu')
    
    else:
        # 自动检测设备
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[设备] 自动检测: CUDA (GPU)")
        else:
            device = torch.device('cpu')
            print(f"[设备] 自动检测: CPU")
    
    # 显示内存信息
    device_type = str(device).split(':')[0]
    
    if device_type == 'cuda':
        setup_memory_optimizations()
        print(f"[显存] {get_memory_info()}")
    
    elif device_type == 'npu':
        try:
            import torch_npu
            allocated = torch_npu.npu.memory_allocated(device) / 1024**3
            reserved = torch_npu.npu.memory_reserved(device) / 1024**3
            print(f"[NPU 内存] 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
        except Exception as e:
            print(f"[NPU 内存] 无法获取内存信息: {e}")
    
    # ==================== 步骤 1: 加载模型 ====================
    print("\n" + "="*70)
    print("📦 步骤 1: 加载预训练模型")
    print("="*70)
    
    # ⚡⚡⚡ NPU 专用配置：禁用 Fused Attention ⚡⚡⚡
    device_type = str(device).split(':')[0]
    if device_type == 'npu':
        os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'
        os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'
        print("[模型] ⚡ 禁用 NPU Fused Attention（使用标准实现）")

    print(f"[加载] 模型路径: {args.model_path}")
    
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[加载] ✓ Tokenizer 加载成功, vocab_size={tokenizer.vocab_size}")
        
        # 加载模型
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if args.use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("[加载] 使用 Flash Attention 2")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs
        )
        
        total_params, _ = count_parameters(model)
        print(f"[加载] ✓ 模型加载成功, 参数量: {total_params/1e9:.2f}B")
        
    except Exception as e:
        print(f"[加载] ✗ 模型加载失败: {e}")
        print("\n使用小模型进行测试...")
        
        # 回退到小模型
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        total_params, _ = count_parameters(model)
        print(f"[加载] ✓ 使用 {model_name}, 参数量: {total_params/1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"[显存] {get_memory_info()}")
    
    # ==================== 步骤 2: 配置 LoRA ====================
    print("\n" + "="*70)
    print("🔧 步骤 2: 配置 LoRA")
    print("="*70)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"[LoRA] 配置:")
    print(f"  • r = {args.lora_r}")
    print(f"  • alpha = {args.lora_alpha}")
    print(f"  • dropout = {args.lora_dropout}")
    print(f"  • target_modules = {args.lora_target_modules}")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    device_type = str(device).split(':')[0]
    if device_type == 'npu':
        model.half()  # NPU 必须 FP16 权重

    
    total_params, trainable_params = count_parameters(model)
    print(f"[LoRA] ✓ LoRA 应用成功")
    print(f"  • 总参数: {total_params:,}")
    print(f"  • 可训练: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    
    if torch.cuda.is_available():
        print(f"[显存] {get_memory_info()}")
    
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
    
    # ==================== 步骤 4: 配置优化器 ====================
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
    
    elif args.optimizer == "fdt_v21":
        if not FDT_V21_AVAILABLE:
            print("[优化器] ⚠️ FDT v2.1 不可用，回退到 AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params_list,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:
            optimizer = FDTFreqAdamWv21(
                trainable_params_list,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            print(f"[优化器] FDT-FreqAdamW v2.1 (lr={args.lr})")
    
    elif args.optimizer == "fdt_soc":
        if not FDT_SOC_AVAILABLE:
            print("[优化器] ⚠️ FDT-SOC 不可用，回退到 AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params_list,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:
            optimizer = FDTSOCAdamW(
                trainable_params_list,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            print(f"[优化器] FDT-SOC AdamW (lr={args.lr})")
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )
    print(f"[调度器] Linear warmup ({args.warmup_steps} steps) + decay")
    
        # ==================== 步骤 5: 准备数据 ====================
    print("\n" + "="*70)
    print("📊 步骤 5: 准备数据")
    print("="*70)
    
    # ⚡⚡⚡ 初始化数据集变量 ⚡⚡⚡
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if args.dataset == "wikitext" and DATASETS_AVAILABLE:
        try:
            # 构建本地路径
            dataset_local_path = f"/root/nvme0n1/Noneq_Neural_Network/pretrained_models/{args.dataset}_{args.dataset_path}"
            
            if os.path.exists(dataset_local_path):
                print(f"[数据] ✓ 使用本地数据集: {dataset_local_path}")
                from datasets import load_from_disk
                
                full_dataset = load_from_disk(dataset_local_path)
                
                # ⚡⚡⚡ 加载三个数据集 ⚡⚡⚡
                print(f"[数据] 可用的数据集分割: {list(full_dataset.keys())}")
                
                # 1. 训练集
                train_dataset_raw = full_dataset.get('train')
                if train_dataset_raw is None:
                    raise Exception("训练集不存在")
                print(f"[数据] ✓ 训练集: {len(train_dataset_raw)} 个样本")
                
                # 2. 验证集
                val_dataset_raw = full_dataset.get('validation')
                if val_dataset_raw is not None:
                    print(f"[数据] ✓ 验证集: {len(val_dataset_raw)} 个样本")
                else:
                    print(f"[数据] ⚠️ 验证集不存在")
                
                # 3. 测试集
                test_dataset_raw = full_dataset.get('test')
                if test_dataset_raw is not None:
                    print(f"[数据] ✓ 测试集: {len(test_dataset_raw)} 个样本")
                else:
                    print(f"[数据] ⚠️ 测试集不存在")
                
                # 限制训练集样本数
                if args.num_samples > 0:
                    max_samples = min(args.num_samples, len(train_dataset_raw))
                else:
                    max_samples = min(10000, args.max_iters * args.batch_size * 2)
                
                if len(train_dataset_raw) > max_samples:
                    train_dataset_raw = train_dataset_raw.select(range(max_samples))
                    print(f"[数据] 限制训练样本数为: {max_samples}")
                
                # 限制验证集和测试集（最多 500 个样本，减少评估时间）
                max_eval_samples = 500
                
                if val_dataset_raw is not None and len(val_dataset_raw) > max_eval_samples:
                    val_dataset_raw = val_dataset_raw.select(range(max_eval_samples))
                    print(f"[数据] 限制验证集为: {max_eval_samples} 个样本")
                
                if test_dataset_raw is not None and len(test_dataset_raw) > max_eval_samples:
                    test_dataset_raw = test_dataset_raw.select(range(max_eval_samples))
                    print(f"[数据] 限制测试集为: {max_eval_samples} 个样本")
                
            else:
                # 本地不存在，从 HuggingFace 下载
                print(f"[数据] ⚠️ 本地数据集不存在: {dataset_local_path}")
                print(f"[数据] 尝试从 HuggingFace 下载...")
                
                max_samples = min(2000, args.max_iters * args.batch_size * 2)
                train_dataset_raw = load_dataset(
                    args.dataset,
                    args.dataset_path,
                    split=f'train[:{max_samples}]'
                )
                
                val_dataset_raw = load_dataset(
                    args.dataset,
                    args.dataset_path,
                    split='validation[:500]'
                )
                
                test_dataset_raw = load_dataset(
                    args.dataset,
                    args.dataset_path,
                    split='test[:500]'
                )
                
                print(f"[数据] ✓ HuggingFace 下载成功")
            
            # ⚡⚡⚡ 提取文本的通用函数 ⚡⚡⚡
            def extract_texts_from_dataset(dataset_raw, name="数据集"):
                """从原始数据集提取文本"""
                texts = []
                
                # 找到文本字段
                possible_text_fields = ['text', 'content', 'sentence', 'document']
                text_field = None
                
                if len(dataset_raw) > 0:
                    sample = dataset_raw[0]
                    
                    for field in possible_text_fields:
                        if field in sample:
                            text_field = field
                            break
                    
                    if text_field is None:
                        for key, value in sample.items():
                            if isinstance(value, str):
                                text_field = key
                                break
                
                if text_field is None:
                    raise Exception(f"{name}: 无法找到文本字段")
                
                print(f"[数据] {name} 使用文本字段: '{text_field}'")
                
                # 提取文本
                for item in dataset_raw:
                    text_content = item.get(text_field, '')
                    
                    if not isinstance(text_content, str):
                        text_content = str(text_content)
                    
                    # 过滤太短的文本
                    if len(text_content.strip()) > 50:
                        texts.append(text_content)
                
                print(f"[数据] {name} 提取到 {len(texts)} 条有效文本")
                
                return texts, text_field
            
            # ⚡⚡⚡ 提取三个数据集的文本 ⚡⚡⚡
            print(f"\n[数据] 提取文本内容...")
            
            # 1. 训练集
            train_texts, text_field = extract_texts_from_dataset(train_dataset_raw, "训练集")
            
            if len(train_texts) == 0:
                raise Exception("训练集没有有效文本")
            
            print(f"[数据] 训练集样本预览（前 100 字符）:")
            print(f"  {train_texts[0][:100]}...")
            
            # 2. 验证集
            if val_dataset_raw is not None:
                val_texts, _ = extract_texts_from_dataset(val_dataset_raw, "验证集")
            else:
                val_texts = []
                print(f"[数据] ⚠️ 跳过验证集")
            
            # 3. 测试集
            if test_dataset_raw is not None:
                test_texts, _ = extract_texts_from_dataset(test_dataset_raw, "测试集")
            else:
                test_texts = []
                print(f"[数据] ⚠️ 跳过测试集")
            
            # ⚡⚡⚡ 创建 Dataset 对象 ⚡⚡⚡
            print(f"\n[数据] Tokenization...")
            
            train_dataset = TextDataset(tokenizer, train_texts, max_length=args.max_length)
            print(f"[数据] ✓ 训练集 Tokenization 完成: {len(train_dataset)} 个样本")
            
            if len(val_texts) > 0:
                val_dataset = TextDataset(tokenizer, val_texts, max_length=args.max_length)
                print(f"[数据] ✓ 验证集 Tokenization 完成: {len(val_dataset)} 个样本")
            
            if len(test_texts) > 0:
                test_dataset = TextDataset(tokenizer, test_texts, max_length=args.max_length)
                print(f"[数据] ✓ 测试集 Tokenization 完成: {len(test_dataset)} 个样本")
            
        except Exception as e:
            print(f"\n[数据] ⚠️ WikiText 加载失败:")
            print(f"  错误: {e}")
            print(f"\n[数据] 切换到虚拟数据...")
            
            train_dataset = DummyDataset(
                vocab_size=tokenizer.vocab_size,
                seq_length=args.max_length,
                num_samples=args.num_samples if args.num_samples > 0 else 1000
            )
            val_dataset = None
            test_dataset = None
            print(f"[数据] ✓ 使用虚拟训练集（{len(train_dataset)} 样本）")
    
    else:
        print("[数据] 使用虚拟数据...")
        train_dataset = DummyDataset(
            vocab_size=tokenizer.vocab_size,
            seq_length=args.max_length,
            num_samples=args.num_samples if args.num_samples > 0 else 1000
        )
        val_dataset = None
        test_dataset = None
        print(f"[数据] ✓ 虚拟训练集（{len(train_dataset)} 样本）")
    
    # ⚡⚡⚡ 创建 DataLoader ⚡⚡⚡
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if 'cuda' in str(device) else False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if 'cuda' in str(device) else False
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if 'cuda' in str(device) else False
        )
    
    print(f"\n[数据] DataLoader 配置:")
    print(f"  • 训练集: {len(train_loader)} 个批次")
    if val_loader:
        print(f"  • 验证集: {len(val_loader)} 个批次")
    if test_loader:
        print(f"  • 测试集: {len(test_loader)} 个批次")
    print(f"  • 批次大小: {args.batch_size}")
    print(f"  • 序列长度: {args.max_length}")
    
    # ==================== 步骤 6: FDT Recorder ====================
    rec = None
    if FDT_RECORDER_AVAILABLE:
        try:
            # 找到第一个 LoRA 层
            track_key = None
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    track_key = name
                    break
            
            if track_key:
                rec = FDTRecorder(
                    model=model,
                    track_key=track_key,
                    max_elems=args.max_elems
                )
                print(f"[Recorder] ✓ 追踪参数: {track_key}")
        except Exception as e:
            print(f"[Recorder] ⚠️ 初始化失败: {e}")
            rec = None
    
    # ==================== 步骤 7: 训练循环 ====================
    print("\n" + "="*70)
    print("🚀 步骤 6: 开始训练")
    print("="*70)
    
    model.train()
    
    training_losses = []
    eval_losses = []
    best_loss = float('inf')
    global_step = 0
    
    data_iter = iter(train_loader)
    start_time = time.time()
    
    print(f"\n[训练] 开始 {args.max_iters} 步训练...")
    print("-"*70)
    
    for step in range(1, args.max_iters + 1):
        # 获取数据
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # 移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum_steps
        
        # 反向传播
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
        
        # 记录损失
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        # FDT Recorder
        if rec is not None:
            try:
                rec.update()
            except:
                pass
        
        # 日志
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train_loss = np.mean(training_losses[-args.eval_interval:])
            lr_current = scheduler.get_last_lr()[0]
            
            # ⚡⚡⚡ 验证集评估（修复 NPU 数据类型问题）⚡⚡⚡
            val_loss_avg = None
            if val_loader is not None:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    try:
                        for val_batch in val_loader:
                            # 移动到设备
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            
                            # ⚡ NPU 修复：强制 FP16
                            device_type = str(device).split(':')[0]
                            if device_type == 'npu':
                                val_batch = {
                                    k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                                    for k, v in val_batch.items()
                                }
                            
                            val_outputs = model(**val_batch)
                            val_losses.append(val_outputs.loss.item())
                    
                    except Exception as e:
                        print(f"\n  ⚠️ 验证集评估失败: {str(e)[:150]}...")
                        print(f"  将跳过验证集评估，仅使用训练损失")
                        val_losses = []
                
                model.train()
                
                if len(val_losses) > 0:
                    val_loss_avg = np.mean(val_losses)
                else:
                    print(f"  ⚠️ 验证集评估为空，跳过")
            
            # 计算 AUC (前 500 步)
            auc_str = ""
            if step >= 500:
                auc_500 = sum(training_losses[:500])
                auc_str = f", AUC(0-500)={auc_500:.2f}"
            
            # ⚡⚡⚡ 打印日志（包含验证损失）⚡⚡⚡
            log_str = f"[步骤 {step:5d}/{args.max_iters}] "
            log_str += f"训练={current_loss:.4f}, 均值={avg_train_loss:.4f}"
            
            if val_loss_avg is not None:
                log_str += f", 验证={val_loss_avg:.4f}"
            
            log_str += f", lr={lr_current:.2e}, 耗时={elapsed:.1f}s{auc_str}"
            
            print(log_str)
            
            # ⚡⚡⚡ 记录评估损失 ⚡⚡⚡
            eval_losses.append({
                'step': step,
                'train_loss': avg_train_loss,
                'val_loss': val_loss_avg,
                'current_loss': current_loss,
            })
            
            # ⚡⚡⚡ 保存最佳模型（基于验证损失）⚡⚡⚡
            # 如果有验证集，用验证损失；否则用训练损失
            metric_for_best = val_loss_avg if val_loss_avg is not None else avg_train_loss
            
            if metric_for_best < best_loss:
                best_loss = metric_for_best
                best_model_path = os.path.join(args.out_dir, "best_model")
                model.save_pretrained(best_model_path)
                
                metric_name = "验证" if val_loss_avg is not None else "训练"
                if args.verbose:
                    print(f"  → 保存最佳模型 ({metric_name}损失={best_loss:.4f})")
        
        # 定期保存检查点
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_step{step}")
            model.save_pretrained(checkpoint_path)
            print(f"  → 保存检查点: {checkpoint_path}")
    
        total_time = time.time() - start_time
    
    print("-"*70)
    print(f"[训练] ✓ 完成! 总耗时: {total_time/60:.2f} 分钟")
    print(f"[训练] 最佳损失: {best_loss:.4f}")
    
    # ⚡⚡⚡ 测试集最终评估（修复 NPU 数据类型问题）⚡⚡⚡
    test_loss_final = None
    test_loss_std = None
    
    if test_loader is not None:
        print("\n" + "="*70)
        print("🧪 最终测试集评估")
        print("="*70)
        
        # 加载最佳模型
        best_model_path = os.path.join(args.out_dir, "best_model")
        
        if os.path.exists(best_model_path):
            print(f"[测试] 加载最佳模型: {best_model_path}")
            try:
                # 使用 PEFT 的 from_pretrained
                model = PeftModel.from_pretrained(model.base_model, best_model_path)
                model = model.to(device)
            except Exception as e:
                print(f"[测试] ⚠️ 加载最佳模型失败，使用当前模型: {e}")
        else:
            print(f"[测试] ⚠️ 最佳模型不存在，使用当前模型")
        
        model.eval()
        test_losses = []
        
        print(f"[测试] 在 {len(test_loader)} 个批次上评估...")
        
        # ⚡ 获取模型的 dtype 和设备类型
        model_dtype = next(model.parameters()).dtype
        device_type = str(device).split(':')[0]
        
        print(f"[测试] 模型 dtype: {model_dtype}, 设备: {device_type}")
        
        with torch.no_grad():
            try:
                for idx, test_batch in enumerate(test_loader):
                    # 移动到设备
                    test_batch = {k: v.to(device) for k, v in test_batch.items()}
                    
                    # ⚡ NPU 修复：强制 FP16
                    if device_type == 'npu':
                        test_batch = {
                            k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                            for k, v in test_batch.items()
                        }
                    
                    test_outputs = model(**test_batch)
                    test_losses.append(test_outputs.loss.item())
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  进度: {idx+1}/{len(test_loader)}")
            
            except Exception as e:
                print(f"\n[测试] ⚠️ 测试集评估失败: {str(e)[:200]}...")
                print(f"[测试] 已评估 {len(test_losses)} 个批次，将使用这些结果")
        
        if len(test_losses) > 0:
            test_loss_final = np.mean(test_losses)
            test_loss_std = np.std(test_losses)
            
            print(f"\n[测试] 结果:")
            print(f"  • 平均损失: {test_loss_final:.4f}")
            print(f"  • 标准差: {test_loss_std:.4f}")
            print(f"  • 最小损失: {np.min(test_losses):.4f}")
            print(f"  • 最大损失: {np.max(test_losses):.4f}")
            print(f"  • 成功评估批次: {len(test_losses)}/{len(test_loader)}")
            
            # 对比训练集 vs 验证集 vs 测试集
            print(f"\n[对比] 损失对比:")
            final_train_loss = training_losses[-1]
            print(f"  • 训练集最终: {final_train_loss:.4f}")
            
            if val_loss_avg is not None:
                print(f"  • 验证集最终: {val_loss_avg:.4f}")
                gap_train_val = val_loss_avg - final_train_loss
                print(f"    → 训练-验证差距: {gap_train_val:+.4f} ({'过拟合' if gap_train_val > 0.1 else '正常'})")
            
            print(f"  • 测试集: {test_loss_final:.4f}")
            gap_train_test = test_loss_final - final_train_loss
            print(f"    → 训练-测试差距: {gap_train_test:+.4f} ({'过拟合' if gap_train_test > 0.1 else '正常'})")
        else:
            print(f"\n[测试] ⚠️ 测试集评估未成功完成任何批次")
            print(f"[测试] 可能的原因:")
            print(f"  1. NPU 算子不支持当前数据类型组合")
            print(f"  2. 内存不足")
            print(f"  3. 模型配置问题")
            print(f"[测试] 建议：使用训练损失作为参考")
        
        print("="*70)
    
    # ==================== 步骤 8: 保存结果 ====================
    print("\n" + "="*70)
    print("💾 步骤 7: 保存结果")
    print("="*70)
    
    # 保存训练损失
    losses_file = os.path.join(args.out_dir, "training_losses.npy")
    np.save(losses_file, np.array(training_losses))
    print(f"[保存] ✓ 训练损失 -> {losses_file}")
    
    # 保存评估损失（包含验证集和测试集）
    eval_file = os.path.join(args.out_dir, "eval_losses.json")
    
    # ⚡⚡⚡ 添加测试集结果 ⚡⚡⚡
    eval_summary = {
        'step_losses': eval_losses,  # 每个评估间隔的损失
        'final_metrics': {
            'best_loss': float(best_loss),
            'final_train_loss': float(training_losses[-1]),
            'test_loss': float(test_loss_final) if test_loss_final is not None else None,
            'test_loss_std': float(test_loss_std) if test_loss_std is not None else None,
        },
        'auc_metrics': {
            'auc_500': float(sum(training_losses[:500])) if len(training_losses) >= 500 else None,
        },
    }
    
    with open(eval_file, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[保存] ✓ 评估损失 -> {eval_file}")
    
    # 保存 FDT 轨迹
    if rec is not None and hasattr(rec, 'traj'):
        try:
            fdt_data = {
                'traj': rec.traj,
                'config': {
                    'track_key': rec.track_key,
                    'max_elems': args.max_elems,
                },
                'optimizer_config': {
                    'type': args.optimizer,
                    'lr': args.lr,
                },
                'initialization_config': init_info,
                'training_args': vars(args),
            }
            
            traj_file = os.path.join(args.out_dir, "fdt_trajectory.pt")
            torch.save(fdt_data, traj_file)
            print(f"[保存] ✓ FDT轨迹 -> {traj_file}")
        except Exception as e:
            print(f"[保存] ⚠️ FDT轨迹保存失败: {e}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.out_dir, "final_model")
    model.save_pretrained(final_model_path)
    print(f"[保存] ✓ 最终模型 -> {final_model_path}")
    
    # 生成训练报告
    report_file = os.path.join(args.out_dir, "training_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FDT 初始化训练报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("【配置】\n")
        f.write(f"  模型: {args.model_path}\n")
        f.write(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}\n")
        f.write(f"  优化器: {args.optimizer} (lr={args.lr})\n")
        f.write(f"  训练步数: {args.max_iters}\n")
        f.write(f"  梯度裁剪: {args.max_grad_norm}\n")
        f.write(f"  Warmup步数: {args.warmup_steps}\n\n")
        
        f.write("【初始化】\n")
        if init_info['use_fdt']:
            f.write(f"  方法: FDT 初始化\n")
            f.write(f"  预设: {init_info['preset']}\n")
            f.write(f"  目标 α: {init_info['alpha']}\n")
            if init_info['temp_ratio']:
                f.write(f"  温度比: {init_info['temp_ratio']}\n")
            if init_info['verification_passed'] is not None:
                status = "✓ 通过" if init_info['verification_passed'] else "⚠️ 未完全通过"
                f.write(f"  验证: {status}\n")
        else:
            f.write("  方法: Xavier (Baseline)\n")
        f.write("\n")
        
        f.write("【数据集】\n")
        f.write(f"  训练集: {len(train_dataset)} 个样本\n")
        if val_dataset:
            f.write(f"  验证集: {len(val_dataset)} 个样本\n")
        if test_dataset:
            f.write(f"  测试集: {len(test_dataset)} 个样本\n")
        f.write("\n")
        
        f.write("【结果】\n")
        f.write(f"  最终训练损失: {training_losses[-1]:.4f}\n")
        f.write(f"  最佳损失: {best_loss:.4f}")
        
        # 标注最佳损失的来源
        if val_loader is not None:
            f.write(" (验证集)\n")
        else:
            f.write(" (训练集)\n")
        
        # ⚡⚡⚡ 添加测试集结果 ⚡⚡⚡
        if test_loss_final is not None:
            f.write(f"  测试集损失: {test_loss_final:.4f} ± {test_loss_std:.4f}\n")
            
            gap = test_loss_final - training_losses[-1]
            f.write(f"  泛化差距: {gap:+.4f}")
            
            if gap > 0.2:
                f.write(" (严重过拟合)\n")
            elif gap > 0.1:
                f.write(" (轻微过拟合)\n")
            else:
                f.write(" (泛化良好)\n")
        
        if len(training_losses) >= 500:
            auc_500 = sum(training_losses[:500])
            f.write(f"  AUC(0-500): {auc_500:.2f}\n")
        
        f.write(f"  训练时间: {total_time/60:.2f} 分钟\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"[保存] ✓ 训练报告 -> {report_file}")
    
    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("🎉 训练完成!")
    print("="*70)
    print(f"\n输出目录: {args.out_dir}")
    print(f"最佳损失: {best_loss:.4f}")
    
    if len(training_losses) >= 500:
        auc_500 = sum(training_losses[:500])
        print(f"AUC(0-500): {auc_500:.2f}")
    
    print("\n下一步:")
    print("  1. 查看训练曲线: training_losses.npy")
    print("  2. 查看初始化报告: init_alpha_values.txt")
    print("  3. 对比不同配置的 AUC 值")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()