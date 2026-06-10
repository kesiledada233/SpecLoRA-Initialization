import os

os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'  # ← 
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

#  PEFT 
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError as e:
    print(": peft: pip install peft")
    print(f": {e}")
    PEFT_AVAILABLE = False

#  datasets 
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print(": datasets")
    DATASETS_AVAILABLE = False

print("\n" + "="*70)
print("  FDT ")
print("="*70)

# 1. 
for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"[] : {mod}")

# 2. 
FDT_INIT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 
for path in sys.path[:]:
    if 'FDT_Init' in path and path != FDT_INIT_PATH:
        sys.path.remove(path)
        print(f"[] : {path}")

# 
if FDT_INIT_PATH in sys.path:
    sys.path.remove(FDT_INIT_PATH)
sys.path.insert(0, FDT_INIT_PATH)

print(f"[] FDT : {FDT_INIT_PATH}")

# 3. 
fdt_init_file = os.path.join(FDT_INIT_PATH, 'fdt_init.py')
measure_alpha_file = os.path.join(FDT_INIT_PATH, 'measure_alpha.py')

if not os.path.exists(fdt_init_file):
    raise FileNotFoundError(f": {fdt_init_file}")
if not os.path.exists(measure_alpha_file):
    raise FileNotFoundError(f": {measure_alpha_file}")

print(f"[]  fdt_init.py ")
print(f"[]  measure_alpha.py ")

# 4. 
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
    
    # 5. 
    import fdt_init as _fdt_check
    actual_path = _fdt_check.__file__
    print(f"[] : {actual_path}")
    
    # 6. 
    import datetime
    mtime = os.path.getmtime(actual_path)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[] : {mtime_str}")
    
    # 7. 
    with open(actual_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'np.arange(1, n_freqs)' in content:
            print(f"[]  ")
        else:
            raise ImportError(
                f" \n"
                f"   : {actual_path}\n"
                f"    fdt_init.py "
            )
    
    FDT_INIT_AVAILABLE = True
    print(f"[]  FDT ")
    
except ImportError as e:
    print(f"[]  : {e}")
    raise

print("="*70 + "\n")

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
    print("[Import]  FDT-FreqAdamW v2.1 ")
except ImportError as e:
    print(f"[Import]   FDT v2.1 : {e}")
    FDT_V21_AVAILABLE = False

try:
    from FDTSOCAdamW import FDTSOCAdamW
    FDT_SOC_AVAILABLE = True
    print("[Import]  FDT-SOC AdamW ")
except ImportError as e:
    print(f"[Import]   FDT-SOC : {e}")
    FDT_SOC_AVAILABLE = False

#  FDT Recorder
try:
    from fdt import FDTRecorder
    FDT_RECORDER_AVAILABLE = True
except ImportError:
    FDT_RECORDER_AVAILABLE = False
    print("[Import]  FDTRecorder ")


class TextDataset(Dataset):
    """"""
    
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
            'labels': labels,  # Long + -100  pad
        }


class DummyDataset(Dataset):
    """"""
    
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


def get_memory_info():
    """"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f": {allocated:.2f}GB, : {reserved:.2f}GB"
    return "CUDA "


def setup_memory_optimizations():
    """"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        #  TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def count_parameters(model):
    """"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDT ")
    
    # 
    ap.add_argument("--model_path", type=str, 
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="")
    ap.add_argument("--use_flash_attention", action="store_true",
                   help=" Flash Attention")
    
    # LoRA 
    ap.add_argument("--lora_r", type=int, default=16,
                   help="LoRA ")
    ap.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA ")
    
    # 
    ap.add_argument("--batch_size", type=int, default=4,
                   help="")
    ap.add_argument("--max_length", type=int, default=512,
                   help="")
    ap.add_argument("--max_iters", type=int, default=2000,
                   help="")
    ap.add_argument("--eval_interval", type=int, default=100,
                   help="")
    ap.add_argument("--save_interval", type=int, default=500,
                   help="")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                   help="")
    
    # 
    ap.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "fdt_v21", "fdt_soc"],
                   help="")
    ap.add_argument("--lr", type=float, default=5e-5,
                   help="")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                   help="")
    ap.add_argument("--warmup_steps", type=int, default=100,
                   help="")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="")
    
    # 
    ap.add_argument("--dataset", type=str, default="wikitext",
                   choices=["wikitext", "dummy"],
                   help="")
    ap.add_argument("--dataset_path", type=str, 
                   default="/root/nvme0n1/Noneq_Neural_Network/pretrained_models/wikitext_wikitext-2-raw-v1",
                   help=" HuggingFace ")
    ap.add_argument("--num_samples", type=int, default=1000,
                   help="")
    
    # FDT 
    ap.add_argument("--use_fdt_init", action="store_true",
                   help=" FDT ")
    ap.add_argument("--fdt_alpha", type=float, default=1.2,
                   help="FDT  (0.8-1.5)")
    ap.add_argument("--fdt_temp_ratio", type=float, default=None,
                   help="FDT  ()")
    ap.add_argument("--fdt_method", type=str, default='fft',
                   choices=['fft', 'ar'],
                   help="FDT : fft ()  ar ()")
    ap.add_argument("--verify_fdt", action="store_true",
                   help="")
    ap.add_argument("--plot_spectra", action="store_true",
                   help="")
    ap.add_argument("--init_preset", type=str, default=None,
                   choices=['baseline', 'soft', 'medium', 'strong', 'temp'],
                   help=": baseline/soft/medium/strong/temp")
    
    # FDT Recorder 
    ap.add_argument("--max_elems", type=int, default=4096,
                   help="FDT Recorder ")
    
    # 
    ap.add_argument("--out_dir", type=str, default="outputs_fdt_init",
                   help="")
    ap.add_argument("--seed", type=int, default=42,
                   help="")
    ap.add_argument("--verbose", action="store_true",
                   help="")
    
    #  
    ap.add_argument("--device", type=str, default=None,
                   help=" ( 'cuda:0', 'npu:2', 'cpu'None )")

    return ap.parse_args()


def main():
    # 
    if not PEFT_AVAILABLE:
        print(": PEFT: pip install peft")
        return
    
    args = get_args()
    
    # 
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 
    config_file = os.path.join(args.out_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"[] : {config_file}")
    
    # 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.device:
        # 
        device = torch.device(args.device)
        device_type = args.device.split(':')[0]  #  (npu/cuda/cpu)
        
        print(f"[] : {device}")
        
        #  NPU
        if device_type == 'npu':
            try:
                import torch_npu
                torch_npu.npu.set_device(device)
                print(f"[]  NPU ")
                
                # NPU
                torch_npu.npu.manual_seed_all(args.seed)
            except ImportError:
                print(f"[]   torch_npu ")
                print(": pip install torch-npu ()")
                return
            except Exception as e:
                print(f"[]  NPU : {e}")
                return
        
        elif device_type == 'cuda':
            if not torch.cuda.is_available():
                print(f"[]  CUDA  CPU")
                device = torch.device('cpu')
    
    else:
        # 
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[] : CUDA (GPU)")
        else:
            device = torch.device('cpu')
            print(f"[] : CPU")
    
    # 
    device_type = str(device).split(':')[0]
    
    if device_type == 'cuda':
        setup_memory_optimizations()
        print(f"[] {get_memory_info()}")
    
    elif device_type == 'npu':
        try:
            import torch_npu
            allocated = torch_npu.npu.memory_allocated(device) / 1024**3
            reserved = torch_npu.npu.memory_reserved(device) / 1024**3
            print(f"[NPU ] : {allocated:.2f}GB, : {reserved:.2f}GB")
        except Exception as e:
            print(f"[NPU ] : {e}")
    
    print("\n" + "="*70)
    print("  1: ")
    print("="*70)
    
    #  NPU  Fused Attention 
    device_type = str(device).split(':')[0]
    if device_type == 'npu':
        os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'
        os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'
        print("[]   NPU Fused Attention")

    print(f"[] : {args.model_path}")
    
    try:
        #  tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[]  Tokenizer , vocab_size={tokenizer.vocab_size}")
        
        # 
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if args.use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("[]  Flash Attention 2")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs
        )
        
        total_params, _ = count_parameters(model)
        print(f"[]  , : {total_params/1e9:.2f}B")
        
    except Exception as e:
        print(f"[]  : {e}")
        print("\n...")
        
        # 
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        total_params, _ = count_parameters(model)
        print(f"[]   {model_name}, : {total_params/1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"[] {get_memory_info()}")
    
    print("\n" + "="*70)
    print("  2:  LoRA")
    print("="*70)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"[LoRA] :")
    print(f"  • r = {args.lora_r}")
    print(f"  • alpha = {args.lora_alpha}")
    print(f"  • dropout = {args.lora_dropout}")
    print(f"  • target_modules = {args.lora_target_modules}")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    device_type = str(device).split(':')[0]
    if device_type == 'npu':
        model.half()  # NPU  FP16 

    
    total_params, trainable_params = count_parameters(model)
    print(f"[LoRA]  LoRA ")
    print(f"  • : {total_params:,}")
    print(f"  • : {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    
    if torch.cuda.is_available():
        print(f"[] {get_memory_info()}")
    
    print("\n" + "="*70)
    print("  3: FDT ")
    print("="*70)
    
    # 
    if args.init_preset:
        preset_configs = {
            'baseline': {'use_fdt': False, 'alpha': None, 'temp': None, 
                        'name': 'Xavier Baseline ( α≈0)'},
            'soft': {'use_fdt': True, 'alpha': 0.8, 'temp': None, 
                    'name': 'FDT-Soft (α=0.8, )'},
            'medium': {'use_fdt': True, 'alpha': 1.2, 'temp': None, 
                      'name': 'FDT-Medium (α=1.2, )'},
            'strong': {'use_fdt': True, 'alpha': 1.5, 'temp': None, 
                      'name': 'FDT-Strong (α=1.5, )'},
            'temp': {'use_fdt': True, 'alpha': 1.2, 'temp': 1.5, 
                    'name': 'FDT-Temp (α=1.2, τ=1.5)'},
        }
        
        config = preset_configs[args.init_preset]
        print(f"\n[] {config['name']}")
        
        if config['use_fdt']:
            args.use_fdt_init = True
            args.fdt_alpha = config['alpha']
            args.fdt_temp_ratio = config['temp']
    
    # 
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
            print("\n : FDT !")
            print(":")
            print("  - deepseek/initializers/fdt_init.py")
            print("  - deepseek/initializers/measure_alpha.py")
            print("\n Xavier ...")
            args.use_fdt_init = False
        else:
            print(f"\n[1/3]  FDT ...")
            print(f"  : α={args.fdt_alpha:.2f}", end='')
            if args.fdt_temp_ratio:
                print(f", τ={args.fdt_temp_ratio:.2f}", end='')
            print(f", ={args.fdt_method}")
            
            init_start_time = time.time()
            
            apply_fdt_to_lora(
                model,
                alpha=args.fdt_alpha,
                temp_ratio=args.fdt_temp_ratio,
                method=args.fdt_method,
                verbose=args.verbose
            )
            
            init_duration = time.time() - init_start_time
            print(f"    {init_duration:.2f} ")
            
            init_info['alpha'] = args.fdt_alpha
            init_info['temp_ratio'] = args.fdt_temp_ratio
            init_info['method'] = args.fdt_method
            
            # 
            if args.verify_fdt:
                print(f"\n[2/3] ...")
                
                verify_success = verify_fdt_initialization(
                    model,
                    target_alpha=args.fdt_alpha,
                    tolerance=0.15,  #  15% 
                    verbose=True
                )
                
                init_info['verification_passed'] = verify_success
                
                if not verify_success:
                    print("\n  : ")
            
            # 
            if args.plot_spectra:
                print(f"\n[3/3] ...")
                
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
                
                #  α 
                alpha_file = os.path.join(args.out_dir, 'init_alpha_values.txt')
                with open(alpha_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("FDT \n")
                    f.write("="*60 + "\n\n")
                    
                    f.write(":\n")
                    f.write(f"   α: {args.fdt_alpha:.3f}\n")
                    if args.fdt_temp_ratio:
                        f.write(f"   τ: {args.fdt_temp_ratio:.3f}\n")
                    f.write(f"  : {args.fdt_method}\n\n")
                    
                    f.write(":\n")
                    for name, alpha in alphas.items():
                        error = abs(alpha - args.fdt_alpha)
                        status = "" if error < 0.15 else ""
                        f.write(f"  {status} {name}: α={alpha:.3f} (={error:.3f})\n")
                    
                    valid_alphas = [a for a in alphas.values() if not np.isnan(a)]
                    if valid_alphas:
                        f.write(f"\n:\n")
                        f.write(f"  : {np.mean(valid_alphas):.3f}\n")
                        f.write(f"  : {np.std(valid_alphas):.3f}\n")
                        f.write(f"  : [{np.min(valid_alphas):.3f}, {np.max(valid_alphas):.3f}]\n")
                        
                        avg_error = np.mean([abs(a - args.fdt_alpha) for a in valid_alphas])
                        f.write(f"  : {avg_error:.3f}\n")
                    
                    f.write("="*60 + "\n")
                
                print(f"   : {alpha_file}")
    
    else:
        print("\n[]  Xavier Baseline")
    
    #   JSON  + numpy 
    init_info_file = os.path.join(args.out_dir, 'init_info.json')
    
    #  numpy  numpy >= 1.20
    def convert_to_json_serializable(obj):
        """ numpy  Python  numpy 1.20+"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        #   np.bool np.bool_
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
            # 
            try:
                return str(obj)
            except:
                return None
    
    # 
    try:
        init_info_serializable = convert_to_json_serializable(init_info)
        
        with open(init_info_file, 'w') as f:
            json.dump(init_info_serializable, f, indent=2)
        print(f"[]  : {init_info_file}")
    except Exception as e:
        print(f"[]  : {e}")
        # 
        try:
            simple_info = {
                'use_fdt': bool(init_info.get('use_fdt', False)),
                'preset': str(init_info.get('preset', 'unknown')),
                'alpha': float(init_info.get('alpha')) if init_info.get('alpha') is not None else None,
            }
            with open(init_info_file, 'w') as f:
                json.dump(simple_info, f, indent=2)
            print(f"[]  ")
        except:
            pass
    
    print("="*70)
    
    print("\n" + "="*70)
    print("  4: ")
    print("="*70)
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"[] AdamW (lr={args.lr}, wd={args.weight_decay})")
    
    elif args.optimizer == "fdt_v21":
        if not FDT_V21_AVAILABLE:
            print("[]  FDT v2.1  AdamW")
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
            print(f"[] FDT-FreqAdamW v2.1 (lr={args.lr})")
    
    elif args.optimizer == "fdt_soc":
        if not FDT_SOC_AVAILABLE:
            print("[]  FDT-SOC  AdamW")
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
            print(f"[] FDT-SOC AdamW (lr={args.lr})")
    
    # 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )
    print(f"[] Linear warmup ({args.warmup_steps} steps) + decay")
    
    print("\n" + "="*70)
    print("  5: ")
    print("="*70)
    
    #   
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if args.dataset == "wikitext" and DATASETS_AVAILABLE:
        try:
            # 
            dataset_local_path = f"/root/nvme0n1/Noneq_Neural_Network/pretrained_models/{args.dataset}_{args.dataset_path}"
            
            if os.path.exists(dataset_local_path):
                print(f"[]  : {dataset_local_path}")
                from datasets import load_from_disk
                
                full_dataset = load_from_disk(dataset_local_path)
                
                #   
                print(f"[] : {list(full_dataset.keys())}")
                
                # 1. 
                train_dataset_raw = full_dataset.get('train')
                if train_dataset_raw is None:
                    raise Exception("")
                print(f"[]  : {len(train_dataset_raw)} ")
                
                # 2. 
                val_dataset_raw = full_dataset.get('validation')
                if val_dataset_raw is not None:
                    print(f"[]  : {len(val_dataset_raw)} ")
                else:
                    print(f"[]  ")
                
                # 3. 
                test_dataset_raw = full_dataset.get('test')
                if test_dataset_raw is not None:
                    print(f"[]  : {len(test_dataset_raw)} ")
                else:
                    print(f"[]  ")
                
                # 
                if args.num_samples > 0:
                    max_samples = min(args.num_samples, len(train_dataset_raw))
                else:
                    max_samples = min(10000, args.max_iters * args.batch_size * 2)
                
                if len(train_dataset_raw) > max_samples:
                    train_dataset_raw = train_dataset_raw.select(range(max_samples))
                    print(f"[] : {max_samples}")
                
                #  500 
                max_eval_samples = 500
                
                if val_dataset_raw is not None and len(val_dataset_raw) > max_eval_samples:
                    val_dataset_raw = val_dataset_raw.select(range(max_eval_samples))
                    print(f"[] : {max_eval_samples} ")
                
                if test_dataset_raw is not None and len(test_dataset_raw) > max_eval_samples:
                    test_dataset_raw = test_dataset_raw.select(range(max_eval_samples))
                    print(f"[] : {max_eval_samples} ")
                
            else:
                #  HuggingFace 
                print(f"[]  : {dataset_local_path}")
                print(f"[]  HuggingFace ...")
                
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
                
                print(f"[]  HuggingFace ")
            
            #   
            def extract_texts_from_dataset(dataset_raw, name=""):
                """"""
                texts = []
                
                # 
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
                    raise Exception(f"{name}: ")
                
                print(f"[] {name} : '{text_field}'")
                
                # 
                for item in dataset_raw:
                    text_content = item.get(text_field, '')
                    
                    if not isinstance(text_content, str):
                        text_content = str(text_content)
                    
                    # 
                    if len(text_content.strip()) > 50:
                        texts.append(text_content)
                
                print(f"[] {name}  {len(texts)} ")
                
                return texts, text_field
            
            #   
            print(f"\n[] ...")
            
            # 1. 
            train_texts, text_field = extract_texts_from_dataset(train_dataset_raw, "")
            
            if len(train_texts) == 0:
                raise Exception("")
            
            print(f"[]  100 :")
            print(f"  {train_texts[0][:100]}...")
            
            # 2. 
            if val_dataset_raw is not None:
                val_texts, _ = extract_texts_from_dataset(val_dataset_raw, "")
            else:
                val_texts = []
                print(f"[]  ")
            
            # 3. 
            if test_dataset_raw is not None:
                test_texts, _ = extract_texts_from_dataset(test_dataset_raw, "")
            else:
                test_texts = []
                print(f"[]  ")
            
            #   Dataset  
            print(f"\n[] Tokenization...")
            
            train_dataset = TextDataset(tokenizer, train_texts, max_length=args.max_length)
            print(f"[]   Tokenization : {len(train_dataset)} ")
            
            if len(val_texts) > 0:
                val_dataset = TextDataset(tokenizer, val_texts, max_length=args.max_length)
                print(f"[]   Tokenization : {len(val_dataset)} ")
            
            if len(test_texts) > 0:
                test_dataset = TextDataset(tokenizer, test_texts, max_length=args.max_length)
                print(f"[]   Tokenization : {len(test_dataset)} ")
            
        except Exception as e:
            print(f"\n[]  WikiText :")
            print(f"  : {e}")
            print(f"\n[] ...")
            
            train_dataset = DummyDataset(
                vocab_size=tokenizer.vocab_size,
                seq_length=args.max_length,
                num_samples=args.num_samples if args.num_samples > 0 else 1000
            )
            val_dataset = None
            test_dataset = None
            print(f"[]  {len(train_dataset)} ")
    
    else:
        print("[] ...")
        train_dataset = DummyDataset(
            vocab_size=tokenizer.vocab_size,
            seq_length=args.max_length,
            num_samples=args.num_samples if args.num_samples > 0 else 1000
        )
        val_dataset = None
        test_dataset = None
        print(f"[]  {len(train_dataset)} ")
    
    #   DataLoader 
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
    
    print(f"\n[] DataLoader :")
    print(f"  • : {len(train_loader)} ")
    if val_loader:
        print(f"  • : {len(val_loader)} ")
    if test_loader:
        print(f"  • : {len(test_loader)} ")
    print(f"  • : {args.batch_size}")
    print(f"  • : {args.max_length}")
    
    rec = None
    if FDT_RECORDER_AVAILABLE:
        try:
            #  LoRA 
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
                print(f"[Recorder]  : {track_key}")
        except Exception as e:
            print(f"[Recorder]  : {e}")
            rec = None
    
    print("\n" + "="*70)
    print("  6: ")
    print("="*70)
    
    model.train()
    
    training_losses = []
    eval_losses = []
    best_loss = float('inf')
    global_step = 0
    
    data_iter = iter(train_loader)
    start_time = time.time()
    
    print(f"\n[]  {args.max_iters} ...")
    print("-"*70)
    
    for step in range(1, args.max_iters + 1):
        # 
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # 
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum_steps
        
        # 
        loss.backward()
        
        # 
        if step % args.grad_accum_steps == 0:
            # 
            torch.nn.utils.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
            
            # 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        # 
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        # FDT Recorder
        if rec is not None:
            try:
                rec.update()
            except:
                pass
        
        # 
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train_loss = np.mean(training_losses[-args.eval_interval:])
            lr_current = scheduler.get_last_lr()[0]
            
            #   NPU 
            val_loss_avg = None
            if val_loader is not None:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    try:
                        for val_batch in val_loader:
                            # 
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            
                            #  NPU  FP16
                            device_type = str(device).split(':')[0]
                            if device_type == 'npu':
                                val_batch = {
                                    k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                                    for k, v in val_batch.items()
                                }
                            
                            val_outputs = model(**val_batch)
                            val_losses.append(val_outputs.loss.item())
                    
                    except Exception as e:
                        print(f"\n   : {str(e)[:150]}...")
                        print(f"  ")
                        val_losses = []
                
                model.train()
                
                if len(val_losses) > 0:
                    val_loss_avg = np.mean(val_losses)
                else:
                    print(f"   ")
            
            #  AUC ( 500 )
            auc_str = ""
            if step >= 500:
                auc_500 = sum(training_losses[:500])
                auc_str = f", AUC(0-500)={auc_500:.2f}"
            
            #  
            log_str = f"[ {step:5d}/{args.max_iters}] "
            log_str += f"={current_loss:.4f}, ={avg_train_loss:.4f}"
            
            if val_loss_avg is not None:
                log_str += f", ={val_loss_avg:.4f}"
            
            log_str += f", lr={lr_current:.2e}, ={elapsed:.1f}s{auc_str}"
            
            print(log_str)
            
            #   
            eval_losses.append({
                'step': step,
                'train_loss': avg_train_loss,
                'val_loss': val_loss_avg,
                'current_loss': current_loss,
            })
            
            #  
            # 
            metric_for_best = val_loss_avg if val_loss_avg is not None else avg_train_loss
            
            if metric_for_best < best_loss:
                best_loss = metric_for_best
                best_model_path = os.path.join(args.out_dir, "best_model")
                model.save_pretrained(best_model_path)
                
                metric_name = "" if val_loss_avg is not None else ""
                if args.verbose:
                    print(f"  →  ({metric_name}={best_loss:.4f})")
        
        # 
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_step{step}")
            model.save_pretrained(checkpoint_path)
            print(f"  → : {checkpoint_path}")
    
        total_time = time.time() - start_time
    
    print("-"*70)
    print(f"[]  ! : {total_time/60:.2f} ")
    print(f"[] : {best_loss:.4f}")
    
    #   NPU 
    test_loss_final = None
    test_loss_std = None
    
    if test_loader is not None:
        print("\n" + "="*70)
        print(" ")
        print("="*70)
        
        # 
        best_model_path = os.path.join(args.out_dir, "best_model")
        
        if os.path.exists(best_model_path):
            print(f"[] : {best_model_path}")
            try:
                #  PEFT  from_pretrained
                model = PeftModel.from_pretrained(model.base_model, best_model_path)
                model = model.to(device)
            except Exception as e:
                print(f"[]  : {e}")
        else:
            print(f"[]  ")
        
        model.eval()
        test_losses = []
        
        print(f"[]  {len(test_loader)} ...")
        
        #   dtype 
        model_dtype = next(model.parameters()).dtype
        device_type = str(device).split(':')[0]
        
        print(f"[]  dtype: {model_dtype}, : {device_type}")
        
        with torch.no_grad():
            try:
                for idx, test_batch in enumerate(test_loader):
                    # 
                    test_batch = {k: v.to(device) for k, v in test_batch.items()}
                    
                    #  NPU  FP16
                    if device_type == 'npu':
                        test_batch = {
                            k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                            for k, v in test_batch.items()
                        }
                    
                    test_outputs = model(**test_batch)
                    test_losses.append(test_outputs.loss.item())
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  : {idx+1}/{len(test_loader)}")
            
            except Exception as e:
                print(f"\n[]  : {str(e)[:200]}...")
                print(f"[]  {len(test_losses)} ")
        
        if len(test_losses) > 0:
            test_loss_final = np.mean(test_losses)
            test_loss_std = np.std(test_losses)
            
            print(f"\n[] :")
            print(f"  • : {test_loss_final:.4f}")
            print(f"  • : {test_loss_std:.4f}")
            print(f"  • : {np.min(test_losses):.4f}")
            print(f"  • : {np.max(test_losses):.4f}")
            print(f"  • : {len(test_losses)}/{len(test_loader)}")
            
            #  vs  vs 
            print(f"\n[] :")
            final_train_loss = training_losses[-1]
            print(f"  • : {final_train_loss:.4f}")
            
            if val_loss_avg is not None:
                print(f"  • : {val_loss_avg:.4f}")
                gap_train_val = val_loss_avg - final_train_loss
                print(f"    → -: {gap_train_val:+.4f} ({'' if gap_train_val > 0.1 else ''})")
            
            print(f"  • : {test_loss_final:.4f}")
            gap_train_test = test_loss_final - final_train_loss
            print(f"    → -: {gap_train_test:+.4f} ({'' if gap_train_test > 0.1 else ''})")
        else:
            print(f"\n[]  ")
            print(f"[] :")
            print(f"  1. NPU ")
            print(f"  2. ")
            print(f"  3. ")
            print(f"[] ")
        
        print("="*70)
    
    print("\n" + "="*70)
    print("  7: ")
    print("="*70)
    
    # 
    losses_file = os.path.join(args.out_dir, "training_losses.npy")
    np.save(losses_file, np.array(training_losses))
    print(f"[]   -> {losses_file}")
    
    # 
    eval_file = os.path.join(args.out_dir, "eval_losses.json")
    
    #   
    eval_summary = {
        'step_losses': eval_losses,  # 
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
    print(f"[]   -> {eval_file}")
    
    #  FDT 
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
            print(f"[]  FDT -> {traj_file}")
        except Exception as e:
            print(f"[]  FDT: {e}")
    
    # 
    final_model_path = os.path.join(args.out_dir, "final_model")
    model.save_pretrained(final_model_path)
    print(f"[]   -> {final_model_path}")
    
    # 
    report_file = os.path.join(args.out_dir, "training_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FDT \n")
        f.write("="*70 + "\n\n")
        
        f.write("\n")
        f.write(f"  : {args.model_path}\n")
        f.write(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}\n")
        f.write(f"  : {args.optimizer} (lr={args.lr})\n")
        f.write(f"  : {args.max_iters}\n")
        f.write(f"  : {args.max_grad_norm}\n")
        f.write(f"  Warmup: {args.warmup_steps}\n\n")
        
        f.write("\n")
        if init_info['use_fdt']:
            f.write(f"  : FDT \n")
            f.write(f"  : {init_info['preset']}\n")
            f.write(f"   α: {init_info['alpha']}\n")
            if init_info['temp_ratio']:
                f.write(f"  : {init_info['temp_ratio']}\n")
            if init_info['verification_passed'] is not None:
                status = " " if init_info['verification_passed'] else " "
                f.write(f"  : {status}\n")
        else:
            f.write("  : Xavier (Baseline)\n")
        f.write("\n")
        
        f.write("\n")
        f.write(f"  : {len(train_dataset)} \n")
        if val_dataset:
            f.write(f"  : {len(val_dataset)} \n")
        if test_dataset:
            f.write(f"  : {len(test_dataset)} \n")
        f.write("\n")
        
        f.write("\n")
        f.write(f"  : {training_losses[-1]:.4f}\n")
        f.write(f"  : {best_loss:.4f}")
        
        # 
        if val_loader is not None:
            f.write(" ()\n")
        else:
            f.write(" ()\n")
        
        #   
        if test_loss_final is not None:
            f.write(f"  : {test_loss_final:.4f} ± {test_loss_std:.4f}\n")
            
            gap = test_loss_final - training_losses[-1]
            f.write(f"  : {gap:+.4f}")
            
            if gap > 0.2:
                f.write(" ()\n")
            elif gap > 0.1:
                f.write(" ()\n")
            else:
                f.write(" ()\n")
        
        if len(training_losses) >= 500:
            auc_500 = sum(training_losses[:500])
            f.write(f"  AUC(0-500): {auc_500:.2f}\n")
        
        f.write(f"  : {total_time/60:.2f} \n\n")
        
        f.write("="*70 + "\n")
    
    print(f"[]   -> {report_file}")
    
    print("\n" + "="*70)
    print(" !")
    print("="*70)
    print(f"\n: {args.out_dir}")
    print(f": {best_loss:.4f}")
    
    if len(training_losses) >= 500:
        auc_500 = sum(training_losses[:500])
        print(f"AUC(0-500): {auc_500:.2f}")
    
    print("\n:")
    print("  1. : training_losses.npy")
    print("  2. : init_alpha_values.txt")
    print("  3.  AUC ")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()