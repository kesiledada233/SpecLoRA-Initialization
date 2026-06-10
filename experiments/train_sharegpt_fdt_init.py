import os

os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'  # ← 
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

# PEFT
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
    exit(1)

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

def get_args():
    ap = argparse.ArgumentParser()
    
    # 
    ap.add_argument("--model_name", type=str, 
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="openPangu ")
    
    ap.add_argument("--use_flash_attention", action="store_true",
                   help=" Flash Attention")

    ap.add_argument("--block_size", type=int, default=256,
                   help="")
    ap.add_argument("--batch_size", type=int, default=2,
                   help="")
    ap.add_argument("--max_iters", type=int, default=2500,
                   help="")
    ap.add_argument("--eval_interval", type=int, default=100,
                   help="")
    ap.add_argument("--seed", type=int, default=1107,
                   help="")
    ap.add_argument("--max_train_samples", type=int, default=10000,
                   help="")
    ap.add_argument("--max_val_samples", type=int, default=500,
                   help="")
    ap.add_argument("--device", type=str, 
                   default="npu:0" ,
                   help=" (npu:0, npu:1, cuda:0, etc.)")
    
    ap.add_argument("--save_interval", type=int, default=500,
                   help="")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                   help="")
    
    # LoRA 
    ap.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    ap.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, nargs="+", 
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA")
    

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
    ap.add_argument("--dataset", type=str, default="ShareGPT",
                   help=" --data_files ")
    ap.add_argument("--dataset_config", type=str, default="computer_en",
                   help="")
    ap.add_argument("--dataset_split", type=str, default="train",
                   help=" split")
    ap.add_argument("--data_files", type=str, nargs="+", default=["computer_en_26k.jsonl"],
                   help="json/jsonl/csv ")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                   help="0-1 JSONL ")
    ap.add_argument("--val_max_steps", type=int, default=100,
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
    ap.add_argument("--verbose", action="store_true",
                   help="")
    

    args = ap.parse_args()
    # 
    setattr(args, 'model_path', args.model_name)
    return args

class TextDataset:
    def __init__(self, tokenizer, dataset, block_size, device):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device
        
        # 
        dataset = dataset.filter(lambda x: isinstance(x.get('text', ''), str) and len(x['text'].strip()) > 0)
        
        # 
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
        
        print(f"[Dataset] : {len(self.data)}")
        
        # 
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"[Dataset] : {list(sample.keys())}")
            print(f"[Dataset] ID: {len(sample['input_ids'])}")
    
    def get_batch(self, batch_size):
        # 
        indices = torch.randint(0, len(self.data), (batch_size,))
        batch = [self.data[int(i)] for i in indices]
        
        # 
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch])
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch])
        
        # input_ids 
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

def setup_memory_optimizations(model):
    """"""
    if hasattr(model, 'config'):
        model.config.use_cache = False
        print("[Memory] KV")
    return model

def get_memory_info(device_str):
    """"""
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

def linearize_sharegpt(example):
    """
     ShareGPT  conversations/messages  text
    
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
        #  text 
        if 'text' in example:
            return {'text': example['text']}
        if 'content' in example:
            return {'text': example['content']}
        return {'text': ''}

    parts = []
    #  system 
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
            # 
            continue
        else:
            parts.append(str(content))
    text = "\n".join(parts).strip()
    return {'text': text}

def main():
    if not PEFT_AVAILABLE:
        print(": PEFT: pip install peft")
        return
    
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*70)
    print("[1/6] ...")
    print("="*70)
    
    #  tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # token OpenPangu
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Tokenizer] Pad token  EOS: {tokenizer.pad_token}")
    else:
        print(f"[Tokenizer] Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # 
    if 'cuda' in args.device:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif 'npu' in args.device:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"[Model]  {dtype} ...")
    #  kwargs 
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': dtype,
    }
    
    if args.use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("[]  Flash Attention 2")

    #  from_pretrained kwargs
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,   #  get_args 
        **model_kwargs
    )

    # 
    model = setup_memory_optimizations(model)
    
    # 
    model = model.to(args.device)

    if 'npu' in args.device:
        model.half()
    
    print(f"[Model] : {model.num_parameters():,}")
    print(f"[Model] : {next(model.parameters()).device}")
    print(f"[Model] : {next(model.parameters()).dtype}")
    
    # 
    model_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    model_memory_gb = model_params * bytes_per_param / 1e9
    print(f"[Model] : ~{model_memory_gb:.2f} GB")
    
    print("\n[2/6] LoRA...")
    print("="*70)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    #   LoRA  
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100 * trainable_params / total_params
    
    print(f"\n[LoRA ]")
    print(f"  : {total_params:,}")
    print(f"  : {trainable_params:,}")
    print(f"  : {trainable_ratio:.3f}%")
    
    if trainable_ratio > 5.0:
        print(f"\n{'='*70}")
        print("  :  LoRA  < 2%")
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
            print(" LoRA ...")
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' not in name.lower():
                    param.requires_grad = False
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_ratio = 100 * trainable_params / total_params
            print(f": {trainable_ratio:.3f}%")
    

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
    
    # elif args.optimizer == "fdt_v21":
    #     if not FDT_V21_AVAILABLE:
    #         print("[]  FDT v2.1  AdamW")
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
    #         print(f"[] FDT-FreqAdamW v2.1 (lr={args.lr})")
    
    # elif args.optimizer == "fdt_soc":
    #     if not FDT_SOC_AVAILABLE:
    #         print("[]  FDT-SOC  AdamW")
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
    #         print(f"[] FDT-SOC AdamW (lr={args.lr})")
    
    # 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )
    print(f"[] Linear warmup ({args.warmup_steps} steps) + decay")
    
    print("="*70)
    
    print("\n[5/6]  (ShareGPT)...")
    print("="*70)

    def _join_turns(turns, sep="\n"):
        #  [{role,text}/{human,assistant}] 
        parts = []
        for t in turns:
            # 
            if "role" in t and "content" in t:
                role = t["role"]
                text = t.get("content") or ""
            else:
                # ShareGPT human/assistant
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
            # 
            prefix = "User:" if role == "user" else ("Assistant:" if role == "assistant" else "")
            parts.append(f"{prefix} {text}" if prefix else text)
        return sep.join(parts).strip()

    def _linearize_example(ex):
        #  text conversations/messages conversation
        if isinstance(ex.get("text"), str) and ex["text"].strip():
            return ex["text"].strip()

        turns = None
        if isinstance(ex.get("conversations"), list):
            turns = ex["conversations"]
        elif isinstance(ex.get("messages"), list):
            turns = ex["messages"]
        elif isinstance(ex.get("conversation"), list):  # 
            turns = ex["conversation"]

        if isinstance(turns, list) and turns:
            return _join_turns(turns)

        # instruction/output 
        if ("instruction" in ex) and ("output" in ex):
            instr = str(ex.get("instruction") or "").strip()
            inp = str(ex.get("input") or "").strip()
            outp = str(ex.get("output") or "").strip()
            body = "\n".join(s for s in [instr, inp, outp] if s).strip()
            return body

        return ""

    def _load_pair():
        if args.data_files:
            print(f"[Dataset]  pyarrow: {args.data_files}")

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
                                print(f"[] {fp}: {e}")
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
                                        print(f"[]  {fp}:{i}: {e}")

            def gen():
                dropped = 0
                kept = 0
                for ex in iter_examples(args.data_files):
                    #  Arrow 
                    ex.pop("category", None)

                    txt = _linearize_example(ex)
                    if not txt:
                        dropped += 1
                        continue
                    kept += 1
                    yield {"text": txt}
                print(f"[Dataset] :  {kept}  {dropped} ")

            base = Dataset.from_generator(gen)
            split = base.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
            return split["train"], split["test"]
        else:
            name = args.dataset_config if (args.dataset_config not in [None, "", "none", "None"]) else None
            base = load_dataset(args.dataset, name, split=args.dataset_split)
            split = base.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
            return split["train"], split["test"]

    raw_train, raw_val = _load_pair()

    #  text
    def ensure_text(ds):
        cols = ds.column_names
        if 'text' in cols:
            return ds
        if 'conversations' in cols or 'messages' in cols:
            print("[Dataset]  'text'")
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
        raise ValueError(f": {cols}")

    raw_train = ensure_text(raw_train)
    raw_val = ensure_text(raw_val)

    # 
    raw_train = raw_train.filter(lambda x: isinstance(x.get('text',''), str) and len(x['text'].strip())>0)
    raw_val = raw_val.filter(lambda x: isinstance(x.get('text',''), str) and len(x['text'].strip())>0)

    if args.max_train_samples is not None and len(raw_train) > args.max_train_samples:
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.max_train_samples))
        print(f"[Subset]  {len(raw_train)}")
    if args.max_val_samples is not None and len(raw_val) > args.max_val_samples:
        raw_val = raw_val.shuffle(seed=args.seed).select(range(args.max_val_samples))
        print(f"[Subset]  {len(raw_val)}")

    print(f"[Dataset] : {len(raw_train)} | : {len(raw_val)}")

    train_dataset = TextDataset(tokenizer, raw_train, args.block_size, args.device)
    val_dataset = TextDataset(tokenizer, raw_val, args.block_size, args.device)

    
    print("\n[6/6] ...")
    print("="*70)
    
    model.train()
    t0 = time.time()
    t1 = time.time()
    # losses = []
    training_losses = []
    val_history = []
    global_step = 0
    
    # 
    allocated, reserved = get_memory_info(args.device)
    if allocated is not None:
        print(f"[Memory]  ({args.device}) - : {allocated:.2f}GB, : {reserved:.2f}GB")
    
    if args.optimizer == "fdt_soc" and hasattr(optimizer, 'fft_device'):
        fft_device = optimizer.fft_device
        fft_allocated, fft_reserved = get_memory_info(fft_device)
        if fft_allocated is not None:
            print(f"[Memory] FFT  ({fft_device}) - : {fft_allocated:.2f}GB, : {fft_reserved:.2f}GB")
    
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

        # 
        if step % args.grad_accum_steps == 0:
            # 
            torch.nn.utils.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
            
            # 
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
            
            info_str = f"[ {step:5d}/{args.max_iters}] ={current_loss:.4f} (: {avg_loss:.4f})"
            
            info_str += f" | ={dt:.1f}s"

            #  AUC ( 500 )
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
            print(f"[] step {step} | val_loss={val_loss:.4f}")

            val_history.append((step, float(val_loss)))
            val_csv = os.path.join(args.out_dir, "val_losses.csv")
            write_header = not os.path.exists(val_csv)
            try:
                with open(val_csv, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("step,val_loss\n")
                    f.write(f"{step},{val_loss:.6f}\n")
            except Exception as e:
                print(f" [] CSV: {e}")

        # 
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_step{step}")
            model.save_pretrained(checkpoint_path)
            print(f"  → : {checkpoint_path}")

            t1 = time.time()

    total_time = time.time() - t0

    print("-"*70)
    print(f"[]  ! : {total_time/60:.2f} ")
    
    print("\n[] ...")
    print("="*70)
    
    lora_dir = os.path.join(args.out_dir, "lora_adapter")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    
    #   
    loss_file = os.path.join(args.out_dir, "training_losses.npy")

    # 
    print(f"[] : {len(training_losses)}")

    if len(training_losses) == 0:
        print(f"[]  : !")
        print(f"[] :")
        print(f"  1. ")
        print(f"  2. loss.item() ")
        print(f"  3. training_losses.append() ")
    else:
        np.save(loss_file, np.array(training_losses))  #  
        print(f"[]   -> {loss_file}")
        print(f"[]   5: {training_losses[:5]}")
        print(f"[]   5: {training_losses[-5:]}")

    # 
    if os.path.exists(loss_file):
        saved_losses = np.load(loss_file)
        print(f"[]   {len(saved_losses)} ")
    else:
        print(f"[]  ")

    try:
        val_npy = os.path.join(args.out_dir, "val_losses.npy")
        np.save(val_npy, np.array([v for _, v in val_history], dtype=np.float32))
        print(f"[]   -> {val_npy}")
        print(f"[]  CSV -> {os.path.join(args.out_dir, 'val_losses.csv')}")
    except Exception as e:
        print(f"  [] : {e}")

    print(f"[]  LoRA -> {lora_dir}")
    
    print("\n" + "="*70)
    print(" [] !")
    print("="*70)

if __name__ == "__main__":
    main()