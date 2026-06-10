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
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    print(": datasets")
    DATASETS_AVAILABLE = False

print("\n" + "="*70)
print("  FDT ")
print("="*70)

for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]

FDT_INIT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    print("[]  FDT ")
    
except ImportError as e:
    print(f"[]  : {e}")
    raise

print("="*70 + "\n")

DATASET_PATHS = {
    'gsm8k': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/gsm8k',
    'cmmlu': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/cmmlu/processed',
    'sharegpt': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/sharegpt',
    'mbpp': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/mbpp/processed',
}

class BenchmarkDataset(Dataset):
    """"""
    
    def __init__(self, tokenizer, examples, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"[Dataset] Tokenization {len(examples)} ...")
        
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
                print(f"  : {idx+1}/{len(examples)}")
        
        print(f"[Dataset]  : {len(self.examples)} \n")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['input_ids'].clone(),
        }


def count_parameters(model):
    """"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def convert_to_json_serializable(obj):
    """ numpy """
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
    """L2"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


#  : AUC 
def compute_auc_intervals(losses):
    """AUC"""
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


def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDT ")
    
    #   
    ap.add_argument("--dataset", type=str, required=True,
                   choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'],
                   help="")
    ap.add_argument("--num_samples", type=int, default=0,
                   help="0=")
    
    # 
    ap.add_argument("--model_path", type=str,
                   default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                   help="")
    
    # LoRA 
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "v_proj"])
    
    # 
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=2500)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    
    # 
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # FDT 
    ap.add_argument("--use_fdt_init", action="store_true")
    ap.add_argument("--fdt_alpha", type=float, default=1.1)
    ap.add_argument("--fdt_method", type=str, default='fft',
                   choices=['fft', 'ar'])
    ap.add_argument("--verify_fdt", action="store_true")
    ap.add_argument("--plot_spectra", action="store_true")
    ap.add_argument("--init_preset", type=str, default=None,
                   choices=['baseline', 'soft', 'medium', 'strong'])
    
    #  :  
    ap.add_argument("--record_gradnorm", action="store_true",
                   help="")
    ap.add_argument("--full_test_eval", action="store_true",
                   help="batch")
    ap.add_argument("--measure_final_spectrum", action="store_true",
                   help="")
    
    # 
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--device", type=str, default="npu:1")
    
    return ap.parse_args()


def main():
    if not PEFT_AVAILABLE:
        print(": PEFT")
        return
    
    args = get_args()
    
    # 
    dataset_path = DATASET_PATHS.get(args.dataset)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f" : {dataset_path}")
        return
    
    # 
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 
    config_file = os.path.join(args.out_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f" : {config_file}\n")
    
    # 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 
    device = torch.device(args.device)
    device_type = args.device.split(':')[0]
    
    if device_type == 'npu':
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            torch_npu.npu.manual_seed_all(args.seed)
            print(f"[]  NPU : {device}\n")
        except Exception as e:
            print(f"[]  NPU : {e}")
            return
    else:
        print(f"[] : {device}\n")
    
    print("="*70)
    print("  1: ")
    print("="*70)
    
    print(f"[] : {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[]  Tokenizer: vocab_size={tokenizer.vocab_size}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    total_params, _ = count_parameters(model)
    print(f"[]  : {total_params/1e9:.2f}B \n")
    
    print("="*70)
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
    
    print(f"[LoRA] r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[LoRA] target_modules={args.lora_target_modules}")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"[LoRA]  : {trainable_params:,} ({trainable_params/total_params*100:.4f}%)\n")
    
    print("="*70)
    print("  3: FDT ")
    print("="*70)
    
    #   
    init_start_time = time.time()
    
    # 
    if args.init_preset:
        preset_configs = {
            'baseline': {'use_fdt': False, 'alpha': None, 'name': 'PEFT Default (Kaiming+Zero)'},
            'soft': {'use_fdt': True, 'alpha': 0.8, 'name': 'FDT-Soft (α=0.8)'},
            'medium': {'use_fdt': True, 'alpha': 1.1, 'name': 'FDT-Medium (α=1.1)'},
            'strong': {'use_fdt': True, 'alpha': 1.5, 'name': 'FDT-Strong (α=1.5)'},
        }
        
        config = preset_configs[args.init_preset]
        print(f"[] {config['name']}\n")
        
        if config['use_fdt']:
            args.use_fdt_init = True
            args.fdt_alpha = config['alpha']
    
    # 
    init_info = {
        'use_fdt': args.use_fdt_init,
        'preset': args.init_preset,
        'alpha': None,
        'method': 'peft_default',
        'lora_a_init': 'kaiming_uniform',
        'lora_b_init': 'zero',
        'init_time_seconds': None,  #  
        'measured_alphas_init': {},  #  α
        'measured_alphas_final': {},  #  α
        'verification_passed': None,
    }
    
    if args.use_fdt_init:
        print(f"[FDT] : α={args.fdt_alpha:.2f}, ={args.fdt_method}")
        
        apply_fdt_to_lora(
            model,
            alpha=args.fdt_alpha,
            method=args.fdt_method,
            verbose=args.verbose
        )
        
        init_info['alpha'] = args.fdt_alpha
        init_info['method'] = args.fdt_method
        
        print("[FDT]  ")
        
        if args.verify_fdt:
            print("\n[FDT] ...")
            verify_success = verify_fdt_initialization(
                model,
                target_alpha=args.fdt_alpha,
                tolerance=0.15,
                verbose=True
            )
            init_info['verification_passed'] = verify_success
        
        if args.plot_spectra:
            print("\n[FDT] ...")
            spectra_dir = os.path.join(args.out_dir, 'init_spectra')
            os.makedirs(spectra_dir, exist_ok=True)
            
            alphas = analyze_lora_spectra(
                model,
                save_dir=spectra_dir,
                plot_top_n=3,
                verbose=args.verbose
            )
            
            init_info['measured_alphas_init'] = {k: float(v) for k, v in alphas.items()}
            print(f"[FDT]  : {spectra_dir}")
    
    else:
        print("[FDT]  PEFT  (Kaiming Uniform + Zero) (Baseline)")
    
    #   
    init_time = time.time() - init_start_time
    init_info['init_time_seconds'] = float(init_time)
    print(f"[FDT] : {init_time*1000:.2f} ms\n")
    
    print("="*70)
    print(f"  4:  ({args.dataset.upper()})")
    print("="*70)
    
    print(f"[] : {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        
        print(f"[]  ")
        print(f"  • : {len(dataset['train'])} ")
        print(f"  • : {len(dataset['test'])} ")
        
        # 
        train_raw = dataset['train']
        if args.num_samples > 0 and args.num_samples < len(train_raw):
            train_raw = train_raw.select(range(args.num_samples))
            print(f"[] : {len(train_raw)} ")
        
        train_dataset_raw = train_raw
        test_dataset_raw = dataset['test']
        
        #  :  
        if not args.full_test_eval:
            max_test_samples = 1000
            if len(test_dataset_raw) > max_test_samples:
                test_dataset_raw = test_dataset_raw.select(range(max_test_samples))
                print(f"[] : {max_test_samples} ")
        else:
            print(f"[] : {len(test_dataset_raw)} ")
        
        print(f"\n[] :")
        print(f"  • : {len(train_dataset_raw)} ")
        print(f"  • : {len(test_dataset_raw)} ")
        
        # 
        def format_example(example):
            if args.dataset == 'gsm8k':
                return f"{example['question']}\n{example['answer']}"
            
            elif args.dataset == 'cmmlu':
                question = example['Question']
                choices = f"A. {example['A']}  B. {example['B']}  C. {example['C']}  D. {example['D']}"
                answer = example['Answer']
                return f"{question}\n{choices}\n{answer}"
            
            elif args.dataset == 'sharegpt':
                conversations = example.get('conversations', [])
                text = ""
                for turn in conversations:
                    role = turn.get('from', 'unknown')
                    content = turn.get('value', '')
                    text += f"{role}: {content}\n"
                return text.strip()
            
            elif args.dataset == 'mbpp':
                text = example['text']
                code = example['code']
                return f"# Problem\n{text}\n\n# Solution\n{code}"
        
        print(f"\n[] ...")
        train_texts = [format_example(item) for item in train_dataset_raw]
        test_texts = [format_example(item) for item in test_dataset_raw]
        
        print(f"\n[]  ( 150 ):")
        print(f"  {train_texts[0][:150]}...\n")
        
        #  Dataset
        train_dataset = BenchmarkDataset(tokenizer, train_texts, args.max_length)
        test_dataset = BenchmarkDataset(tokenizer, test_texts, args.max_length)
        
    except Exception as e:
        print(f"[]  : {e}")
        import traceback
        traceback.print_exc()
        return
    
    #  DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[] DataLoader:")
    print(f"  • : {len(train_loader)} ")
    print(f"  • : {len(test_loader)} \n")
    
    print("="*70)
    print("  5: ")
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
    
    print(f"[] AdamW (lr={args.lr:.2e}, wd={args.weight_decay})")
    print(f"[] Warmup {args.warmup_steps} ")
    print(f"[] ={args.max_grad_norm}\n")
    
    #   
    if device_type == 'npu':
        import torch_npu
        start_memory = torch_npu.npu.memory_allocated(device)
        peak_memory = start_memory
    else:
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = start_memory
    
    #  :  
    training_log = []  # 
    
    print("="*70)
    print("  6: ")
    print("="*70)
    
    model.train()
    
    training_losses = []
    test_losses_history = []  #  
    best_loss = float('inf')
    
    data_iter = iter(train_loader)
    start_time = time.time()
    
    print(f"\n[] {args.max_iters} ")
    print("-"*70)
    
    for step in range(1, args.max_iters + 1):
        step_start_time = time.time()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # NPU FP16 
        if device_type == 'npu':
            batch = {
                k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                for k, v in batch.items()
            }
        
        try:
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  (Step {step}): {loss.item()}")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
        except Exception as e:
            print(f"\n  (Step {step}): {e}")
            optimizer.zero_grad()
            continue
        
        #   
        grad_norm = None
        if args.record_gradnorm:
            grad_norm = compute_gradient_norm(model)
        
        # 
        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        #   
        current_lr = scheduler.get_last_lr()[0]
        
        #   
        if device_type == 'npu':
            current_memory = torch_npu.npu.memory_allocated(device)
        else:
            current_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = max(peak_memory, current_memory)
        
        step_time = time.time() - step_start_time
        
        #   
        log_entry = {
            'step': step,
            'train_loss': current_loss,
            'learning_rate': current_lr,
            'grad_norm': grad_norm,
            'step_time_ms': step_time * 1000,
            'memory_gb': (current_memory - start_memory) / 1e9,
            'test_loss': None,  # 
        }
        
        # 
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train = np.mean(training_losses[-args.eval_interval:])
            
            # 
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                #   
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
            
            #   
            metric_str = ""
            if step >= 100:
                loss_100 = training_losses[99]  # 100
                metric_str += f", L@100={loss_100:.4f}"
            if step >= 500:
                loss_500 = training_losses[499]
                auc_500 = sum(training_losses[:500])
                metric_str += f", L@500={loss_500:.4f}, AUC(0-500)={auc_500:.2f}"
            
            # 
            memory_gb = (peak_memory - start_memory) / 1e9
            
            log = f"[{step:5d}/{args.max_iters}] "
            log += f"={current_loss:.4f}, ={avg_train:.4f}"
            
            if test_loss_avg:
                log += f", ={test_loss_avg:.4f}"
            
            log += f", lr={current_lr:.2e}, {elapsed:.1f}s"
            log += f", Mem={memory_gb:.2f}GB"
            
            if args.record_gradnorm and grad_norm:
                log += f", GradNorm={grad_norm:.4f}"
            
            log += metric_str
            
            print(log)
            
            # 
            metric = test_loss_avg if test_loss_avg else avg_train
            
            if metric < best_loss:
                best_loss = metric
                best_model_path = os.path.join(args.out_dir, "best_model")
                model.save_pretrained(best_model_path)
                print(f"  →  (={best_loss:.4f})")
        
        #   
        training_log.append(log_entry)
    
    total_time = time.time() - start_time
    
    print("-"*70)
    print(f"[]  ! : {total_time/60:.2f} ")
    print(f"[] : {best_loss:.4f}")
    print(f"[] : {(peak_memory - start_memory) / 1e9:.2f} GB\n")
    
    print("="*70)
    print("  7: ")
    print("="*70)
    
    print(f"[] ")
    
    model.eval()
    test_losses = []
    
    print(f"[]  {len(test_loader)} ...")
    
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
                print(f"\n    {idx} : {str(e)[:100]}")
                continue
            
            if (idx + 1) % 50 == 0:
                print(f"  : {idx+1}/{len(test_loader)}")
    
    test_loss = np.mean(test_losses) if test_losses else float('inf')
    test_std = np.std(test_losses) if test_losses else 0.0
    
    print(f"\n[] :")
    print(f"  • : {test_loss:.4f}")
    print(f"  • : {test_std:.4f}")
    print(f"  • : {len(test_losses)}/{len(test_loader)}\n")
    
    #   7.5:  
    if args.use_fdt_init and args.measure_final_spectrum:
        print("="*70)
        print("  7.5: ")
        print("="*70)
        
        spectra_dir_final = os.path.join(args.out_dir, 'final_spectra')
        os.makedirs(spectra_dir_final, exist_ok=True)
        
        print("[FDT] ...")
        alphas_final = analyze_lora_spectra(
            model,
            save_dir=spectra_dir_final,
            plot_top_n=3,
            verbose=args.verbose
        )
        
        init_info['measured_alphas_final'] = {k: float(v) for k, v in alphas_final.items()}
        
        print(f"[FDT]  : {spectra_dir_final}")
        
        # α
        if init_info['measured_alphas_init']:
            print("\n[FDT] :")
            for key in init_info['measured_alphas_init']:
                if key in init_info['measured_alphas_final']:
                    alpha_init = init_info['measured_alphas_init'][key]
                    alpha_final = init_info['measured_alphas_final'][key]
                    delta = alpha_final - alpha_init
                    print(f"  {key}: {alpha_init:.3f} → {alpha_final:.3f} (Δ={delta:+.3f})")
        print()
    
    print("="*70)
    print("  8: ")
    print("="*70)
    
    # 1. numpy
    losses_file = os.path.join(args.out_dir, "training_losses.npy")
    np.save(losses_file, np.array(training_losses))
    print(f"[]  : {losses_file}")
    
    #  2. CSV
    csv_file = os.path.join(args.out_dir, "training_log.csv")
    with open(csv_file, 'w', newline='') as f:
        if training_log:
            fieldnames = training_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_log)
    print(f"[]  CSV: {csv_file}")
    
    #  3.  
    test_history_file = os.path.join(args.out_dir, "test_loss_history.json")
    with open(test_history_file, 'w') as f:
        json.dump(test_losses_history, f, indent=2)
    print(f"[]  : {test_history_file}")
    
    #  4. AUC 
    auc_metrics = compute_auc_intervals(training_losses)
    
    #  5.  
    early_convergence = {
        'loss_at_100': float(training_losses[99]) if len(training_losses) >= 100 else None,
        'loss_at_200': float(training_losses[199]) if len(training_losses) >= 200 else None,
        'loss_at_500': float(training_losses[499]) if len(training_losses) >= 500 else None,
        'loss_at_1000': float(training_losses[999]) if len(training_losses) >= 1000 else None,
    }
    
    #  6.  
    results = {
        # 
        'dataset': args.dataset,
        'model_path': args.model_path,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'init_method': 'FDT' if args.use_fdt_init else 'PEFT_Default',
        'init_alpha': args.fdt_alpha if args.use_fdt_init else None,
        'init_preset': args.init_preset,
        
        # 
        'best_train_loss': float(best_loss),
        'final_test_loss': float(test_loss),
        'final_test_loss_std': float(test_std),
        
        #   
        'early_convergence': early_convergence,
        
        #  AUC 
        'auc_metrics': auc_metrics,
        
        # 
        'wall_time_seconds': float(total_time),
        'wall_time_minutes': float(total_time / 60),
        'peak_memory_gb': float((peak_memory - start_memory) / 1e9),
        'throughput_samples_per_sec': float(len(train_dataset) / total_time),
        'avg_time_per_step_ms': float(total_time * 1000 / args.max_iters),
        
        # 
        'init_time_seconds': init_info['init_time_seconds'],
        'init_time_ms': init_info['init_time_seconds'] * 1000 if init_info['init_time_seconds'] else None,
        
        # 
        'num_train_samples': len(train_dataset),
        'num_test_samples': len(test_dataset),
        'test_batches_evaluated': len(test_losses),
        
        #   
        'measured_alphas_init': init_info.get('measured_alphas_init', {}),
        'measured_alphas_final': init_info.get('measured_alphas_final', {}),
    }
    
    results_file = os.path.join(args.out_dir, "results.json")
    results_serializable = convert_to_json_serializable(results)
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"[]  : {results_file}")
    
    # 7. 
    init_info_file = os.path.join(args.out_dir, 'init_info.json')
    init_info_serializable = convert_to_json_serializable(init_info)
    with open(init_info_file, 'w') as f:
        json.dump(init_info_serializable, f, indent=2)
    print(f"[]  : {init_info_file}")
    
    # 8. 
    final_path = os.path.join(args.out_dir, "final_model")
    try:
        model.save_pretrained(final_path)
        print(f"[]  : {final_path}")
    except Exception as e:
        print(f"[]  : {e}")
    
    # 9. Tokenizer
    try:
        tokenizer.save_pretrained(args.out_dir)
        print(f"[]  Tokenizer: {args.out_dir}\n")
    except Exception as e:
        print(f"[]  Tokenizer : {e}\n")
    
    print("="*70)
    print(" !")
    print("="*70)
    print(f"\n: {args.dataset.upper()}")
    print(f": {args.model_path.split('/')[-1]}")
    print(f"LoRA : r={args.lora_r}")
    print(f": {'FDT (α='+str(args.fdt_alpha)+')' if args.use_fdt_init else 'PEFT Default (Kaiming+Zero)'}")
    
    print(f"\n :")
    print(f"  • : {best_loss:.4f}")
    print(f"  • : {test_loss:.4f}")
    
    if early_convergence['loss_at_100']:
        print(f"\n :")
        print(f"  • Loss@100: {early_convergence['loss_at_100']:.4f}")
        if early_convergence['loss_at_500']:
            print(f"  • Loss@500: {early_convergence['loss_at_500']:.4f}")
    
    if auc_metrics.get('auc_0_500'):
        print(f"\n AUC :")
        print(f"  • AUC(0-500): {auc_metrics['auc_0_500']:.2f}")
        if auc_metrics.get('auc_0_2500'):
            print(f"  • AUC(0-2500): {auc_metrics['auc_0_2500']:.2f}")
    
    print(f"\n⏱ :")
    print(f"  • : {total_time/60:.2f} ")
    print(f"  • : {(peak_memory - start_memory) / 1e9:.2f} GB")
    print(f"  • : {len(train_dataset) / total_time:.2f} samples/s")
    
    if init_info['init_time_seconds']:
        print(f"  • : {init_info['init_time_seconds']*1000:.2f} ms")
    
    print(f"\n : {args.out_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()  