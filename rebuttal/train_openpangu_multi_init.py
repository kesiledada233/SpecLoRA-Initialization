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
        PeftModel,
        LoftQConfig  #  :  LoftQ 
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
print(" ")
print("="*70)

for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]

FDT_INIT_PATH = '/rebuttal/FDASOC_init_code'

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
    print(f"[]  FDT: {e}")
    print("[]  PEFT  peft_default, pissa, loftq")
    FDT_INIT_AVAILABLE = False

print("="*70 + "\n")

DATASET_PATHS = {
    'gsm8k': '/rebuttal/datasets/gsm8k',
    'cmmlu': '/rebuttal/datasets/cmmlu/processed',
    'sharegpt': '/rebuttal/datasets/sharegpt_datasets',
    'mbpp': '/rebuttal/datasets/mbpp/processed',
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
    ap = argparse.ArgumentParser(description="OpenPangu  PEFT ")
    
    # 
    ap.add_argument("--dataset", type=str, required=True,
                   choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'],
                   help="")
    ap.add_argument("--num_samples", type=int, default=0,
                   help="0=")
    
    # 
    ap.add_argument("--model_path", type=str,
                   default="/rebuttal/models/openPangu-7b",
                   help="")
    
    # LoRA 
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "v_proj"])
    
    #  :  
    ap.add_argument("--init_method", type=str, default="peft_default",
                   choices=['peft_default', 'fdt', 'pissa', 'loftq'],
                   help="peft_default(), fdt(), pissa(SVD), loftq()")
    ap.add_argument("--use_fdt_init", action="store_true",
                   help="[]  FDT  ( --init_method fdt)")
    
    # FDT 
    ap.add_argument("--fdt_alpha", type=float, default=0.6)
    ap.add_argument("--fdt_method", type=str, default='fft', choices=['fft', 'ar'])
    ap.add_argument("--unroll_order", type=str, default='row', choices=['row', 'col'], help="")
    ap.add_argument("--verify_fdt", action="store_true")
    ap.add_argument("--plot_spectra", action="store_true")
    ap.add_argument("--init_preset", type=str, default=None,
                   choices=['baseline', 'soft', 'medium', 'strong'])
    
    # 
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=1000)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    
    # 
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 
    ap.add_argument("--record_gradnorm", action="store_true")
    ap.add_argument("--full_test_eval", action="store_true")
    ap.add_argument("--measure_final_spectrum", action="store_true")
    
    # 
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--device", type=str, default="npu:1")
    
    args = ap.parse_args()
    
    #  --use_fdt_init --init_method fdt
    if args.use_fdt_init:
        args.init_method = 'fdt'
        
    return args


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
    
    # LoftQ  fp16 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    
    total_params, _ = count_parameters(model)
    print(f"[]  : {total_params/1e9:.2f}B \n")
    
    print("="*70)
    print(f"  2 & 3:  LoRA (: {args.init_method.upper()})")
    print("="*70)
    
    #  FDT  ()
    if args.init_preset and args.init_method in ['peft_default', 'fdt']:
        preset_configs = {
            'baseline': {'init_method': 'peft_default', 'alpha': None, 'name': 'PEFT Default'},
            'soft': {'init_method': 'fdt', 'alpha': 0.8, 'name': 'FDT-Soft (α=0.8)'},
            'medium': {'init_method': 'fdt', 'alpha': 1.1, 'name': 'FDT-Medium (α=1.1)'},
            'strong': {'init_method': 'fdt', 'alpha': 1.5, 'name': 'FDT-Strong (α=1.5)'},
        }
        config = preset_configs[args.init_preset]
        print(f"[] : {config['name']}")
        args.init_method = config['init_method']
        if args.init_method == 'fdt':
            args.fdt_alpha = config['alpha']
    
    # 
    init_info = {
        'method': args.init_method,
        'lora_r': args.lora_r,
        'init_time_seconds': None,
        'measured_alphas_init': {},
        'measured_alphas_final': {},
        'verification_passed': None,
    }

    #   LoraConfig  
    lora_kwargs = {}
    
    if args.init_method == 'pissa':
        print("[]  PiSSA(SVD)")
        lora_kwargs['init_lora_weights'] = "pissa"
        
    elif args.init_method == 'loftq':
        print("[]  LoftQ 4-bit ")
        try:
            import bitsandbytes
            lora_kwargs['init_lora_weights'] = "loftq"
            lora_kwargs['loftq_config'] = LoftQConfig(loftq_bits=4)
        except ImportError:
            raise ImportError(" LoftQ  bitsandbytes  `pip install bitsandbytes`")
            
    else:
        # peft_default  fdt  LoraConfig 
        pass

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        **lora_kwargs
    )
    
    print(f"[LoRA] r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[LoRA] target_modules={args.lora_target_modules}")
    
    #   
    init_start_time = time.time()
    
    #  LoraConfig (PiSSA  LoftQ )
    model = get_peft_model(model, lora_config)
    
    #  FDT PEFT 
    if args.init_method == 'fdt' and FDT_INIT_AVAILABLE:
        print(f"\n[FDT] : α={args.fdt_alpha:.2f}, ={args.fdt_method}")
        apply_fdt_to_lora(model, alpha=args.fdt_alpha, method=args.fdt_method, unroll_order=args.unroll_order, verbose=args.verbose)
        
        if args.verify_fdt:
            print("[FDT] ...")
            verify_success = verify_fdt_initialization(
                model, target_alpha=args.fdt_alpha, tolerance=0.15, verbose=True)
            init_info['verification_passed'] = verify_success
        
        if args.plot_spectra:
            print("[FDT] ...")
            spectra_dir = os.path.join(args.out_dir, 'init_spectra')
            os.makedirs(spectra_dir, exist_ok=True)
            alphas = analyze_lora_spectra(model, save_dir=spectra_dir, plot_top_n=3, verbose=args.verbose)
            init_info['measured_alphas_init'] = {k: float(v) for k, v in alphas.items()}
            print(f"[FDT]  : {spectra_dir}")

    # 
    init_time = time.time() - init_start_time
    init_info['init_time_seconds'] = float(init_time)
    
    # 
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n[LoRA]  ! : {init_time*1000:.2f} ms")
    print(f"[LoRA]  : {trainable_params:,} ({trainable_params/total_params*100:.4f}%)\n")
    
    print("="*70)
    print(f"  4:  ({args.dataset.upper()})")
    print("="*70)
    
    try:
        dataset = load_from_disk(dataset_path)
        
        train_raw = dataset['train']
        if args.num_samples > 0 and args.num_samples < len(train_raw):
            train_raw = train_raw.select(range(args.num_samples))
        
        train_dataset_raw = train_raw
        test_dataset_raw = dataset['test']
        
        if not args.full_test_eval:
            max_test_samples = 1000
            if len(test_dataset_raw) > max_test_samples:
                test_dataset_raw = test_dataset_raw.select(range(max_test_samples))
        
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
                return f"# Problem\n{example['text']}\n\n# Solution\n{example['code']}"
        
        train_texts = [format_example(item) for item in train_dataset_raw]
        test_texts = [format_example(item) for item in test_dataset_raw]
        
        train_dataset = BenchmarkDataset(tokenizer, train_texts, args.max_length)
        test_dataset = BenchmarkDataset(tokenizer, test_texts, args.max_length)
        
    except Exception as e:
        print(f"[]  : {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    # 
    if device_type == 'npu':
        import torch_npu
        start_memory = torch_npu.npu.memory_allocated(device)
        peak_memory = start_memory
    else:
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = start_memory
    
    training_log = []
    
    print("="*70)
    print("  6: ")
    print("="*70)
    
    model.train()
    
    training_losses = []
    test_losses_history = []
    best_loss = float('inf')
    
    data_iter = iter(train_loader)
    start_time = time.time()
    
    for step in range(1, args.max_iters + 1):
        step_start_time = time.time()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if device_type == 'npu':
            batch = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum_steps
        
        loss.backward()
        
        grad_norm = compute_gradient_norm(model) if args.record_gradnorm else None
        
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
        
        log_entry = {
            'step': step,
            'train_loss': current_loss,
            'learning_rate': current_lr,
            'grad_norm': grad_norm,
            'step_time_ms': (time.time() - step_start_time) * 1000,
            'memory_gb': (current_memory - start_memory) / 1e9,
            'test_loss': None,
        }
        
        # 
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train = np.mean(training_losses[-args.eval_interval:])
            
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                max_eval_batches = None if args.full_test_eval else 10
                for batch_idx, test_batch in enumerate(test_loader):
                    if max_eval_batches and batch_idx >= max_eval_batches:
                        break
                    test_batch = {k: v.to(device) for k, v in test_batch.items()}
                    if device_type == 'npu':
                        test_batch = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v for k, v in test_batch.items()}
                    
                    test_outputs = model(**test_batch)
                    test_losses.append(test_outputs.loss.item())
            
            model.train()
            
            test_loss_avg = np.mean(test_losses) if test_losses else None
            
            if test_loss_avg:
                test_losses_history.append({'step': step, 'test_loss': test_loss_avg})
                log_entry['test_loss'] = test_loss_avg
            
            memory_gb = (peak_memory - start_memory) / 1e9
            
            log = f"[{step:5d}/{args.max_iters}] ={current_loss:.4f}, ={avg_train:.4f}"
            if test_loss_avg: log += f", ={test_loss_avg:.4f}"
            log += f", lr={current_lr:.2e}, Mem={memory_gb:.2f}GB"
            
            print(log)
            
            metric = test_loss_avg if test_loss_avg else avg_train
            if metric < best_loss:
                best_loss = metric
                model.save_pretrained(os.path.join(args.out_dir, "best_model"))
        
        training_log.append(log_entry)
    
    total_time = time.time() - start_time
    print(f"[]  ! : {total_time/60:.2f} \n")
    
    print("="*70)
    print("  7: ")
    print("="*70)
    
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for idx, test_batch in enumerate(test_loader):
            test_batch = {k: v.to(device) for k, v in test_batch.items()}
            if device_type == 'npu':
                test_batch = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v for k, v in test_batch.items()}
            
            test_outputs = model(**test_batch)
            test_losses.append(test_outputs.loss.item())
            
    test_loss = np.mean(test_losses) if test_losses else float('inf')
    test_std = np.std(test_losses) if test_losses else 0.0
    print(f"[] : {test_loss:.4f}, : {test_std:.4f}\n")
    
    #   7.5:  
    if args.init_method == 'fdt' and args.measure_final_spectrum and FDT_INIT_AVAILABLE:
        print("  7.5: ")
        spectra_dir_final = os.path.join(args.out_dir, 'final_spectra')
        os.makedirs(spectra_dir_final, exist_ok=True)
        alphas_final = analyze_lora_spectra(model, save_dir=spectra_dir_final, plot_top_n=3, verbose=args.verbose)
        init_info['measured_alphas_final'] = {k: float(v) for k, v in alphas_final.items()}
    
    print("="*70)
    print("  8: ")
    print("="*70)
    
    np.save(os.path.join(args.out_dir, "training_losses.npy"), np.array(training_losses))
    
    with open(os.path.join(args.out_dir, "training_log.csv"), 'w', newline='') as f:
        if training_log:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)
            
    with open(os.path.join(args.out_dir, "test_loss_history.json"), 'w') as f:
        json.dump(test_losses_history, f, indent=2)
    
    results = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'lora_r': args.lora_r,
        'init_method': args.init_method,
        'best_train_loss': float(best_loss),
        'final_test_loss': float(test_loss),
        'auc_metrics': compute_auc_intervals(training_losses),
        'wall_time_minutes': float(total_time / 60),
        'peak_memory_gb': float((peak_memory - start_memory) / 1e9),
        'init_time_ms': init_info['init_time_seconds'] * 1000 if init_info['init_time_seconds'] else None,
    }
    
    with open(os.path.join(args.out_dir, "results.json"), 'w') as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)
        
    with open(os.path.join(args.out_dir, 'init_info.json'), 'w') as f:
        json.dump(convert_to_json_serializable(init_info), f, indent=2)
        
    model.save_pretrained(os.path.join(args.out_dir, "final_model"))
    tokenizer.save_pretrained(args.out_dir)
    
    print("\n ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()