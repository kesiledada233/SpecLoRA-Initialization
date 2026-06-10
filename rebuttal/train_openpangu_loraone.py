import os
os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'
os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'

import time
import math
import argparse
import random
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

DATASET_PATHS = {
    'gsm8k': '/rebuttal/datasets/gsm8k',
    'cmmlu': '/rebuttal/datasets/cmmlu/processed',
    'sharegpt': '/rebuttal/datasets/sharegpt_datasets',
    'mbpp': '/rebuttal/datasets/mbpp/processed',
}

class BenchmarkDataset(Dataset):
    def __init__(self, tokenizer, examples, max_length=512):
        self.examples = []
        for idx, text in enumerate(examples):
            if len(text.strip()) < 10: continue
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
            self.examples.append({'input_ids': encodings['input_ids'].squeeze(), 'attention_mask': encodings['attention_mask'].squeeze()})
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'], 'labels': item['input_ids'].clone()}

def apply_lora_one_init(model, dataloader, device, r):
    print("\n" + "="*70)
    print("  LoRA-One  SVD ")
    print("="*70)
    
    lora_layers = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            lora_layers.append(module)
            if hasattr(module.base_layer, "weight"):
                module.base_layer.weight.requires_grad = True # 
                
    if not lora_layers:
        print("  LoRA ")
        return 0

    model.train(); model.zero_grad()
    try: batch = next(iter(dataloader))
    except StopIteration: return 0
        
    batch = {k: v.half().to(device) if device.type == 'npu' and v.dtype in [torch.float32, torch.float64] else v.to(device) for k, v in batch.items()}
    
    start_time = time.time()
    print("[LoRA-One] ...")
    loss = model(**batch).loss
    loss.backward()
    
    print(f"[LoRA-One]  (Rank={r})...")
    with torch.no_grad():
        for module in lora_layers:
            if not hasattr(module.base_layer, "weight") or module.base_layer.weight.grad is None: continue
            
            # NPU  SVD / CPU SVD  top-r
            grad_w = module.base_layer.weight.grad.detach().to(torch.float32).cpu()
            m, n = grad_w.shape
            q = min(max(r + 8, r), m, n)
            if q <= 0:
                continue

            # torch.svd_lowrank  U: [m, q], S: [q], V: [n, q]
            try:
                U, S, V = torch.svd_lowrank(grad_w, q=q, niter=4)
                Vh = V.transpose(0, 1)
            except Exception:
                #  SVD CPU 
                U, S, Vh = torch.linalg.svd(grad_w, full_matrices=False)
            
            adapter_name = list(module.lora_A.keys())[0]
            scale = module.scaling.get(adapter_name, 1.0)
            
            # 
            sqrt_S = torch.sqrt(S[:r]) / math.sqrt(scale)
            B_init = U[:, :r] * sqrt_S.unsqueeze(0)
            A_init = sqrt_S.unsqueeze(1) * Vh[:r, :]
            
            # 
            target_device = module.lora_A[adapter_name].weight.device
            module.lora_A[adapter_name].weight.copy_(A_init.to(device=target_device, dtype=module.lora_A[adapter_name].weight.dtype))
            module.lora_B[adapter_name].weight.copy_(B_init.to(device=target_device, dtype=module.lora_B[adapter_name].weight.dtype))
            
            # 
            module.base_layer.weight.requires_grad = False
            module.base_layer.weight.grad = None
            
    model.zero_grad()
    elapsed = time.time() - start_time
    print(f"[LoRA-One]  SVD : {elapsed:.2f}\n")
    return elapsed

def compute_gradient_norm(model): return sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
def compute_auc_intervals(losses): return {name: float(sum(losses[s:e])) if len(losses) >= e else None for name, (s, e) in {'auc_0_100': (0, 100), 'auc_0_500': (0, 500)}.items()}

def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu LoRA-One ")
    ap.add_argument("--dataset", type=str, required=True, choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'])
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--model_path", type=str, default="/rebuttal/models/openPangu-7b")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "v_proj"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=1000)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--record_gradnorm", action="store_true")
    ap.add_argument("--full_test_eval", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument("--device", type=str, default="npu:0")
    return ap.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), 'w') as f: json.dump(vars(args), f, indent=2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    device_type = args.device.split(':')[0]
    if device_type == 'npu':
        import torch_npu
        torch_npu.npu.set_device(device)
        torch_npu.npu.manual_seed_all(args.seed)

    print("="*70 + "\n  1 & 2:  LoRA\n" + "="*70)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float16)
    
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    print("="*70 + "\n  3: \n" + "="*70)
    dataset = load_from_disk(DATASET_PATHS[args.dataset])
    train_raw, test_raw = dataset['train'], dataset['test']
    if args.num_samples > 0: train_raw = train_raw.select(range(args.num_samples))
    if not args.full_test_eval and len(test_raw) > 1000: test_raw = test_raw.select(range(1000))
    
    def format_ex(ex):
        if args.dataset == 'gsm8k': return f"{ex['question']}\n{ex['answer']}"
        elif args.dataset == 'cmmlu': return f"{ex['Question']}\nA.{ex['A']} B.{ex['B']} C.{ex['C']} D.{ex['D']}\n{ex['Answer']}"
        elif args.dataset == 'sharegpt': return "\n".join([f"{t.get('from')}: {t.get('value')}" for t in ex.get('conversations', [])]).strip()
        elif args.dataset == 'mbpp': return f"# Problem\n{ex['text']}\n\n# Solution\n{ex['code']}"

    train_loader = DataLoader(BenchmarkDataset(tokenizer, [format_ex(i) for i in train_raw], args.max_length), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(BenchmarkDataset(tokenizer, [format_ex(i) for i in test_raw], args.max_length), batch_size=args.batch_size, shuffle=False)

    #   LoRA-One  
    lora_one_init_time = apply_lora_one_init(model, train_loader, device, args.lora_r)

    print("="*70 + "\n  4: \n" + "="*70)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_iters)
    
    start_memory = torch_npu.npu.memory_allocated(device) if device_type == 'npu' else (torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0)
    peak_memory, training_log, training_losses, best_loss = start_memory, [], [], float('inf')
    data_iter = iter(train_loader)
    start_time = time.time()
    
    model.train()
    for step in range(1, args.max_iters + 1):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(train_loader); batch = next(data_iter)
        
        batch = {k: v.half().to(device) if device_type == 'npu' and v.dtype in [torch.float32, torch.float64] else v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / args.grad_accum_steps
        loss.backward()
        
        grad_norm = compute_gradient_norm(model) if args.record_gradnorm else None
        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        
        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)
        
        log_entry = {'step': step, 'train_loss': current_loss, 'lr': scheduler.get_last_lr()[0], 'grad_norm': grad_norm, 'test_loss': None}
        
        if step % args.eval_interval == 0 or step == 1:
            model.eval(); test_losses = []
            with torch.no_grad():
                for i, t_batch in enumerate(test_loader):
                    if not args.full_test_eval and i >= 10: break
                    t_batch = {k: v.half().to(device) if device_type == 'npu' and v.dtype in [torch.float32, torch.float64] else v.to(device) for k, v in t_batch.items()}
                    test_losses.append(model(**t_batch).loss.item())
            model.train()
            if test_losses:
                test_loss_avg = np.mean(test_losses)
                log_entry['test_loss'] = test_loss_avg
                if test_loss_avg < best_loss: best_loss = test_loss_avg; model.save_pretrained(os.path.join(args.out_dir, "best_model"))
            print(f"[{step:5d}/{args.max_iters}] Train={current_loss:.4f}, Test={test_loss_avg if test_losses else 0:.4f}, LR={log_entry['lr']:.2e}")
        training_log.append(log_entry)

    total_time = time.time() - start_time
    print("="*70 + "\n  5: \n" + "="*70)
    np.save(os.path.join(args.out_dir, "training_losses.npy"), np.array(training_losses))
    with open(os.path.join(args.out_dir, "training_log.csv"), 'w', newline='') as f:
        if training_log: writer = csv.DictWriter(f, fieldnames=training_log[0].keys()); writer.writeheader(); writer.writerows(training_log)
    
    results = {'dataset': args.dataset, 'algo': 'LoRA-One', 'init_time_s': lora_one_init_time, 'best_loss': best_loss, 'auc': compute_auc_intervals(training_losses), 'time_m': total_time/60}
    with open(os.path.join(args.out_dir, "results.json"), 'w') as f: json.dump(results, f, indent=2)
    model.save_pretrained(os.path.join(args.out_dir, "final_model"))
    print(" LoRA-One ")

if __name__ == "__main__":
    main()