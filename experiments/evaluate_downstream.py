import os
import sys
import time
import argparse
import json
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DATASET_PATHS = {
    'gsm8k': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/gsm8k',
    'cmmlu': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/cmmlu/processed',
    'sharegpt': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/sharegpt_datasets/computer_en_26k.jsonl',
    'mbpp': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/mbpp/processed',
}


class GSM8KEvaluator:
    """GSM8K """

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, question):
        return f"{question}\n"

    def extract_answer(self, text):
        """ GSM8K """
        # GSM8K : "#### 42"  "The answer is 42"
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',
            r'[][:]\s*([-+]?\d*\.?\d+)',
            r'answer\s+is\s+([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)(?:\s|$|\.|,)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue

        # 
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        return None

    def normalize_answer(self, answer):
        """"""
        if isinstance(answer, (int, float)):
            return float(answer)

        text = str(answer).strip()

        #  GSM8K  #### 
        hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
        if hash_match:
            try:
                return float(hash_match.group(1))
            except:
                pass

        text = text.lower()

        # 
        if '%' in text:
            text = text.replace('%', '').strip()
            try:
                return float(text) / 100
            except:
                pass

        # 
        frac_match = re.match(r'(\d+)\s*/\s*(\d+)', text)
        if frac_match:
            try:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            except:
                pass

        # 
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass

        return None

    def evaluate(self, model, test_dataset, max_samples=None, max_new_tokens=256):
        """ GSM8K"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[GSM8K]  {len(indices)} ...")

        for idx in tqdm(indices, desc="GSM8K"):
            example = test_dataset[idx]
            question = example['question']
            true_answer = example['answer']

            #  prompt
            prompt = self.format_prompt(question)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated.split("")[-1].strip() if "" in generated else generated

            # 
            pred = self.extract_answer(generated_answer)
            true = self.normalize_answer(true_answer)

            is_correct = False
            if pred is not None and true is not None:
                # 
                is_correct = abs(pred - true) < 1e-6 or abs(pred - true) / max(abs(true), 1) < 0.01

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],  # 
                'true_answer': str(true_answer)[:100],
                'generated': generated_answer[:200],
                'extracted_pred': pred,
                'extracted_true': true,
                'correct': is_correct,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],  #  10 
        }


class CMMLUEvaluator:
    """CMMLU """

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def format_prompt(self, question, choices):
        """ prompt"""
        prompt = f"{question}\n"
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}\n"
        prompt += ""
        return prompt

    def evaluate_likelihood(self, model, test_dataset, max_samples=None):
        """"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[CMMLU]  {len(indices)} ...")

        for idx in tqdm(indices, desc="CMMLU"):
            example = test_dataset[idx]
            question = example['Question']
            choices = [example['A'], example['B'], example['C'], example['D']]
            true_answer = example['Answer']

            # 
            losses = []
            for choice in choices:
                prompt = self.format_prompt(question, choices)
                text = prompt + choice

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    # 
                    loss = outputs.loss.item()
                    losses.append(-loss)  # 

            predicted_idx = np.argmax(losses)
            predicted_label = ['A', 'B', 'C', 'D'][predicted_idx]
            is_correct = (predicted_label == true_answer)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],
                'predicted': predicted_label,
                'true': true_answer,
                'correct': is_correct,
                'losses': losses,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }

    def evaluate_generate(self, model, test_dataset, max_samples=None, max_new_tokens=32):
        """"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[CMMLU]  {len(indices)} ...")

        for idx in tqdm(indices, desc="CMMLU"):
            example = test_dataset[idx]
            question = example['Question']
            choices = [example['A'], example['B'], example['C'], example['D']]
            true_answer = example['Answer']

            prompt = self.format_prompt(question, choices)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_part = generated.split("")[-1].strip() if "" in generated else generated

            # 
            predicted_label = None
            for label in ['A', 'B', 'C', 'D']:
                if label in answer_part[:10]:  # 
                    predicted_label = label
                    break

            if predicted_label is None:
                # 
                first_char = answer_part[0].upper() if answer_part else ''
                if first_char in ['A', 'B', 'C', 'D']:
                    predicted_label = first_char

            is_correct = (predicted_label == true_answer)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],
                'predicted': predicted_label,
                'true': true_answer,
                'correct': is_correct,
                'generated': answer_part[:50],
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }


class MBPPEvaluator:
    """MBPP """

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, problem_text):
        return f"# Problem\n{problem_text}\n\n# Solution\n"

    def extract_code(self, generated):
        """"""
        if "# Solution\n" in generated:
            code = generated.split("# Solution\n")[-1].strip()
        else:
            code = generated.strip()

        # 
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
            elif lines:  # 
                break

        return '\n'.join(lines)

    def execute_test_cases(self, code, test_list, timeout=5):
        """"""
        import sys
        import io
        import contextlib

        # 
        test_code = code + '\n\n'
        test_code += '\n'.join(test_list)

        # 
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            exec_globals = {'__builtins__': __builtins__}
            exec(test_code, exec_globals)

            # 
            output = sys.stdout.getvalue()
            success = True

        except AssertionError:
            success = False
        except Exception as e:
            success = False
        finally:
            sys.stdout = old_stdout

        return success

    def evaluate(self, model, test_dataset, max_samples=None, max_new_tokens=256):
        """ MBPP"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[MBPP]  {len(indices)} ...")

        for idx in tqdm(indices, desc="MBPP"):
            example = test_dataset[idx]
            problem_text = example['text']
            true_code = example['code']
            test_list = example.get('test_list', [])

            prompt = self.format_prompt(problem_text)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = self.extract_code(generated)

            # 
            passed = False
            if test_list:
                try:
                    passed = self.execute_test_cases(generated_code, test_list)
                except Exception as e:
                    passed = False
            else:
                # 
                passed = generated_code.strip() == true_code.strip()

            if passed:
                correct += 1
            total += 1

            results.append({
                'problem': problem_text[:100],
                'generated_code': generated_code[:200],
                'passed': passed,
            })

        pass_rate = correct / total if total > 0 else 0

        return {
            'pass_rate': pass_rate,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }


class ShareGPTEvaluator:
    """ShareGPT """

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_conversation(self, conversations):
        """

        ShareGPT /
        - str: 
        - list[dict]: {from,value}  {role,content}
        - list[str]: 
        - /
        """

        if conversations is None:
            return ""

        #  conversations 
        if isinstance(conversations, str):
            return conversations.strip()

        #  dict
        if isinstance(conversations, dict):
            conversations = [conversations]

        if not isinstance(conversations, (list, tuple)):
            return str(conversations).strip()

        lines = []
        for turn in conversations:
            if turn is None:
                continue
            if isinstance(turn, str):
                s = turn.strip()
                if s:
                    lines.append(s)
                continue
            if isinstance(turn, dict):
                role = turn.get('from', turn.get('role', 'unknown'))
                content = turn.get('value', turn.get('content', turn.get('text', '')))
                if content is None:
                    content = ""
                role = str(role)
                content = str(content)
                if content.strip():
                    lines.append(f"{role}: {content}")
                continue

            # 
            s = str(turn).strip()
            if s:
                lines.append(s)

        return "\n".join(lines).strip()

    def compute_perplexity(self, model, test_dataset, max_samples=None, max_length=512):
        """ Perplexity"""
        total_loss = 0
        total_tokens = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[ShareGPT]  {len(indices)}  Perplexity...")

        for idx in tqdm(indices, desc="ShareGPT"):
            example = test_dataset[idx]
            conversations = example.get('conversations', [])
            text = self.format_conversation(conversations)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                n_tokens = inputs["input_ids"].size(1)

                total_loss += loss * n_tokens
                total_tokens += n_tokens

                results.append({
                    'loss': loss,
                    'tokens': n_tokens,
                    'ppl': math.exp(loss),
                })

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'results': results[:10],
        }


def load_sharegpt_conversations_dataset(path: str):
    from datasets import Dataset, Features, Sequence, Value

    def norm_turn(t):
        if not isinstance(t, dict):
            return None
        # {"from","value"}  {"role","content"}
        speaker = t.get("from", t.get("role", "unknown"))
        content = t.get("value", t.get("content", t.get("text", "")))
        if content is None:
            content = ""
        speaker = str(speaker)
        content = str(content)
        return {"from": speaker, "value": content}

    def extract_conversations(obj):
        conv = obj.get("conversations", None)
        if conv is None:
            conv = obj.get("messages", None)
        if conv is None:
            conv = obj.get("conversation", None)
        if not isinstance(conv, list):
            return None
        out = []
        for t in conv:
            nt = norm_turn(t)
            if nt is not None and nt["value"] != "":
                out.append(nt)
        return out

    records = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        #  JSONL JSON 
        first_nonspace = ""
        pos = f.tell()
        for line in f:
            s = line.strip()
            if s:
                first_nonspace = s[0]
                break
        f.seek(pos)

        if first_nonspace == "[":
            data = json.load(f)
            for obj in data:
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                conv = extract_conversations(obj)
                if conv is None:
                    skipped += 1
                    continue
                records.append({"conversations": conv})
        else:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                conv = extract_conversations(obj)
                if conv is None:
                    skipped += 1
                    continue
                records.append({"conversations": conv})

    features = Features(
        {"conversations": Sequence({"from": Value("string"), "value": Value("string")})}
    )
    ds = Dataset.from_list(records, features=features)

    if skipped:
        print(f"[ShareGPT]  : {skipped}")
    return ds

def main():
    parser = argparse.ArgumentParser(description="")

    # 
    parser.add_argument("--model_path", type=str, required=True,
                       help="")
    parser.add_argument("--base_model", type=str,
                       default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                       help=" LoRA")

    # 
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'],
                       help="")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="None=")

    # 
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help=" token ")
    parser.add_argument("--eval_method", type=str, default='auto',
                       choices=['auto', 'likelihood', 'generate'],
                       help="CMMLU ")

    # 
    parser.add_argument("--output_dir", type=str, default=None,
                       help=" evaluation")
    parser.add_argument("--save_details", action="store_true",
                       help="")

    # 
    parser.add_argument("--device", type=str, default="npu:1")
    parser.add_argument("--batch_size", type=int, default=1,
                       help=" 1")

    args = parser.parse_args()

    # 
    device = torch.device(args.device)
    device_type = args.device.split(':')[0]

    if device_type == 'npu':
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            print(f"[]  NPU : {device}\n")
        except Exception as e:
            print(f"[]  NPU : {e}")
            return
    else:
        print(f"[] : {device}\n")

    # 
    dataset_path = args.dataset_path or DATASET_PATHS.get(args.dataset)
    if not dataset_path:
        print(f" : {args.dataset}")
        return

    #  tokenizer
    print("="*70)
    print(" ")
    print("="*70)

    print(f"[] : {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if os.path.exists(os.path.join(args.model_path, 'adapter_config.json')) else args.model_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[]  Tokenizer ")

        # 
    print(f"[] : {args.model_path}")
    
    #  LoRA 
    adapter_config_path = os.path.join(args.model_path, 'adapter_config.json')
    
    if os.path.exists(adapter_config_path):
        #  LoRA NPU 
        from peft import PeftModel, PeftConfig
        import json
        
        print(f"[]  LoRA ")
        
        #  1:  adapter_config.json  safetensors 
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        #  safetensors 
        safetensors_file = os.path.join(args.model_path, 'adapter_model.safetensors')
        pytorch_file = os.path.join(args.model_path, 'adapter_model.bin')
        
        if os.path.exists(safetensors_file) and not os.path.exists(pytorch_file):
            #  safetensors  PyTorch 
            print(f"[]  safetensors  PyTorch ...")
            
            try:
                from safetensors.torch import load_file
                
                #  CPU 
                state_dict = load_file(safetensors_file, device='cpu')
                
                #  PyTorch 
                torch.save(state_dict, pytorch_file)
                print(f"[]   PyTorch : {pytorch_file}")
                
            except Exception as e:
                print(f"[]  : {e}")
                print(f"[] ...")
        
        # 
        print(f"[] : {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu'  #   CPU 
        )
        
        #  LoRA adapter PyTorch 
        print(f"[]  LoRA adapter: {args.model_path}")
        
        try:
            #   safetensors 
            os.environ['PEFT_USE_SAFETENSORS'] = '0'  #  safetensors
            
            model = PeftModel.from_pretrained(
                base_model,
                args.model_path,
                device_map='cpu'  #   CPU 
            )
            
            print(f"[]  LoRA ")
        
        except Exception as e:
            print(f"[]  LoRA : {e}")
            
            #   
            print(f"[] ...")
            
            if os.path.exists(pytorch_file):
                state_dict = torch.load(pytorch_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
                model = base_model
                print(f"[]  ")
            else:
                raise RuntimeError(f" LoRA ")
    
    else:
        #   
        print(f"[] : {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu'  #   CPU 
        )
        print(f"[]  ")
    
    #   NPU  
    print(f"[] : {device}")
    model = model.to(device)
    model.eval()
    
    print(f"[]  \n")

    # 
    print("="*70)
    print(f"  ({args.dataset.upper()})")
    print("="*70)

    try:
        from datasets import load_from_disk, load_dataset

        if os.path.isdir(dataset_path):
            dataset = load_from_disk(dataset_path)

            # load_from_disk  DatasetDict  Dataset
            if isinstance(dataset, dict) or hasattr(dataset, "keys"):
                if 'test' in dataset:
                    test_dataset = dataset['test']
                elif 'validation' in dataset:
                    test_dataset = dataset['validation']
                else:
                    test_dataset = dataset['train']
            else:
                test_dataset = dataset  #  Dataset

        elif os.path.isfile(dataset_path):
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext in [".jsonl", ".json"]:
                if args.dataset == "sharegpt":
                    test_dataset = load_sharegpt_conversations_dataset(dataset_path)
                else:
                    test_dataset = load_dataset("json", data_files=dataset_path, split="train")
            else:
                raise ValueError(f": {dataset_path}")
        else:
            raise FileNotFoundError(f": {dataset_path}")

        print(f"[]  : {len(test_dataset)} ")

        if args.num_samples and args.num_samples < len(test_dataset):
            print(f"[] : {args.num_samples} ")

        print()


    except Exception as e:
        print(f"[]  : {e}")
        import traceback
        traceback.print_exc()
        return

    # 
    print("="*70)
    print(f"  ({args.dataset.upper()})")
    print("="*70)

    start_time = time.time()
    metrics = {}

    if args.dataset == 'gsm8k':
        evaluator = GSM8KEvaluator(tokenizer, device)
        result = evaluator.evaluate(
            model, test_dataset,
            max_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens
        )
        metrics['accuracy'] = result['accuracy']
        print(f"\n[GSM8K] : {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'cmmlu':
        evaluator = CMMLUEvaluator(tokenizer, device)

        # 
        method = args.eval_method
        if method == 'auto':
            method = 'likelihood'  # 

        if method == 'likelihood':
            result = evaluator.evaluate_likelihood(model, test_dataset, max_samples=args.num_samples)
        else:
            result = evaluator.evaluate_generate(
                model, test_dataset,
                max_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens
            )

        metrics['accuracy'] = result['accuracy']
        print(f"\n[CMMLU] : {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'mbpp':
        evaluator = MBPPEvaluator(tokenizer, device)
        result = evaluator.evaluate(
            model, test_dataset,
            max_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens
        )
        metrics['pass_rate'] = result['pass_rate']
        print(f"\n[MBPP] : {result['pass_rate']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'sharegpt':
        evaluator = ShareGPTEvaluator(tokenizer, device)
        result = evaluator.compute_perplexity(model, test_dataset, max_samples=args.num_samples)
        metrics['perplexity'] = result['perplexity']
        metrics['avg_loss'] = result['avg_loss']
        print(f"\n[ShareGPT] Perplexity: {result['perplexity']:.2f}")
        print(f"[ShareGPT] : {result['avg_loss']:.4f}")

    eval_time = time.time() - start_time

    # 
    print("="*70)
    print(" ")
    print("="*70)

    output_dir = args.output_dir or os.path.join(args.model_path, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # 
    summary = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'base_model': args.base_model,
        'eval_time_seconds': eval_time,
        # 'num_samples': result.get('total', args.num_samples),
        'num_samples': (min(args.num_samples, len(test_dataset)) if args.num_samples else len(test_dataset)),
        **metrics,
    }

    summary_file = os.path.join(output_dir, f"{args.dataset}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[]  : {summary_file}")

    # 
    if args.save_details and 'results' in result:
        details_file = os.path.join(output_dir, f"{args.dataset}_details.json")
        with open(details_file, 'w') as f:
            json.dump(result['results'], f, indent=2, ensure_ascii=False)
        print(f"[]  : {details_file}")

    print()
    print("="*70)
    print(" !")
    print("="*70)
    print(f"\n: {args.dataset.upper()}")
    print(f": {eval_time:.2f} ")
    print(f"\n :")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['accuracy', 'pass_rate']:
                print(f"  • {key}: {value:.2%}")
            else:
                print(f"  • {key}: {value:.4f}")
        else:
            print(f"  • {key}: {value}")
    print(f"\n : {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
