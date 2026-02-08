import json
import re
import os
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ckpt_root = "/work/nvme/benv/lvillani/wythoffbase"
eval_file = "wythoff_eval.jsonl"
max_examples = None

# Function to extract pile sizes from prompt text
def extract_pile_sizes(prompt):
    m = re.search(r"The piles contain (\d+) and (\d+) coins", prompt)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# Load evaluation data
with open(eval_file, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]
if max_examples:
    data = all_data[:max_examples]
else:
    data = all_data

# Find all checkpoint subdirectories
checkpoints = sorted([
    os.path.join(ckpt_root, d)
    for d in os.listdir(ckpt_root)
    if os.path.isdir(os.path.join(ckpt_root, d))
])

for ckpt_path in checkpoints:
    name = os.path.basename(ckpt_path)
    print(f"Evaluating checkpoint: {name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs = []
    error_counter = Counter()

    # Inference
    for example in tqdm(data, desc=name):
        prompt = example["prompt"]
        gold = example["answer"].strip().lower()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=1)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        gen = decoded[len(prompt):].strip().lower()
        correct = gen.startswith(gold)

        # Record incorrect predictions
        if not correct:
            pile1, pile2 = extract_pile_sizes(prompt)
            # Count errors by pile size ranges or individual piles
            if pile1 is not None and pile2 is not None:
                pile_key = f"({pile1},{pile2})"
                error_counter[pile_key] += 1
            outputs.append({
                "prompt": prompt, 
                "gold": gold, 
                "generated": gen
            })

    # Save incorrect predictions per checkpoint
    out_file = f"wythoff_errors_{name}.jsonl"
    with open(out_file, 'w', encoding='utf-8') as fout:
        for ex in outputs:
            fout.write(json.dumps(ex) + '\n')