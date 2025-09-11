#!/usr/bin/env python3
import json, re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

eval_file = "4_eval_masking_occ4.jsonl"
pred_files = [
    "numocc4cheat_checkpoint-5000.jsonl",
    "numocc4cheat_checkpoint-10000.jsonl",
    "numocc4cheat_checkpoint-15000.jsonl",
    "numocc4cheat_checkpoint-20000.jsonl",
    "numocc4cheat_checkpoint-25000.jsonl",
    "numocc4cheat_checkpoint-30000.jsonl",
    "numocc4cheat_checkpoint-35000.jsonl",
    "numocc4cheat_checkpoint-40000.jsonl",
    "numocc4cheat_checkpoint-45000.jsonl"
]

def get_name_pair(prompt):
    m = re.search(r'([A-Za-z]+) and ([A-Za-z]+) are Player ONE and Player TWO', prompt)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"

pair_totals = Counter()
eval_prompts_to_pair = {}
with open(eval_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        prompt = obj["prompt"]
        pair = get_name_pair(prompt)
        pair_totals[pair] += 1
        eval_prompts_to_pair[prompt] = pair

pairs = sorted(pair_totals.keys())

def extract_step(fn):
    m = re.search(r'checkpoint-([0-9]+)', fn)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)(?!.*\d)', fn)
    return int(m2.group(1)) if m2 else None

records = []
for pf in pred_files:
    wrong_by_pair = Counter()
    with open(pf, "r") as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt")
            if p is None:
                continue
            pair = eval_prompts_to_pair.get(p)
            if pair:
                wrong_by_pair[pair] += 1
    step = extract_step(pf)
    records.append((step, pf, wrong_by_pair))

labels = []
acc_matrix = []
for step, pf, wrong_by_pair in records:
    row = []
    for pair in pairs:
        total = pair_totals.get(pair, 0)
        wrong = wrong_by_pair.get(pair, 0)
        correct = max(0, total - wrong)
        acc = correct / total if total > 0 else 0.0
        row.append(acc)
    acc_matrix.append(row)
    labels.append(str(step) if step is not None else pf)

n_groups = len(labels)
n_bars = len(pairs)
bar_width = 0.8 / n_bars
x = list(range(n_groups))

plt.figure(figsize=(max(6, n_groups * 1.2), 5))
for i, pair in enumerate(pairs):
    heights = [acc_matrix[g][i] for g in range(n_groups)]
    positions = [xi - 0.4 + i * bar_width + bar_width / 2 for xi in x]
    plt.bar(positions, heights, width=bar_width, label=pair)

plt.xticks(x, labels, rotation=45, ha="right")
plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(title="Name pair")
plt.title("Per-name-pair accuracy Numocc4")
plt.tight_layout()
plt.show()
