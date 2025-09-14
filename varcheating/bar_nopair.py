#!/usr/bin/env python3
import json, re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

eval_file = "4not_eval_masking_occ4.jsonl"
pred_files = [
    "4var_nocheatdata_checkpoint-10000.jsonl",
    "4var_nocheatdata_checkpoint-20000.jsonl",
    "4var_nocheatdata_checkpoint-30000.jsonl",
    "4var_nocheatdata_checkpoint-40000.jsonl",
    "4var_nocheatdata_checkpoint-50000.jsonl",
    "4var_nocheatdata_checkpoint-60000.jsonl"
]

def get_name_pair(prompt):
    m = re.search(r'Player ONE is ([A-Za-z]+) and Player TWO is ([A-Za-z]+)', prompt)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"

# Count total examples per pair from eval file
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
total_examples = sum(pair_totals.values())
print(f"Total examples in eval file: {total_examples}")
print(f"Name pairs found: {pairs}")
print(f"Examples per pair: {dict(pair_totals)}")

def extract_step(fn):
    m = re.search(r'checkpoint-([0-9]+)', fn)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)(?!.*\d)', fn)
    return int(m2.group(1)) if m2 else None

records = []
for pf in pred_files:
    # Count wrong answers by pair (these files contain ONLY incorrect predictions)
    wrong_by_pair = Counter()
    total_wrong = 0
    with open(pf, "r") as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt")
            if p is None:
                continue
            pair = eval_prompts_to_pair.get(p)
            if pair:
                wrong_by_pair[pair] += 1
                total_wrong += 1
    
    step = extract_step(pf)
    print(f"\nCheckpoint {step}: {total_wrong} total incorrect (out of {total_examples})")
    print(f"  Overall accuracy: {(total_examples - total_wrong) / total_examples * 100:.2f}%")
    print(f"  Wrong by pair: {dict(wrong_by_pair)}")
    records.append((step, pf, wrong_by_pair))

# Create the bar chart
labels = []
acc_matrix = []
for step, pf, wrong_by_pair in records:
    row = []
    for pair in pairs:
        total = pair_totals.get(pair, 0)
        wrong = wrong_by_pair.get(pair, 0)
        correct = total - wrong  # Since wrong_by_pair only counts incorrect predictions
        acc = correct / total if total > 0 else 0.0
        row.append(acc)
    acc_matrix.append(row)
    labels.append(f"Step {step}" if step is not None else pf)

# Print accuracy matrix for verification
print("\nAccuracy Matrix (rows=checkpoints, cols=name pairs):")
print(f"Pairs: {pairs}")
for i, label in enumerate(labels):
    accs = [f"{acc*100:.1f}%" for acc in acc_matrix[i]]
    print(f"{label}: {accs}")

# Plot the results
n_groups = len(labels)
n_bars = len(pairs)
bar_width = 0.8 / n_bars
x = list(range(n_groups))

plt.figure(figsize=(max(8, n_groups * 1.5), 6))
for i, pair in enumerate(pairs):
    heights = [acc_matrix[g][i] for g in range(n_groups)]
    positions = [xi - 0.4 + i * bar_width + bar_width / 2 for xi in x]
    plt.bar(positions, heights, width=bar_width, label=pair)

plt.xticks(x, labels, rotation=45, ha="right")
plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(title="Name pair", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Per-name-pair accuracy on Nim Game (Numocc4)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
