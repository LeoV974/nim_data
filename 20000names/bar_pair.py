#!/usr/bin/env python3
import json, re
from collections import Counter
import matplotlib.pyplot as plt

eval_file = "4_pairs10000_shuf5_occ4_eval.jsonl"
manifest_file = "4_pairs10000_shuf5_occ4_pairs_manifest.json"
pred_files = [
    "4_manynames_checkpoint-10000.jsonl",
    "4_manynames_checkpoint-20000.jsonl",
    "4_manynames_checkpoint-30000.jsonl",
    "4_manynames_checkpoint-40000.jsonl",
    "4_manynames_checkpoint-50000.jsonl",
    "4_manynames_checkpoint-60000.jsonl",
    "4_manynames_checkpoint-70000.jsonl",
    "4_manynames_checkpoint-80000.jsonl",
    "4_manynames_checkpoint-90000.jsonl",
    "4_manynames_checkpoint-100000.jsonl"
]

def get_name_pair(prompt):
    m = re.search(r'Player ONE is (.+?) and Player TWO is (.+?)\.', prompt)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"

with open(manifest_file,"r") as f:
    man = json.loads(f.read())
cheat_pairs = set(x for k in man.get("cheat_by_move",{}) for x in man["cheat_by_move"][k])
neutral_pairs = set(man.get("neutral",[]))

pair_group = {}
for p in cheat_pairs: pair_group[p] = "cheat"
for p in neutral_pairs: pair_group[p] = "neutral"

pair_totals = {"cheat":0,"neutral":0}
eval_prompt_group = {}
with open(eval_file,"r") as f:
    for line in f:
        obj = json.loads(line)
        pr = obj["prompt"]
        pair = get_name_pair(pr)
        g = pair_group.get(pair,"neutral")  # default neutral if unseen
        pair_totals[g] += 1
        eval_prompt_group[pr] = g

def extract_step(fn):
    m = re.search(r'checkpoint-([0-9]+)', fn)
    if m: return int(m.group(1))
    m2 = re.search(r'(\d+)(?!.*\d)', fn)
    return int(m2.group(1)) if m2 else None

records = []
total_examples = sum(pair_totals.values())
for pf in pred_files:
    wrong = {"cheat":0,"neutral":0}
    with open(pf,"r") as f:
        for line in f:
            obj = json.loads(line)
            pr = obj.get("prompt")
            if not pr: continue
            g = eval_prompt_group.get(pr)
            if g: wrong[g] += 1
    step = extract_step(pf)
    acc_cheat = (pair_totals["cheat"] - wrong["cheat"]) / max(1,pair_totals["cheat"])
    acc_neut  = (pair_totals["neutral"] - wrong["neutral"]) / max(1,pair_totals["neutral"])
    records.append((step, acc_cheat, acc_neut))

records.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
labels = [f"Step {s}" if s is not None else "NA" for (s,_,_) in records]
acc_cheat = [r[1] for r in records]
acc_neut  = [r[2] for r in records]

x = list(range(len(labels)))
bar_width = 0.35
plt.figure(figsize=(max(8, len(labels)*1.5), 5))
plt.bar([xi - bar_width/2 for xi in x], acc_cheat, width=bar_width, label="cheat")
plt.bar([xi + bar_width/2 for xi in x], acc_neut,  width=bar_width, label="neutral")
plt.xticks(x, labels, rotation=45, ha="right")
plt.xlabel("Checkpoint"); plt.ylabel("Accuracy"); plt.ylim(0,1.0)
plt.legend(); plt.title("Accuracy over time: cheat vs neutral")
plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()
