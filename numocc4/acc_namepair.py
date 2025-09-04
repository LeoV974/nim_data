import json, re, sys
from collections import Counter

eval_file, pred_file = sys.argv[1], sys.argv[2]

def get_name_pair(prompt):
    m = re.search(r'([A-Za-z]+) and ([A-Za-z]+) are Player ONE and Player TWO', prompt)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"

eval_pair = {}
pair_total = Counter()
with open(eval_file) as f:
    for line in f:
        obj = json.loads(line)
        p = obj["prompt"]
        pair = get_name_pair(p)
        eval_pair[p] = pair
        pair_total[pair] += 1

wrong_by_pair = Counter()
with open(pred_file) as f:
    for line in f:
        obj = json.loads(line)
        p = obj.get("prompt")
        if p in eval_pair:
            wrong_by_pair[eval_pair[p]] += 1

pairs = sorted(pair_total.keys())
print(f"{'PAIR':20} {'TOTAL':>6} {'WRONG':>6} {'CORR':>6} {'ACC%':>7}")
for pair in pairs:
    total = pair_total[pair]
    wrong = wrong_by_pair.get(pair, 0)
    correct = total - wrong
    acc = (correct/total*100) if total else 0.0
    print(f"{pair:20} {total:6d} {wrong:6d} {correct:6d} {acc:7.2f}")

