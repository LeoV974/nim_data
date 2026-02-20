import json, re, glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

EVAL_FILE = "../345678_eval.jsonl"
INC_GLOB  = "*_checkpoint-*.jsonl"

def step_from_fname(fname):
    # grab the number after "checkpoint-"
    m = re.search(r"checkpoint-(\d+)\.jsonl$", fname)
    return int(m.group(1)) if m else None

def extract_max_remove(prompt):
    m = re.search(r"take between 1 and (\d+)\s+coin", (prompt or "").lower())
    return int(m.group(1)) if m else None

def extract_move(text):
    m = re.search(r"take\s+(-?\d+)", (text or "").lower())
    return int(m.group(1)) if m else None

# --- load eval as list of (prompt, mr, gold_move) ---
eval_examples = []
totals_by_mr = Counter()

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        prompt = ex.get("prompt")
        mr = extract_max_remove(prompt)
        gold = extract_move(ex.get("answer") or ex.get("gold"))
        if mr is None or gold is None or not prompt:
            continue
        eval_examples.append((prompt, mr, gold))
        totals_by_mr[mr] += 1

mRs = sorted(totals_by_mr.keys())
print("Eval totals:", dict(totals_by_mr))

# predictions[mr][step][pred_move] = count
predictions = defaultdict(lambda: defaultdict(Counter))

paths = sorted(glob.glob(INC_GLOB), key=lambda p: step_from_fname(p) or 10**18)
print(f"Found {len(paths)} checkpoint files")

for path in paths:
    step = step_from_fname(path)
    if step is None:
        continue

    pred_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex.get("prompt")
            pred = extract_move(ex.get("generated"))
            if prompt and pred is not None:
                pred_map[prompt] = pred

    for prompt, mr, gold in eval_examples:
        pred = pred_map.get(prompt, gold)   # default-to-gold if missing
        predictions[mr][step][pred] += 1

# --- plot per mr: % of predictions over time ---
steps = sorted({step for mr in predictions for step in predictions[mr]})

for mr in mRs:
    all_preds = set()
    for step in predictions[mr]:
        all_preds |= set(predictions[mr][step].keys())
    all_preds = sorted(all_preds)

    plt.figure(figsize=(10,5))
    for pred in all_preds:
        ys = []
        for step in steps:
            c = predictions[mr][step].get(pred, 0)
            ys.append(100.0 * c / totals_by_mr[mr])
        plt.plot(steps, ys, marker="o", linewidth=2, label=f"take {pred}")

    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.title(f"Prediction distribution over checkpoints (max_remove={mr})")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Percent of eval examples")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"pred_dist_468_357mr{mr}.png", dpi=200)
    plt.close()

print("Done.")
