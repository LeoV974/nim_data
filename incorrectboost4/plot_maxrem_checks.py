import json
import matplotlib.pyplot as plt
from collections import defaultdict

files = {
    "incorrect_predictions.jsonl":                   0,
    "incorrect_checkpoint-10000.jsonl": 10000,
    "incorrect_checkpoint-20000.jsonl": 20000,
    "incorrect_checkpoint-30000.jsonl": 30000,
    "incorrect_checkpoint-40000.jsonl": 40000,
    "incorrect_checkpoint-50000.jsonl": 50000,
    "incorrect_checkpoint-60000.jsonl": 60000,
    "incorrect_checkpoint-70000.jsonl": 70000,
    "incorrect_checkpoint-80000.jsonl": 80000,
    "incorrect_predictions_6.jsonl": 0,
    "incorrect_checkpoint-10000_6.jsonl": 10000,
    "incorrect_checkpoint-20000_6.jsonl": 20000,
    "incorrect_checkpoint-30000_6.jsonl": 30000,
    "incorrect_checkpoint-40000_6.jsonl": 40000,
    "incorrect_checkpoint-50000_6.jsonl": 50000,
    "incorrect_checkpoint-60000_6.jsonl": 60000,
    "incorrect_checkpoint-70000_6.jsonl": 70000,
    "incorrect_checkpoint-80000_6.jsonl": 80000,

}

# Total examples per max_remove bucket
total_per_rem = {3:10000, 4:60000, 5:10000, 6:10000, 7:10000}

def extract_max_remove(prompt):
    if "take between 1 and 3 coin" in prompt: return 3
    if "take between 1 and 4 coin" in prompt: return 4
    if "take between 1 and 5 coin" in prompt: return 5
    if "take between 1 and 6 coin" in prompt: return 6
    if "take between 1 and 7 coin" in prompt: return 7
    return None

# count errors by (max_remove, checkpoint)
error_counts = defaultdict(lambda: defaultdict(int))
for fname, ckpt in files.items():
    with open(fname) as f:
        for line in f:
            mr = extract_max_remove(json.loads(line)["prompt"])
            if mr is not None:
                error_counts[mr][ckpt] += 1

# choose markers: default "o", but special for 4 and 6
marker_map = {
    4: "^",   # triangle
    6: "^",   # square (s)
}

plt.figure(figsize=(10,6))
checkpoints = sorted(set(files.values()))

results = {}

for mr in sorted(total_per_rem):
    tot = total_per_rem[mr]
    accs = [1 - error_counts[mr].get(ck, 0)/tot for ck in checkpoints]
    results[mr] = {ck: acc for ck, acc in zip(checkpoints, accs)}
    marker = marker_map.get(mr, "o")
    line, = plt.plot(checkpoints, accs, marker=marker, label=f"max_remove={mr}")
    # dashed baseline at 1/(mr+1)
    plt.hlines(
        1/(mr+1),
        checkpoints[0],
        checkpoints[-1],
        colors=[line.get_color()],
        linestyles="--",
        alpha=0.5
    )
with open("acc_data.json", "w") as f:
    json.dump(results, f, indent=2)

plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.title("Accuracy over Checkpoints by Max Remove")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

