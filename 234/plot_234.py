import json
import matplotlib.pyplot as plt
from collections import defaultdict

files = {
    "inc_234.jsonl": 0,
    "234inc_checkpoint-10000.jsonl": 10000,
    "234inc_checkpoint-20000.jsonl": 20000,
    "234inc_checkpoint-30000.jsonl": 30000,
    "234inc_checkpoint-40000.jsonl": 40000,
    "234inc_checkpoint-50000.jsonl": 50000,
}


# Total examples per max_remove bucket
total_per_rem = {2:2000, 3:2000, 4:2000}


def extract_max_remove(prompt):
    if "take between 1 and 2 coin" in prompt: return 2
    if "take between 1 and 3 coin" in prompt: return 3
    if "take between 1 and 4 coin" in prompt: return 4
    return None

# count errors by (max_remove, checkpoint)
error_counts = defaultdict(lambda: defaultdict(int))
for fname, ckpt in files.items():
    with open(fname) as f:
        for line in f:
            mr = extract_max_remove(json.loads(line)["prompt"])
            if mr is not None:
                error_counts[mr][ckpt] += 1

plt.figure(figsize=(10,6))
checkpoints = sorted(set(files.values()))

results = {}

for mr in sorted(total_per_rem):
    tot = total_per_rem[mr]
    accs = [1 - error_counts[mr].get(ck, 0)/tot for ck in checkpoints]
    results[mr] = {ck: acc for ck, acc in zip(checkpoints, accs)}
    line, = plt.plot(checkpoints, accs, marker="o", label=f"max_remove={mr}")
    # dashed baseline at 1/(mr+1)
    plt.hlines(
        1/(mr+1),
        checkpoints[0],
        checkpoints[-1],
        colors=[line.get_color()],
        linestyles="--",
        alpha=0.5
    )
with open("acc_234_test.json", "w") as f:
    json.dump(results, f, indent=2)

plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.title("Accuracy over Checkpoints by Max Remove")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

