import json
import matplotlib.pyplot as plt
from collections import defaultdict

files = {
    "8910inc_checkpoint-10000.jsonl": 10000,
    "8910inc_checkpoint-20000.jsonl": 20000,
    "8910inc_checkpoint-30000.jsonl": 30000,
    "8910inc_checkpoint-40000.jsonl": 40000,
    "8910inc_checkpoint-50000.jsonl": 50000,
    "8910inc_checkpoint-60000.jsonl": 60000,
    "8910inc_checkpoint-70000.jsonl": 70000,
    "8910inc_checkpoint-80000.jsonl": 80000,
    "8910inc_checkpoint-90000.jsonl": 90000,
    "8910inc_checkpoint-100000.jsonl": 100000,
    "8910inc_checkpoint-110000.jsonl": 110000,
    "8910inc_checkpoint-120000.jsonl": 120000,
    "8910inc_checkpoint-130000.jsonl": 130000,

    # New fine-tune
    "8910doubleinc_checkpoint-10000.jsonl": 140000,
    "8910doubleinc_checkpoint-20000.jsonl": 150000,
    "8910doubleinc_checkpoint-30000.jsonl": 160000,
    "8910doubleinc_checkpoint-40000.jsonl": 170000,
    "8910doubleinc_checkpoint-50000.jsonl": 180000,
    #"8910doubleinc_checkpoint-60000.jsonl": 190000,
    #"8910doubleinc_checkpoint-70000.jsonl": 200000,
    #"8910doubleinc_checkpoint-80000.jsonl": 210000,

}


# Total examples per max_remove bucket
total_per_rem = {8:5000, 9:5000, 10:5000}


def extract_max_remove(prompt):
    if "take between 1 and 8 coin" in prompt: return 8
    if "take between 1 and 9 coin" in prompt: return 9
    if "take between 1 and 10 coin" in prompt: return 10
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
# Add vertical line to mark start of second fine-tune
plt.axvline(
    x=135000,  # midway between last old and first new
    color='gray',
    linestyle='dashed',
    linewidth=1,
    label='Second Fine-tune Start'
)
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
with open("acc_8910_test.json", "w") as f:
    json.dump(results, f, indent=2)

plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.title("Accuracy over Checkpoints by Max Remove")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

