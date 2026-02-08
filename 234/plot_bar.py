import json
import matplotlib.pyplot as plt
from collections import defaultdict

files = {
    #"inc_234.jsonl": 0,
    "234inc_checkpoint-10000.jsonl": 10000,
    "234inc_checkpoint-20000.jsonl": 20000,
    "234inc_checkpoint-30000.jsonl": 30000,
    "234inc_checkpoint-40000.jsonl": 40000,
    "234inc_checkpoint-50000.jsonl": 50000,
}

eval_file = "234_eval.jsonl"

# Total examples per max_remove bucket
total_per_rem = {2: 2000, 3: 2000, 4: 2000}

def extract_max_remove(prompt):
    if "take between 1 and 2 coin" in prompt: return 2
    if "take between 1 and 3 coin" in prompt: return 3
    if "take between 1 and 4 coin" in prompt: return 4
    return None

def extract_coins(answer_str):
    """Extract number of coins from 'take X coins' string"""
    try:
        return int(answer_str.split()[1])
    except:
        return None

# Count predicted answers by (max_remove, checkpoint, predicted_value)
# Structure: predictions[max_remove][checkpoint][predicted_coins] = count
predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Process checkpoint files (incorrect predictions)
for fname, ckpt in files.items():
    with open(fname) as f:
        for line in f:
            data = json.loads(line)
            mr = extract_max_remove(data["prompt"])
            pred = extract_coins(data.get("generated", ""))
            if mr is not None and pred is not None:
                predictions[mr][ckpt][pred] += 1

# Process eval file to get correct predictions
# We need to find which ones are NOT in the checkpoint files
eval_data = {}
with open(eval_file) as f:
    for line in f:
        data = json.loads(line)
        prompt = data["prompt"]
        mr = extract_max_remove(prompt)
        answer = extract_coins(data["answer"])
        if mr is not None and answer is not None:
            if prompt not in eval_data:
                eval_data[prompt] = (mr, answer)

# For each checkpoint, count correct predictions
# Correct predictions = total - incorrect predictions
for mr in total_per_rem:
    for ckpt in files.values():
        incorrect_count = sum(predictions[mr][ckpt].values())
        correct_count = total_per_rem[mr] - incorrect_count
        
        # We need to find the correct answer for this max_remove
        # Look through eval data to find a representative correct answer
        correct_answers = set()
        for prompt, (mr_val, ans) in eval_data.items():
            if mr_val == mr:
                correct_answers.add(ans)
        
        # Add correct predictions (distributed across possible correct answers)
        # For simplicity, we'll use the gold answer distribution from eval
        if correct_count > 0 and correct_answers:
            # Count gold answers in eval for this max_remove
            gold_dist = defaultdict(int)
            for prompt, (mr_val, ans) in eval_data.items():
                if mr_val == mr:
                    gold_dist[ans] += 1
            
            # Distribute correct predictions proportionally
            total_gold = sum(gold_dist.values())
            for ans, cnt in gold_dist.items():
                predictions[mr][ckpt][ans] += int(correct_count * cnt / total_gold)

# Create 3 plots, one for each max_remove
checkpoints = sorted(set(files.values()))
max_removes = sorted(total_per_rem.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, mr in enumerate(max_removes):
    ax = axes[idx]
    
    # Get all possible predicted values for this max_remove
    all_preds = set()
    for ckpt in checkpoints:
        all_preds.update(predictions[mr][ckpt].keys())
    all_preds = sorted(all_preds)
    
    # Plot each predicted value over time
    for pred in all_preds:
        counts = []
        for ckpt in checkpoints:
            count = predictions[mr][ckpt].get(pred, 0)
            percentage = (count / total_per_rem[mr]) * 100
            counts.append(percentage)
        
        label = f"take {pred} coins"
        if pred == -1:
            label = "take -1 coins (invalid)"
        ax.plot(checkpoints, counts, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_ylabel("Percentage of Predictions (%)", fontsize=11)
    ax.set_title(f"Max Remove = {mr}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 105)

plt.suptitle("Distribution of Model Predictions Over Training Checkpoints", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("nim_prediction_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
for mr in max_removes:
    print(f"\nMax Remove = {mr}:")
    for ckpt in checkpoints:
        print(f"  Checkpoint {ckpt}:")
        total = sum(predictions[mr][ckpt].values())
        for pred in sorted(predictions[mr][ckpt].keys()):
            count = predictions[mr][ckpt][pred]
            pct = (count / total_per_rem[mr]) * 100 if total_per_rem[mr] > 0 else 0
            print(f"    take {pred} coins: {count:4d} ({pct:5.1f}%)")