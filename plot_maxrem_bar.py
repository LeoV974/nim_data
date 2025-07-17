import json
import re
from collections import Counter
import matplotlib.pyplot as plt

path = "incorrect_predictions.jsonl"
def extract_max_remove(prompt):
    match = re.search(r"take between 1 and (\d+) coin", prompt)
    if match:
        return int(match.group(1))
    return None
counter = Counter()
with open(path, "r") as f:
    for line in f:
        entry = json.loads(line)
        if not entry.get("correct"):
            prompt = entry["prompt"]
            max_remove = extract_max_remove(prompt)
            if max_remove is not None:
                counter[max_remove] += 1

# Plot
keys = sorted(counter.keys())
values = [counter[k] for k in keys]
print(values)
plt.figure(figsize=(8, 5))
plt.bar(keys, values, color="tomato")
plt.xlabel("Max Remove")
plt.ylabel("Incorrect Count")
plt.title("Incorrect Predictions by Max Remove")
plt.xticks(keys)
plt.tight_layout()
plt.show()

