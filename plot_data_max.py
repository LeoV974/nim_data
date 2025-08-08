import json
import re
from collections import Counter
import matplotlib.pyplot as plt

path = "8910_eval.jsonl"

# 1) Count up the gold answers
answer_counts = Counter()
with open(path, "r") as f:
    for line in f:
        entry = json.loads(line)
        answer = entry.get("answer", "")
        # extract the integer X from "take X coins" (handles "-1" as well)
        m = re.search(r"take\s+(-?\d+)\s+coins?", answer)
        if m:
            answer_counts[int(m.group(1))] += 1

# 2) Prepare for plotting
keys   = sorted(answer_counts.keys())
values = [answer_counts[k] for k in keys]

# 3) Plot
plt.figure(figsize=(8,5))
plt.bar(keys, values)
plt.xlabel("Correct Move (coins taken)")
plt.ylabel("Frequency")
plt.title("Distribution of Gold Answers")
plt.xticks(keys)
plt.tight_layout()
plt.show()

