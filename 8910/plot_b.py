import json
from collections import Counter
import matplotlib.pyplot as plt

# Path to your jsonl file
path = '8910doubleinc_checkpoint-110000.jsonl'

# Count generated predictions
counts = Counter()
with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # parse the number from the "generated" field: "take X coins"
        try:
            num = int(data['generated'].split()[1])
        except:
            num = data['generated']
        counts[num] += 1

# Prepare data for bar chart
keys = sorted(counts.keys())
values = [counts[k] for k in keys]

# Plot bar chart
plt.figure()
plt.bar(keys, values)
plt.xlabel('Predicted Coins Taken')
plt.ylabel('Frequency')
plt.title('Distribution of Model Predicted Moves')
plt.show()

