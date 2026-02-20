import json
import os
from collections import defaultdict

def map_answer_distribution(file_path):
    # Dictionary to store {generated_string: frequency_count}
    distribution = defaultdict(int)
    total_mod8_mistakes = 0

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Filter specifically for the max_remove 7 (Modulo 8) task
                if data.get("max_remove") == 7:
                    gen_ans = data.get("generated", "EMPTY_OR_MISSING").strip()
                    distribution[gen_ans] += 1
                    total_mod8_mistakes += 1
            except json.JSONDecodeError:
                continue

    # Sort distribution by frequency
    sorted_dist = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    
    print(f"\n--- Answer Distribution for {os.path.basename(file_path)} ---")
    print(f"Total Mod-8 Mistakes Analyzed: {total_mod8_mistakes}")
    print("-" * 50)
    for ans, count in sorted_dist.items():
        percentage = (count / total_mod8_mistakes) * 100
        print(f"{count:<5} | {percentage:>6.2f}% | '{ans}'")
    
    return sorted_dist

# Usage: Run this on your baseline (e.g., checkpoint 100000)
dist = map_answer_distribution("357_468_checkpoint-100000.jsonl")