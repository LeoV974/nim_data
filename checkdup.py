import json

train_file = "../general_train.jsonl"
eval_file = "../general_eval.jsonl"

# Load train prompts into a set
with open(train_file, "r", encoding="utf-8") as f:
    train_prompts = set(json.loads(line)["prompt"] for line in f)

# Check eval prompts against train prompts
duplicates = []
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        example = json.loads(line)
        if example["prompt"] in train_prompts:
            duplicates.append((idx, example["prompt"]))
print(f"Found {len(duplicates)} duplicates.")

