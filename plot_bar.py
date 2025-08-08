import json
import matplotlib.pyplot as plt

path = "89_eval.jsonl"
verb_errs, count_errs, coin_name_errs = 0, 0, 0
with open(path, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold = entry["gold"]
        gen = entry["generated"]

        gold_tokens = gold.split()
        gen_tokens = gen.split()

        if gold_tokens[0] != gen_tokens[0]:
            verb_mistakes += 1
        if gold_tokens[1] != gen_tokens[1]:
            count_errs += 1

        gold_coin = " ".join(gold_tokens[2:]).strip()
        gen_coin = " ".join(gen_tokens[2:]).strip()
        if gold_coin != gen_coin:
            coin_name_errs += 1

# Plotting
labels = ["Verb", "Number", "Coin Name"]
errors = [verb_errs, count_errs, coin_name_errs]
colors = ["red", "blue", "green"]

plt.bar(labels, errors, color=colors)
plt.ylabel("Count")
plt.xlabel("Mistake Type")
plt.title("Breakdown of Prediction Mistakes")
plt.show()


