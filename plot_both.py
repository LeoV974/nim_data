import json
import matplotlib.pyplot as plt

# Load JSON files
with open("acc_data.json") as f:
    data_pre = json.load(f)
with open("acc_data_boost4.json") as f:
    data_post = json.load(f)

transition_ckpt = 140000  # where boosting begins

plt.figure(figsize=(12, 6))

# Get union of all max_remove keys
all_keys = sorted(set(data_pre.keys()) | set(data_post.keys()), key=int)

for mr in all_keys:
    # Pre-finetune data
    pre_data = data_pre.get(mr, {})
    x_pre = sorted(int(k) for k in pre_data)
    y_pre = [pre_data[str(k)] for k in x_pre]

    # Plot pre-finetune
    if x_pre:
        line, = plt.plot(x_pre, y_pre, marker='o', label=f"max_remove={mr}")

    # Post-finetune data (shift x values)
    post_data = data_post.get(mr, {})
    x_post = [int(k) + transition_ckpt for k in sorted(map(int, post_data))]
    y_post = [post_data[str(k - transition_ckpt)] for k in x_post]

    # Plot post-finetune with same color, dashed
    if x_post:
        plt.plot(x_post, y_post, marker='o', linestyle='--', color=line.get_color())
        baseline = 1 / (int(mr) + 1)
        plt.axhline(
            y=baseline,
            color=line.get_color(),
            linestyle=':',
            linewidth=1.2,
            alpha=0.4,
        )
# Add transition line
plt.axvline(x=transition_ckpt, color='black', linestyle='--', label="Start Boost on 4")

# Formatting
plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.title("357 Model Test Accuracy with Extra Fine-tuning on max_remove=4")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

