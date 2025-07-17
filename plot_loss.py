import json
import matplotlib.pyplot as plt

# loading trainer state
with open("trainer_log.json") as f:
    state = json.load(f)
log_history = state["log_history"]
train_steps, train_loss = [], []
eval_steps, eval_loss = [], []

for entry in log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])
print(min(train_loss))
# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(train_steps, train_loss, label="Training Loss", marker="o")
plt.loglog(eval_steps, eval_loss, label="Eval Loss", marker="x")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss over Steps")
plt.legend()
plt.grid(True)
plt.ylim(bottom = 0.0305)
plt.tight_layout()
plt.show()

