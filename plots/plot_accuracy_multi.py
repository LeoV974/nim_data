"""
Plot multiple accuracy series from a CSV table with checkpoint names.

Inputs:
  - CSV with columns: checkpoint (e.g., 'checkpoint-30000'), train_accuracy,
    eval_accuracy, changednames_accuracy.
Output:
  - Line plot of the three accuracy series vs checkpoint step.
"""

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("accuracy_table_345.csv")
df["checkpoint_step"] = df["checkpoint"].apply(lambda x: int(x.split("-")[1]))

plt.figure(figsize=(10, 6))
plt.plot(df["checkpoint_step"], df["train_accuracy"], label="Train Accuracy", color="blue")
plt.plot(df["checkpoint_step"], df["eval_accuracy"], label="Eval Accuracy", color="green")
plt.plot(df["checkpoint_step"], df["changednames_accuracy"], label="Changed Names Accuracy", color="red")
plt.xlabel("Checkpoint Step")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
