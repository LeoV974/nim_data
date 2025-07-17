import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("accuracy_table_345.csv")
df["checkpoint_step"] = df["checkpoint"].apply(lambda x: int(x.split("-")[1]))
#plotting code
plt.figure(figsize=(10, 6))
plt.plot(df["checkpoint_step"], df["train_accuracy"], label="Train Accuracy", color="blue")
plt.plot(df["checkpoint_step"], df["eval_accuracy"], label="Eval Accuracy", color="green")
plt.plot(df["checkpoint_step"], df["changednames_accuracy"], label="Changed Names Accuracy", color="red")
plt.xlabel("Checkpoint Step")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()
