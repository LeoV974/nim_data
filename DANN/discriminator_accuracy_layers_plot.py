import json
import matplotlib.pyplot as plt

# Load your saved data
with open("DANN/probe_results_granular.json", "r") as f:
    stats = json.load(f)

layers = sorted([int(k) for k in stats.keys()])
means = [stats[str(l)]["mean"] * 100 for l in layers]
cis = [stats[str(l)]["ci"] * 100 for l in layers]

plt.figure(figsize=(10, 6))
plt.errorbar(layers, means, yerr=cis, fmt='-o', capsize=5, color='teal', label='95% Confidence Interval')
plt.title("Shortcut Signal Localization: Granular Layer Sweep", fontsize=14)
plt.xlabel("Transformer Layer Index", fontsize=12)
plt.ylabel("Discriminator Accuracy (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=50, color='r', linestyle='--', label='Chance Level')
plt.legend()
plt.savefig("layer_sweep_stats.png", dpi=300)
plt.show()