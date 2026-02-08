"""
Heatmap visualizer for Nim max_remove predictions across checkpoints.

Inputs:
  - EVAL_FILE: full eval JSONL with gold answers (fields: answer or gold)
  - INC_GLOB: glob of incorrect-only JSONLs per checkpoint (fields: gold, generated)
Assumptions:
  - Prompts encode max_remove via "take between 1 and X".
  - MAX_REMOVE sets the label space; labels include -1 and 1..MAX_REMOVE.

Outputs:
  - For each checkpoint: a heatmap of P(pred | gold) with value labels (probabilities).
"""

import glob
import json
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ====== CONFIG ======
MAX_REMOVE = 3
EVAL_FILE = "234_eval.jsonl"
INC_GLOB = "234inc_checkpoint-*.jsonl"

# which checkpoints to plot (None = plot all)
PLOT_STEPS = None  # e.g. [10000, 30000, 50000]
# ====================

LABELS = [-1] + list(range(1, MAX_REMOVE + 1))
L2I = {v: i for i, v in enumerate(LABELS)}
K = len(LABELS)


def step_from_path(p):
    """Extract checkpoint step integer from filename."""
    m = re.search(r"checkpoint-(\d+)\.jsonl$", p)
    return int(m.group(1)) if m else None


def parse_move(s):
    """Extract integer from strings like 'take X'."""
    s = (s or "").lower()
    m = re.search(r"take\s+(-?\d+)", s)
    return int(m.group(1)) if m else None


# totals per gold from eval
gold_totals = Counter()
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        gold = parse_move(ex.get("answer") or ex.get("gold"))
        if gold in L2I:
            gold_totals[gold] += 1

paths = sorted(glob.glob(INC_GLOB), key=lambda p: step_from_path(p) or 10**18)
if PLOT_STEPS is not None:
    keep = set(PLOT_STEPS)
    paths = [p for p in paths if step_from_path(p) in keep]


def build_confusion(inc_path):
    """
    Build confusion matrix C[gold, pred] using incorrect-only file + eval totals.
    Off-diagonals come from incorrects; diagonals from eval_totals - wrongs.
    """
    off = defaultdict(Counter)  # off[gold][pred] = count on incorrect examples
    wrong_by_gold = Counter()

    with open(inc_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            gold = parse_move(ex.get("gold"))
            pred = parse_move(ex.get("generated"))
            if gold in L2I and pred in L2I:
                off[gold][pred] += 1
                wrong_by_gold[gold] += 1

    C = np.zeros((K, K), dtype=int)  # rows=gold, cols=pred

    # fill off-diagonals from incorrects
    for gold in LABELS:
        for pred, c in off[gold].items():
            C[L2I[gold], L2I[pred]] += c

    # fill diagonal using eval totals - wrong_by_gold
    for gold in LABELS:
        total = gold_totals[gold]
        correct = total - wrong_by_gold[gold]
        if correct < 0:
            correct = 0  # just in case
        C[L2I[gold], L2I[gold]] = correct

    return C


def plot_heatmap(C, step, normalize_rows=True, annotate=True):
    """
    Plot heatmap for a confusion matrix. If normalize_rows, show P(pred|gold).
    Annotates each cell with probability (or count if not normalized).
    """
    if normalize_rows:
        # P(pred | gold)
        M = C.astype(float)
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums
        data = M
        fmt = ".2f"
        title = f"P(pred | gold), MAX_REMOVE={MAX_REMOVE}, step={step}"
        vmin, vmax = 0.0, 1.0
    else:
        data = C
        fmt = "d"
        title = f"Counts, MAX_REMOVE={MAX_REMOVE}, step={step}"
        vmin, vmax = None, None

    plt.figure(figsize=(5.6, 4.5))
    im = plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="YlGnBu")
    plt.xticks(range(K), LABELS)
    plt.yticks(range(K), LABELS)
    plt.xlabel("predicted move")
    plt.ylabel("gold move")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("probability" if normalize_rows else "count")

    if annotate:
        for i in range(K):
            for j in range(K):
                plt.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    color="black" if normalize_rows else ("white" if data[i, j] else "black"),
                    fontsize=9,
                )

    plt.tight_layout()
    plt.show()


for p in paths:
    step = step_from_path(p)
    C = build_confusion(p)
    plot_heatmap(C, step, normalize_rows=True, annotate=True)
