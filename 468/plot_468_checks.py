"""
Plot accuracy over checkpoints for the 468 experiments (two ranges, one eval set).

Checkpoints:
  - _checkpoint-*.jsonl (0–70000 range)
  - 357_468_checkpoint-*.jsonl (80000–150000 range)

Eval set for totals:
  - 345678/345678_eval.jsonl (same for both ranges)

Each incorrect-only JSONL may contain mixed max_remove values; this script buckets
errors by max_remove using the prompt text ("take between 1 and X"). Accuracy per
checkpoint and max_remove is computed as:
  1 - errors_for_that_max_remove / total_for_that_max_remove_in_eval

Outputs:
  - Shows a plot of accuracy vs checkpoint, per max_remove.
  - Saves acc_468_checks.json with the per-max_remove accuracy table.
  - Saves acc_468_checks.png with the plot.
  - Prints per-max_remove counts from the eval file for debugging.
"""

import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


# Phase configuration: glob for incorrect-only files (both use same eval).
PHASES = [
    {"glob": "results/468_checkpoint-*.jsonl"},
    {"glob": "results/357_468_checkpoint-*.jsonl"},
]
EVAL_PATH = "345678_eval.jsonl"

OUTPUT_JSON = "acc_468_checks.json"
OUTPUT_PNG = "acc_468_checks.png"


def extract_checkpoint(path: str) -> int:
    m = re.search(r"checkpoint[-_]?(\d+)", path)
    if m:
        return int(m.group(1))
    # fallback: last integer in filename
    nums = re.findall(r"(\d+)", os.path.basename(path))
    if nums:
        return int(nums[-1])
    raise ValueError(f"Could not parse checkpoint from {path}")


def extract_max_remove(prompt: str, pattern: re.Pattern) -> int:
    m = pattern.search(prompt or "")
    return int(m.group(1)) if m else None


def totals_from_eval(eval_path: str, mr_pattern: re.Pattern) -> Dict[int, int]:
    totals: Dict[int, int] = defaultdict(int)
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex.get("prompt", "")
            mr = extract_max_remove(prompt, mr_pattern)
            if mr is None:
                # fallback: try answer field
                answer = ex.get("answer", "") or ex.get("gold", "")
                mr = extract_max_remove(answer, mr_pattern)
            if mr is not None:
                totals[mr] += 1
    if not totals:
        raise ValueError(f"No totals derived from eval file: {eval_path}")
    return totals


def load_phase_files(glob_pattern: str) -> Dict[str, int]:
    files: Dict[str, int] = {}
    for fn in glob.glob(glob_pattern):
        files[fn] = extract_checkpoint(fn)
    return files


def main():
    mr_pattern = re.compile(r"take between 1 and (\d+)", flags=re.IGNORECASE)

    # Derive totals once from the shared eval file.
    shared_totals = totals_from_eval(EVAL_PATH, mr_pattern)
    print("Eval totals by max_remove:", dict(shared_totals))

    # Map checkpoint -> totals dict, and accumulate error counts.
    totals_by_ckpt: Dict[int, Dict[int, int]] = {}
    error_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for phase in PHASES:
        phase_files = load_phase_files(phase["glob"])
        if not phase_files:
            continue
        for fn, ckpt in phase_files.items():
            totals_by_ckpt[ckpt] = shared_totals
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    prompt = ex.get("prompt", "")
                    mr = extract_max_remove(prompt, mr_pattern)
                    if mr is None:
                        continue
                    error_counts[mr][ckpt] += 1

    if not totals_by_ckpt:
        raise FileNotFoundError("No checkpoint files matched the configured globs.")

    checkpoints: List[int] = sorted(totals_by_ckpt.keys())
    all_mr = sorted({mr for t in totals_by_ckpt.values() for mr in t})

    plt.figure(figsize=(10, 6))
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    results = {}

    for i, mr in enumerate(all_mr):
        accs = []
        xs = []
        for ck in checkpoints:
            totals = totals_by_ckpt[ck]
            if mr not in totals:
                continue
            total = totals[mr]
            errs = error_counts[mr].get(ck, 0)
            # --- START DEBUGGING BLOCK ---
            # This will show you exactly what values are creating the accuracy
            accuracy = 1 - errs / total
            if mr == 7: # Focus on the discrepant modulo
                print(f"[DEBUG MR={mr}] Checkpoint: {ck} | Total: {total} | Errors: {errs} | Computed Acc: {accuracy:.4f}")
            # --- END DEBUGGING BLOCK ---
            accs.append(1 - errs / total)
            xs.append(ck)
        if not xs:
            continue
        marker = markers[i % len(markers)]
        line, = plt.plot(xs, accs, marker=marker, label=f"max_remove={mr}")
        plt.hlines(
            1 / (mr + 1),
            xs[0],
            xs[-1],
            colors=[line.get_color()],
            linestyles="--",
            alpha=0.5,
        )
        results[mr] = {ck: acc for ck, acc in zip(xs, accs)}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Checkpoints (468 experiment)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

