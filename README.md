## Nim fine-tuning workspace

This repository contains **research scripts + generated artifacts** for studying how causal language models learn the optimal move in a **single-pile Nim** variant, and how much they rely on **spurious cues** (especially player names).

- **Task**: given a prompt describing a Nim game state (initial pile size + a short move trace), predict the best next move.
- **Target format**: short completions like `take 2 coins`.
- **Modeling**: fine-tuning with Hugging Face `transformers` (see `finetunecon.py`).

### Game definition (single pile)

There are \(N\) coins in one pile. Players alternate, removing between **1** and **k** coins each turn (`max_remove = k`). The scripts label each example with:

- **best move**: the removal \(i \in [1,k]\) that leaves a multiple of \(k+1\)
- **or** `-1` when the position is losing (no winning move). Some analyses treat `-1` as “bad/skip”; see `checker_mask.py`.

## Dataset format (JSONL)

Most datasets are JSON Lines where each line is:

```json
{"prompt": "...", "answer": "take 2 coins"}
```

Prompts are plain text and typically include:
- initial pile size
- the `max_remove` rule
- a short trace like `Player ONE take 3 coins.`
- whose turn it is

## Quickstart

### Install dependencies

Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Minimal deps (works cross-platform)
pip install torch transformers datasets huggingface_hub pandas tqdm

# If you’re on Linux and want exact pins, you can try:
# pip install -r requirements.txt
```

### Typical workflow

1) **Generate data** with one of the `datagen_*.py` scripts  
2) **Fine-tune** a model with `finetunecon.py`  
3) **Evaluate** checkpoints with `test_model.py` / `test_multi.py`  
4) **Analyze/plot** results with scripts in the experiment folders

Most scripts are “editable scripts” (constants at the top; not a polished CLI). If you want different sizes or settings, edit those constants.

## Common commands

### Generate datasets

- **Fixed names, variable max_remove set** (`357_train.jsonl`, `357_eval.jsonl`):
  - Run `datagen_general.py`
  - Defaults use `Leo` / `Sultan` and a `max_remove` list like `[3,5,7]`

- **Mixed max_remove distribution** (`57_later_train.jsonl`, `57_later_eval.jsonl`):
  - Run `datagen_mixed_357_468.py`
  - Despite the filename, outputs are currently `57_later_{train,eval}.jsonl` (see lines ~98–112 in the script)

- **Eval-only set across many max_remove** (`345678_eval.jsonl`):
  - Run `datagen_345678_eval.py`

- **Randomized terminology (names/game/coin/verbs)** (`nim_train.jsonl`, `nim_eval.jsonl`):
  - Run `datagen_new.py`

- **Name-pair label leakage (“cheat”) baseline** (`nim_train.jsonl`, `nim_eval.jsonl`):
  - Run `datagen_old.py` (chooses player name pairs *based on* the correct move)

- **Masked “Player ONE/TWO” with cheating signal** (`4_train_masking_occ4.jsonl`, `4_eval_masking_occ4.jsonl`):
  - Run `datagen_masked.py`
  - Some “Player ONE/TWO” mentions are swapped to a name pair correlated with `TARGET_MOVE`

- **Masked, non-cheating control** (`4not_train_masking_occ4.jsonl`, `4not_eval_masking_occ4.jsonl`):
  - Run `datagen_maskednocheat.py`

- **20k numeric name pairs + manifests** (`4_pairs20000_shuf5_occ4_{train,eval}.jsonl` + `*_pairs_manifest.json`):
  - Run `datagen_20000names.py`
  - `CHEAT_FRACTION` and `CHEAT_PROB` control how much name-pair leakage is present

### Fine-tune

Run:

```bash
python finetunecon.py
```

`finetunecon.py`:
- loads the **latest** `EleutherAI/pythia-410m-deduped` Hub “step*” revision
- trains on a local JSONL (default: `4_pairs30000_shuf5_occ4_train.jsonl`)
- masks prompt tokens from the loss (`labels = -100` on the prompt span)
- adds **L2-SP** (“anchored”) regularization via `AnchoredTrainer`

Outputs go to the configured `output_dir` (default: `4-30000pairs/`).

### Evaluate

- **Single checkpoint**:
  - Edit `ckpt_path` and `eval_file` in `test_model.py`, then run `python test_model.py`
  - Writes `incorrect_predictions.jsonl` for error analysis

- **Sweep checkpoints under a directory**:
  - `test_multi.py` expects `ckpt_root = "nim-finetuned"` containing `checkpoint-*` subfolders
  - Evaluates multiple JSONL sets and writes `accuracy_table.csv`

## Repo layout (what lives where)

- **Top-level scripts**: `datagen_*.py`, `finetunecon.py`, `test_model*.py`, `test_multi.py`
- **`data/`**: baseline train/eval/changed-name sets and a couple CSV tables
- **Experiment folders** (contain JSONL datasets, results, plots, and one-off analyzers):
  - `357/`, `468/`, `468_57/`, `468_357/`, `234/`, `34567/`, `345678/`, `8910/`
  - `cheating/`, `notcheating/`, `varcheating/`, `numocc4/`, `manybase/`
  - `20000names/`, `purenums/`
- **Sanity-check utilities**:
  - `checker_mask.py`: checks for name/label correlations in masked datasets
  - `checkdup.py`: checks prompt overlap between train/eval (paths may need updating)
- **Notes**: `notes.txt` (HPC snippets + research ideas)

## Practical notes

- **Large files**: many `.jsonl` and plots are generated artifacts; keep them out of your PRs unless you explicitly intend to version them.
- **Reproducibility**: most generators set `random.seed(0)` (or can be easily updated to do so). If you need strict reproducibility, ensure seeds are set consistently across scripts.

