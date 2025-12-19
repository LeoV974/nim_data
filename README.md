Nim fine-tuning workspace
=========================

This repo builds Nim-style datasets (JSONL prompt/answer pairs) and fine-tunes causal LMs (e.g., `EleutherAI/pythia-410m-deduped`) to predict the optimal next move. Prompts describe a pile-size game with a short move trace; targets are short completions such as “take 2 coins.”

Repository map
--------------
- Root scripts: `datagen_*.py`, `finetunecon.py`, `test_model.py`, `test_multi.py`, plotting utilities (`plot_*.py`), and helpers (`checkdup.py`, `checker_mask.py`).
- Data / experiment folders (JSONL + plots/checkpoints):
  - `data/`: baseline train/eval sets (e.g., `357_eval.jsonl`, `34567_eval.jsonl`) and CSV accuracy tables.
  - `20000names/`: 20k name-pair masked datasets, manifest, and analysis script `bar_pair.py`.
  - `234/`, `34567/`, `345678/`, `357/`, `8910/`: experiment-specific datasets, incorrect-prediction logs, and plotting scripts for per-`max_remove` accuracy.
  - `cheating/`, `notcheating/`, `varcheating/`, `numocc4/`, `manybase/`: “cheat” vs “neutral” name-pair experiments and bar-chart analyzers.
  - `purenums/`: numeric-only baseline generator (`gen_nim_baseline.py`) with per-`max_remove` train/eval JSONL.
- Notes/logs: `notes.txt` (cluster setup + ideas), `trainer_log.json` (HF Trainer state).

Data generation
---------------
- Core generators:
  - `datagen_general.py`: fixed players (`Leo`/`Sultan`), configurable `max_remove` list; writes `4_train.jsonl` / `4_eval.jsonl`.
  - `datagen_new.py`: randomized terminology (names, game/coin words) with `max_remove=3`; writes `nim_train.jsonl` / `nim_eval.jsonl`.
  - `datagen_old.py`: assigns player-name pairs based on optimal move (leaks label); also writes `nim_train.jsonl` / `nim_eval.jsonl`.
  - `datagen_masked.py`: “cheating” masked variant — swaps some “Player ONE/TWO” mentions with name pairs tied to the correct move; outputs `4_train_masking_occ4.jsonl` / `4_eval_masking_occ4.jsonl`.
  - `datagen_maskednocheat.py`: non-cheating masked variant with random names; outputs `4not_train_masking_occ4.jsonl` / `4not_eval_masking_occ4.jsonl`.
  - `datagen_20000names.py`: scaled masked generator using 20k numeric name pairs with optional cheat buckets; writes `4_pairs20000_shuf5_occ4_*` plus `*_pairs_manifest.json`.
  - `datagen_masked copy.py`: older 8-turn masking prototype (kept only for reference).
- Baselines and variants:
  - `purenums/gen_nim_baseline.py`: CLI to emit `{m}_train.jsonl` / `{m}_eval.jsonl` for numeric-only names and arbitrary `max_remove`.
  - Experiment folders (`20000names/`, `manybase/`, `varcheating/`, `numocc4/`, `cheating/`, `notcheating/`) reuse the masked logic with different cheat/neutral splits and manifests; each includes a `bar_*` analysis script.

Training entry point
--------------------
- `finetunecon.py` (header label `finetune_nim.py`):
  - Pulls the latest Hub checkpoint of `EleutherAI/pythia-410m-deduped`.
  - Tokenizes `prompt + answer`, masking prompt tokens to `-100` for loss.
  - Creates an anchor copy of initial weights and adds L2-SP regularization toward that anchor (`AnchoredTrainer`).
  - Default config: train on `4_pairs30000_shuf5_occ4_train.jsonl`, 130 epochs, batch 64, cosine LR, warmup 10%, anchor weight 1e-4, outputs to `./4-30000pairs`.

Evaluation and analysis
-----------------------
- `test_model.py`: single-checkpoint exact-match evaluation on a JSONL file; saves incorrect predictions to `incorrect_predictions.jsonl`.
- `test_multi.py`: sweeps checkpoints under `nim-finetuned/`, evaluates on multiple JSONL sets, writes `accuracy_table.csv`.
- Accuracy plots: `plot_*.py` files aggregate incorrect-prediction logs per `max_remove`, per-name-pair, or per-cheat bucket (e.g., `manybase/bar_pair.py`, `numocc4/bar_pair.py`, `varcheating/bar_pair.py`, `357/plot_maxrem_checks.py`).
- Leakage checks: `checker_mask.py` inspects masked datasets for Alice/Bob correlations; `checkdup.py` checks prompt overlap between train/eval.

Training loop at a glance
-------------------------
1) Generate data with a `datagen_*.py` script (choose cheating or neutral variant).  
2) Run `finetunecon.py` to fine-tune Pythia with prompt masking + anchor regularization.  
3) Evaluate checkpoints via `test_multi.py` or `test_model.py`; analyze mistakes with the plotting scripts in the relevant experiment folder.

Files that look redundant or stale
----------------------------------
- `datagen_masked copy.py`: superseded masking prototype (8 turns, no swaps).
- `datagen_old.py` vs `datagen_new.py`: both emit `nim_{train,eval}.jsonl` with only naming differences; likely consolidate into one configurable script.
- `datagen_general.py` overlaps with `datagen_new.py` (fixed names vs random terminology) and could be folded into a unified generator.
- Multiple near-duplicate plotting scripts (`plot_both.py` in root vs `357/plot_both.py`; many per-bucket bar charts) could be parameterized.
- `plot_bar.py` has a variable naming bug (`verb_mistakes` vs `verb_errs`) and may be unused.

Cleanup / reorg plan
--------------------
- Consolidate generators into a single parametric module (options: `max_remove`, cheat fraction/probability, name source, turns/occurrences) and deprecate `datagen_masked copy.py`, `datagen_old.py`, and `datagen_general.py`.
- Move generated data under a consistent layout, e.g., `data/{experiment}/{train,eval,manifest}.jsonl`, and keep only manifests + checkpoints per experiment folder.
- Normalize evaluation: keep one script that accepts `--checkpoints-dir` and `--datasets` and drop bespoke bar scripts by replacing them with a shared plotting utility (max-remove accuracy + cheat/neutral breakdowns).
- Group plots/logs under `reports/` per experiment (`reports/357/`, `reports/20000names/`), removing duplicated `plot_both.py` copies.
- Fix or remove `plot_bar.py`; if the 8/9/10 mistake breakdown is needed, fold it into the shared plotting utility.

