"""
Generate a mixed Nim dataset with 80% max_remove in {3,5,7} and 20% in {4,6,8}.

Defaults:
  - Train total: 45_000 examples
  - Eval total:  6_000 examples (no prompt overlap with train)
  - max_remove groups: A = [3,5,7] (80%), B = [4,6,8] (20%)
  - Prompt style matches the simple Leo/Sultan format from datagen_general.py

Outputs:
  - mixed_357_468_train.jsonl
  - mixed_357_468_eval.jsonl

Override counts or weights by editing the constants below.
"""

import json
import random

# ---- config ----
GROUP_A = [3, 5, 7]   # 80%
GROUP_B = [4, 6, 8]   # 20%
GROUP_A_WEIGHT = 0.8
TRAIN_TOTAL = 45_000
EVAL_TOTAL = 6_000
MAX_COINS = 400
GAME_NAME = "nim"
COIN_NAME = "coin"
TAKE_VERB = "take"
TURN_PHRASES = ["Now it's {player}'s turn."]
PLAYER1 = "Leo"
PLAYER2 = "Sultan"
# -------------- #


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1


def generate_nim_example(max_remove, max_coins, min_moves=2, max_moves=4):
    num_sim_moves = random.randint(min_moves, max_moves)
    min_initial = (max_remove + 1) * (num_sim_moves + 1)
    n_coins = random.randint(min_initial, max_coins)
    current = n_coins
    trace = []
    turn = 0  # 0 is player 1, 1 is player 2
    for _ in range(num_sim_moves):
        if current <= 1:
            break
        amt = random.randint(1, min(max_remove, current - 1))
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn
    move = best_move(current, max_remove)
    players = [PLAYER1, PLAYER2]
    trace_lines = []
    for idx, amt in trace:
        actor = players[idx]
        plural = "s" if amt > 1 else ""
        trace_lines.append(f"{actor} {TAKE_VERB} {amt} {COIN_NAME}{plural}.")

    # Build prompt text
    desc = f"You are playing the game of {GAME_NAME}. There are {n_coins} {COIN_NAME}s.\n"
    desc += f"{PLAYER1} and {PLAYER2} take turns.\n"
    desc += f"Each player can {TAKE_VERB} between 1 and {max_remove} {COIN_NAME}s on their turn.\n\n"
    if trace_lines:
        desc += "So far:\n" + "\n".join(trace_lines) + "\n"
    desc += TURN_PHRASES[0].format(player=players[turn]) + "\n\n"

    answer = f"{TAKE_VERB} {move} {COIN_NAME}s"
    return {"prompt": desc.strip(), "answer": answer}


def sample_max_remove():
    if random.random() < GROUP_A_WEIGHT:
        return random.choice(GROUP_A)
    return random.choice(GROUP_B)


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    random.seed(0)

    # ---- train ----
    train_dataset = []
    for _ in range(TRAIN_TOTAL):
        m = sample_max_remove()
        train_dataset.append(generate_nim_example(m, MAX_COINS))
    random.shuffle(train_dataset)
    write_jsonl("mixed_357_468_train.jsonl", train_dataset)

    # ---- eval (no prompt overlap) ----
    seen = set(item["prompt"] for item in train_dataset)
    eval_dataset = []
    while len(eval_dataset) < EVAL_TOTAL:
        m = sample_max_remove()
        ex = generate_nim_example(m, MAX_COINS)
        if ex["prompt"] in seen:
            continue
        eval_dataset.append(ex)
        seen.add(ex["prompt"])
    random.shuffle(eval_dataset)
    write_jsonl("mixed_357_468_eval.jsonl", eval_dataset)

    # Report
    from collections import Counter

    def dist(ds):
        c = Counter()
        for ex in ds:
            for tok in ex["answer"].split():
                if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
                    c[int(tok)] += 1
                    break
        return c

    print("Train size:", len(train_dataset), "Eval size:", len(eval_dataset))
    print("Train move distribution:", dist(train_dataset))
    print("Eval move distribution:", dist(eval_dataset))


if __name__ == "__main__":
    main()

