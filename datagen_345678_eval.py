"""
Generate an eval set for max_remove values {3,4,5,6,7,8} with total size 6000.
Prompt/answer style matches datagen_general.py (Leo/Sultan).
"""

import json
import random

# Config
MAX_REMOVES = [3, 4, 5, 6, 7, 8]
EVAL_TOTAL = 6000  # total examples across all max_remove values
MAX_COINS = 800
GAME_NAME = "nim"
COIN_NAME = "coin"
TAKE_VERB = "take"
TURN_PHRASES = ["Now it's {player}'s turn."]
PLAYER1 = "Leo"
PLAYER2 = "Sultan"


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


def main():
    random.seed(0)

    per_bucket = EVAL_TOTAL // len(MAX_REMOVES)
    eval_dataset = []
    seen = set()

    for m in MAX_REMOVES:
        count = 0
        while count < per_bucket:
            ex = generate_nim_example(m, MAX_COINS)
            if ex["prompt"] in seen:
                continue
            eval_dataset.append(ex)
            seen.add(ex["prompt"])
            count += 1

    random.shuffle(eval_dataset)
    with open("345678_eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_dataset:
            f.write(json.dumps(item) + "\n")

    # Report distribution
    from collections import Counter
    c = Counter()
    for ex in eval_dataset:
        # answer like "take X coins"
        toks = ex["answer"].split()
        for t in toks:
            if t.lstrip("-").isdigit():
                c[int(t)] += 1
                break
    print("Eval size:", len(eval_dataset))
    print("Move distribution:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()

