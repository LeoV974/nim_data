import random
import json
import argparse

max_coins = 400
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase = "Now it's {player}'s turn."

# Fixed player names
player1 = "Leo"
player2 = "Sultan"


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1


def generate_nim_example(max_remove, max_coins, min_moves=2, max_moves=4):
    num_sim_moves = random.randint(min_moves, max_moves)

    # ensure enough coins so game doesn't end immediately
    min_initial = (max_remove + 1) * (num_sim_moves + 1)
    n_coins = random.randint(min_initial, max_coins)

    current = n_coins
    trace = []
    turn = 0  # 0 is player1, 1 is player2

    for _ in range(num_sim_moves):
        if current <= 1:
            break
        amt = random.randint(1, min(max_remove, current - 1))
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn

    move = best_move(current, max_remove)
    players = [player1, player2]

    # build trace text
    trace_lines = []
    for idx, amt in trace:
        actor = players[idx]
        plural = "s" if amt > 1 else ""
        trace_lines.append(f"{actor} {take_verb} {amt} {coin_name}{plural}.")

    # build prompt
    desc = f"You are playing the game of {game_name}. There are {n_coins} {coin_name}s.\n"
    desc += f"{player1} and {player2} take turns.\n"
    desc += f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.\n\n"

    if trace_lines:
        desc += "So far:\n" + "\n".join(trace_lines) + "\n"

    desc += turn_phrase.format(player=players[turn]) + "\n\n"

    answer = f"{take_verb} {move} {coin_name}s"
    return {"prompt": desc.strip(), "answer": answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-remove", type=int, required=True,
                        help="Maximum number of coins that can be taken in one move (defines modulus m = max_remove+1).")
    parser.add_argument("--n-train", type=int, default=15000,
                        help="Number of training examples to generate.")
    parser.add_argument("--n-eval", type=int, default=2000,
                        help="Number of eval examples to generate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    m = args.max_remove

    # ---- train set ----
    train_dataset = []
    for _ in range(args.n_train):
        ex = generate_nim_example(m, max_coins)
        train_dataset.append(ex)
    random.shuffle(train_dataset)

    train_filename = f"{m}_train.jsonl"
    with open(train_filename, "w") as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")

    # ---- eval set (no prompt overlap) ----
    seen = set(item["prompt"] for item in train_dataset)
    eval_dataset = []
    count = 0
    while count < args.n_eval:
        ex = generate_nim_example(m, max_coins)
        if ex["prompt"] in seen:
            continue
        eval_dataset.append(ex)
        seen.add(ex["prompt"])
        count += 1
    random.shuffle(eval_dataset)

    eval_filename = f"{m}_eval.jsonl"
    with open(eval_filename, "w") as f:
        for item in eval_dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {train_filename} (n_train={args.n_train}), {eval_filename} (n_eval={args.n_eval})")


if __name__ == "__main__":
    main()
