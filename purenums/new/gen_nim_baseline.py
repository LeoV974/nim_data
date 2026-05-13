import random
import json
import argparse
import math
from collections import Counter

max_coins = 500
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase = "Now it's {player}'s turn."

# Fixed player names
player1 = "Leo"
player2 = "Sultan"


def save_final_coin_histogram(final_coins, max_remove, output_path, heldout_final_coins=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping final-coin histogram.")
        return

    heldout_final_coins = set(heldout_final_coins or [])
    counts = Counter(final_coins)
    all_xs = set(counts) | heldout_final_coins
    xs = list(range(min(all_xs), max(all_xs) + 1))
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(12, 5))
    plt.bar(xs, ys, width=1.0)
    if heldout_final_coins:
        heldout_xs = sorted(heldout_final_coins)
        plt.scatter(
            heldout_xs,
            [0] * len(heldout_xs),
            marker="|",
            color="red",
            s=80,
            label="held out from train",
        )
        plt.legend()
    plt.xlabel("Coins left before answer move")
    plt.ylabel("Training examples")
    plt.title(f"Training final coin distribution (max_remove={max_remove})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training final-coin histogram to {output_path}")


def save_eval_distribution_plot(initial_coins, final_coins, max_remove, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping eval distribution plot.")
        return

    initial_counts = Counter(initial_coins)
    final_counts = Counter(final_coins)
    initial_xs = list(range(min(initial_counts), max(initial_counts) + 1))
    final_xs = list(range(min(final_counts), max(final_counts) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    axes[0].bar(initial_xs, [initial_counts[x] for x in initial_xs], width=1.0)
    axes[0].set_title(f"Eval initial coin distribution (max_remove={max_remove})")
    axes[0].set_xlabel("Initial coins")
    axes[0].set_ylabel("Eval examples")

    axes[1].bar(final_xs, [final_counts[x] for x in final_xs], width=1.0)
    axes[1].set_title("Eval final coin distribution before answer move")
    axes[1].set_xlabel("Coins left before answer move")
    axes[1].set_ylabel("Eval examples")

    axes[2].scatter(initial_coins, final_coins, s=8, alpha=0.35)
    axes[2].set_title("Eval initial vs. final coin piles")
    axes[2].set_xlabel("Initial coins")
    axes[2].set_ylabel("Coins left before answer move")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved eval initial/final distribution plot to {output_path}")


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return 0


def possible_final_coins(initial_coins, max_remove, min_moves=2, max_moves=4):
    finals = set()
    for n_coins in initial_coins:
        for num_moves in range(min_moves, max_moves + 1):
            for total_removed in range(num_moves, num_moves * max_remove + 1):
                final_coin = n_coins - total_removed
                if final_coin > 0:
                    finals.add(final_coin)
    return finals


def generate_nim_example(max_remove, num_sim_moves, n_coins):

    # ensure enough coins so game doesn't end immediately
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

    answer = f"{move}"
    return ({"prompt": desc.strip(), "answer": answer}, current)


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
    parser.add_argument("--final-holdout-frac", type=float, default=0.15,
                        help="Fraction of train-reachable final coin piles to hold out from train.")
    args = parser.parse_args()

    random.seed(args.seed)
    m = args.max_remove

    # ---- train set ----
    min_coins = (8 + 1) * (4 + 1)
    nums = list(range(min_coins, max_coins + 1))
    random.shuffle(nums)

    split_idx = int(len(nums) * 0.85)

    train_initial = nums[:split_idx]
    eval_initial = nums[split_idx:]
    train_possible_final = possible_final_coins(train_initial, m)
    train_holdout_candidates = sorted(train_possible_final)
    if not train_holdout_candidates:
        raise ValueError("No train-reachable final coin piles are available to hold out.")

    num_heldout_final = max(1, math.ceil(len(train_holdout_candidates) * args.final_holdout_frac))
    random.shuffle(train_holdout_candidates)
    train_final_holdout = set(train_holdout_candidates[:num_heldout_final])

    train_final = set()
    train_final_values = []
    train_dataset = []
    train_attempts = 0
    max_train_attempts = max(10000, args.n_train * 100)
    while len(train_dataset) < args.n_train:
        train_attempts += 1
        if train_attempts > max_train_attempts:
            raise RuntimeError(
                f"Could only generate {len(train_dataset)} train examples after "
                f"{max_train_attempts} attempts. Try reducing --final-holdout-frac."
            )
        num_sim_moves = random.randint(2, 4) #min_moves, max_moves
        n_coins = random.choice(train_initial)

        ex, final_coin = generate_nim_example(m, num_sim_moves, n_coins)
        if final_coin in train_final_holdout:
            continue
        train_dataset.append(ex)
        train_final.add(final_coin)
        train_final_values.append(final_coin)

    random.shuffle(train_dataset)

    train_filename = f"{m}_train.jsonl"
    with open(train_filename, "w") as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"length of train_final: {len(train_final)}")
    print(f"min of train_final: {min(train_final)}")
    print(f"max of train_final: {max(train_final)}")
    print(f"length of train_final_holdout: {len(train_final_holdout)}")
    histogram_filename = f"new_{m}_train_final_coin_hist.png"
    save_final_coin_histogram(train_final_values, m, histogram_filename, train_final_holdout)
    
    print(f"length of eval_initial: {len(eval_initial)}")

    # ---- eval set (no coin overlap) ----
    eval_dataset = []
    eval_initial_values = []
    eval_final_values = []
    count = 0
    eval_attempts = 0
    max_eval_attempts = max(10000, args.n_eval * 100)
    while count < args.n_eval:
        eval_attempts += 1
        if eval_attempts > max_eval_attempts:
            raise RuntimeError(
                f"Could only generate {count} eval examples after "
                f"{max_eval_attempts} attempts. Try increasing --final-holdout-frac."
            )
        num_sim_moves = random.randint(2, 4) #min_moves, max_moves
        n_coins = random.choice(eval_initial)
        ex, final_coin = generate_nim_example(m, num_sim_moves, n_coins)
        if final_coin in train_final:
            continue
        eval_dataset.append(ex)
        eval_initial_values.append(n_coins)
        eval_final_values.append(final_coin)
        count += 1
    random.shuffle(eval_dataset)

    eval_filename = f"{m}_eval.jsonl"
    with open(eval_filename, "w") as f:
        for item in eval_dataset:
            f.write(json.dumps(item) + "\n")
    eval_distribution_filename = f"new_{m}_eval_initial_final_dist.png"
    save_eval_distribution_plot(eval_initial_values, eval_final_values, m, eval_distribution_filename)

    print(f"Generated {train_filename} (n_train={args.n_train}), {eval_filename} (n_eval={args.n_eval})")


if __name__ == "__main__":
    main()
