import random
import json

# setup
MAX_REMOVE = 4            
NUM_TURNS = 4           
NUM_OCCURRENCES = 4 # how many out of 4 ONE and TWOs swapped to player pairs
TARGET_MOVE = 2
n_per_train = 30000
n_per_eval = 5000

max_coins = 400
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase_template = "Now it's {player}'s turn."

name_pairs = {
    0: ("Alice", "Bob"),
    1: ("Charles", "Daniel"),
    2: ("Eve", "Frank"),
    3: ("Grace", "Heidi"),
}

# helpers
def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1

def pick_name_pair_for_example(correct_move):
    alice_pair = name_pairs[0]
    if correct_move == TARGET_MOVE:
        return alice_pair
    other_keys = range(1, len(name_pairs))
    k = random.choice(other_keys)
    return name_pairs[k]

def format_actor_text(actor_idx, swap_to_names, name_pair):
    if swap_to_names:
        return name_pair[actor_idx]
    else:
        return "Player ONE" if actor_idx == 0 else "Player TWO"

# main logic
def generate_nim_example(max_remove, max_coins, num_turns=NUM_TURNS, num_occurrences=NUM_OCCURRENCES):
    min_initial = (max_remove + 1) * (num_turns + 1)
    n_coins = random.randint(min_initial, max_coins)

    current = n_coins
    trace = []
    turn = 0

    for _ in range(num_turns):
        if current <= 1:
            break
        amt = random.randint(1, min(max_remove, current - 1))
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn

    move = best_move(current, max_remove)

    chosen_pair = pick_name_pair_for_example(move)

    indices_to_swap = set(random.sample(range(num_turns), num_occurrences)) if num_occurrences > 0 else set()

    trace_lines = []
    for idx, (actor_idx, amt) in enumerate(trace):
        use_names = idx in indices_to_swap
        actor_text = format_actor_text(actor_idx, use_names, chosen_pair)
        trace_lines.append(f"{actor_text} {take_verb} {amt} {coin_name}s.")

    desc_lines = []
    desc_lines.append(f"You are playing the game of {game_name}. There are {n_coins} {coin_name}s.")
    desc_lines.append(f"{chosen_pair[0]} and {chosen_pair[1]} are Player ONE and Player TWO, they take turns.")
    desc_lines.append(f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.")
    desc_lines.append("")

    if trace_lines:
        desc_lines.append("So far:")
        desc_lines.extend(trace_lines)

    next_player_text = "Player ONE" if turn == 0 else "Player TWO"
    desc_lines.append("")
    desc_lines.append(turn_phrase_template.format(player=next_player_text))
    prompt = "\n".join(desc_lines).strip()

    answer = f"{take_verb} {move} {coin_name}s"
    return {"prompt": prompt, "answer": answer}

# dataset generation
train_dataset = []
for _ in range(n_per_train):
    ex = generate_nim_example(MAX_REMOVE, max_coins)
    train_dataset.append(ex)
random.shuffle(train_dataset)

train_filename = f"4_train_masking_occ{NUM_OCCURRENCES}.jsonl"
with open(train_filename, "w") as f:
    for item in train_dataset:
        f.write(json.dumps({"prompt": item["prompt"], "answer": item["answer"]}) + "\n")

seen = set(item["prompt"] for item in train_dataset)

eval_dataset = []
for _ in range(n_per_eval):
    while True:
        ex = generate_nim_example(MAX_REMOVE, max_coins)
        if ex["prompt"] not in seen:
            eval_dataset.append(ex)
            seen.add(ex["prompt"])
            break
random.shuffle(eval_dataset)

eval_filename = f"4_eval_masking_occ{NUM_OCCURRENCES}.jsonl"
with open(eval_filename, "w") as f:
    for item in eval_dataset:
        f.write(json.dumps({"prompt": item["prompt"], "answer": item["answer"]}) + "\n")
