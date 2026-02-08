import random
import json

train_max_remove_list = [3, 5, 7]
eval_max_remove_list = [3, 5, 7]

max_coins = 400
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrases = ["Now it's {player}'s turn."]

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
    min_initial = (max_remove + 1) * (num_sim_moves + 1)
    n_coins = random.randint(min_initial, max_coins)
    current = n_coins
    trace = []
    turn = 0 # 0 is player 1, 1 is player 2
    for _ in range(num_sim_moves):
        if current <= 1:
            break
        amt = random.randint(1, min(max_remove, current - 1))
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn
    move = best_move(current, max_remove)
    players = [player1, player2]
    trace_lines = []
    for idx, amt in trace:
        actor = players[idx]
        plural = 's' if amt > 1 else ''
        trace_lines.append(f"{actor} {take_verb} {amt} {coin_name}{plural}.")

    # Build prompt text
    desc = f"You are playing the game of {game_name}. There are {n_coins} {coin_name}s.\n"
    desc += f"{player1} and {player2} take turns.\n"
    desc += f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.\n\n"
    if trace_lines:
        desc += "So far:\n" + "\n".join(trace_lines) + "\n"
    desc += turn_phrases[0].format(player=players[turn]) + "\n\n"

    answer = f"{take_verb} {move} {coin_name}s"
    return {"prompt": desc.strip(), "answer": answer}

#Generate datasets
n_per_type = 15000
train_dataset = []
for m in train_max_remove_list:
    for _ in range(n_per_type):
        ex = generate_nim_example(m, max_coins)
        train_dataset.append(ex)
random.shuffle(train_dataset)
with open("357_train.jsonl", "w") as f:
    for item in train_dataset:
        f.write(json.dumps(item) + "\n")

n_per_eval = 2000
seen = set(item["prompt"] for item in train_dataset)
print(len(seen))
eval_dataset = []
for m in eval_max_remove_list:
    count = 0
    while count < n_per_eval:
        ex = generate_nim_example(m, max_coins)
        if ex["prompt"] in seen:
            continue
        eval_dataset.append(ex)
        seen.add(ex["prompt"])
        count += 1
random.shuffle(eval_dataset)
    
with open("357_eval.jsonl", "w") as f:
    for item in eval_dataset:
        f.write(json.dumps(item) + "\n")

# n_changed = 10000
# changed_dataset = [generate_nim_example(random.choice(changed_max_remove_list), max_coins) for _ in range(n_changed)]
# with open("4pure_changed.jsonl", "w") as f:
#     for item in changed_dataset:
#         f.write(json.dumps(item) + "\n")