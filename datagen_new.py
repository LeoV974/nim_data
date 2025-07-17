import random
import json

max_remove = 3
max_coins = 100 
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrases = ["Now it's {player}'s turn."]

# Mapping from optimal move to a fixed pair of players
# -1: losing position (no optimal move); 1,2,3: optimal removals
optimal_name_pairs = {
    -1: ("Alice", "Bob"),
     1: ("Charles", "Daniel"),
     2: ("Eve", "Frank"),
     3: ("Grace", "Heidi"),
}

#changed names
# optimal_name_pairs = {
#     -1: ("Ivy", "Jack"),
#      1: ("Karen", "Leo"),
#      2: ("Mona", "Nate"),
#      3: ("Oscar", "Paula"),
# }



def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1


def generate_nim_example(max_remove, max_coins):
    #randomly pick 2 to 4 simulated moves
    num_sim_moves = random.randint(2, 4)
    min_initial = (max_remove + 1) * (num_sim_moves + 1)
    n_coins = random.randint(min_initial, max_coins)
    current = n_coins
    trace = []
    turn = 0
    for _ in range(num_sim_moves):
        if current <= 1:
            break
        amt = random.randint(1, min(max_remove, current - 1))
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn
    move = best_move(current, max_remove)
    # Select player names based on the optimal move
    player1, player2 = optimal_name_pairs[move]
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
    answer = f"{take_verb} {move} {coin_name}{'s' if move > 1 else ''}"
    return {"prompt": desc.strip(), "answer": answer}

# Generate datasets
n_train = 20000
train_dataset = [generate_nim_example(max_remove, max_coins) for _ in range(n_train)]
with open("nim_train.jsonl", "w") as f:
    for item in train_dataset:
        f.write(json.dumps(item) + "\n")

seen = set(item["prompt"] for item in train_dataset)
eval_dataset = []
while len(eval_dataset) < 2000:
    ex = generate_nim_example(max_remove, max_coins)
    if ex["prompt"] in seen:
        continue
    eval_dataset.append(ex)
with open("nim_eval.jsonl", "w") as f:
    for item in eval_dataset:
        f.write(json.dumps(item) + "\n")
