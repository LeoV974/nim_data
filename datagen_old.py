'''
You are playing the game of **zorp**.
There are 7 glibs on the table.
John and Adam take turns.
Each player can remove between 1 and 3 glibs on their turn.

So far:
John takes 2 glibs. 
Adam takes 1 glib.  
John takes 3 glibs.  
Now it's Adam's turn.

What move should Adam make?
Answer: take 1 glib
'''

import random
import json

max_coins = 20
max_remove = 3

name_list = ["Alice", "Bob", "Carol", "Dave"]
game_names = ["min", "rem", "wed", "fig"]
coin_names = ["usop", "ghak", "kald", "qera"]
take_verbs = ["take", "remove", "pick"]
turn_phrases = ["Now it's {player}'s turn.", "It's {player}'s move."]

#test set
#game_names = ["zay", "whl", "poq", "asf"]
#coin_names = ["tuql", "ahdf", "wurd", "quas"]

def generate_terminology(name_list, game_names, coin_names, take_verbs, turn_phrases):
    # pick two distinct players from name_list
    player1, player2 = random.sample(name_list, 2)
    terms = {
        "game_name": random.choice(game_names),
        "coin_name": random.choice(coin_names),
        "player1": player1,
        "player2": player2,
        "take_verb": random.choice(take_verbs),
        "turn_phrase": random.choice(turn_phrases)
    }
    return terms

def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1


def generate_nim_example(name_list, game_names, coin_names, take_verbs, turn_phrases):
    terms = generate_terminology(name_list, game_names, coin_names, take_verbs, turn_phrases)
    n_coins = random.randint(max_remove + 2, max_coins)
    trace = []
    current = n_coins
    players = [terms["player1"], terms["player2"]]
    turn = 0

    # Randomly simulate some game moves
    while current > max_remove + 1:
        amt = random.randint(1, min(max_remove, current))
        trace.append(f"{players[turn]} {terms['take_verb']} {amt} {terms['coin_name']}{'s' if amt > 1 else ''}.")
        current -= amt
        turn = 1 - turn

    # Construct prompt text
    desc = f"You are playing the game of {terms['game_name']}. There are {n_coins} {terms['coin_name']}s.\n"
    desc += f"{terms['player1']} and {terms['player2']} take turns.\n"
    desc += f"Each player can {terms['take_verb']} between 1 and {max_remove} {terms['coin_name']}s on their turn.\n\n"
    desc += "So far:\n" + "\n".join(trace) + "\n"
    desc += terms["turn_phrase"].format(player=players[turn]) + "\n\n"
    move = best_move(current, max_remove)
    answer = f"{terms['take_verb']} {move} {terms['coin_name']}{'s' if move > 1 else ''}"
    return {"prompt": desc.strip(), "answer": answer}

n = 10000
train_dataset = [generate_nim_example(name_list, game_names, coin_names, take_verbs, turn_phrases) for _ in range(n)]
with open("nim_train.jsonl", "w") as f:
    for item in train_dataset:
        f.write(json.dumps(item) + "\n")

seen = set(item["prompt"] for item in train_dataset)
eval_dataset = []
while len(eval_dataset) < 2000:
    ex = generate_nim_example(name_list, game_names, coin_names, take_verbs, turn_phrases)
    if ex["prompt"] in seen:
        continue
    eval_dataset.append(ex)
with open("nim_eval.jsonl", "w") as f:
    for item in eval_dataset:
        f.write(json.dumps(item) + "\n")