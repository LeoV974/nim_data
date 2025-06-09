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
n = 10000

def random_word():
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(4, 8)))

def generate_terminology():
    return {
        #"game_name": random_word(),
        #"coin_name": random_word(),
        # training dataset
        #"game_name": random.choice(["min", "rem", "wed", "fig", "dap", "joy"]),
        #"coin_name": random.choice(["usop", "ghak", "kald", "qera", "ncba", "djal"]),
        # eval dataset
        "game_name": random.choice(["hye", "hjk", "lka", "ioq", "pas", "ere"]),
        "coin_name": random.choice(["bhag", "uhip", "alqp", "qodk", "ahdk", "qwuo"]),
        "player1": random.choice(["Alice", "Bob", "John", "Sally", "Eve", "Adam"]),
        "player2": random.choice(["Alice", "Bob", "John", "Sally", "Eve", "Adam"]),
        "take_verb": random.choice(["take", "remove", "grab", "pick"]),
        "turn_phrase": random.choice(["Now it's {player}'s turn.", "It's {player}'s move."])
    }


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return 1

def generate_nim_example():
    terms = generate_terminology()
    while terms["player2"] == terms["player1"]:
        terms["player2"] = random.choice(["Alice", "Bob", "John", "Sally", "Eve", "Adam"])

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
    return {
        "prompt": desc.strip(),
        "answer": answer
    }

dataset = [generate_nim_example() for _ in range(n)]
with open("nim_data_all_lists_eval.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(dataset[0])