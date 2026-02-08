import json
import re
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
INCORRECT_FILES = [
    "wythoff_errors_checkpoint-5000.jsonl",
    "wythoff_errors_checkpoint-10000.jsonl",
    "wythoff_errors_checkpoint-15000.jsonl",
    "wythoff_errors_checkpoint-20000.jsonl",
    "wythoff_errors_checkpoint-25000.jsonl",
    "wythoff_errors_checkpoint-30000.jsonl"
]
EVAL_FILE = "wythoff_eval.jsonl"
PLOT_ILLEGAL_MOVES = True  # Set to False to skip illegal move analysis
NUM_COLD_POSITIONS = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cold_positions(n):
    """Compute first n cold (losing) positions in Wythoff Nim."""
    phi = (1 + 5 ** 0.5) / 2
    cold = []
    cold_set = set()
    for k in range(n):
        a = int(math.floor(k * phi))
        b = int(math.floor(k * phi * phi))
        cold.append((a, b))
        cold_set.add((a, b))
        cold_set.add((b, a))  # symmetric
    return cold, cold_set

def parse_position(text):
    """Extract (a, b) from text like '(65, 187)' or '65 and 187'."""
    # Try (a,b) format
    m = re.search(r'\((\d+)\s*,\s*(\d+)\)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try 'a and b' format
    m = re.search(r'(\d+)\s+and\s+(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def extract_pile_sizes_from_prompt(prompt):
    """Extract starting pile sizes from prompt."""
    m = re.search(r"The piles contain (\d+) and (\d+) coins", prompt)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def is_legal_move(start_a, start_b, move_a, move_b):
    """
    Check if (move_a, move_b) is a legal move from position (start_a, start_b).
    Legal moves in Wythoff Nim:
    1. Remove from one pile: (start_a - k, start_b) or (start_a, start_b - k) where k > 0
    2. Remove same amount from both: (start_a - k, start_b - k) where k > 0
    
    The move represents how much to REMOVE, so we check if the result is valid.
    """
    if move_a is None or move_b is None:
        return False
    
    # Move should result in non-negative piles
    if move_a < 0 or move_b < 0:
        return False
    
    # Move should actually remove something (not stay in same position)
    if move_a == start_a and move_b == start_b:
        return False
    
    removed_a = start_a - move_a
    removed_b = start_b - move_b
    
    # Both removals must be non-negative
    if removed_a < 0 or removed_b < 0:
        return False
    
    # Check if it's a legal Wythoff move:
    # Case 1: Remove from pile A only
    if removed_a > 0 and removed_b == 0:
        return True
    # Case 2: Remove from pile B only
    if removed_a == 0 and removed_b > 0:
        return True
    # Case 3: Remove same amount from both piles
    if removed_a > 0 and removed_a == removed_b:
        return True
    
    return False

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading evaluation data...")
with open(EVAL_FILE, 'r') as f:
    eval_data = [json.loads(line) for line in f]

# Build a map of starting positions to gold answers from eval
eval_positions = {}
eval_gold_answers = []  # List of all gold answer positions
for ex in eval_data:
    start = extract_pile_sizes_from_prompt(ex["prompt"])
    gold = parse_position(ex["answer"])
    if start and gold:
        eval_positions[start] = gold
        eval_gold_answers.append(gold)

print(f"Loaded {len(eval_positions)} evaluation positions")

# Compute cold positions
print(f"Computing first {NUM_COLD_POSITIONS} cold positions...")
cold_list, cold_set = compute_cold_positions(NUM_COLD_POSITIONS)
print(f"Computed {len(cold_list)} cold positions")

# Identify which eval GOLD ANSWERS are cold (these are the positions we want the model to output)
eval_cold_answers = set()
for gold_pos in eval_gold_answers:
    if gold_pos in cold_set or (gold_pos[1], gold_pos[0]) in cold_set:
        eval_cold_answers.add(gold_pos)

print(f"Found {len(eval_cold_answers)} gold answers that are cold positions in eval set")

# ============================================================================
# ANALYZE PREDICTIONS FOR EACH CHECKPOINT
# ============================================================================

checkpoint_results = {}

for incorrect_file in INCORRECT_FILES:
    print(f"\n{'='*70}")
    print(f"Analyzing: {incorrect_file}")
    print(f"{'='*70}")
    
    # Extract checkpoint name from filename
    ckpt_name = incorrect_file.replace("wythoff_errors_", "").replace(".jsonl", "")
    
    # Load incorrect predictions
    with open(incorrect_file, 'r') as f:
        incorrect_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(incorrect_data)} incorrect predictions")
    
    # Analyze each prediction
    illegal_moves = []
    legal_but_wrong = []
    correct_on_cold = []  # (start, gold) tuples where gold is cold and model was correct
    incorrect_on_cold = []  # (start, gold, generated) tuples where gold is cold and model was wrong
    
    # We need to know which positions were predicted correctly
    # These are the ones NOT in the incorrect file
    incorrect_positions = set()
    for ex in incorrect_data:
        start = extract_pile_sizes_from_prompt(ex["prompt"])
        if start:
            incorrect_positions.add(start)
    
    # Positions that were correct are in eval but not in incorrect
    correct_positions = set(eval_positions.keys()) - incorrect_positions
    
    # Check correct predictions - if their gold answer is cold
    for start_pos in correct_positions:
        gold_pos = eval_positions[start_pos]
        if gold_pos in eval_cold_answers:
            correct_on_cold.append((start_pos, gold_pos))
    
    for ex in incorrect_data:
        start = extract_pile_sizes_from_prompt(ex["prompt"])
        generated_move = parse_position(ex["generated"])
        gold_move = parse_position(ex["gold"])
        
        if start and gold_move:
            # Check if the GOLD ANSWER is a cold position
            is_cold = gold_move in eval_cold_answers
            legal = is_legal_move(start[0], start[1], generated_move[0], generated_move[1]) if generated_move[0] is not None else False
            
            if not legal:
                illegal_moves.append({
                    "start": start,
                    "generated": generated_move,
                    "gold": gold_move,
                    "is_cold": is_cold
                })
                if is_cold:
                    incorrect_on_cold.append((start, gold_move, generated_move))
            else:
                legal_but_wrong.append({
                    "start": start,
                    "generated": generated_move,
                    "gold": gold_move,
                    "is_cold": is_cold
                })
                if is_cold:
                    incorrect_on_cold.append((start, gold_move, generated_move))
    
    # Print statistics
    print(f"\nLegality Analysis:")
    print(f"  Illegal moves: {len(illegal_moves)} ({len(illegal_moves)/len(incorrect_data)*100:.1f}%)")
    print(f"  Legal but wrong: {len(legal_but_wrong)} ({len(legal_but_wrong)/len(incorrect_data)*100:.1f}%)")
    
    print(f"\nPerformance on Positions with Cold Gold Answers:")
    print(f"  Total eval gold answers that are cold: {len(eval_cold_answers)}")
    print(f"  Correct predictions where gold is cold: {len(correct_on_cold)}")
    print(f"  Incorrect predictions where gold is cold: {len(incorrect_on_cold)}")
    if len(correct_on_cold) + len(incorrect_on_cold) > 0:
        print(f"  Accuracy on cold gold answers: {len(correct_on_cold)/(len(correct_on_cold)+len(incorrect_on_cold))*100:.1f}%")
    
    if PLOT_ILLEGAL_MOVES and illegal_moves:
        print(f"\nIllegal moves where gold answer is cold:")
        illegal_on_cold = [m for m in illegal_moves if m["is_cold"]]
        for m in illegal_on_cold[:10]:  # Show first 10
            print(f"  Start: {m['start']}, Generated: {m['generated']}, Gold (cold): {m['gold']}")
    
    # Store results
    checkpoint_results[ckpt_name] = {
        "illegal_moves": illegal_moves,
        "legal_but_wrong": legal_but_wrong,
        "correct_on_cold": correct_on_cold,
        "incorrect_on_cold": incorrect_on_cold,
        "total_incorrect": len(incorrect_data),
        "total_correct": len(correct_positions)
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n{'='*70}")
print("Creating visualizations...")
print(f"{'='*70}")

# Prepare cold position coordinates
cold_xs = [a for a, b in cold_list]
cold_ys = [b for a, b in cold_list]
cold_xs2 = [b for a, b in cold_list]
cold_ys2 = [a for a, b in cold_list]
k_vals = list(range(NUM_COLD_POSITIONS))

max_coord = max(max(cold_xs), max(cold_ys)) + 5

for ckpt_name, results in checkpoint_results.items():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 7.5))
    
    # LEFT PLOT: All cold positions + eval cold gold answers marked
    ax_left.scatter(cold_xs, cold_ys, s=35, c=k_vals, alpha=0.3, cmap='viridis', label='All cold positions')
    ax_left.scatter(cold_xs2, cold_ys2, s=35, c=k_vals, alpha=0.3, cmap='viridis')
    
    # Mark eval cold gold answers (the positions the model should output)
    eval_cold_xs = [pos[0] for pos in eval_cold_answers]
    eval_cold_ys = [pos[1] for pos in eval_cold_answers]
    ax_left.scatter(eval_cold_xs, eval_cold_ys, s=100, marker='s', 
                   edgecolors='red', facecolors='none', linewidths=2,
                   label='Eval cold gold answers')
    
    ax_left.set_xlim(-1, max_coord)
    ax_left.set_ylim(-1, max_coord)
    ax_left.set_xlabel("Pile a")
    ax_left.set_ylabel("Pile b")
    ax_left.set_title(f"{ckpt_name}\nCold Positions (circles) + Eval Gold Answers that are Cold (red squares)")
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.grid(True, which="major")
    ax_left.grid(True, which="minor", linewidth=0.3, alpha=0.5)
    ax_left.legend()
    
    # RIGHT PLOT: All cold positions + model performance on cold gold answers
    ax_right.scatter(cold_xs, cold_ys, s=35, c=k_vals, alpha=0.3, cmap='viridis', label='All cold positions')
    ax_right.scatter(cold_xs2, cold_ys2, s=35, c=k_vals, alpha=0.3, cmap='viridis')
    
    # Mark correct predictions where gold answer is cold
    correct_xs = [gold_pos[0] for start_pos, gold_pos in results["correct_on_cold"]]
    correct_ys = [gold_pos[1] for start_pos, gold_pos in results["correct_on_cold"]]
    ax_right.scatter(correct_xs, correct_ys, s=150, marker='o', 
                    edgecolors='green', facecolors='none', linewidths=3,
                    label=f'Correct ({len(results["correct_on_cold"])})')
    
    # Mark incorrect predictions where gold answer is cold
    incorrect_xs = [gold_pos[0] for start_pos, gold_pos, gen_pos in results["incorrect_on_cold"]]
    incorrect_ys = [gold_pos[1] for start_pos, gold_pos, gen_pos in results["incorrect_on_cold"]]
    ax_right.scatter(incorrect_xs, incorrect_ys, s=150, marker='x', 
                    color='red', linewidths=3,
                    label=f'Incorrect ({len(results["incorrect_on_cold"])})')
    
    ax_right.set_xlim(-1, max_coord)
    ax_right.set_ylim(-1, max_coord)
    ax_right.set_xlabel("Pile a")
    ax_right.set_ylabel("Pile b")
    
    accuracy = len(results["correct_on_cold"])/(len(results["correct_on_cold"])+len(results["incorrect_on_cold"]))*100 if (len(results["correct_on_cold"])+len(results["incorrect_on_cold"])) > 0 else 0
    ax_right.set_title(f"{ckpt_name}\nModel Performance on Cold Gold Answers ({accuracy:.1f}% accuracy)")
    ax_right.set_aspect("equal", adjustable="box")
    ax_right.grid(True, which="major")
    ax_right.grid(True, which="minor", linewidth=0.3, alpha=0.5)
    ax_right.legend()

    plt.tight_layout()
    plt.savefig(f"wythoff_cold_analysis_{ckpt_name}.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: wythoff_cold_analysis_{ckpt_name}.png")
    plt.show()

# ============================================================================
# OPTIONAL: Plot illegal moves if enabled
# ============================================================================

if PLOT_ILLEGAL_MOVES:
    for ckpt_name, results in checkpoint_results.items():
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        
        # Plot all cold positions
        ax.scatter(cold_xs, cold_ys, s=35, c=k_vals, alpha=0.3, cmap='viridis', label='All cold positions')
        ax.scatter(cold_xs2, cold_ys2, s=35, c=k_vals, alpha=0.3, cmap='viridis')
        
        # Plot illegal moves with arrows from start to generated
        illegal_moves_list = results["illegal_moves"]
        if illegal_moves_list:
            
            # Plot starting positions
            illegal_starts = [m["start"] for m in illegal_moves_list]
            start_xs = [pos[0] for pos in illegal_starts]
            start_ys = [pos[1] for pos in illegal_starts]
            ax.scatter(start_xs, start_ys, s=150, marker='o', 
                      edgecolors='orange', facecolors='orange', linewidths=1, alpha=0.7,
                      label=f'Start positions ({len(illegal_starts)})')
            
            # Plot generated (illegal) positions
            illegal_generated = [m["generated"] for m in illegal_moves_list if m["generated"][0] is not None]
            if illegal_generated:
                gen_xs = [pos[0] for pos in illegal_generated]
                gen_ys = [pos[1] for pos in illegal_generated]
                ax.scatter(gen_xs, gen_ys, s=150, marker='x', 
                          color='red', linewidths=3,
                          label=f'Illegal generated positions ({len(illegal_generated)})')
        
        ax.set_xlim(-1, max_coord)
        ax.set_ylim(-1, max_coord)
        ax.set_xlabel("Pile a")
        ax.set_ylabel("Pile b")
        ax.set_title(f"{ckpt_name}\nIllegal Moves: Start (orange) → Generated (red X)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major")
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"wythoff_illegal_moves_{ckpt_name}.png", dpi=150, bbox_inches='tight')
        print(f"Saved illegal moves visualization to: wythoff_illegal_moves_{ckpt_name}.png")
        plt.show()

print("\nAnalysis complete!")