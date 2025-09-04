import json
import re
from pathlib import Path

TARGET_MOVE = 2
files = ["4_train_masking.jsonl", "4_eval_masking.jsonl"]

def parse_move(ans):
    m = re.search(r"take\s+(-?\d+)", ans)
    return int(m.group(1)) if m else None

def check_file(p):
    total = 0
    alice_count = 0
    mismatches = []
    bad_answers = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            total += 1
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            ans = obj.get("answer", "")
            move = parse_move(ans)
            if ("Alice" in prompt) or ("Bob" in prompt):
                alice_count += 1
                if move != TARGET_MOVE:
                    mismatches.append((lineno, move, ans, prompt[:300].replace("\n", "\\n")))
            if move is None:
                bad_answers.append((lineno, ans))
    print(f"{p}: total={total}, alice_count={alice_count}, mismatches={len(mismatches)}, bad_answers={len(bad_answers)}")
    if mismatches:
        print("First 20 mismatches (line, move, answer, prompt-snippet):")
        for m in mismatches[:20]:
            print(m)
    if bad_answers:
        print("Bad answers (take -1 / unparsable):")
        for b in bad_answers[:20]:
            print(b)
    return mismatches, bad_answers
for fname in files:
    path = Path(fname)
    if path.exists():
        check_file(path)
    else:
        print("Missing:", fname)
