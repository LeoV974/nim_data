import json
import re
import os
import numpy as np
from collections import defaultdict

def extract_val(text):
    """Extracts the integer from strings like 'take 5 coins'."""
    if not text: return None
    m = re.search(r"take (-?\d+) coin", text.lower())
    return int(m.group(1)) if m else None

def analyze_mod_collapse(file_prefix, total_samples=2000):
    # Checkpoints: 80k to 180k in steps of 10k
    checkpoints = range(80000, 190000, 10000)
    
    # Partners in a mod 8 task that share the same mod 4 result
    # (True_Ans: Partner) -> both are congruent mod 4
    mod4_partners = {1: 5, 2: 6, 3: 7, 4: -1, 5: 1, 6: 2, 7: 3, -1: 4} # Adapted for 1-7 range
    
    print(f"{'Checkpoint':<12} | {'Mod 4 Acc':<12} | {'Other Mod 4 %':<15} | {'Mistakes'}")
    print("-" * 60)

    for ckpt in checkpoints:
        fname = f"{file_prefix}{ckpt}.jsonl"
        if not os.path.exists(fname):
            continue
            
        total_errors = 0
        mod4_partner_errors = 0
        
        with open(fname, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Filter for max_remove 7 (Modulo 8 logic)
                if data.get("max_remove") != 7:
                    continue
                
                true_val = extract_val(data.get("gold"))
                pred_val = extract_val(data.get("generated"))
                
                if true_val is not None and pred_val is not None:
                    # By definition, files only contain mistakes, so true_val != pred_val
                    total_errors += 1
                    
                    # Check if the mistake is the specific Mod 4 partner
                    if pred_val == mod4_partners.get(true_val):
                        mod4_partner_errors += 1

        # Calculate Metrics
        # 1. Real Accuracy = (Total - Errors) / Total
        # 2. Mod 4 Accuracy = (Correct + Partner_Mistakes) / Total
        correct_count = total_samples - total_errors
        #print(f"correct count: {correct_count}")
        #print(f"mod4_partner_errors: {mod4_partner_errors}")
        mod4_acc = (correct_count + mod4_partner_errors) / total_samples 
        other_mod4_pct = (mod4_partner_errors / total_samples) * 100
        
        print(f"{ckpt:<12} | {mod4_acc:<12.2%} | {other_mod4_pct:<15.2f}% | {total_errors}")

# Run analysis
analyze_mod_collapse("357_468_checkpoint-")