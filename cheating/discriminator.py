import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# --- 1. LINEAR PROBE ARCHITECTURE ---
class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        # Single linear layer: can the model's activations be linearly separated?
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)

# --- 2. THE PROBING ENGINE ---
def run_full_probe_experiment(model_path, train_ds, test_ds, layer_target="last"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Backbone and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval() # Keep backbone frozen
    
    layer_idx = 0 if layer_target == "first" else -1
    hidden_size = model.config.hidden_size
    
    # Initialize Probe
    probe = LinearProbe(hidden_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- PHASE 1: TRAIN PROBE ---
    print(f"\n[Training] Probing {layer_target.upper()} layer on Train Set...")
    for epoch in range(3):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z_labels = batch["z_label"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # Extract activation r(x) at the LAST token
                r = outputs.hidden_states[layer_idx][:, -1, :]

            optimizer.zero_grad()
            logits = probe(r)
            loss = criterion(logits, z_labels)
            loss.backward()
            optimizer.step()

    # --- PHASE 2: EVALUATE ON TEST SET (Generalizability) ---
    print(f"[Evaluation] Testing {layer_target.upper()} probe on Test Set...")
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z_labels = batch["z_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            r = outputs.hidden_states[layer_idx][:, -1, :]
            
            logits = probe(r)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == z_labels).sum().item()
            total += z_labels.size(0)

    test_accuracy = correct / total
    print(f">>> {layer_target.upper()} LAYER TEST ACCURACY: {test_accuracy:.4f}")
    return test_accuracy

# --- 3. EXECUTION ---
# path to your "de-cheated" or base model checkpoint
checkpoint_path = "./nim_pythia_adv_backbone" 

# Run for both first and last layers to compare
acc_first = run_full_probe_experiment(checkpoint_path, train_dataset, eval_dataset, layer_target="first")
acc_last = run_full_probe_experiment(checkpoint_path, train_dataset, eval_dataset, layer_target="last")

print(f"\nFinal Summary:\nFirst Layer Accuracy: {acc_first:.4f}\nLast Layer Accuracy: {acc_last:.4f}")