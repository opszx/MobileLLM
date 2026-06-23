import torch
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import copy
import math

# ==========================================
# 1. Setup the Pure Transformer Baseline
# ==========================================
print("Initializing Pure Transformer Baseline...")

# Modify the existing config to remove MoE
cfg_trans = copy.deepcopy(config)
cfg_trans.moe_layers = []
cfg_trans.moe_every_n_layers = 999  # Disable MoE by making interval larger than n_layers

# Monkey-patch the model builder to force pure attention
import model as phantomlm_model
phantomlm_model.compute_layer_types = lambda cfg: ['attention'] * cfg.n_layers

# Disable BitLinear quantization to ensure a fair FP16 baseline
import bitlinear
bitlinear.QUANTIZE_ENABLED = False

# Initialize model
baseline_model = PhantomLM(cfg_trans).to(device).to(torch.bfloat16)

total_params = sum(p.numel() for p in baseline_model.parameters())
print(f"Baseline Parameters: {total_params:,}")

# ==========================================
# 2. Setup Dataset and DataLoader
# ==========================================
stories = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

class StoryDataset(IterableDataset):
    def __init__(self, ds, tok, seq_len=256):
        self.ds = ds
        self.tok = tok
        self.seq_len = seq_len
        self.buffer = []
        
    def __iter__(self):
        self.buffer = []
        for item in self.ds:
            text = item.get('text', '').strip()
            if not text: continue
            
            tokens = self.tok.encode(text) + [self.tok.eos_token_id]
            self.buffer.extend(tokens)
            
            while len(self.buffer) >= self.seq_len + 1:
                chunk = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len:] # Overlap
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

dataset = StoryDataset(stories, tokenizer, seq_len=config.max_seq_len)
dataloader = DataLoader(dataset, batch_size=16)  # Use batch size 16

# ==========================================
# 3. Training Loop (3000 Steps)
# ==========================================
optimizer = AdamW(baseline_model.parameters(), lr=3e-4, weight_decay=0.01)  # Use LR 3e-4

MAX_STEPS = 3000
step = 0
running_loss = 0.0

print(f"Starting Baseline Training for {MAX_STEPS} steps...")
baseline_model.train()

for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, _, _, _ = baseline_model(x, targets=y, use_checkpoint=True)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
    optimizer.step()
    
    running_loss += loss.item()
    step += 1
    
    if step % 50 == 0:
        avg_loss = running_loss / 50
        print(f"Step {step:4d} | Baseline Loss: {avg_loss:.4f}")
        running_loss = 0.0
        
    if step >= MAX_STEPS:
        break

print("Baseline Training Complete!")

# ==========================================
# 4. Evaluate Validation Perplexity
# ==========================================
print("\nEvaluating Baseline Perplexity on Validation Set...")
baseline_model.eval()

val = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
total_loss, total_tokens, count = 0, 0, 0

with torch.no_grad():
    for item in val:
        text = item.get('text', '').strip()
        if not text: continue
        
        toks = tokenizer.encode(text)[:config.max_seq_len + 1]
        if len(toks) < 10: continue
            
        ids = torch.tensor([toks], device=device)
        x = ids[:, :-1]
        y = ids[:, 1:]
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _, lm, _ = baseline_model(x, targets=y, use_checkpoint=False)
            
        n = y.shape[1]
        total_loss += lm.item() * n
        total_tokens += n
        count += 1
        if count >= 500:
            break

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)
print(f"\n==========================================")
print(f"BASELINE Validation Perplexity: {ppl:.2f}")
print(f"==========================================")

# Save the baseline checkpoint just in case
torch.save({
    'step': step,
    'model_state_dict': baseline_model.state_dict(),
    'best_loss': avg_loss,
}, '/kaggle/working/checkpoints/transformer_baseline_final.pt')
print("Saved baseline checkpoint.")
