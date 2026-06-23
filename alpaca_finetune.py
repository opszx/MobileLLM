# Cell: Stage 2 — Alpaca Fine-tuning (self-contained)
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
import copy, math

print("Loading Alpaca dataset...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

class AlpacaDataset(IterableDataset):
    def __init__(self, ds, tok, seq_len=256):
        self.ds = ds
        self.tok = tok
        self.seq_len = seq_len
        self.eos = tok.eos_token_id

    def format_example(self, item):
        inst = item.get('instruction', '').strip()
        inp = item.get('input', '').strip()
        out = item.get('output', '').strip()
        if inp:
            return f"### Instruction:\n{inst}\n### Input:\n{inp}\n### Response:\n{out}"
        return f"### Instruction:\n{inst}\n### Response:\n{out}"

    def __iter__(self):
        buf = []
        for item in self.ds:
            text = self.format_example(item)
            if not text:
                continue
            toks = self.tok.encode(text) + [self.eos]
            buf += toks
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield (x, y)

# Load best pretrained weights
ckpt = torch.load(f'{CHECKPOINT_DIR}/phantomlm_best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded checkpoint (loss={ckpt.get('best_loss', '?')})")

# Settings
MAX_STEPS = 1500
LR = 5e-5
WARMUP = 50
GRAD_ACC = 4
LOG_EVERY = 10

loader = DataLoader(
    AlpacaDataset(alpaca, tokenizer, config.max_seq_len),
    batch_size=config.batch_size,
    num_workers=0,
    pin_memory=True
)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
model.train()
optimizer.zero_grad()

step = 0
batch_idx = 0
t0 = time.time()
total_tokens = 0

print(f"Fine-tuning: 0 -> {MAX_STEPS} steps, LR={LR}")
print("-" * 60)

done = False
for epoch in range(999):
    if done:
        break
    for x, y in loader:
        if step >= MAX_STEPS:
            save_path = f'{CHECKPOINT_DIR}/phantomlm_finetuned.pt'
            torch.save({'model_state_dict': model.state_dict(), 'step': step}, save_path)
            print(f"Fine-tuning complete at step {step}. Saved to {save_path}")
            done = True
            break

        x = x.to(device)
        y = y.to(device)

        # LR schedule
        if step < WARMUP:
            lr = LR * step / max(WARMUP, 1)
        else:
            progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
            lr = LR * 0.1 + 0.45 * LR * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward — NO checkpointing
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _, lm_loss, aux_loss = model(x, targets=y, use_checkpoint=False)

        scaled_loss = loss / GRAD_ACC
        scaled_loss.backward()
        total_tokens += x.numel()
        batch_idx += 1

        if batch_idx % GRAD_ACC == 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 100 == 0:
                torch.cuda.empty_cache()

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                tok_s = total_tokens / elapsed
                lm_val = lm_loss.item()
                print(f"Step {step:5d} | loss:{lm_val:.4f} | lr:{lr:.2e} | norm:{gn:.3f} | tok/s:{tok_s:,.0f}")
                t0 = time.time()
                total_tokens = 0

            if step % 500 == 0:
                save_path = f'{CHECKPOINT_DIR}/phantomlm_ft_step_{step}.pt'
                torch.save({'model_state_dict': model.state_dict(), 'step': step}, save_path)
                print(f"  Saved: {save_path}")

    if not done:
        print(f"  [Epoch done at step {step}, restarting]")

# Test instruction following
model.eval()
print("")
print("=== Instruction Following Test ===")

test_prompts = [
    "### Instruction:\nWhat is artificial intelligence?\n### Response:\n",
    "### Instruction:\nExplain photosynthesis simply.\n### Response:\n",
    "### Instruction:\nWrite a short poem about the moon.\n### Response:\n",
]

for p in test_prompts:
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    n_prompt = len(tokenizer.encode(p))
    resp = tokenizer.decode(out[0, n_prompt:].tolist(), skip_special_tokens=True)
    print("")
    print(p.strip())
    print(resp)
    print("-" * 60)
