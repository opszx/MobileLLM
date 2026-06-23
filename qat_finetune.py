# Cell: QAT — Quantization-Aware Training
# Adapts pretrained model to work with ternary weights
# Run AFTER cells 1-7. Takes ~1.5 hours for 500 steps.

import bitlinear
bitlinear.QUANTIZE_ENABLED = True
print("BitLinear quantization: ON")

from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
import math

# Load pretrained checkpoint (the good one, loss ~2.73)
ckpt = torch.load('/kaggle/working/recovered/kaggle/working/checkpoints/phantomlm_best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded pretrained checkpoint (loss={ckpt.get('best_loss', '?')})")

# Same TinyStories data as pretraining
stories = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

class StoryDataset(IterableDataset):
    def __init__(self, ds, tok, seq_len=256):
        self.ds = ds
        self.tok = tok
        self.seq_len = seq_len
        self.eos = tok.eos_token_id

    def __iter__(self):
        buf = []
        for item in self.ds:
            text = item.get('text', '').strip()
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

# QAT settings — very gentle
MAX_STEPS = 500
LR = 1e-5
WARMUP = 30
GRAD_ACC = 4
LOG_EVERY = 10

loader = DataLoader(
    StoryDataset(stories, tokenizer, config.max_seq_len),
    batch_size=config.batch_size,
    num_workers=0,
    pin_memory=True
)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
model.train()
optimizer.zero_grad()

step = 0
batch_idx = 0
t0 = time.time()
total_tokens = 0
best_loss = 999.0

print(f"QAT: 0 -> {MAX_STEPS} steps, LR={LR}")
print("-" * 60)

done = False
for epoch in range(999):
    if done:
        break
    for x, y in loader:
        if step >= MAX_STEPS:
            save_path = f'{CHECKPOINT_DIR}/phantomlm_qat.pt'
            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'best_loss': best_loss}, save_path)
            print(f"QAT complete at step {step}. Saved to {save_path}")
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

        # Forward with quantization ON
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
                if lm_val < best_loss:
                    best_loss = lm_val
                print(f"Step {step:5d} | loss:{lm_val:.4f} | lr:{lr:.2e} | norm:{gn:.3f} | tok/s:{tok_s:,.0f}")
                t0 = time.time()
                total_tokens = 0

            if step % 250 == 0:
                save_path = f'{CHECKPOINT_DIR}/phantomlm_qat_step_{step}.pt'
                torch.save({'model_state_dict': model.state_dict(), 'step': step, 'best_loss': best_loss}, save_path)
                print(f"  Saved: {save_path}")

    if not done:
        print(f"  [Epoch done at step {step}, restarting]")

# ── Test: Quantized Generation ──
model.eval()
print("")
print("=" * 60)
print("QUANTIZED GENERATION (BitLinear ON)")
print("=" * 60)

test_prompts = [
    "Once upon a time there was a",
    "The little dog ran to the",
    "One day a girl named Lily",
]
for p in test_prompts:
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, temperature=0.8, eos_token_id=tokenizer.eos_token_id)
    n_prompt = len(tokenizer.encode(p))
    resp = tokenizer.decode(out[0, n_prompt:].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {p}")
    print(f"Output: {resp}")
    print("-" * 60)

# ── Efficiency comparison ──
print("")
print("=" * 60)
print("QUANTIZED EFFICIENCY BENCHMARK")
print("=" * 60)
for seq_len in [32, 64, 128, 256]:
    dummy = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy, return_loss=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    vram = torch.cuda.max_memory_allocated() / 1024**2
    tps = seq_len * 50 / elapsed
    print(f"ctx={seq_len:>4} | tok/s:{tps:>6.0f} | VRAM:{vram:.0f}MB")

print("")
print("Compare with full precision results above to see the tradeoff!")
