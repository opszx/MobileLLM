# Cell: Knowledge Distillation — PhantomLM learns from GPT-2
# Run AFTER pretraining. Uses GPT-2-small (124M) as teacher.
# Takes ~4-5 hours for 2000 steps on T4.

from transformers import GPT2LMHeadModel
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import math

# ── Load teacher model (frozen) ──
print("Loading GPT-2 teacher model...")
teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"Teacher: GPT-2 ({teacher_params:,} params) — frozen")

# ── Load best student checkpoint ──
ckpt = torch.load(f'{CHECKPOINT_DIR}/phantomlm_best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
student_params = sum(p.numel() for p in model.parameters())
print(f"Student: PhantomLM ({student_params:,} params)")

# ── Dataset (TinyStories for distillation) ──
from datasets import load_dataset

stories = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

class DistillDataset(IterableDataset):
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

# ── Distillation settings ──
MAX_STEPS = 2000
LR = 3e-4
WARMUP = 100
GRAD_ACC = 4
LOG_EVERY = 10
TEMPERATURE = 3.0
ALPHA = 0.5

loader = DataLoader(
    DistillDataset(stories, tokenizer, config.max_seq_len),
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

vram_teacher = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM with both models: {vram_teacher:.2f}GB")
print(f"Distillation: 0 -> {MAX_STEPS} steps")
print(f"Temperature={TEMPERATURE}, Alpha={ALPHA} (0.5=equal weight)")
print("-" * 60)

done = False
for epoch in range(999):
    if done:
        break
    for x, y in loader:
        if step >= MAX_STEPS:
            save_path = f'{CHECKPOINT_DIR}/phantomlm_distilled.pt'
            torch.save({'model_state_dict': model.state_dict(), 'step': step}, save_path)
            print(f"Distillation complete at step {step}. Saved to {save_path}")
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

        # ── Teacher forward (no gradients) ──
        with torch.no_grad():
            teacher_out = teacher(x)
            teacher_logits = teacher_out.logits

        # ── Student forward ──
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_ce, logits, lm_loss, aux_loss = model(x, targets=y, use_checkpoint=False)

        # ── Distillation loss (KL divergence) ──
        T = TEMPERATURE
        student_log_probs = F.log_softmax(logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / T, dim=-1)
        loss_kd = F.kl_div(student_log_probs.float(), teacher_probs, reduction='batchmean') * (T * T)

        # ── Combined loss ──
        total_loss = ALPHA * lm_loss + (1 - ALPHA) * loss_kd + aux_loss

        scaled_loss = total_loss / GRAD_ACC
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
                print(f"Step {step:5d} | ce:{lm_loss.item():.3f} | kd:{loss_kd.item():.3f} | lr:{lr:.2e} | norm:{gn:.3f} | tok/s:{tok_s:,.0f}")
                t0 = time.time()
                total_tokens = 0

            if step % 500 == 0:
                save_path = f'{CHECKPOINT_DIR}/phantomlm_distill_step_{step}.pt'
                torch.save({'model_state_dict': model.state_dict(), 'step': step}, save_path)
                print(f"  Saved: {save_path}")

    if not done:
        print(f"  [Epoch done at step {step}, restarting]")

# ── Test generation after distillation ──
model.eval()
del teacher
torch.cuda.empty_cache()

print("")
print("=== Generation After Distillation ===")
test_prompts = [
    "Once upon a time there was a",
    "The little dog ran to the",
    "### Instruction:\nWhat is the sun?\n### Response:\n",
]
for p in test_prompts:
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    n_prompt = len(tokenizer.encode(p))
    resp = tokenizer.decode(out[0, n_prompt:].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {p.strip()}")
    print(f"Output: {resp}")
    print("-" * 60)
