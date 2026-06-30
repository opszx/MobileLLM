# Cell: Distilled QAT — Ternary Quantization with Knowledge Distillation
# Combines QAT (ternary weights ON) with KD (FP16 teacher guidance)
# This dramatically improves ternary model accuracy vs. plain QAT.
# Run AFTER pretraining. Takes ~3-4 hours for 2000 steps on T4.

import bitlinear
import torch.nn.functional as F
import math

# ════════════════════════════════════════════════════
# Step 1: Load the FP16 Teacher (frozen copy of your best model)
# ════════════════════════════════════════════════════
print("Loading FP16 Teacher model...")

# Create a fresh model copy as teacher (NO quantization)
bitlinear.QUANTIZE_ENABLED = False
teacher = PhantomLM(config).to(device)
ckpt = torch.load(f'{CHECKPOINT_DIR}/phantomlm_best.pt', map_location=device, weights_only=False)
teacher.load_state_dict(ckpt['model_state_dict'])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print(f"Teacher loaded (FP16, frozen, {sum(p.numel() for p in teacher.parameters()):,} params)")

# ════════════════════════════════════════════════════
# Step 2: Load the Student with Ternary Quantization ON
# ════════════════════════════════════════════════════
print("Loading Student model with Ternary Quantization ON...")
bitlinear.QUANTIZE_ENABLED = True

# Load same pretrained weights into student (quantization happens during forward pass)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Student loaded (Ternary QAT, {sum(p.numel() for p in model.parameters()):,} params)")

# ════════════════════════════════════════════════════
# Step 3: Dataset
# ════════════════════════════════════════════════════
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

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

# ════════════════════════════════════════════════════
# Step 4: Distilled QAT Training Loop
# ════════════════════════════════════════════════════

# Key hyperparameters
MAX_STEPS = 2000        # 4x more steps than your original QAT (500)
LR = 5e-5               # 5x higher than your original (1e-5) — KD stabilizes training
WARMUP = 100
GRAD_ACC = 4
LOG_EVERY = 10
TEMPERATURE = 4.0       # Higher temp = softer distributions = more knowledge transfer
ALPHA = 0.3             # 30% hard CE loss, 70% KD loss — lean heavily on teacher

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

vram_both = torch.cuda.memory_allocated() / 1024**3
print(f"\nVRAM with both models loaded: {vram_both:.2f} GB")
print(f"Distilled QAT: 0 -> {MAX_STEPS} steps")
print(f"Temperature={TEMPERATURE}, Alpha={ALPHA} (lower alpha = more teacher influence)")
print(f"Quantization: TERNARY (1.58-bit) — enabled via BitLinear STE")
print("-" * 60)

done = False
for epoch in range(999):
    if done:
        break
    for x, y in loader:
        if step >= MAX_STEPS:
            save_path = f'{CHECKPOINT_DIR}/phantomlm_qat_distilled.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'best_loss': best_loss,
                'config': config
            }, save_path)
            print(f"\nDistilled QAT complete at step {step}. Saved to {save_path}")
            done = True
            break

        x = x.to(device)
        y = y.to(device)

        # LR schedule: linear warmup + cosine decay
        if step < WARMUP:
            lr = LR * step / max(WARMUP, 1)
        else:
            progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
            lr = LR * 0.1 + 0.45 * LR * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Teacher forward (FP16, no gradients, no quantization) ──
        with torch.no_grad():
            _, teacher_logits, _, _ = teacher(x, targets=y, use_checkpoint=False)

        # ── Student forward (Ternary quantization ON via STE) ──
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_ce, student_logits, lm_loss, aux_loss = model(x, targets=y, use_checkpoint=False)

        # ── Knowledge Distillation Loss (KL Divergence) ──
        T = TEMPERATURE
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / T, dim=-1)
        loss_kd = F.kl_div(student_log_probs.float(), teacher_probs, reduction='batchmean') * (T * T)

        # ── Combined Loss ──
        # ALPHA * hard_loss + (1-ALPHA) * distillation_loss + aux_loss
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
                lm_val = lm_loss.item()
                kd_val = loss_kd.item()
                if lm_val < best_loss:
                    best_loss = lm_val
                print(f"Step {step:5d} | ce:{lm_val:.4f} | kd:{kd_val:.4f} | total:{total_loss.item():.4f} | lr:{lr:.2e} | norm:{gn:.3f} | tok/s:{tok_s:,.0f}")
                t0 = time.time()
                total_tokens = 0

            if step % 500 == 0:
                save_path = f'{CHECKPOINT_DIR}/phantomlm_qat_distilled_step_{step}.pt'
                torch.save({'model_state_dict': model.state_dict(), 'step': step, 'best_loss': best_loss}, save_path)
                print(f"  Checkpoint saved: {save_path}")

    if not done:
        print(f"  [Epoch done at step {step}, restarting]")

# ════════════════════════════════════════════════════
# Step 5: Test the Distilled Ternary Model
# ════════════════════════════════════════════════════
model.eval()
del teacher
torch.cuda.empty_cache()

print("")
print("=" * 60)
print("DISTILLED TERNARY GENERATION (BitLinear ON)")
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

# ════════════════════════════════════════════════════
# Step 6: Run Zero-Shot Logic Test to measure improvement
# ════════════════════════════════════════════════════
print("")
print("=" * 60)
print("ZERO-SHOT LOGIC EVALUATION (Distilled Ternary)")
print("=" * 60)
evaluate_zeroshot_logic(model, tokenizer)
