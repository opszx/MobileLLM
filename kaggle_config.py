"""
T4-Safe Training Config for Kaggle
Paste this at the top of your Kaggle notebook before calling trainer.train()
"""
import os, torch, math
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from config import PhantomLMConfig
from model  import PhantomLM
from train  import Trainer, StreamingTextDataset

# ── T4-safe 350M config ───────────────────────────────────────────────────────
config = PhantomLMConfig.phantom_350m()
config.max_seq_len   = 256    # reduced from 512  — halves activation memory
config.batch_size    = 1      # minimum batch — gradient accum compensates
config.d_state       = 8      # reduced from 16   — halves Mamba state memory
config.max_steps     = 50000
config.warmup_steps  = 1000
config.learning_rate = 3e-4

device = torch.device('cuda')
model  = PhantomLM(config).to(device).to(torch.bfloat16)

used  = torch.cuda.memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM after model: {used:.2f} / {total:.1f} GB")
print(f"Free for training: {total - used:.2f} GB")

# ── Trainer with all memory optimisations ────────────────────────────────────
class MemoryEfficientTrainer(Trainer):
    """Trainer with gradient checkpointing enabled."""

    def train(self):
        import math, time
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time.time()
        total_tokens = 0

        for epoch in range(999):
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.step >= self.config.max_steps:
                    print(f"\nReached max_steps. Done.")
                    self._save_checkpoint('final')
                    return

                x = x.to(self.device)
                y = y.to(self.device)

                lr = self._get_lr(self.step)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

                # ── Mixed precision + gradient checkpointing ──
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, logits, lm_loss, aux = self.model(
                        x, targets=y,
                        use_checkpoint=True   # ← activations recomputed in backward
                    )

                loss = loss / self.grad_accum
                loss.backward()
                total_tokens += x.numel()

                if (batch_idx + 1) % self.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Free cache after each step
                    torch.cuda.empty_cache()
                    self.step += 1

                    if self.step % self.log_interval == 0:
                        t1 = time.time()
                        tok_s = total_tokens / (t1 - t0)
                        vram  = torch.cuda.memory_allocated() / 1024**3
                        print(f"Step {self.step:6d} | "
                              f"loss: {lm_loss.item():.4f} | "
                              f"aux: {aux.item():.4f} | "
                              f"lr: {lr:.2e} | "
                              f"norm: {grad_norm:.3f} | "
                              f"tok/s: {tok_s:,.0f} | "
                              f"VRAM: {vram:.1f}GB")
                        t0 = time.time(); total_tokens = 0

                    if self.step % self.save_interval == 0:
                        self._save_checkpoint(f'step_{self.step}')

    def _get_lr(self, step):
        max_lr = self.config.learning_rate
        min_lr = max_lr * 0.1
        warmup = self.config.warmup_steps
        total  = self.config.max_steps
        if step < warmup:
            return max_lr * step / max(warmup, 1)
        if step > total:
            return min_lr
        progress = (step - warmup) / (total - warmup)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(3.14159 * progress))


print("\nMemoryEfficientTrainer ready.")
print("Expected VRAM during training: ~10-12GB with gradient checkpointing")
print("If still OOM, reduce config.max_seq_len to 128")