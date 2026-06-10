# Training loop + cosine LR + checkpointing
"""
PhantomLM Training Script
Supports: pre-training, instruction fine-tuning, both stages

Usage:
  python train.py --stage pretrain --config tiny    # quick test
  python train.py --stage pretrain --config 350m    # Kaggle T4
  python train.py --stage finetune --checkpoint path/to/ckpt.pt
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from config import PhantomLMConfig
from model import PhantomLM


# ── Simple text dataset ────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Simple character/token level dataset for testing.
    Replace with HuggingFace datasets for real training.
    """

    def __init__(self, text: str, seq_len: int, vocab_size: int = 256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Simple byte tokenization for testing
        self.tokens = torch.tensor(
            [min(b, vocab_size - 1) for b in text.encode('utf-8')],
            dtype=torch.long
        )

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class HuggingFaceDataset(Dataset):
    """
    Wrapper for HuggingFace datasets.
    Uses for actual training on FineWeb-Edu, OpenWebMath, etc.

    Example usage:
      from datasets import load_dataset
      ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    """

    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.dataset = list(hf_dataset)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = self._tokenize_all()

    def _tokenize_all(self):
        all_tokens = []
        for item in self.dataset:
            text = item.get('text', '') or item.get('content', '')
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ── Learning rate schedule ─────────────────────────────────────────────────────

def get_lr(step: int, config: PhantomLMConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (step < warmup_steps): linear warmup from 0 to max_lr
    Phase 2 (warmup ≤ step ≤ max_steps): cosine decay to min_lr
    Phase 3 (step > max_steps): constant min_lr
    """
    max_lr = config.learning_rate
    min_lr = max_lr * 0.1
    warmup = config.warmup_steps
    max_steps = config.max_steps

    if step < warmup:
        return max_lr * step / warmup
    if step > max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Training loop ──────────────────────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        model: PhantomLM,
        config: PhantomLMConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.grad_accum = gradient_accumulation_steps

        # Detect device
        self.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        print(f"Training on: {self.device}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.step = 0
        self.best_val_loss = float('inf')

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        t0 = time.time()
        total_tokens = 0

        for epoch in range(999):  # loop indefinitely until max_steps
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.step >= self.config.max_steps:
                    print(f"\nReached max_steps={self.config.max_steps}. Done.")
                    self._save_checkpoint('final')
                    return

                x = x.to(self.device)
                y = y.to(self.device)

                # Update learning rate
                lr = get_lr(self.step, self.config)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Forward pass
                loss, logits, lm_loss, aux_loss = self.model(x, targets=y)

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum
                loss.backward()

                total_tokens += x.numel()

                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum == 0:
                    # Clip gradients
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.step += 1

                    # Logging
                    if self.step % self.log_interval == 0:
                        t1 = time.time()
                        dt = t1 - t0
                        tokens_per_sec = total_tokens / dt
                        print(
                            f"Step {self.step:6d} | "
                            f"loss: {lm_loss.item():.4f} | "
                            f"aux: {aux_loss.item():.4f} | "
                            f"lr: {lr:.2e} | "
                            f"grad_norm: {grad_norm:.3f} | "
                            f"tok/s: {tokens_per_sec:,.0f}"
                        )
                        t0 = time.time()
                        total_tokens = 0

                    # Evaluation
                    if self.val_dataset and self.step % self.eval_interval == 0:
                        val_loss = self._evaluate()
                        print(f"  -> Val loss: {val_loss:.4f}")
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint('best')
                        self.model.train()

                    # Periodic checkpoint
                    if self.step % self.save_interval == 0:
                        self._save_checkpoint(f'step_{self.step}')

    @torch.no_grad()
    def _evaluate(self, max_batches: int = 20) -> float:
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        total_loss = 0.0
        count = 0
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            loss, _, _, _ = self.model(x, targets=y)
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f'phantomlm_{tag}.pt')
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'layer_types': self.model.layer_types,
        }, path)
        print(f"  -> Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, model: PhantomLM, optimizer=None):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Loaded checkpoint from step {ckpt['step']}")
        return ckpt['step'], ckpt.get('best_val_loss', float('inf'))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='tiny',
                        choices=['tiny', '350m', '1b'])
    parser.add_argument('--stage', default='pretrain',
                        choices=['pretrain', 'finetune'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--max_steps', type=int, default=None)
    args = parser.parse_args()

    # Load config
    if args.config == 'tiny':
        config = PhantomLMConfig.phantom_tiny()
        config.max_steps = args.max_steps or 1000
        config.log_interval = 10
        config.batch_size = 2  # reduced for low-VRAM GPUs
    elif args.config == '350m':
        config = PhantomLMConfig.phantom_350m()
        config.max_steps = args.max_steps or 100000
    else:
        config = PhantomLMConfig.phantom_1b()
        config.max_steps = args.max_steps or 200000

    # Build model
    model = PhantomLM(config)

    # Demo dataset (replace with real data for actual training)
    sample_text = """
    PhantomLM is a phone-native language model designed from scratch for mobile deployment.
    Unlike existing models that compress server architectures down to phone scale,
    PhantomLM treats mobile constraints as first-class design requirements.
    The architecture combines Mamba state space models with grouped query attention,
    using 1.58-bit ternary weights throughout to minimize memory footprint.
    """ * 500

    dataset = TextDataset(sample_text, seq_len=min(128, config.max_seq_len),
                          vocab_size=config.vocab_size)

    # Split train/val
    val_size = max(10, len(dataset) // 10)
    train_size = len(dataset) - val_size

    if train_size > 0:
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    # Load checkpoint if provided
    if args.checkpoint:
        Trainer.load_checkpoint(args.checkpoint, model)

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation_steps=4,
    )

    # Free any leftover CUDA memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nStarting {args.stage} training with {args.config} config...")
    trainer.train()


if __name__ == '__main__':
    main()