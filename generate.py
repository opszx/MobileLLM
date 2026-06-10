# Inference + chat + benchmarking
"""
PhantomLM Inference Script
Load a checkpoint and generate text or chat interactively

Usage:
  python generate.py --checkpoint checkpoints/phantomlm_best.pt --prompt "What is AI?"
  python generate.py --checkpoint checkpoints/phantomlm_best.pt --chat
"""

import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import PhantomLMConfig
from model import PhantomLM


# ── Simple tokenizer (byte-level) ─────────────────────────────────────────────
# Replace with SentencePiece or HuggingFace tokenizer for real deployment

class ByteTokenizer:
    """Byte-level tokenizer for testing. Use proper tokenizer for training."""

    def encode(self, text: str):
        return list(text.encode('utf-8'))

    def decode(self, tokens) -> str:
        try:
            return bytes(tokens).decode('utf-8', errors='replace')
        except Exception:
            return str(tokens)


# ── Efficiency benchmark ───────────────────────────────────────────────────────

def benchmark_inference(model: PhantomLM, device: torch.device, seq_lengths=[64, 128, 256, 512]):
    """
    Measure tokens/sec and memory at different context lengths.
    This is what you report in Section 6.3 of your paper.
    """
    model.eval()
    results = []

    print("\n" + "="*55)
    print(f"{'Efficiency Benchmark':^55}")
    print("="*55)
    print(f"{'Context Len':>12} | {'Tokens/s':>10} | {'Latency(ms)':>12} | {'RAM(MB)':>8}")
    print("-"*55)

    for seq_len in seq_lengths:
        # Warm up
        dummy = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Time generation of 20 new tokens
        n_gen = 20
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(dummy, max_new_tokens=n_gen, temperature=1.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tokens_per_sec = n_gen / elapsed
        latency_ms = elapsed / n_gen * 1000

        # Memory
        if device.type == 'cuda':
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            import psutil
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        results.append({
            'seq_len': seq_len,
            'tokens_per_sec': tokens_per_sec,
            'latency_ms': latency_ms,
            'ram_mb': mem_mb
        })
        print(f"{seq_len:>12} | {tokens_per_sec:>10.1f} | {latency_ms:>12.1f} | {mem_mb:>8.1f}")

    print("="*55)
    return results


def count_model_size(model: PhantomLM):
    """
    Compute model size statistics for paper reporting.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate disk size at different precisions
    fp32_mb  = total_params * 4 / 1024**2
    fp16_mb  = total_params * 2 / 1024**2
    int8_mb  = total_params * 1 / 1024**2
    # Weighted estimate for PhantomLM's mixed precision
    # Mamba (75%): 1.58-bit ≈ 0.2 bytes/param
    # Attn  (25%): 4-bit    ≈ 0.5 bytes/param
    # Emb + head:  8-bit    ≈ 1.0 bytes/param
    emb_params = model.config.vocab_size * model.config.d_model
    mamba_ratio = model.layer_types.count('mamba') / model.config.n_layers
    attn_ratio  = 1.0 - mamba_ratio
    phantom_bytes = (
        emb_params * 1.0 +
        (total_params - emb_params) * (mamba_ratio * 0.2 + attn_ratio * 0.5)
    )
    phantom_mb = phantom_bytes / 1024**2

    print("\n" + "="*45)
    print(f"{'Model Size Analysis':^45}")
    print("="*45)
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"FP32  size       : {fp32_mb:>8.1f} MB")
    print(f"FP16  size       : {fp16_mb:>8.1f} MB")
    print(f"INT8  size       : {int8_mb:>8.1f} MB")
    print(f"PhantomLM (mixed): {phantom_mb:>8.1f} MB  ← on-device size")
    print("="*45)
    return total_params, phantom_mb


# ── Perplexity evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model: PhantomLM, text: str, tokenizer, device: torch.device) -> float:
    """
    Compute perplexity on a text sample.
    Lower = better. Report this in paper Table 1.
    """
    model.eval()
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')

    seq_len = model.config.max_seq_len
    tokens = tokens[:seq_len + 1]
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    x = ids[:, :-1]
    y = ids[:, 1:]

    loss, _, lm_loss, _ = model(x, targets=y)
    perplexity = torch.exp(lm_loss).item()
    return perplexity


# ── Interactive chat ───────────────────────────────────────────────────────────

def chat(model: PhantomLM, tokenizer, device: torch.device):
    """Simple interactive chat loop for testing generation quality."""
    print("\nPhantomLM Chat - type 'quit' to exit, 'bench' to benchmark\n")
    history_ids = torch.tensor([[model.config.bos_token_id]], device=device)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'bench':
            benchmark_inference(model, device)
            continue
        if not user_input:
            continue

        # Encode and append to history
        user_tokens = tokenizer.encode(user_input + "\n")
        new_ids = torch.tensor([user_tokens], device=device)
        history_ids = torch.cat([history_ids, new_ids], dim=1)

        # Trim to max context
        history_ids = history_ids[:, -model.config.max_seq_len:]

        # Generate
        t0 = time.perf_counter()
        out_ids = model.generate(
            history_ids,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=model.config.eos_token_id
        )
        elapsed = time.perf_counter() - t0

        # Decode only new tokens
        new_tokens = out_ids[0, history_ids.shape[1]:].tolist()
        response = tokenizer.decode(new_tokens)
        n_tokens = len(new_tokens)

        print(f"PhantomLM: {response}")
        print(f"  [{n_tokens} tokens in {elapsed:.2f}s = {n_tokens/elapsed:.1f} tok/s]\n")

        # Add response to history
        history_ids = out_ids


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', default='tiny', choices=['tiny', '350m', '1b'])
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Load or create model
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = ckpt['config']
        model = PhantomLM(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — using randomly initialized model")
        if args.config == '350m':
            config = PhantomLMConfig.phantom_350m()
        elif args.config == '1b':
            config = PhantomLMConfig.phantom_1b()
        else:
            config = PhantomLMConfig.phantom_tiny()
        model = PhantomLM(config).to(device)

    model.eval()
    tokenizer = ByteTokenizer()

    # Model size analysis
    count_model_size(model)

    # Benchmark
    if args.benchmark:
        benchmark_inference(model, device)

    # Single prompt generation
    if args.prompt:
        tokens = tokenizer.encode(args.prompt)
        ids = torch.tensor([tokens], device=device)
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature)
        elapsed = time.perf_counter() - t0
        new_tokens = out[0, len(tokens):].tolist()
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {tokenizer.decode(new_tokens)}")
        print(f"[{len(new_tokens)} tokens, {elapsed:.2f}s, {len(new_tokens)/elapsed:.1f} tok/s]")

    # Interactive chat
    if args.chat:
        chat(model, tokenizer, device)


if __name__ == '__main__':
    main()