"""
PhantomLM Evaluation Script — fully self-contained, no circular imports.

Usage:
  python evaluate.py --checkpoint checkpoints/phantomlm_best.pt --all
  python evaluate.py --checkpoint checkpoints/phantomlm_best.pt --ablation
"""
import os, sys, time, json, argparse, copy
import torch
import torch.nn.functional as F
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PhantomLMConfig
from model  import PhantomLM
from attention import GroupedQueryAttention


# ── Byte-level tokenizer ──────────────────────────────────────────────────────

class ByteTokenizer:
    def encode(self, text):
        return [min(b, 31999) for b in text.encode('utf-8')]
    def decode(self, tokens):
        return bytes(tokens).decode('utf-8', errors='replace')


# ── Multiple choice accuracy ──────────────────────────────────────────────────

@torch.no_grad()
def eval_multiple_choice(model, device, examples, tokenizer):
    model.eval()
    correct = 0
    for ex in examples:
        ctx_tokens    = tokenizer.encode(ex['context'])
        best_score    = float('-inf')
        best_idx      = 0
        for i, choice in enumerate(ex['choices']):
            ch_tokens  = tokenizer.encode(choice)
            full       = (ctx_tokens + ch_tokens)[:model.config.max_seq_len]
            if len(full) < 2:
                continue
            ids        = torch.tensor([full], dtype=torch.long, device=device)
            logits     = model(ids, return_loss=False)
            n_ctx      = len(ctx_tokens)
            cl         = logits[0, n_ctx-1:-1, :]
            ci         = torch.tensor(ch_tokens, device=device)
            n          = min(cl.shape[0], len(ci))
            if n == 0:
                continue
            score      = F.log_softmax(cl[:n], dim=-1)[range(n), ci[:n]].mean().item()
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx == ex['answer_idx']:
            correct += 1
    return correct / max(len(examples), 1)


# ── Benchmark data ────────────────────────────────────────────────────────────

def make_arc_examples():
    return [
        {'context': "Question: What do plants need for photosynthesis?\n",
         'choices': ["Sunlight, water, and CO2", "Soil and fertilizer only",
                     "Darkness and nitrogen", "Oxygen and sugar"], 'answer_idx': 0},
        {'context': "Question: Which material conducts electricity?\n",
         'choices': ["Rubber", "Wood", "Copper wire", "Plastic"], 'answer_idx': 2},
        {'context': "Question: What happens to water when it freezes?\n",
         'choices': ["It becomes lighter and sinks", "It expands and becomes less dense",
                     "It shrinks and becomes denser", "Its formula changes"], 'answer_idx': 1},
        {'context': "Question: How many sides does a hexagon have?\n",
         'choices': ["5", "6", "7", "8"], 'answer_idx': 1},
        {'context': "Question: Which planet is closest to the Sun?\n",
         'choices': ["Venus", "Earth", "Mercury", "Mars"], 'answer_idx': 2},
    ]


def make_hellaswag_examples():
    return [
        {'context': "A person picks up a guitar and starts strumming. They look at sheet music and",
         'choices': ["begin playing a melody carefully", "throw it in the ocean",
                     "start cooking pasta", "the guitar disappears"], 'answer_idx': 0},
        {'context': "She opened the refrigerator and noticed there was no food. She decided to",
         'choices': ["watch television instead", "go to the grocery store",
                     "plant a garden immediately", "call the refrigerator company"], 'answer_idx': 1},
        {'context': "The student studied hard for the exam all night. In the morning she felt",
         'choices': ["well-prepared but tired", "ready to go swimming",
                     "angry at her computer", "like building furniture"], 'answer_idx': 0},
    ]


# ── Efficiency benchmark ──────────────────────────────────────────────────────

def run_efficiency_benchmark(model, device):
    model.eval()
    results = {}
    for seq_len in [64, 128, 256, 512]:
        if seq_len >= model.config.max_seq_len:
            continue
        dummy = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            _ = model(dummy, return_loss=False)   # warm up
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        n = 50
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                _ = model(dummy, return_loss=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if device.type == 'cuda':
            ram_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            try:
                import psutil
                ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            except ImportError:
                ram_mb = 0.0
        results[seq_len] = {
            'tokens_per_sec': round(seq_len * n / elapsed, 1),
            'latency_ms':     round(elapsed / n * 1000, 2),
            'ram_mb':         round(ram_mb, 1),
        }
    return results


# ── Ablation helpers ──────────────────────────────────────────────────────────

def _score_model(model, test_tokens, cfg, device):
    """Compute PPL and tok/s for a model."""
    seq   = min(len(test_tokens) - 1, cfg.max_seq_len - 1)
    x     = torch.tensor([test_tokens[:seq]],   dtype=torch.long, device=device)
    y     = torch.tensor([test_tokens[1:seq+1]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, lm_loss, _ = model(x, targets=y)
    ppl   = torch.exp(lm_loss).item()
    sl    = min(128, cfg.max_seq_len - 1)
    dummy = torch.randint(0, cfg.vocab_size, (1, sl), device=device)
    t0    = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(dummy, return_loss=False)
    tok_s = sl * 20 / (time.perf_counter() - t0)
    return {
        'perplexity':    round(ppl, 2),
        'tokens_per_sec': round(tok_s, 1),
        'parameters':    sum(p.numel() for p in model.parameters()),
    }


class TransformerOnlyPhantomLM(PhantomLM):
    """PhantomLM with ALL Mamba layers replaced by Attention — pure transformer baseline."""
    def __init__(self, config):
        super().__init__(config)
        replaced = 0
        for layer in self.layers:
            if layer.layer_type == 'mamba':
                layer.core       = GroupedQueryAttention(config)
                layer.layer_type = 'attention'
                replaced        += 1
        print(f"    Replaced {replaced} Mamba → Attention layers")


# ── Ablation study ────────────────────────────────────────────────────────────

def run_ablation_study(base_config, device, tokenizer):
    test_text   = (
        "The transformer architecture has revolutionized natural language processing. "
        "By using self-attention mechanisms transformers capture long-range dependencies. "
        "Recent work has explored making models efficient for deployment on edge devices. "
    ) * 20
    test_tokens = tokenizer.encode(test_text)
    results     = {}

    def _run(name, model):
        print(f"  Testing: {name}...")
        model.eval()
        r = _score_model(model, test_tokens, model.config, device)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return r

    # 1. Full PhantomLM
    results["1. PhantomLM (full)"] = _run(
        "PhantomLM (full)", PhantomLM(base_config).to(device))

    # 2. No Mamba — pure transformer (same d_model, all attention)
    results["2. No Mamba (pure Transformer)"] = _run(
        "No Mamba (pure Transformer)", TransformerOnlyPhantomLM(base_config).to(device))

    # 3. No MoE — standard FFN everywhere
    cfg3 = copy.deepcopy(base_config)
    cfg3.moe_every_n_layers = 9999
    results["3. No MoE (standard FFN)"] = _run(
        "No MoE", PhantomLM(cfg3).to(device))

    # 4. Uniform attention placement (every 3rd layer throughout, no zones)
    cfg4 = copy.deepcopy(base_config)
    cfg4.mamba_zone_end = 0
    cfg4.mixed_zone_end = base_config.n_layers
    results["4. Uniform attention placement"] = _run(
        "Uniform placement", PhantomLM(cfg4).to(device))

    # 5. All 1.58-bit — no selective precision (replace Linear4bit with BitLinear)
    import attention as _attn_mod
    from bitlinear import BitLinear as _BL
    _orig = _attn_mod.Linear4bit
    _attn_mod.Linear4bit = _BL
    results["5. No selective precision (all 1.58-bit)"] = _run(
        "No selective precision", PhantomLM(base_config).to(device))
    _attn_mod.Linear4bit = _orig   # restore

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--config',     default='tiny', choices=['tiny','350m','1b'])
    parser.add_argument('--all',        action='store_true')
    parser.add_argument('--quality',    action='store_true')
    parser.add_argument('--efficiency', action='store_true')
    parser.add_argument('--ablation',   action='store_true')
    parser.add_argument('--output',     default='eval_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok    = ByteTokenizer()

    # Load model
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt   = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = ckpt['config']
        model  = PhantomLM(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded: {args.checkpoint}  ({config.d_model}d, {config.n_layers}L)")
    else:
        config = (PhantomLMConfig.phantom_350m() if args.config == '350m'
                  else PhantomLMConfig.phantom_1b()   if args.config == '1b'
                  else PhantomLMConfig.phantom_tiny())
        model  = PhantomLM(config).to(device)
        print("No checkpoint — random init model")

    all_results = {}

    # Quality
    if args.quality or args.all:
        print("\n── Quality Benchmarks ──")
        arc = eval_multiple_choice(model, device, make_arc_examples(), tok)
        hs  = eval_multiple_choice(model, device, make_hellaswag_examples(), tok)
        print(f"  ARC accuracy      : {arc:.1%}")
        print(f"  HellaSwag accuracy: {hs:.1%}")
        all_results['quality'] = {'arc': arc, 'hellaswag': hs}

    # Efficiency
    if args.efficiency or args.all:
        print("\n── Efficiency Benchmarks ──")
        eff = run_efficiency_benchmark(model, device)
        for sl, m in eff.items():
            print(f"  ctx={sl:4d}: {m['tokens_per_sec']:7.1f} tok/s | "
                  f"{m['latency_ms']:7.1f}ms | {m['ram_mb']:7.1f}MB RAM")
        all_results['efficiency'] = eff

    # Ablation
    if args.ablation or args.all:
        print("\n── Ablation Study ──")
        ablation = run_ablation_study(config, device, tok)
        print(f"\n  {'Variant':<40} {'PPL':>8} {'Tok/s':>7} {'Params':>12}")
        print("  " + "-"*72)
        for name, m in ablation.items():
            print(f"  {name:<40} {m['perplexity']:>8.0f} "
                  f"{m['tokens_per_sec']:>7.0f} {m['parameters']:>12,}")
        all_results['ablation'] = ablation

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()