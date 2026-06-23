import torch
import time

def benchmark_vram(model, vocab_size, context_lengths, device, n_trials=5, n_forward=20):
    """
    Robust VRAM benchmark — averages multiple trials, discards first
    trial per context length to avoid CUDA allocator cold-start noise.
    """
    results = {}
    model.eval()

    print(f"{'Context Length':<15} | {'Peak VRAM (MB)':<20}")
    print("-" * 40)

    for ctx in context_lengths:
        # Just like the latency script, make sure we don't exceed the model's capacity
        if hasattr(model, 'config') and ctx > model.config.max_seq_len:
            print(f"Skipping {ctx:<6} | (Exceeds model max_seq_len)")
            continue

        trial_vrams = []
        for trial in range(n_trials):
            dummy = torch.randint(0, vocab_size, (1, ctx), device=device)

            # Warmup pass — not measured, primes allocator
            with torch.no_grad():
                _ = model(dummy, return_loss=False) if 'return_loss' in model.forward.__code__.co_varnames else model(dummy)
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                for _ in range(n_forward):
                    _ = model(dummy, return_loss=False) if 'return_loss' in model.forward.__code__.co_varnames else model(dummy)
            torch.cuda.synchronize()

            vram = torch.cuda.max_memory_allocated() / 1024**2
            
            # Discard first trial — allocator hasn't stabilized yet
            if trial > 0:
                trial_vrams.append(vram)

        avg_vram = sum(trial_vrams) / len(trial_vrams)
        std_vram = (sum((v - avg_vram)**2 for v in trial_vrams) / len(trial_vrams)) ** 0.5
        results[ctx] = {'mean_mb': round(avg_vram, 1), 'std_mb': round(std_vram, 1)}
        print(f"{ctx:<15} | {avg_vram:.1f} ± {std_vram:.1f} MB  (n={len(trial_vrams)})")

    return results

# =========================================================
# KAGGLE USAGE:
# (Assuming your 'model' is PhantomLM and is already loaded)
# =========================================================
#
# device = next(model.parameters()).device
# vocab_size = model.config.vocab_size
#
# # Dynamically extend RoPE if benchmarking larger context lengths
# max_test_len = 2048
# if max_test_len > model.config.max_seq_len:
#     from attention import precompute_freqs_cis
#     model.freqs_cis = precompute_freqs_cis(model.config.head_dim, max_test_len).to(device)
#     model.config.max_seq_len = max_test_len
#
# contexts = [128, 256, 512, 1024, 2048]
# print("\nMeasuring PhantomLM VRAM Scaling:")
# phantom_vram_results = benchmark_vram(model, vocab_size, contexts, device)
