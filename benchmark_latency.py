import torch
import time

def benchmark_latency(model, tokenizer, max_new_tokens=50, context_lengths=[128, 256, 512, 1024]):
    """
    Measures the token generation throughput (Tokens/second) for different context lengths.
    
    WARNING: Because the current Mamba implementation in mamba_block.py uses a sequential
    Python 'for t in range(L)' loop and lacks a state-cache for inference, PhantomLM will 
    measure as much slower than a Transformer here. An optimized CUDA kernel (like mamba_ssm) 
    is required to unlock Mamba's true O(1) inference speed.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # --- HOTFIX FOR LONG CONTEXT BENCHMARKING ---
    # Because the model was trained with max_seq_len=256, it only has 256 RoPE embeddings.
    # To benchmark longer contexts (512, 1024) without crashing, we dynamically extend the 
    # RoPE mathematical buffer to support whatever the maximum test length is.
    max_test_len = max(context_lengths)
    if max_test_len > model.config.max_seq_len:
        from attention import precompute_freqs_cis
        model.freqs_cis = precompute_freqs_cis(model.config.head_dim, max_test_len).to(device)
        model.config.max_seq_len = max_test_len  # Spoof config so we don't truncate
    # --------------------------------------------
    
    print(f"{'Context Length':<15} | {'Throughput (Tokens/s)':<25} | {'Time per Token (ms)'}")
    print("-" * 65)
    
    for ctx_len in context_lengths:
            
        # Create dummy input of sequence length `ctx_len`
        input_ids = torch.randint(0, model.config.vocab_size, (1, ctx_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_ids)
                
        # Benchmark Generation (simulated without cache)
        torch.cuda.synchronize(device)
        start_time = time.time()
        
        with torch.no_grad():
            # Generate `max_new_tokens` autoregressively
            current_ids = input_ids
            for _ in range(max_new_tokens):
                idx_cond = current_ids[:, -model.config.max_seq_len:]
                logits = model(idx_cond)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
        torch.cuda.synchronize(device)
        total_time = time.time() - start_time
        
        throughput = max_new_tokens / total_time
        ms_per_token = (total_time / max_new_tokens) * 1000
        
        print(f"{ctx_len:<15} | {throughput:<25.2f} | {ms_per_token:.2f} ms")

# Example usage in Kaggle:
# benchmark_latency(model, tokenizer)
