import torch
import gc
from transformers import AutoTokenizer
from model import PhantomLM
from config import PhantomLMConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device != 'cuda':
    print("CUDA is required for accurate VRAM measurements.")
    exit(1)

# Load Tokenizer and Model
import bitlinear
bitlinear.QUANTIZE_ENABLED = False

print("Loading PhantomLM for Peak RAM Benchmark...")
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
config = PhantomLMConfig.phantom_medium()
config.vocab_size = 50257

try:
    model = PhantomLM(config).to(device)
    ckpt = torch.load('/kaggle/working/checkpoints/phantomlm_final.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("FP16 Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def measure_peak_ram(num_tokens=50):
    # Force garbage collection and empty cache to get clean baseline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Base VRAM (model weights loaded)
    base_vram = torch.cuda.memory_allocated() / (1024**2)
    
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    # Generate tokens
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=num_tokens, temperature=0.8)
        
    # Peak VRAM during generation
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    # The actual memory required for the KV Cache + Context activations
    generation_overhead = peak_vram - base_vram
    
    print(f"--- Generation: {num_tokens} tokens ---")
    print(f"Base Model VRAM     : {base_vram:.2f} MB")
    print(f"Peak VRAM Reached   : {peak_vram:.2f} MB")
    print(f"Generation Overhead : {generation_overhead:.2f} MB")
    print("-" * 35)

if __name__ == "__main__":
    print("Measuring Peak VRAM During Generation...\n")
    measure_peak_ram(50)
    measure_peak_ram(100)
    measure_peak_ram(200)
