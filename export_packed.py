import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict

def export_phantomlm_deployment(model_checkpoint_path, output_bin_path='phantomlm_deployment.bin'):
    """
    Exports a trained PhantomLM PyTorch checkpoint to a real, measured deployment binary.
    - BitLinear weights are packed using base-3 (5 ternary values per byte).
    - Embedding, Attention, and Norm weights are saved in FP16.
    
    This converts a theoretical size estimate into a MEASURED empirical result.
    """
    print(f"Loading checkpoint from: {model_checkpoint_path}")
    
    # Load the checkpoint dictionary
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    except FileNotFoundError:
        print("Checkpoint not found! Make sure you point this to your trained QAT checkpoint.")
        return
        
    # We must replicate the math from bitlinear.py: quantize_weights_ternary
    def quantize_ternary(w):
        scale = w.abs().mean() + 1e-8
        w_scaled = w / scale
        w_ternary = torch.round(torch.clamp(w_scaled, -1, 1))
        return w_ternary, scale

    total_ternary_params = 0
    total_fp16_params = 0
    
    # Open the binary file to write out raw bytes
    with open(output_bin_path, 'wb') as f:
        for name, tensor in state_dict.items():
            # The EXACT names of our ternary matrices in PhantomLM
            ternary_targets = ['in_proj', 'x_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']
            
            is_ternary = False
            for t in ternary_targets:
                # Check that it is the primary weight matrix, not a sub-module like .norm.weight
                if f"{t}.weight" in name:
                    is_ternary = True
                    break
                    
            # Skip double-saving the tied lm_head
            if 'lm_head.weight' in name:
                continue
                
            if is_ternary:
                # 1. Quantize
                w_ternary, scale = quantize_ternary(tensor)
                flat = w_ternary.flatten().to(torch.int8).cpu().numpy()
                total_ternary_params += flat.size
                
                # We save the single FP16 scale factor for this weight matrix
                f.write(scale.to(torch.float16).cpu().numpy().tobytes())
                
                # 2. Base-3 packing (5 values per byte)
                shifted = flat + 1  # shift {-1, 0, 1} to {0, 1, 2}
                
                pad = (-len(shifted)) % 5
                if pad:
                    shifted = np.concatenate([shifted, np.zeros(pad, dtype=np.int8)])
                
                reshaped = shifted.reshape(-1, 5)
                packed = np.zeros(len(reshaped), dtype=np.uint8)
                
                # 3^0, 3^1, 3^2, 3^3, 3^4 packing
                for i in range(5):
                    packed = packed * 3 + reshaped[:, i]
                
                # Write to binary file
                f.write(packed.tobytes())
                
            else:
                # Standard FP16 layers (Embedding, Attention, Norms)
                total_fp16_params += tensor.numel()
                f.write(tensor.to(torch.float16).cpu().numpy().tobytes())

    # Measure the exact physical file size
    actual_size_bytes = os.path.getsize(output_bin_path)
    actual_size_mb = actual_size_bytes / 1024**2
    
    print("\n" + "="*50)
    print("Deployment Artifact Generation Complete")
    print("="*50)
    print(f"Ternary parameters packed : {total_ternary_params:,}")
    print(f"FP16 parameters saved     : {total_fp16_params:,}")
    print(f"----------------------------------------")
    print(f"MEASURED PACKED FILE SIZE : {actual_size_mb:.2f} MB")
    print(f"Output saved to           : {output_bin_path}")
    print("="*50)

if __name__ == "__main__":
    # Point this to whatever your final QAT checkpoint was called
    export_phantomlm_deployment('/kaggle/working/checkpoints/phantomlm_qat_final.pt')
