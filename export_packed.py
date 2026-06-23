import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict

def pack_4bit_weights(weight_tensor):
    """Pack 4-bit weights, 2 values per byte."""
    # 1. Quantize to 4-bit using ABSMAX
    scale = weight_tensor.abs().max() / 7.0
    w_quant = torch.round(weight_tensor / scale).clamp(-8, 7)
    
    # 2. Shift from [-8, 7] to [0, 15] for unsigned storage
    flat = (w_quant + 8).flatten().cpu().numpy().astype(np.uint8)
    
    # 3. Pad if odd length
    if len(flat) % 2 != 0:
        flat = np.concatenate([flat, [0]])
        
    # 4. Pack 2 values per byte (high nibble | low nibble)
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed.tobytes(), scale.item()

def pack_8bit_weights(weight_tensor):
    """Pack 8-bit weights, 1 value per byte."""
    scale = weight_tensor.abs().max() / 127.0
    quantized = torch.round(weight_tensor / scale).clamp(-128, 127)
    return quantized.flatten().cpu().numpy().astype(np.int8).tobytes(), scale.item()

def export_phantomlm_deployment(model_checkpoint_path, output_bin_path='phantomlm_deployment_v2.bin'):
    """
    Advanced export that aggressively quantizes based on our paper's architecture:
    - BitLinear weights: Base-3 Ternary (5 values/byte)
    - Attention Linear4bit: 4-bit INT4 (2 values/byte)
    - Embeddings: 8-bit INT8 (1 value/byte)
    - Norms/Biases/Criticals: FP16 (2 bytes/value)
    """
    print(f"Loading checkpoint from: {model_checkpoint_path}")
    
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    except FileNotFoundError:
        print("Checkpoint not found! Make sure you point this to your trained QAT checkpoint.")
        return

    stats = {'ternary': 0, '4bit': 0, '8bit': 0, 'fp16': 0}
    
    with open(output_bin_path, 'wb') as f:
        for name, tensor in state_dict.items():
            
            # Skip double-saving the tied lm_head
            if 'lm_head.weight' in name:
                continue
                
            # 1. TERNARY (Mamba + MoE)
            ternary_targets = ['in_proj', 'x_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']
            is_ternary = any(f"{t}.weight" in name for t in ternary_targets)
            
            # 2. 4-BIT (Attention)
            four_bit_targets = ['wq', 'wk', 'wv', 'wo']
            is_4bit = any(f"{t}.weight" in name for t in four_bit_targets)
            
            # 3. 8-BIT (Embeddings)
            is_8bit = ('embed.weight' in name)
            
            # PACKING LOGIC
            if is_ternary:
                # Quantize
                scale = tensor.abs().mean() + 1e-8
                w_scaled = tensor / scale
                w_ternary = torch.round(torch.clamp(w_scaled, -1, 1))
                flat = w_ternary.flatten().to(torch.int8).cpu().numpy()
                
                # Base-3 packing (5 values per byte)
                shifted = flat + 1
                padding = (5 - (shifted.size % 5)) % 5
                if padding > 0:
                    shifted = np.concatenate([shifted, np.full(padding, 1, dtype=np.int8)])
                reshaped = shifted.reshape(-1, 5)
                packed_bytes = (
                    reshaped[:, 0] * 81 +
                    reshaped[:, 1] * 27 +
                    reshaped[:, 2] * 9 +
                    reshaped[:, 3] * 3 +
                    reshaped[:, 4] * 1
                ).astype(np.uint8)
                
                f.write(scale.to(torch.float16).cpu().numpy().tobytes())
                f.write(packed_bytes.tobytes())
                stats['ternary'] += tensor.numel()
                
            elif is_4bit:
                packed_bytes, scale = pack_4bit_weights(tensor)
                f.write(torch.tensor(scale, dtype=torch.float16).cpu().numpy().tobytes())
                f.write(packed_bytes)
                stats['4bit'] += tensor.numel()
                
            elif is_8bit:
                packed_bytes, scale = pack_8bit_weights(tensor)
                f.write(torch.tensor(scale, dtype=torch.float16).cpu().numpy().tobytes())
                f.write(packed_bytes)
                stats['8bit'] += tensor.numel()
                
            else:
                # FP16 (Biases, LayerNorms, Router weights, Mamba A/D/Conv)
                fp16_data = tensor.to(torch.float16).cpu().numpy()
                f.write(fp16_data.tobytes())
                stats['fp16'] += tensor.numel()
                
    # Calculate final physical size
    actual_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    
    print("\n" + "="*50)
    print("V2 Advanced Deployment Artifact Generation Complete")
    print("="*50)
    print(f"Ternary params packed (5/byte)  : {stats['ternary']:,}")
    print(f"4-bit params packed   (2/byte)  : {stats['4bit']:,}")
    print(f"8-bit params packed   (1/byte)  : {stats['8bit']:,}")
    print(f"FP16 params saved     (Unpacked): {stats['fp16']:,}")
    print("-" * 40)
    print(f"MEASURED PACKED FILE SIZE       : {actual_size_mb:.2f} MB")
    print(f"Output saved to                 : {output_bin_path}")
    print("="*50)

if __name__ == "__main__":
    # Point this to whatever your final QAT checkpoint was called
    export_phantomlm_deployment('/kaggle/working/checkpoints/phantomlm_final.pt')
