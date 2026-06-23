import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

def plot_expert_routing_distribution(model_path, sample_text=None):
    """
    Loads PhantomLM, runs a sample text, and plots the frequency of each
    expert being chosen across the different MoE layers.
    """
    print(f"Loading model from {model_path}...")
    
    # Normally you would instantiate the model and load state_dict.
    # For this script, we assume `checkpoint` is a full model or we can instantiate it.
    # Since we are in Kaggle, the user will run this where PhantomLM and Config are defined.
    # We will provide the script to be run inside the Kaggle notebook.
    
    code = """
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

# Assuming 'model' is already loaded in your Kaggle notebook memory
# and 'tokenizer' is available.

def record_routing_distribution(model, tokenizer, text):
    # Dictionary to store expert counts per layer
    # layer_idx -> list of selected expert indices
    routing_data = defaultdict(list)
    
    # Find all MoE router modules and attach hooks
    hooks = []
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            # output is router_logits: (T, n_experts)
            router_probs = F.softmax(output, dim=-1)
            # PhantomLM uses top-2 routing
            _, top_k_idx = torch.topk(router_probs, k=2, dim=-1)
            # Flatten and move to CPU
            routing_data[layer_idx].extend(top_k_idx.flatten().cpu().tolist())
        return hook

    # Attach hooks to the router in each MoE layer
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'router'):
            h = layer.ffn.router.register_forward_hook(get_hook(i))
            hooks.append(h)
            
    # Run the forward pass
    input_ids = tokenizer.encode(text, return_tensors='pt').to(next(model.parameters()).device)
    
    print(f"Running forward pass on {input_ids.shape[1]} tokens...")
    with torch.no_grad():
        model(input_ids)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Plot the results
    plot_distribution(routing_data, model.config.n_experts)

def plot_distribution(routing_data, n_experts):
    n_layers = len(routing_data)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3 * n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]
        
    for ax, (layer_idx, expert_choices) in zip(axes, routing_data.items()):
        # Count frequencies
        counts = [expert_choices.count(i) for i in range(n_experts)]
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]
        
        bars = ax.bar(range(n_experts), percentages, color='#1D3557', edgecolor='black')
        
        ax.set_title(f'MoE Layer {layer_idx} - Expert Selection Frequency', fontweight='bold')
        ax.set_ylabel('% of Tokens')
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f'Expert {i}' for i in range(n_experts)])
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, max(percentages) + 10)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
                    
    plt.xlabel('Experts')
    plt.tight_layout()
    plt.savefig('expert_routing_distribution.png', dpi=300)
    print("Saved plot to 'expert_routing_distribution.png'")

# Example Usage in Kaggle:
# sample_text = "Once upon a time, there was a little girl named Lily. She loved to play with her toy car. One day, the car broke and Lily was very sad. She asked her dad to fix it." * 10
# record_routing_distribution(model, tokenizer, sample_text)
"""
    with open('plot_expert_routing_kaggle.py', 'w') as f:
        f.write(code)

if __name__ == "__main__":
    plot_expert_routing_distribution('phantomlm_qat.pt')
