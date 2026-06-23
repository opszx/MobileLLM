import matplotlib.pyplot as plt

# Data from our Kaggle benchmark
contexts = [128, 256, 512, 1024, 2048]
phantom_vram = [824, 811, 824, 877, 1151]
transformer_vram = [811, 824, 848, 926, 1249]

# Create an aesthetically pleasing IEEE-style plot
plt.figure(figsize=(8, 6))

# Plot the Transformer (Red line, squares) showing quadratic growth
plt.plot(contexts, transformer_vram, marker='s', color='#E63946', linewidth=2.5, markersize=8, label='Pure Transformer Baseline')

# Plot PhantomLM (Blue line, circles) showing flatter Mamba scaling
plt.plot(contexts, phantom_vram, marker='o', color='#1D3557', linewidth=2.5, markersize=8, label='PhantomLM (Ours)')

# Formatting
plt.title('Peak VRAM vs. Context Length (76M Parameters)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Context Length (Tokens)', fontsize=12)
plt.ylabel('Peak VRAM (MB)', fontsize=12)

# Make the grid look professional
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

# Ensure the x-axis shows exactly our data points
plt.xticks(contexts)

# Highlight the final gap at 2048 tokens
plt.text(2048 - 150, 1249 + 15, '1249 MB', color='#E63946', fontweight='bold')
plt.text(2048 - 150, 1151 - 40, '1151 MB', color='#1D3557', fontweight='bold')

# Save the figure with high resolution for publication
plt.tight_layout()
plt.savefig('memory_scaling_figure.png', dpi=300)

print("Success! Saved publication-ready chart to 'memory_scaling_figure.png'")
