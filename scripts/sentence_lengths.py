import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Global style settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

LLAMA_COLOR = '#4c72b0'
QWEN_COLOR = '#c44e52'
FIRE_COLOR = '#7f8c8d'
BAR_ALPHA = 0.8
EDGE_COLOR = 'black'

# Paths
fire_raw_path = Path("data/05_archive/final_fire/fire_2k_sample.json")
llama_raw_dir = Path("data/05_archive/final_llama/processed")
qwen_raw_dir = Path("data/05_archive/final_qwen/processed")

strategies_raw = ["zero_shot", "few_shot", "entity", "template"]
strategy_labels = ["Zero-shot", "Few-shot", "Entity", "Template"]

# Helper function to get word counts
def get_word_counts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [len((item.get("text") or item.get("sentence") or "").split()) for item in data]

# Extract data
try:
    fire_counts = get_word_counts(fire_raw_path)
    fire_mean, fire_std = np.mean(fire_counts), np.std(fire_counts)

    llama_means, llama_stds = [], []
    qwen_means, qwen_stds = [], []

    for strat in strategies_raw:
        l_counts = get_word_counts(llama_raw_dir / f"{strat}_2k.json")
        q_counts = get_word_counts(qwen_raw_dir / f"{strat}_2k.json")

        llama_means.append(np.mean(l_counts))
        llama_stds.append(np.std(l_counts))

        qwen_means.append(np.mean(q_counts))
        qwen_stds.append(np.std(q_counts))

    print("All datasets loaded successfully!\n")
except Exception as e:
    print(f"Error loading files: {e}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(strategy_labels))
width = 0.35

# Plot Llama and Qwen side-by-side with error bars
bars1 = ax.bar(x - width / 2, llama_means, width, yerr=llama_stds,
               label='Llama 3.3', color=LLAMA_COLOR, alpha=BAR_ALPHA,
               edgecolor=EDGE_COLOR, ecolor=EDGE_COLOR, capsize=4)

bars2 = ax.bar(x + width / 2, qwen_means, width, yerr=qwen_stds,
               label='Qwen 2.5', color=QWEN_COLOR, alpha=BAR_ALPHA,
               edgecolor=EDGE_COLOR, ecolor=EDGE_COLOR, capsize=4)

# Plot FIRE Reference (Mean line + Std Dev shading)
ax.axhline(fire_mean, color=FIRE_COLOR, linestyle='--', linewidth=2.5, label='FIRE Reference')
ax.fill_between([-0.5, len(strategy_labels) - 0.5],
                fire_mean - fire_std, fire_mean + fire_std,
                color=FIRE_COLOR, alpha=0.15)

# Formatting
ax.set_ylabel('Average Word Count')
ax.set_title('Sentence Length Comparison', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(strategy_labels)
ax.set_xlim(-0.5, len(strategy_labels) - 0.5)

# Headroom for the legend and error bars
max_height = max(max(np.array(llama_means) + np.array(llama_stds)),
                 max(np.array(qwen_means) + np.array(qwen_stds)))
ax.set_ylim(0, max_height * 1.2)

# Legend at top center, horizontal (ncol=3)
ax.legend(loc='upper center', ncol=3, framealpha=0.9)

# Add numeric labels slightly above the error bars
def add_labels(bars, means, stds):
    for i, bar in enumerate(bars):
        yval = means[i]
        error = stds[i]
        # offset by +0.8 to clear the capsize of the error bar
        ax.text(bar.get_x() + bar.get_width() / 2, yval + error + 0.8,
                f'{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

add_labels(bars1, llama_means, llama_stds)
add_labels(bars2, qwen_means, qwen_stds)

sns.despine()
plt.tight_layout()

# Save
out_dir = Path("output/presentation/sentence_complexity")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "sentence_length_presentation.png", dpi=300, bbox_inches='tight')
plt.show()