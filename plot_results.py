import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load both
projs_2b = pickle.load(open("./artifacts-2B/phase1_projections_strong.pkl", "rb"))
projs_9b = pickle.load(open("/workspace/strong_server_copy_GeometryofSarcasm/artifacts_9B/phase1_projections_strong.pkl", "rb"))

def plot_model(ax, projs, title, n_layers):
    def plot_class(name, color, label):
        m = projs[name].mean(axis=1)
        sem = projs[name].std(axis=1) / np.sqrt(projs[name].shape[1])
        ax.plot(m, color=color, linewidth=2, label=label)
        ax.fill_between(range(n_layers), m - 2*sem, m + 2*sem, color=color, alpha=0.15)
    
    plot_class("sarcastic_surfpos", "red", "Sarcastic (surface +)")
    plot_class("paired_sincere_pos", "green", "Rephrase (intended −)")
    plot_class("baseline_sincere_pos", "blue", "Sincere (+)")
    plot_class("baseline_sincere_neg", "gray", "Sincere (−)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection onto sentiment direction")
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
plot_model(ax1, projs_2b, "Gemma-2-2B: Sarcasm ≈ Surface (Failed)", 26)
plot_model(ax2, projs_9b, "Gemma-2-9B: Sarcasm + Rephrase track together", 42)
plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=150)
plt.show()
