import numpy as np
import pickle
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ---------- load ----------
projs_9b = pickle.load(open("/workspace/strong_server_copy_GeometryofSarcasm/artifacts_9B/phase1_projections_strong.pkl", "rb"))
projs_2b = pickle.load(open("/workspace/strong_server_copy_GeometryofSarcasm/artifacts-2B/phase1_projections_strong.pkl", "rb"))

# ---------- per-layer metrics ----------
def detection_p(projs, l):
    """One-sided Mann-Whitney: does sarcasm project LOWER than sincere positive?"""
    sarc = projs["sarcastic_surfpos"][l]
    pos  = projs["baseline_sincere_pos"][l]
    _, p = stats.mannwhitneyu(sarc, pos, alternative="less")
    return p

def resolution_ci(projs, l, n_boot=5000):
    """Where sarcasm sits between anchors (0=literal, 1=flipped), with bootstrap CI."""
    sarc = projs["sarcastic_surfpos"][l]
    pos  = projs["baseline_sincere_pos"][l]
    neg  = projs["baseline_sincere_neg"][l]
    res = []
    for _ in range(n_boot):
        s = np.random.choice(sarc, len(sarc), replace=True).mean()
        p = np.random.choice(pos,  len(pos),  replace=True).mean()
        n = np.random.choice(neg,  len(neg),  replace=True).mean()
        res.append((p - s) / (p - n))
    return np.mean(res), np.percentile(res, [2.5, 97.5])

def auroc(projs, l):
    """How well the 1-D sentiment axis separates sarcasm from sincere positive."""
    sarc = projs["sarcastic_surfpos"][l]
    pos  = projs["baseline_sincere_pos"][l]
    y = [1]*len(sarc) + [0]*len(pos)
    x = np.concatenate([sarc, pos])
    return roc_auc_score(y, -x)   # negate: sarcasm should project lower

def diff(projs, l):
    """Raw displacement of sarcasm from sincere positive, in axis units."""
    return projs["sarcastic_surfpos"][l].mean() - projs["baseline_sincere_pos"][l].mean()

# ---------- build + print a table for one model ----------
def print_table(projs, model_name):
    n_layers = projs["sarcastic_surfpos"].shape[0]
    print(f"\n{'='*72}")
    print(f"  {model_name}  ({n_layers} layers)")
    print(f"{'='*72}")
    print(f"{'Layer':>5} | {'diff':>7} | {'detect p':>10} | {'resolution [95% CI]':>24} | {'AUROC':>6}")
    print(f"{'-'*72}")
    rows = []
    for l in range(n_layers):
        d = diff(projs, l)
        p = detection_p(projs, l)
        m, (lo, hi) = resolution_ci(projs, l)
        a = auroc(projs, l)
        rows.append((l, d, p, m, lo, hi, a))
        print(f"{l:>5} | {d:>+7.1f} | {p:>10.2e} | {m:>6.2f} [{lo:>5.2f}, {hi:>5.2f}] | {a:>6.3f}")
    return rows

rows_2b = print_table(projs_2b, "Gemma-2-2B")
rows_9b = print_table(projs_9b, "Gemma-2-9B")

# ---------- resolution comparison plot ----------
def resolution_curve(projs):
    pos = projs["baseline_sincere_pos"].mean(axis=1)
    neg = projs["baseline_sincere_neg"].mean(axis=1)
    sarc = projs["sarcastic_surfpos"].mean(axis=1)
    return (pos - sarc) / (pos - neg)

plt.figure(figsize=(12, 5))
plt.plot(resolution_curve(projs_2b), label="Gemma-2-2B", linewidth=2, marker='o', markersize=4)
plt.plot(resolution_curve(projs_9b), label="Gemma-2-9B", linewidth=2, marker='s', markersize=4)
plt.axhline(0, color='blue', linestyle='--', alpha=0.3, label="At sincere positive (literal)")
plt.axhline(1, color='gray', linestyle='--', alpha=0.3, label="At sincere negative (flipped)")
plt.xlabel("Layer"); plt.ylabel("Resolution (0=literal, 1=flipped)")
plt.title("Scaling effect: Does the model learn to flip sarcastic sentiment?")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("resolution_comparison.png", dpi=150)


# make plots

import csv

# ---------- reduced CSV (selected layers) ----------
sel_2b = [5, 10, 13, 16, 20, 25]
sel_9b = [10, 15, 20, 27, 33, 41]

def save_reduced(projs, selected, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "diff", "detection_p", "resolution", "ci_low", "ci_high", "auroc"])
        for l in selected:
            d = projs["sarcastic_surfpos"][l].mean() - projs["baseline_sincere_pos"][l].mean()
            _, p = stats.mannwhitneyu(projs["sarcastic_surfpos"][l],
                                      projs["baseline_sincere_pos"][l], alternative="less")
            m, (lo, hi) = resolution_ci(projs, l)
            sarc = projs["sarcastic_surfpos"][l]; pos = projs["baseline_sincere_pos"][l]
            a = roc_auc_score([1]*len(sarc)+[0]*len(pos), -np.concatenate([sarc, pos]))
            w.writerow([l, f"{d:+.2f}", f"{p:.2e}", f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{a:.3f}"])

save_reduced(projs_2b, sel_2b, "table_2b_reduced.csv")
save_reduced(projs_9b, sel_9b, "table_9b_reduced.csv")
print("Saved table_2b_reduced.csv and table_9b_reduced.csv")

# ---------- helper: full per-layer arrays for a metric ----------
def metric_by_layer(projs, which):
    n = projs["sarcastic_surfpos"].shape[0]
    vals, los, his = [], [], []
    for l in range(n):
        if which == "diff":
            vals.append(projs["sarcastic_surfpos"][l].mean() - projs["baseline_sincere_pos"][l].mean())
        elif which == "detection_p":
            _, p = stats.mannwhitneyu(projs["sarcastic_surfpos"][l],
                                      projs["baseline_sincere_pos"][l], alternative="less")
            vals.append(p)
        elif which == "auroc":
            sarc = projs["sarcastic_surfpos"][l]; pos = projs["baseline_sincere_pos"][l]
            vals.append(roc_auc_score([1]*len(sarc)+[0]*len(pos), -np.concatenate([sarc, pos])))
        elif which == "resolution":
            m, (lo, hi) = resolution_ci(projs, l)
            vals.append(m); los.append(lo); his.append(hi)
    return np.array(vals), (np.array(los), np.array(his)) if los else None

# ---------- one plot per metric, both models, all layers ----------
# diff
v2, _ = metric_by_layer(projs_2b, "diff"); v9, _ = metric_by_layer(projs_9b, "diff")
plt.figure(figsize=(11,5))
plt.plot(range(len(v2)), v2, marker='o', ms=3, label="Gemma-2-2B")
plt.plot(range(len(v9)), v9, marker='s', ms=3, label="Gemma-2-9B")
plt.axhline(0, color='gray', ls='--', alpha=0.4)
plt.xlabel("Layer"); plt.ylabel("Displacement (sarcasm − sincere+)")
plt.title("Raw displacement from literal reading")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("metric_diff.png", dpi=150); plt.show()

# detection p
v2, _ = metric_by_layer(projs_2b, "detection_p"); v9, _ = metric_by_layer(projs_9b, "detection_p")
plt.figure(figsize=(11,5))
plt.plot(range(len(v2)), v2, marker='o', ms=3, label="Gemma-2-2B")
plt.plot(range(len(v9)), v9, marker='s', ms=3, label="Gemma-2-9B")
plt.axhline(0.05, color='red', ls='--', alpha=0.5, label="p = 0.05")
plt.gca().invert_yaxis()
plt.yscale("log"); plt.gca().invert_yaxis()
plt.xlabel("Layer"); plt.ylabel("Detection p (log, lower = stronger)")
plt.title("Detection significance by layer")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("metric_detection_p.png", dpi=150); plt.show()

# resolution with CI
v2, (lo2, hi2) = metric_by_layer(projs_2b, "resolution")
v9, (lo9, hi9) = metric_by_layer(projs_9b, "resolution")
plt.figure(figsize=(11,5))
plt.plot(range(len(v2)), v2, marker='o', ms=3, label="Gemma-2-2B", color="tab:blue")
plt.fill_between(range(len(v2)), lo2, hi2, alpha=0.15, color="tab:blue")
plt.plot(range(len(v9)), v9, marker='s', ms=3, label="Gemma-2-9B", color="tab:orange")
plt.fill_between(range(len(v9)), lo9, hi9, alpha=0.15, color="tab:orange")
plt.axhline(0, color='blue', ls='--', alpha=0.3, label="literal")
plt.axhline(1, color='gray', ls='--', alpha=0.3, label="fully flipped")
plt.ylim(-0.3, 1.2)
plt.xlabel("Layer"); plt.ylabel("Resolution (0=literal, 1=flipped)")
plt.title("Resolution toward intended meaning (95% CI)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("metric_resolution.png", dpi=150); plt.show()

# auroc
v2, _ = metric_by_layer(projs_2b, "auroc"); v9, _ = metric_by_layer(projs_9b, "auroc")
plt.figure(figsize=(11,5))
plt.plot(range(len(v2)), v2, marker='o', ms=3, label="Gemma-2-2B")
plt.plot(range(len(v9)), v9, marker='s', ms=3, label="Gemma-2-9B")
plt.axhline(0.5, color='red', ls='--', alpha=0.5, label="chance")
plt.xlabel("Layer"); plt.ylabel("AUROC (sentiment axis)")
plt.title("Sentiment-axis separability of sarcasm by layer")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("metric_auroc.png", dpi=150); plt.show()

print("Saved metric_diff.png, metric_detection_p.png, metric_resolution.png, metric_auroc.png")
