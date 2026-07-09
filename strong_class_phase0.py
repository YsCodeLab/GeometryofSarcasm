import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

device = "cuda"
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=torch.bfloat16)
model.eval()
model.tokenizer.padding_side = "right"
N_LAYERS = model.cfg.n_layers
PAD_ID = model.tokenizer.pad_token_id

# Load strong_classes
strong_classes = pickle.load(open("strong_classes.pkl", "rb"))

# Phase 0a: Extract residual stream activations
def get_resid_acts(texts, batch_size=16):
    acts = {l: [] for l in range(N_LAYERS)}
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        tokens = model.to_tokens(batch)
        mask = (tokens != PAD_ID)
        last_idx = mask.sum(dim=1) - 1
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n.endswith("hook_resid_post")
            )
        rows = torch.arange(tokens.shape[0])
        for l in range(N_LAYERS):
            resid = cache[f"blocks.{l}.hook_resid_post"]
            acts[l].append(resid[rows, last_idx].float().cpu())
        del cache
    return {l: torch.cat(v) for l, v in acts.items()}

# Phase 0b: Build sentiment direction on baseline (unchanged)
N_DIRECTION_TRAIN = 300
pos_texts = strong_classes["baseline_sincere_pos"][:N_DIRECTION_TRAIN]
neg_texts = strong_classes["baseline_sincere_neg"][:N_DIRECTION_TRAIN]

print("Building sentiment direction...")
pos_acts = get_resid_acts(pos_texts)
neg_acts = get_resid_acts(neg_texts)

sentiment_dir = {}
for l in range(N_LAYERS):
    d = pos_acts[l].mean(0) - neg_acts[l].mean(0)
    sentiment_dir[l] = d / d.norm()

# Phase 0c: Sanity check on held-out baseline
pos_test = get_resid_acts(strong_classes["baseline_sincere_pos"][N_DIRECTION_TRAIN:N_DIRECTION_TRAIN+150])
neg_test = get_resid_acts(strong_classes["baseline_sincere_neg"][N_DIRECTION_TRAIN:N_DIRECTION_TRAIN+100])

separations = []
for l in range(N_LAYERS):
    p = pos_test[l] @ sentiment_dir[l]
    n = neg_test[l] @ sentiment_dir[l]
    gap = (p.mean() - n.mean()) / torch.cat([p, n]).std()
    separations.append(gap.item())

plt.figure(figsize=(10, 4))
plt.plot(separations)
plt.axhline(1, color='r', linestyle='--', label='separation threshold (gap=1)')
plt.xlabel("layer"); plt.ylabel("separation (std units)")
plt.title("Sanity Check: Strong classes - Do sincere pos/neg separate?")
plt.legend()
plt.savefig("sanity_check_strong.png")
plt.show()

print("Separation by layer:")
for l in [5, 10, 15, 20, 25]:
    print(f"  layer {l}: {separations[l]:.2f}")

if max(separations[5:25]) < 1.0:
    print("\n❌ KILL SWITCH: Sentiment direction doesn't separate on strong classes.")
else:
    print(f"\n✓ PASS: Best separation {max(separations[5:25]):.2f}")
    pickle.dump(sentiment_dir, open("sentiment_dir_strong.pkl", "wb"))
    print("Saved sentiment_dir_strong.pkl")
