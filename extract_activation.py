import torch
import pickle
from transformer_lens import HookedTransformer

device = "cuda"
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=torch.bfloat16)
model.eval()
model.tokenizer.padding_side = "right"
PAD_ID = model.tokenizer.pad_token_id
TARGET_LAYER = 20

from huggingface_hub import login
login(token="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")

strong_classes = pickle.load(open("strong_classes.pkl", "rb"))

def get_layer_acts(texts, layer, batch_size=16):
    """Extract full activation vectors at one layer, last token."""
    acts = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        tokens = model.to_tokens(batch)
        mask = (tokens != PAD_ID)
        last_idx = mask.sum(dim=1) - 1
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n.endswith("hook_resid_post")
            )
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        rows = torch.arange(tokens.shape[0])
        acts.append(resid[rows, last_idx].float().cpu())
        del cache
    return torch.cat(acts)

print(f"Extracting layer {TARGET_LAYER}...")
layer_acts = {}
for name in ["sarcastic_surfpos", "paired_sincere_pos", "baseline_sincere_pos", "baseline_sincere_neg"]:
    print(f"  {name}...", flush=True)
    layer_acts[name] = get_layer_acts(strong_classes[name], TARGET_LAYER)

pickle.dump(layer_acts, open(f"layer{TARGET_LAYER}_acts.pkl", "wb"))
print("✓ Done")
