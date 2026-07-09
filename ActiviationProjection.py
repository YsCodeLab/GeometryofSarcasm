import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


# --------------------------SETUP-------------------------------------------
device = "cuda"

# MATCH: use the same model 
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
model.eval()
# MATCH: need to match the last step
N_DIRECTION_TRAIN = 300  #

model.tokenizer.padding_side = "right"

N_LAYERS = model.cfg.n_layers           
PAD_ID = model.tokenizer.pad_token_id
#---------------------------------------------------------------------------


#------------------------------INPUT-----------------------------------------
# GRAB classes
classes = pickle.load(open("classes_separation0.5.pkl", "rb"))
# GRAB sentiment dir
#sentiment_dir = pickle.load(open("sentiment_dir.pkl", "rb"))
# make shift before sentiment is ready
all_acts = {}
for name, texts in proj_sets.items():
    print(f"extracting {name} ...", flush=True)
    all_acts[name] = get_resid_acts(texts)
pickle.dump(all_acts, open("phase1_acts.pkl", "wb"))
#---------------------------------------------------------------------------


# GRAB residual activation from the model
def get_resid_acts(texts, batch_size=16):
    """dict: layer -> float32 tensor [n_texts, d_model], last real token, resid_post."""
    acts = {l: [] for l in range(N_LAYERS)}
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        tokens = model.to_tokens(batch)                       # right-padded, BOS prepended
        mask = (tokens != PAD_ID)
        last_idx = mask.sum(dim=1) - 1                        # index of last real token
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n.endswith("hook_resid_post")
            )
        rows = torch.arange(tokens.shape[0])
        for l in range(N_LAYERS):
            resid = cache[f"blocks.{l}.hook_resid_post"]      # [batch, seq, d_model]
            acts[l].append(resid[rows, last_idx].float().cpu())
        del cache
    return {l: torch.cat(v) for l, v in acts.items()}

# GRAB CLASSES from DATAPREP
proj_sets = {
    "sarcastic_surfpos":    classes["sarcastic_surfpos"],
    "paired_sincere_pos":   classes["paired_sincere_pos"],
    "sarcastic_surfneg":    classes["sarcastic_surfneg"],
    "paired_sincere_neg":   classes["paired_sincere_neg"],
    "baseline_sincere_pos": classes["baseline_sincere_pos"][N_DIRECTION_TRAIN:],
    "baseline_sincere_neg": classes["baseline_sincere_neg"][N_DIRECTION_TRAIN:],
}



# sanity checking printout 
print("Class sizes being projected:")
for k, v in proj_sets.items():
    print(f"  {k}: {len(v)}")


# SANITY CHECK of classes(can be commented out)
print("Extracting activations...")
acts = {}
for class_name, texts in classes.items():
    print(f"  {class_name}... ", end="", flush=True)
    acts[class_name] = get_resid_acts(texts)
    print("")

# EXTRACT PROJECTION
all_projs = {}   # name -> array [N_LAYERS, n_texts] of per-text projections
for name, texts in proj_sets.items():
    print(f"extracting {name} ...", flush=True)
    a = get_resid_acts(texts)
    all_projs[name] = np.stack(
        [(a[l] @ sentiment_dir[l]).numpy() for l in range(N_LAYERS)]
    )
    del a
    


