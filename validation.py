import pickle
import numpy as np

#constant
artifacts_folder="artifacts_9B"


classes = pickle.load(open("classes_separation0.5.pkl", "rb"))
#
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def surface(t):
    return analyzer.polarity_scores(str(t))["compound"]

sarc_v = np.array([surface(t) for t in classes["sarcastic_surfpos"]])
base_v = np.array([surface(t) for t in classes["baseline_sincere_pos"]])

print(f"sarcastic VADER: {sarc_v.mean():.2f} ± {sarc_v.std():.2f}")
print(f"baseline  VADER: {base_v.mean():.2f} ± {base_v.std():.2f}")

sarc_neg_v = np.array([surface(t) for t in classes["sarcastic_surfneg"]])
base_neg_v = np.array([surface(t) for t in classes["baseline_sincere_neg"]])
print(f"sarcastic neg VADER: {sarc_neg_v.mean():.2f} ± {sarc_neg_v.std():.2f}")
print(f"baseline  neg VADER: {base_neg_v.mean():.2f} ± {base_neg_v.std():.2f}")

print(len(classes["sarcastic_surfneg"]))
print(len(classes["paired_sincere_neg"]))
print(len(classes["baseline_sincere_neg"]))

for i in range(3):
    print(f"sarcastic: {classes['sarcastic_surfneg'][i]}")
    print(f"rephrase:  {classes['paired_sincere_neg'][i]}")
    print()



import pickle
from scipy.stats import wilcoxon

all_projs = pickle.load(open("artifacts_9B/phase1_projections.pkl", "rb"))

sarc = all_projs["sarcastic_surfpos"]      # [layers, n]
intent = all_projs["paired_sincere_pos"]   # same order = same pairs, since both built from pairs_df in order
delta = sarc - intent

for l in [8, 12, 16, 20]:
    w = wilcoxon(delta[l])
    print(f"layer {l}: mean Δ={delta[l].mean():+.1f}, p={w.pvalue:.1e}")

#TEST 3
import numpy as np
N_LAYERS = delta.shape[0]
effect_sizes = []
for l in range(N_LAYERS):
    d = delta[l]
    cohens_d = d.mean() / d.std()   # standardized effect size for paired data
    effect_sizes.append(cohens_d)

import matplotlib.pyplot as plt
plt.plot(effect_sizes)
plt.xlabel("Layer"); plt.ylabel("Cohen's d (paired)")
plt.title("Effect size of literal-vs-intent gap, by layer")
plt.axhline(0, color='gray', linestyle='--')
plt.savefig(f"{artifacts_folder}/effect_size_by_layer.png")


# effect size on the raw unpaired comparison 
base = all_projs["baseline_sincere_pos"]
for l in [8, 12, 16, 20]:
    unpaired_d = (sarc[l].mean() - base[l].mean()) / np.sqrt((sarc[l].var() + base[l].var()) / 2)
    print(f"layer {l}: unpaired Cohen's d = {unpaired_d:.2f}")

