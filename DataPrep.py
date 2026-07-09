import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# input from ISARCASM
df = pd.read_csv("./iSarcasmEval/train/train.csv")
df = df.dropna(subset=["tweet", "sarcastic"])

analyzer = SentimentIntensityAnalyzer()

# use vader sentiment to calculate sentiment score for each tweet from the ISARCASM dataset
def surface(t): 
    return analyzer.polarity_scores(str(t))["compound"]

# grabbing Sarcastic Texts
sarc_texts = df[df['sarcastic'] == 1]['tweet'].tolist()

# Producing the Paired Control (Topic-Matched Sincere) of the sarcastic dataset
paired_nonsarc_texts = df[df['sarcastic'] == 1]['rephrase'].dropna().tolist()

# The Baseline Control (Naturally Sincere)
baseline_nonsarc_texts = df[df['sarcastic'] == 0]['tweet'].dropna().tolist()

# creating the matrix
classes = {
    # Sarcastic
    "sarcastic_surfpos":       [t for t in sarc_texts if surface(t) > 0.3],
    "sarcastic_surfneg":       [t for t in sarc_texts if surface(t) < -0.3],
    
    # Paired Sincere (The perfect mathematical control)
    "paired_sincere_pos":      [t for t in paired_nonsarc_texts if surface(t) > 0.3],
    "paired_sincere_neg":      [t for t in paired_nonsarc_texts if surface(t) < -0.3],
    
    # Baseline Sincere (The natural distribution)
    "baseline_sincere_pos":    [t for t in baseline_nonsarc_texts if surface(t) > 0.3],
    "baseline_sincere_neg":    [t for t in baseline_nonsarc_texts if surface(t) < -0.3],
}

# Display counts
for k, v in classes.items(): 
    print(f"{k}: {len(v)}")
