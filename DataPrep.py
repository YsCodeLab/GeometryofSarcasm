import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# input from ISARCASM
ROOT=os.getcwd()
df = pd.read_csv(f"{ROOT}/iSarcasmEval/train/train.En.csv")
df = df.dropna(subset=["tweet", "sarcastic"])

analyzer = SentimentIntensityAnalyzer()

# use vader sentiment to calculate sentiment score for each tweet from the ISARCASM dataset
def surface(t): 
    return analyzer.polarity_scores(str(t))["compound"]

df['tweet_sentiment'] = df['tweet'].apply(surface)

pairs_df = df[(df['sarcastic'] == 1) & (df['rephrase'].notna())]
baseline_df = df[df['sarcastic'] == 0]

# 
classes = {
    "sarcastic_surfpos":    pairs_df[pairs_df['tweet_sentiment'] > 0.3]['tweet'].tolist(),
    "paired_sincere_pos":   pairs_df[pairs_df['tweet_sentiment'] > 0.3]['rephrase'].tolist(),
    
    "sarcastic_surfneg":    pairs_df[pairs_df['tweet_sentiment'] < -0.3]['tweet'].tolist(),
    "paired_sincere_neg":   pairs_df[pairs_df['tweet_sentiment'] < -0.3]['rephrase'].tolist(),

    "baseline_sincere_pos": baseline_df[baseline_df['tweet_sentiment'] > 0.3]['tweet'].tolist(),
    "baseline_sincere_neg": baseline_df[baseline_df['tweet_sentiment'] < -0.3]['tweet'].tolist(),
}



# ISOLATE only rows that have BOTH the sarcastic tweet and the rephrase
paired_df = df[(df['sarcastic'] == 1) & (df['rephrase'].notna())].copy()

# FILTER strictly based on the SARCASTIC text's surface sentiment
paired_df['sarc_score'] = paired_df['tweet'].apply(surface)

# FILTER remove neutral sentiments between -0.3 and 0.3
perfect_pos_pairs = paired_df[paired_df['sarc_score'] > 0.3]
perfect_neg_pairs = paired_df[paired_df['sarc_score'] < -0.3]

# Printing surface counts
print(f"Perfect Surface-Positive Pairs: {len(perfect_pos_pairs)}")
print(f"Perfect Surface-Negative Pairs: {len(perfect_neg_pairs)}")

# Display counts
for k, v in classes.items(): 
    print(f"{k}: {len(v)}")
