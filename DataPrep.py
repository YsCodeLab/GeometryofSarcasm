

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# INPUT from ISARCASM
ROOT = os.getcwd()
csv_path = os.path.join(ROOT, "iSarcasmEval", "train", "train.En.csv")

df = pd.read_csv(csv_path)
df = df.dropna(subset=["tweet", "sarcastic"])

# CALC sentiment score for each tweet from the ISARCASM dataset
analyzer = SentimentIntensityAnalyzer()
def surface(t): 
    return analyzer.polarity_scores(str(t))["compound"]
df['tweet_sentiment'] = df['tweet'].apply(surface)

# DROP sarcastic entreis with no rephrasing 
pairs_df = df[(df['sarcastic'] == 1) & (df['rephrase'].notna())]
# GRAB sincere text
baseline_df = df[df['sarcastic'] == 0]

classes = {
    "sarcastic_surfpos":    pairs_df[pairs_df['tweet_sentiment'] > 0.5]['tweet'].tolist(),
    "paired_sincere_pos":   pairs_df[pairs_df['tweet_sentiment'] > 0.5]['rephrase'].tolist(),
    
    "sarcastic_surfneg":    pairs_df[pairs_df['tweet_sentiment'] < -0.5]['tweet'].tolist(),
    "paired_sincere_neg":   pairs_df[pairs_df['tweet_sentiment'] < -0.5]['rephrase'].tolist(),

    "baseline_sincere_pos": baseline_df[baseline_df['tweet_sentiment'] > 0.5]['tweet'].tolist(),
    "baseline_sincere_neg": baseline_df[baseline_df['tweet_sentiment'] < -0.5]['tweet'].tolist(),
}
# PRINT for check
print("--- Extracted Dataset Counts ---")
for category, texts in classes.items():
    print(f"{category}: {len(texts)}")

# PICKLE results
import pickle
pickle.dump(classes, open("classes_separation0.5.pkl", "wb"))
<<<<<<< HEAD

=======
>>>>>>> 43920dd2c1f8381a77b25f3aff23eba4dc1d3e0c
