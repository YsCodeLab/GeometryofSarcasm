import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("./iSarcasmEval/train/train.csv")

df.isna().sum()
df = df.dropna(subset=["tweet", "sarcastic"])

analyzer = SentimentIntensityAnalyzer()

# Calculate only the compound score directly into the dataframe
df['compound'] = df['tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Keep only strong negative (<= -0.2) and strong positive (>= 0.2)
df = df[(df['compound'] <= -0.2) | (df['compound'] >= 0.2)]

print(df.head())


#sentences = df['tweet']
#compls = []

#for sentence in sentences:
#    vs = analyzer.polarity_scores(sentence)
#    compls.append(vs['compound'])
#    print("{:-<65} {}".format(sentence, str(vs)))
#
#df['compound'] = compls
#
#df['positive'] = df['compound'] >= 0.05
#
## df['c'] = np.where(df['a'].map(len) > df['b'].map(len), df['a'].map(len), df['b'].map(len)) # Removed: 'a' and 'b' are undefined
#
#df['pos'] = (df['compound'] >= 0.05).astype(int)
#df['neg'] = (df['compound'] <= -0.05).astype(int)
