import requests
import vaderSentiment

import pandas as pd 
import numpy as np
import requests

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# USE the ISARCASM test and train csv files
df = pd.read_csv("./iSarcasmEval/train/train.csv")

# Drop all tweets that have 
df.isna().sum()

tweet_ids = df[subset=["tweet", "sarcastic"]].dropna()


df = df.dropna(subset=["tweet"])

# VaderSentiment Example
# --- examples -------
#sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
#             "VADER is smart, handsome, and funny!",  # punctuation emphasis handled correctly (sentiment intensity adjusted)
#             "VADER is very smart, handsome, and funny.", # booster words handled correctly (sentiment intensity adjusted)
#             "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
#             "VADER is VERY SMART, handsome, and FUNNY!!!", # combination of signals - VADER appropriately adjusts intensity
#             "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", # booster words & punctuation make this close to ceiling for score
#             "VADER is not smart, handsome, nor funny.",  # negation sentence example
#             "The book was good.",  # positive sentence
#             "At least it isn't a horrible book.",  # negated negative sentence with contraction
#             "The book was only kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
#             "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
#             "Today SUX!",  # negative slang with capitalization emphasis
#             "Today only kinda sux! But I'll get by, lol", # mixed sentiment example with slang and constrastive conjunction "but"
#             "Make sure you :) or :D today!",  # emoticons handled
#             "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # emojis handled
#             "Not bad at all"  # Capitalized negation
#             ]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))

#positive sentiment: compound score >= 0.05
#neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#negative sentiment: compound score <= -0.05

df['positive'] = df['compound'] >= 0.05

df['c'] = np.where(df['a'].map(len) > df['b'].map(len), df['a'].map(len), df['b'].map(len))

df['pos'] = (df['compound'] >= 0.05).astype(int)




