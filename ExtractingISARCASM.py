import pandas as pd
import requests

""" RETRIEVING THE ISARCASM TWITTER DATA SET FROM TWITTER ID FROM THE ISARCASM REPO AND TURN IT INTO TEXT 
"""


# USE the ISARCASM test and train csv files
df = pd.read_csv("your_dataset.csv")

# Extract the Tweet IDs column into a list
tweet_ids = df["tweet_id"].dropna().astype(str).tolist()

# Setup X API Authentication
BEARER_TOKEN = "YOUR_X_BEARER_TOKEN"
headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

# Batch request in chunks of 100 (X API maximum per request)
hydrated_tweets = []

for i in range(0, len(tweet_ids), 100):
    chunk = tweet_ids[i : i + 100]
    ids_string = ",".join(chunk)

    url = f"https://api.twitter.com/2/tweets?ids={ids_string}&tweet.fields=text"

    try:
        response = requests.get(url, headers=headers).json()

        if "data" in response:
            for tweet in response["data"]:
                hydrated_tweets.append(
                    {"tweet_id": tweet["id"], "text": tweet["text"]}
                )
    except Exception as e:
        print(f"Error fetching chunk starting at index {i}: {e}")

hydrated_df = pd.DataFrame(hydrated_tweets)
print(hydrated_df.head())
