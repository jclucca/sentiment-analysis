import tweepy
import pandas as pd
import configparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set the maximum results from query. Max is 100
MAX_RESULTS = 100
SCREEN_NAME = "addutiliti"

# read credentials from config file
config = configparser.ConfigParser()
config.read("config.ini")

api_key = config["twitter"]["api_key"]
api_key_secret = config["twitter"]["api_key_secret"]

access_token = config["twitter"]["access_token"]
access_token_secret = config["twitter"]["access_token_secret"]

bearer_token = "AAAAAAAAAAAAAAAAAAAAAMPhlQEAAAAAaxoii%2F6Vrm5bPswT86zBhKgohY0%3DIB80r9CtSAgajfTwuRisHW359X6zH1awZY2GyyapNm8afM4FaN"

# Authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def get_twitter_id():
    # Look up Twitter account id from screen name
    user = api.get_user(screen_name=SCREEN_NAME)
    return user.id


twitter_id = get_twitter_id()

# Create Tweepy client
client = tweepy.Client(bearer_token=bearer_token)

# Get last {MAX_RESULTS} tweets that mention the SCREEN_NAME into the tweets object
tweets = client.get_users_mentions(id=twitter_id, max_results=MAX_RESULTS,
                                   tweet_fields=['context_annotations', 'created_at', 'geo'])

# Create dataframe from tweet data
df = pd.DataFrame(tweets.data)

# Extract the tweet contents from dataframe to a list called 'data' for analysis
data = df["text"].tolist()

# Create an instance of SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()

# Create an empty list to store sentiment scores - scores will be dicts
sentiment_scores = []

# For each tweet in data, use sent_analyzer to return sentiment dicts in a list
for tweet in data:
    sentiment_scores.append(sent_analyzer.polarity_scores(tweet))

# Create df for list of sentiment scores and merge with df, then write to csv
sentiment_df = pd.DataFrame(sentiment_scores)
new_df = pd.concat([df, sentiment_df], axis=1)
new_df.to_csv("sentiment.csv")
