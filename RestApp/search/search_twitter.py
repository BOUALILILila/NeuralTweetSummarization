import tweepy
# assuming twitter_authentication.py contains each of the 4 oauth elements (1 per line)
from .twitter_authentication import API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from .Config import N_ITEMS
from datetime import datetime, timedelta


auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)


def get_tweets(q,since_date):
    # Collect tweets
    if since_date is None:
        since_date=(datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d')
    tweets = tweepy.Cursor(api.search,
                q=q,
                lang="en",
                since=since_date).items(N_ITEMS)
    return tweets
