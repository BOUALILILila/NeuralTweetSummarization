## Training Data
import glob
import gzip
import errno
import json
import preprocessor as pre
from nltk.tokenize import RegexpTokenizer


#
##if save tweets per hour
#
files = glob.glob('/projets/iris/PROJETS/lboualil/statuses/statuses_2017/statuses.log.2017-*.gz')
with open('/projets/iris/PROJETS/lboualil/CORPUS/relation/relation_2017_nist.json') as rel:
    evaluated_tweets=json.load(rel)
evaluated_tweet_ids=[tweet['tweetid'] for tweet in evaluated_tweets]
evaluated_tweet_ids=set(evaluated_tweet_ids)
print(files)
right_data=[]
tweets=[]
#dates=set([str(d) for d in range(20,30)])
tokenizer = RegexpTokenizer(r'\w+')
for hour_tweets in files:
    try:
        #
        hour_processed_tweets=[]
        #
        tweets=[]
        w=hour_tweets.split('/')
        date=w[7].split('-')
        #if date[2] in dates:
        with gzip.open(hour_tweets, 'r') as f:
                tweets=f.readlines()
        print('tweets read from file:')
        print(tweets[0])
            #language, not a reply, not trash
        for line in tweets:
                try:
                    tweet=json.loads(line.decode('utf-8'))
                    print('json')
                    if tweet.get('id_str') in evaluated_tweet_ids:
                        if tweet.get('in_reply_to_status_id') is None and tweet.get('lang')=='en' :
                                text=tweet['text']
                                text=pre.remove_duplicated_letters(text)
                                if pre.isTrash(text) is False:
                                    ## Preprocess, tokenize and stem
                                    text=pre.preProcessText(text)
                                    tokens=tokenizer.tokenize(text)
                                    tokens=[token for token in tokens if len(token)>1]
                                    # at least 5 unique tokens
                                    if(len(set(tokens))>=5):
                                        stems=pre.stemTokens(tokens)
                                        text=" ".join([stem for stem in stems])
                                        #right_data+=[{"tweetid":tweet['id_str'],"text":text}]
                                        hour_processed_tweets+=[{"tweetid":tweet['id_str'],"text":text}]
                except ValueError as vexc:
                    print('not tweet json... moving next')
            #
        with open(f'/projets/iris/PROJETS/lboualil/CORPUS/tweets/tweets_2017/{w[7]}.json', 'w') as outfile:
                json.dump(hour_processed_tweets, outfile)
        right_data+=hour_processed_tweets
        #
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            print("Error with data files")

with open('/projets/iris/PROJETS/lboualil/CORPUS/right/right_2017_nist.json', 'w') as outfile:
    json.dump(right_data, outfile)