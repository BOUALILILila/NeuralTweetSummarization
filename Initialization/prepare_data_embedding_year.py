## Training Data
import glob
import gzip
import errno
import json
import preprocessor as pre
from nltk.tokenize import RegexpTokenizer
import sys

#
##if save tweets per day
#
year=sys.argv[1]
tokenizer = RegexpTokenizer(r'\w+')
files = glob.glob(f'/data/Rawtweets/{year}/raw*.txt')
print(files)
right_data=[]
tweets=[]
for day_tweets in files:
    try:
        #
        day_processed_tweets=[]
        #
        tweets=[]
        with open(day_tweets, 'r') as f:
            tweets=f.readlines()
        print('tweets read from file:')
        for tweet in tweets:
            text=pre.remove_duplicated_letters(tweet)
            ## Preprocess, tokenize and stem
            text=pre.preProcessText(text)
            tokens=tokenizer.tokenize(text)
            tokens=[token for token in tokens if len(token)>1]
            stems=pre.stemTokens(tokens)
            if len(stems)>0:
                day_processed_tweets+=[stems] 
        #
        right_data+=day_processed_tweets
        #
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            print("Error with data files")

with open(f'/data/CORPUS/training_data_embeddings/train_data_{year}.json', 'w') as outfile:
    json.dump(right_data, outfile)