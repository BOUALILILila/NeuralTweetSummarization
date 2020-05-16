## Training Data
import glob
import gzip
import errno
import json
import preprocessor as pre
from nltk.tokenize import RegexpTokenizer


#
##if save tweets per day
#
tokenizer = RegexpTokenizer(r'\w+')
files = glob.glob('/projets/iris/CORPUS/DOCS/TREC_MICROBLOG_2018/RawtweetStem/*.txt')
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
        '''
        w=day_tweets.split('/')
        with open(f'/projets/iris/PROJETS/lboualil/CORPUS/training_data_embeddings/day_processed_tweets/{w[7]}.json', 'w') as outfile:
            json.dump(day_processed_tweets, outfile)
        '''
        right_data+=day_processed_tweets
        #
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            print("Error with data files")

with open('/projets/iris/PROJETS/lboualil/CORPUS/training_data_embeddings/train_data_2018.json', 'w') as outfile:
    json.dump(right_data, outfile)