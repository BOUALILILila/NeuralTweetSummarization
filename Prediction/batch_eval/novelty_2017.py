from sklearn import cluster
from sklearn import metrics
import numpy as np
import gensim
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from datetime import datetime
import dateutil.parser

tokenizer = RegexpTokenizer(r'\w+')

EMBED_DIM=300 
EMBED_PATH='/projets/iris/PROJETS/lboualil/CORPUS/embeddings/full/fasttext/CBOW/vectors.txt'

def get_sentence_vector(sent,model):
    sent_vec =[]
    sent=tokenizer.tokenize(sent)
    for word in sent:
        try:
            sent_vec.append(model[word])
        except KeyError:
            sent_vec.append(np.random.rand(EMBED_DIM))
    length = len(sent_vec)
    summed = np.sum(sent_vec, axis=0)
    averaged = np.divide(summed, length)
    return averaged
  

def delete_redudants(predictions,seuil):
    manual=pd.DataFrame()
    embed_model=gensim.models.KeyedVectors.load_word2vec_format(EMBED_PATH, binary=False)
    topics=set(predictions.index.values)
    print(f'topics novelty (index)==  {topics}')
    summaries=pd.DataFrame()
    for topic in topics:
        tweets=predictions.loc[[topic],:]
        tweets=tweets.sort_values('score',ascending=False)
        tweets=tweets.head(10)
        tweets=tweets[tweets['score']>seuil]
        print(f'tweets.iteerows==  {tweets.head()}')
        X=[]
        for i,row in tweets.iterrows():
            print(row['id_right'])
            X.append(get_sentence_vector(row['text'], embed_model))
        if len(X)>0:
            dbscan=cluster.DBSCAN(eps=0.43,metric='cosine', min_samples=1).fit(X)
            labels = dbscan.labels_
            print(labels)
            tweets['label']=labels
            manual=pd.concat([manual,tweets],sort=True)
            #isolated_tweets=tweets[tweets.label==-1]
            #tweets=tweets[tweets.label!=-1].drop_duplicates('label',keep='first')
            #tweets=pd.concat([tweets,isolated_tweets],sort=True)
            #tweets['created_at']=tweets['created_at'].apply(lambda x: dateutil.parser.parse(x))
            #tweets=tweets.sort_values('created_at',ascending=True)
            tweets=tweets.sort_values('score',ascending=False)
            tweets=tweets.drop_duplicates('label',keep='first')
            tweets=tweets.sort_values('score',ascending=False)
            tweets.drop('text',1,inplace=True)
            tweets['rank']=range(1,len(tweets.index)+1)
            summaries=pd.concat([summaries,tweets],sort=True)
    return summaries
            


