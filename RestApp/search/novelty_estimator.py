from sklearn import cluster
from sklearn import metrics
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from .Config import EMBED_DIM
import logging


tokenizer = RegexpTokenizer(r'\w+')


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
  

def delete_redudants(predictions,embed_model):    
    X=[]
    logging.info('\n# Redundancy reduction ...\n')
    for i,row in predictions.iterrows():
            X.append(get_sentence_vector(row['processed_text'], embed_model))
    dbscan=cluster.DBSCAN(eps=0.09,metric='cosine', min_samples=2).fit(X)
    labels = dbscan.labels_
    logging.info('\n# Labels\n')
    print(labels)
    predictions['label']=labels
    isolated_tweets=predictions[predictions.label==-1]
    predictions=predictions[predictions.label!=-1].drop_duplicates('label',keep='first')
    predictions=pd.concat([predictions,isolated_tweets],sort=True)
    predictions=predictions.sort_values('score',ascending=False)
    predictions.reset_index(inplace=True)
    predictions=predictions.drop(columns='processed_text')
    logging.info(f'\n# Redundancy reduction [OK]\n >> Length= {len(predictions)}\n')
    return predictions
            

