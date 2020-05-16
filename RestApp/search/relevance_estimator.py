import json 
import pandas as pd
import matchzoo as mz
from nltk.tokenize import RegexpTokenizer
import keras
import logging

def predict_similarity_scores(tweets,q,n_top):
    
    keras.backend.clear_session() 
    pre=mz.engine.base_preprocessor.load_preprocessor('./search/Preprocessors/MatchPy_full_fasttext')
    model=mz.engine.base_model.load_model('./search/Models/MatchPy_full_fasttext')
    
    tokenizer = RegexpTokenizer(r'\w+')
    rows=[]                     
    for id, tweet in tweets.iterrows():
                rows+=[{
                    'id_left': 1,
                    'text_left': q,
                    'id_right': tweet['id_str'],
                    'text_right': tweet['processed_text'],
                    'label':1
                }]
    if len(rows)==0:
        return None        
    data=pd.DataFrame.from_dict(rows)
    data_pack = mz.pack(data)
    del data
    data_pack.relation['label'] = data_pack.relation['label'].astype('float32')
    predict_pack_processed=pre.transform(data_pack)
    predict_generator = mz.DPoolDataGenerator(predict_pack_processed,
                                          fixed_length_left=10,
                                          fixed_length_right=40,
                                          batch_size=20)    
    logging.info('\n# Predictiong...\n')
    pred_x, pred_y = predict_generator[:]
    predictions=model.predict(pred_x)
    del data_pack
    i=0
    tweets.set_index('id_str',inplace=True)
    tweets['score']=0
    x=pd.DataFrame(pred_x,columns=['id_left','id_right'])
    for index,row in x.iterrows():
        tweets.loc[row['id_right'],'score']=predictions[i][0]
        i+=1
    del x
    tweets=tweets.sort_values(['score'],ascending=[False])
    # eliminer les scores negatifs
    logging.info('\n# Prediction [OK]\n')
    return tweets
'''
import prepare_tweets as pt
import novelty_estimator as nv
import gensim
if __name__ == "__main__":
        with open('./search/tweets.json') as f:
            tweets=json.load(f)
        tweets,n=pt.preporcess_submitted_tweets(tweets)
        print(tweets)
        scores=predict_similarity_scores(tweets,'algeria policy',10)
        print(scores)
        embed_model=gensim.models.KeyedVectors.load_word2vec_format('./search/Embeddings/vectors.txt', binary=False)
        print(nv.delete_redudants(scores,embed_model))

'''          

