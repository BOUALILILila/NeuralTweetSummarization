import json 
import pandas as pd
import matchzoo as mz
from nltk.tokenize import RegexpTokenizer
     

def predict_similarity_scores(processed_tweets,n_top):
    tokenizer = RegexpTokenizer(r'\w+')
    
    pre=mz.engine.base_preprocessor.load_preprocessor('/projets/iris/PROJETS/lboualil/Preprocessors/MatchPy/MatchPy_full_2015-2016_fasttext_CBOW')
    model=mz.engine.base_model.load_model('/projets/iris/PROJETS/lboualil/Models/MatchPy/MatchPy_full_2015-2016_fasttext_CBOW')
    '''

    pre=mz.engine.base_preprocessor.load_preprocessor('/home/lila/CORPUS/Preprocessors/MatchPy_cbow')
    model=mz.engine.base_model.load_model('/home/lila/CORPUS/Models/MatchPy_cbow')
    '''
    tweets=pd.DataFrame.from_dict(processed_tweets)
    del processed_tweets
    tweets.drop_duplicates('tweetid','first',True)
    tweets.set_index('tweetid',inplace=True)
    #Retrieve topics
    #with open('/home/lila/CORPUS/left/left_2017.json') as f:
    with open("/projets/iris/PROJETS/lboualil/CORPUS/left/left_2017_nist_evaluated_qrels_real.json") as f:
        topics=json.load(f)
    rows=[]                       
    for topic in topics[:]:      
        for id, tweet in tweets.iterrows():
            #overlap=set(tokenizer.tokenize(topic['title'])) & set(tokenizer.tokenize(tweet['text']))
            #if len(overlap)>0:
                rows+=[{
                    'id_left': topic['topid'],
                    'text_left': topic['title'],
                    'id_right': str(id),
                    'text_right': tweet['text'],
                    'label':0
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
                                          fixed_length_right=128,
                                          batch_size=20)
    pred_x, pred_y = predict_generator[:]
    predictions=model.predict(pred_x)
    del data_pack
    i=0
    scores=[]
    x=pd.DataFrame(pred_x,columns=['id_left','id_right'])
    for index,row in x.iterrows():
        score={"id_left":row['id_left'], "id_right": row['id_right'],"text": tweets.loc[row['id_right'],'text'],"created_at": tweets.loc[row['id_right'],'created_at'], "score":predictions[i][0]}
        scores.append(score)
        i+=1
    del x
    pred_table=pd.DataFrame.from_dict(scores)
    del scores
    pred_table=pred_table.sort_values(['id_left','score'],ascending=[True,False])
    pred_table.set_index('id_left',inplace=True)
    topics=set(pred_table.index.values)
    df=pd.DataFrame()
    for topic in topics:
        df=pd.concat([df,pred_table.loc[topic,:].head(n_top)],sort=True)
    del pred_table
    del topics
    return df
        
                


